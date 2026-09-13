"""Supervisor routing loop — the orchestration backbone.

Pattern: an LLM router agent reads the run state and emits a validated
RouterDecision; plain Python dispatches on that enum. LLM output determines
the next step (dynamic routing), but guardrails (iteration caps, retry
limits, terminal checks) live in deterministic code where they belong.

The loop is async internally and `run()` wraps it, so every agent run and the
shared `Deps` HTTP client live on one event loop — the client is opened and
closed on the same loop that the tools used it from.
"""

import asyncio
import logging
import uuid

from pydantic_ai import Agent, ModelRetry, RunContext

from deps import Deps, build_deps
from schemas import (
    BuyerCandidateBatch,
    BuyerLandscape,
    BuyerType,
    NextStep,
    RouterDecision,
    RunState,
    TargetProfile,
)
from tools.comps import comps_lookup
from tools.edgar import edgar_search
from tools.web import web_search

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
)
log = logging.getLogger("supervisor")

MAX_ITERATIONS = 12       # hard cap: the loop can never run away
MAX_DEEPEN_ROUNDS = 2     # low-confidence re-research is bounded

MODEL = "anthropic:claude-sonnet-4-6"  # pick per docs; cheap+fast is fine here

# Every research agent gets the same three tools: the specialists and the
# profiler ask the same kinds of question (who filed what, what was reported,
# what does it trade at) and differ only in what they do with the answers.
# Plain functions attached here rather than `@agent.tool` decorators, so one
# tool serves all three agents — docs/adr/0002-shared-deps-for-tool-resources.md.
RESEARCH_TOOLS = [edgar_search, web_search, comps_lookup]

# `defer_model_check=True` resolves MODEL at first run instead of at import.
# Without it, importing this module raises UserError when ANTHROPIC_API_KEY is
# unset, which would make the agents untestable — tests override the model and
# never reach a provider at all.
DEFER_MODEL_CHECK = True

# ---------------------------------------------------------------------------
# Agents. Each declares its output schema; Pydantic AI validates the model's
# response against it and retries with the validation errors on failure.
# Research tools are attached via constructor `tools=`, and reached through
# `ctx.deps` — docs/adr/0002-shared-deps-for-tool-resources.md.
# ---------------------------------------------------------------------------

router_agent = Agent(
    MODEL,
    output_type=RouterDecision,
    instructions=(
        "You are the supervisor of a buyer-landscape analysis pipeline for an "
        "investment bank. Given the current run state, choose the single next "
        "step.\n"
        "Rules of thumb:\n"
        "- No profile yet -> profile_target.\n"
        "- Profile exists but a buyer list is empty -> find that buyer type.\n"
        "- Many low-confidence buyers and deepen rounds remain -> deepen_research.\n"
        "- Both lists populated with adequate confidence -> synthesize.\n"
        "- Landscape exists but CRM not written -> write_to_crm.\n"
        "- Everything complete -> done."
    ),
    retries=2,  # validation-failure retries
    defer_model_check=DEFER_MODEL_CHECK,
)

profiler_agent = Agent(
    MODEL,
    deps_type=Deps,
    output_type=TargetProfile,
    instructions=(
        "Build a structured profile of the target company. Use your tools "
        "(web search, SEC EDGAR) and cite sources. If revenue is unknowable, "
        "say 'unknown' — do not guess."
    ),
    tools=RESEARCH_TOOLS,
    retries=2,
    defer_model_check=DEFER_MODEL_CHECK,
)

synthesis_agent = Agent(
    MODEL,
    output_type=BuyerLandscape,
    instructions=(
        "Assemble the final buyer landscape from the profile and candidate "
        "lists provided. Rank by fit_score. Write a banker-readable summary. "
        "Do not invent buyers not present in the inputs."
    ),
    retries=2,
    defer_model_check=DEFER_MODEL_CHECK,
)

STRATEGIC_INSTRUCTIONS = (
    "You source STRATEGIC acquirers: operating companies that would buy the "
    "target for fit with a business they already run.\n"
    "Work from evidence, not recall. Use your tools: EDGAR full-text search "
    "for what acquirers have filed and bought, web search for announced deals "
    "and stated strategy, comps lookup for whether a public acquirer can "
    "actually pay.\n"
    "For each buyer give the adjacency (product, channel, or geography), the "
    "precedent activity that shows appetite, and the basis for ability to pay. "
    "Cite what you found. A buyer you cannot evidence is a buyer you leave out."
)

SPONSOR_INSTRUCTIONS = (
    "You source FINANCIAL SPONSORS: private equity firms and similar "
    "investors that would buy the target as a platform or a bolt-on to a "
    "portfolio company.\n"
    "Work from evidence, not recall. Use your tools: web search for fund "
    "activity, stated theses, and portfolio companies, EDGAR full-text search "
    "for filed deal documents, comps lookup for public comparables that frame "
    "the multiple.\n"
    "For each sponsor give the thesis match, the portfolio company it would "
    "bolt onto (or why it is a platform), fund vintage or dry-powder evidence "
    "for ability to pay, and precedent deals in the sector. Cite what you "
    "found. A sponsor you cannot evidence is a sponsor you leave out."
)


def build_specialist_agent(
    buyer_type: BuyerType, instructions: str
) -> Agent[Deps, BuyerCandidateBatch]:
    """Build one buyer-sourcing specialist.

    Strategic and sponsor sourcing differ in what the model is told to look
    for and in the buyer type it must come back with — never in wiring. One
    construction path so a tool added here reaches both, rather than two
    near-copies that drift apart.
    """
    agent = Agent(
        MODEL,
        deps_type=Deps,
        output_type=BuyerCandidateBatch,
        instructions=instructions,
        tools=RESEARCH_TOOLS,
        retries=2,
        defer_model_check=DEFER_MODEL_CHECK,
    )

    @agent.output_validator
    def check_batch_matches_agent(
        ctx: RunContext[Deps], batch: BuyerCandidateBatch
    ) -> BuyerCandidateBatch:
        """Reject a batch sourced against the wrong brief.

        `BuyerCandidateBatch` already checks its candidates against its own
        declared `buyer_type`; only the agent knows which type was asked for,
        so that check lives here. `ModelRetry` sends the model back to work
        instead of letting sponsors land in the strategic list.
        """
        if batch.buyer_type != buyer_type:
            raise ModelRetry(
                f"This batch is declared {batch.buyer_type.value}, but you were "
                f"asked for {buyer_type.value} buyers. Re-source the list as "
                f"{buyer_type.value} buyers."
            )
        return batch

    return agent


strategic_agent = build_specialist_agent(BuyerType.STRATEGIC, STRATEGIC_INSTRUCTIONS)
sponsor_agent = build_specialist_agent(BuyerType.FINANCIAL_SPONSOR, SPONSOR_INSTRUCTIONS)

# The dispatch loop indexes this rather than branching on the buyer type twice.
SPECIALIST_AGENTS: dict[BuyerType, Agent[Deps, BuyerCandidateBatch]] = {
    BuyerType.STRATEGIC: strategic_agent,
    BuyerType.FINANCIAL_SPONSOR: sponsor_agent,
}


# ---------------------------------------------------------------------------
# Dispatch — deterministic. The LLM chooses the step; Python executes it.
# ---------------------------------------------------------------------------

def _specialist_prompt(profile: TargetProfile) -> str:
    """The brief a specialist works from: the profile and nothing else.

    The target input is deliberately left out — the profiler has already
    resolved it into something verified, and handing the specialist the raw
    user string invites it to re-profile rather than source buyers.
    """
    return f"Source buyers for this target.\n\nPROFILE:\n{profile.model_dump_json(indent=2)}"


async def _source_buyers(state: RunState, deps: Deps, buyer_type: BuyerType) -> None:
    """Run the specialist for `buyer_type` and file its candidates.

    Raises rather than returning a failure: the caller turns any exception
    into a `RunState.errors` entry the router can re-route on.
    """
    if state.profile is None:
        raise RuntimeError("no profile yet; profile the target before sourcing buyers")

    agent = SPECIALIST_AGENTS[buyer_type]
    result = await agent.run(_specialist_prompt(state.profile), deps=deps)
    batch = result.output

    # The agent's own output validator already checks this, so reaching here
    # means the dispatch table is wired to the wrong agent — a bug in this
    # module, not a bad model response. Cheap to check, and it keeps a
    # miswiring from quietly filing sponsors under strategic buyers.
    if batch.buyer_type != buyer_type:
        raise RuntimeError(
            f"specialist for {buyer_type.value} returned a {batch.buyer_type.value} batch"
        )

    # Replace rather than extend: a re-run of this step re-sources the list
    # from scratch, and appending would duplicate every buyer found twice.
    # (Deepening low-confidence buyers is its own step — see CONTEXT.md.)
    if buyer_type is BuyerType.STRATEGIC:
        state.strategic_buyers = list(batch.candidates)
    else:
        state.sponsor_buyers = list(batch.candidates)

    log.info(
        "sourced %d %s buyers: run_id=%s",
        len(batch.candidates), buyer_type.value, state.run_id,
    )


async def _run(target_input: str) -> RunState:
    state = RunState(run_id=uuid.uuid4().hex[:8], target_input=target_input)
    deps = build_deps(state.run_id)
    log.info("run started", extra={"run_id": state.run_id})

    try:
        for iteration in range(MAX_ITERATIONS):
            decision = (await router_agent.run(state.summary_for_router())).output
            log.info(
                "routing: iter=%d step=%s reason=%r run_id=%s",
                iteration, decision.next_step.value, decision.reason, state.run_id,
            )
            state.steps_taken.append(decision.next_step.value)

            try:
                if decision.next_step == NextStep.DONE:
                    break

                elif decision.next_step == NextStep.PROFILE_TARGET:
                    state.profile = (
                        await profiler_agent.run(target_input, deps=deps)
                    ).output

                elif decision.next_step == NextStep.FIND_STRATEGIC_BUYERS:
                    await _source_buyers(state, deps, BuyerType.STRATEGIC)

                elif decision.next_step == NextStep.FIND_SPONSOR_BUYERS:
                    await _source_buyers(state, deps, BuyerType.FINANCIAL_SPONSOR)

                elif decision.next_step == NextStep.DEEPEN_RESEARCH:
                    if state.deepen_rounds_used >= MAX_DEEPEN_ROUNDS:
                        state.errors.append("deepen_research requested past cap; ignoring")
                        continue  # guardrail beats LLM enthusiasm
                    state.deepen_rounds_used += 1
                    ...  # re-run low-confidence candidates with a focused prompt

                elif decision.next_step == NextStep.SYNTHESIZE:
                    prompt = (
                        f"PROFILE:\n{state.profile.model_dump_json() if state.profile else 'MISSING'}\n\n"
                        f"STRATEGIC:\n{[b.model_dump() for b in state.strategic_buyers]}\n\n"
                        f"SPONSORS:\n{[b.model_dump() for b in state.sponsor_buyers]}"
                    )
                    state.landscape = (await synthesis_agent.run(prompt)).output

                elif decision.next_step == NextStep.WRITE_TO_CRM:
                    ...  # crm.write_landscape(state.landscape); state.crm_written = True

            except Exception as exc:  # noqa: BLE001 — degrade, don't crash
                log.exception("step failed: %s (run_id=%s)", decision.next_step, state.run_id)
                state.errors.append(f"{decision.next_step.value}: {exc}")
                # The router sees the error in the next state summary and can
                # re-route (retry the step, skip it, or finish degraded).

        else:
            log.warning("hit MAX_ITERATIONS without DONE (run_id=%s)", state.run_id)
    finally:
        await deps.http_client.aclose()

    log.info("run finished: steps=%s errors=%d", state.steps_taken, len(state.errors))
    return state


def run(target_input: str) -> RunState:
    """Sync entry point — the async loop is an implementation detail."""
    return asyncio.run(_run(target_input))


if __name__ == "__main__":
    final = run("Chart Industries — cryogenic equipment for LNG and hydrogen")
    if final.landscape:
        print(final.landscape.model_dump_json(indent=2))

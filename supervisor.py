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
from datetime import UTC, datetime
import logging
from pathlib import Path
import uuid

from pydantic import TypeAdapter
from pydantic_ai import Agent, ModelRetry, RunContext, Tool
from pydantic_ai.tools import ToolDefinition

from deps import Deps, build_deps
from llm import build_model, run_agent
from schemas import (
    MIN_LANDSCAPE_BUYERS,
    BuyerCandidate,
    BuyerCandidateBatch,
    BuyerLandscape,
    BuyerType,
    DeepenedBatch,
    LandscapeSummary,
    NextStep,
    RouterDecision,
    RunState,
    TargetProfile,
    name_key,
)
from tools.comps import comps_lookup
from tools.crm import write_to_crm
from tools.edgar import edgar_search
from tools.web import web_search

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
)
log = logging.getLogger("supervisor")

MAX_ITERATIONS = 12       # hard cap: the loop can never run away
MAX_DEEPEN_ROUNDS = 2     # low-confidence re-research is bounded
RESEARCH_TOOL_CALL_BUDGET = 20  # research tool calls per agent run

MODEL = build_model()  # model choice and its retry/timeout policy live in llm.py


async def within_research_budget(
    ctx: RunContext[Deps], tool_def: ToolDefinition
) -> ToolDefinition | None:
    """Offer a research tool only while the run has budget left.

    Every model request resends the whole tool history, so an agent that
    researches until it feels done pays for its growing context again on every
    call — the first live run's strategic step made 73 calls and read 1.46M
    input tokens before answering. Past the budget the tools are withdrawn and
    the model has to answer with the evidence it has: a thinner result rather
    than a failed step. Checked before each request, so a response issuing
    several calls at once can overshoot by that one batch.
    """
    if ctx.usage.tool_calls >= RESEARCH_TOOL_CALL_BUDGET:
        return None
    return tool_def


# Every research agent gets the same three tools: the specialists and the
# profiler ask the same kinds of question (who filed what, what was reported,
# what does it trade at) and differ only in what they do with the answers.
# Plain functions attached here rather than `@agent.tool` decorators, so one
# tool serves all three agents — docs/adr/0002-shared-deps-for-tool-resources.md.
RESEARCH_TOOLS = [
    Tool(edgar_search, prepare=within_research_budget),
    Tool(web_search, prepare=within_research_budget),
    Tool(comps_lookup, prepare=within_research_budget),
]

# ---------------------------------------------------------------------------
# Agents. Each declares its output schema; Pydantic AI validates the model's
# response against it and retries with the validation errors on failure.
# Research tools are attached via constructor `tools=`, and reached through
# `ctx.deps` — docs/adr/0002-shared-deps-for-tool-resources.md.
# ---------------------------------------------------------------------------

router_agent = Agent(
    MODEL,
    name="router",
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
)

profiler_agent = Agent(
    MODEL,
    name="profiler",
    deps_type=Deps,
    output_type=TargetProfile,
    instructions=(
        "Build a structured profile of the target company. Use your tools "
        "(web search, SEC EDGAR) and cite sources. If revenue is unknowable, "
        "say 'unknown' — do not guess."
    ),
    tools=RESEARCH_TOOLS,
    retries=2,
)

synthesis_agent = Agent(
    MODEL,
    name="synthesis",
    output_type=LandscapeSummary,
    instructions=(
        "Write the banker-readable summary of a buyer landscape. You are given "
        "the target's profile and both buyer lists, already ranked by fit_score. "
        "Summarize who the most credible buyers are and why, across strategic "
        "acquirers and financial sponsors, drawing only on the rationale and "
        "signals provided. Refer only to buyers in the lists."
    ),
    retries=2,
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


def build_buyer_agent[OutputT](
    output_type: type[OutputT], instructions: str, name: str
) -> Agent[Deps, OutputT]:
    """The wiring every buyer-facing agent shares.

    Sourcing strategic buyers, sourcing sponsors and re-researching either
    list differ in what the model is told to look for and in what it must
    hand back — never in wiring. One construction path so a tool or a retry
    setting added here reaches all four, rather than near-copies that drift
    apart.
    """
    return Agent(
        MODEL,
        name=name,
        deps_type=Deps,
        output_type=output_type,
        instructions=instructions,
        tools=RESEARCH_TOOLS,
        retries=2,
    )


def build_specialist_agent(
    buyer_type: BuyerType, instructions: str
) -> Agent[Deps, BuyerCandidateBatch]:
    """Build one buyer-sourcing specialist, held to the type it was asked for."""
    agent = build_buyer_agent(
        BuyerCandidateBatch, instructions, name=f"{buyer_type.value}_specialist"
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

DEEPEN_INSTRUCTIONS = (
    "You re-research buyer candidates that were sourced with LOW confidence, "
    "and decide which of them belong in the landscape at all.\n"
    "Work only from what you can find now. Use your tools: EDGAR full-text "
    "search for what the buyer has filed and bought, web search for announced "
    "deals and stated strategy, comps lookup for whether it can pay.\n"
    "Confirm a candidate only when you found new, citable evidence for it — "
    "then hand back the buyer rewritten around that evidence, with the new "
    "signals and sources, a fit_score that reflects them, and the confidence "
    "the evidence actually supports. Refute it when the evidence is absent, "
    "thin, or contradicts the original rationale: a refuted buyer is dropped "
    "from the landscape, which is the right outcome for a name nobody can "
    "substantiate.\n"
    "Report on every candidate you are given, keep each name exactly as it "
    "was given to you, and introduce no new buyers — sourcing is someone "
    "else's step."
)


def build_deepen_agent(buyer_type: BuyerType) -> Agent[Deps, DeepenedBatch]:
    """Build the deepen-research pass for one buyer list.

    Separate from `build_specialist_agent` because the two ask the model for
    different things: a specialist returns buyers, this returns verdicts on
    buyers it was handed. One agent per buyer type so the output validator can
    hold the findings to the list they were run against.
    """
    agent = build_buyer_agent(
        DeepenedBatch, DEEPEN_INSTRUCTIONS, name=f"{buyer_type.value}_deepen"
    )

    @agent.output_validator
    def check_findings_match_agent(
        ctx: RunContext[Deps], batch: DeepenedBatch
    ) -> DeepenedBatch:
        """Reject findings filed against the wrong list.

        The findings are merged into one buyer list by name, so a batch
        declared as the other type would drop every low-confidence name it was
        asked about — a silent deletion rather than a visible failure.
        """
        if batch.buyer_type != buyer_type:
            raise ModelRetry(
                f"These findings are declared {batch.buyer_type.value}, but you were "
                f"asked about {buyer_type.value} buyers. Re-report them as "
                f"{buyer_type.value} findings."
            )
        return batch

    return agent


strategic_deepen_agent = build_deepen_agent(BuyerType.STRATEGIC)
sponsor_deepen_agent = build_deepen_agent(BuyerType.FINANCIAL_SPONSOR)

DEEPEN_AGENTS: dict[BuyerType, Agent[Deps, DeepenedBatch]] = {
    BuyerType.STRATEGIC: strategic_deepen_agent,
    BuyerType.FINANCIAL_SPONSOR: sponsor_deepen_agent,
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
    batch = await run_agent(
        agent, _specialist_prompt(state.profile), run_id=state.run_id, deps=deps
    )

    # The agent's own output validator already checks this, so reaching here
    # means the dispatch table is wired to the wrong agent — a bug in this
    # module, not a bad model response. Cheap to check, and it keeps a
    # miswiring from quietly filing sponsors under strategic buyers.
    if batch.buyer_type != buyer_type:
        raise RuntimeError(
            f"specialist for {buyer_type.value} returned a {batch.buyer_type.value} batch"
        )

    state.record_buyers(buyer_type, batch.candidates)
    log.info(
        "sourced %d %s buyers: run_id=%s",
        len(batch.candidates), buyer_type.value, state.run_id,
    )


_CANDIDATE_LIST_JSON = TypeAdapter(list[BuyerCandidate])


def _deepen_prompt(profile: TargetProfile, candidates: list[BuyerCandidate]) -> str:
    """The brief a deepen pass works from: the profile and the shaky names.

    Only the low-confidence candidates are handed over. Including the rest
    would invite the model to re-litigate buyers that are already evidenced,
    and spend a bounded round on work the landscape does not need.
    """
    return (
        "Re-research these low-confidence buyer candidates against the target "
        "below. Confirm or refute each one.\n\n"
        f"PROFILE:\n{profile.model_dump_json(indent=2)}\n\n"
        "LOW-CONFIDENCE CANDIDATES:\n"
        f"{_CANDIDATE_LIST_JSON.dump_json(candidates, indent=2).decode()}"
    )


async def _deepen_one_list(
    state: RunState, deps: Deps, buyer_type: BuyerType, profile: TargetProfile
) -> None:
    """Re-research one list's low-confidence candidates and fold the result in."""
    targets = state.low_confidence_buyers(buyer_type)
    agent = DEEPEN_AGENTS[buyer_type]
    batch = await run_agent(
        agent, _deepen_prompt(profile, targets), run_id=state.run_id, deps=deps
    )

    # As in `_source_buyers`: the agent's validator already checked this, so
    # reaching here means DEEPEN_AGENTS is miswired.
    if batch.buyer_type != buyer_type:
        raise RuntimeError(
            f"deepen pass for {buyer_type.value} returned {batch.buyer_type.value} findings"
        )

    # A candidate nothing came back for is dropped, so a pass that answers
    # about the wrong names would quietly delete the list it was meant to
    # resolve. Failing here costs the round and keeps the candidates, which
    # the router can see and act on; a silent deletion it could not.
    asked = {name_key(c.name) for c in targets}
    answered = {name_key(f.original_name) for f in batch.findings}
    if answered != asked:
        raise RuntimeError(
            f"deepen pass for {buyer_type.value} answered about the wrong candidates: "
            f"unasked={sorted(answered - asked)}, unanswered={sorted(asked - answered)}"
        )

    before = len(state.buyers_for(buyer_type))
    state.apply_deepening(buyer_type, batch.findings)
    after = len(state.buyers_for(buyer_type))
    log.info(
        "deepened %d low-confidence %s buyers: %d dropped, %d remain: run_id=%s",
        len(targets), buyer_type.value, before - after, after, state.run_id,
    )


async def _deepen_research(state: RunState, deps: Deps) -> None:
    """Run one bounded deepen-research round over every list that needs one.

    The cap counts rounds, not agent runs: a run where both specialists came
    back shaky gets both lists re-researched together, for the price of the
    one round the router asked for. The two guardrails come first so a round
    the loop refuses to run, or has no work for, costs nothing — the router
    sees the error in the next state summary and routes somewhere useful.

    One list's pass failing is recorded and stepped over rather than raised,
    so a dead source on the strategic side does not leave the sponsor list
    carrying names nobody substantiated.
    """
    if state.deepen_rounds_used >= MAX_DEEPEN_ROUNDS:
        state.errors.append("deepen_research requested past the round cap; ignoring")
        return  # guardrail beats LLM enthusiasm

    pending = [bt for bt in BuyerType if state.low_confidence_buyers(bt)]
    if not pending:
        state.errors.append(
            "deepen_research requested with no low-confidence candidates; ignoring"
        )
        return

    if state.profile is None:
        raise RuntimeError("no profile yet; profile the target before deepening research")

    state.deepen_rounds_used += 1
    for buyer_type in pending:
        try:
            await _deepen_one_list(state, deps, buyer_type, state.profile)
        except Exception as exc:  # noqa: BLE001 — degrade, don't crash
            log.exception(
                "deepen failed for %s (run_id=%s)", buyer_type.value, state.run_id
            )
            state.errors.append(f"deepen_research ({buyer_type.value}): {exc}")


def _ranked(buyers: list[BuyerCandidate]) -> list[BuyerCandidate]:
    """Highest fit first. `sorted` is stable, so buyers that tie keep the order
    they were sourced in."""
    return sorted(buyers, key=lambda buyer: buyer.fit_score, reverse=True)


def _synthesis_prompt(
    profile: TargetProfile, strategic: list[BuyerCandidate], sponsors: list[BuyerCandidate]
) -> str:
    """The buyers go in already ranked, so the summary is written against the
    order the landscape will carry."""
    return (
        f"PROFILE:\n{profile.model_dump_json(indent=2)}\n\n"
        f"STRATEGIC BUYERS (ranked):\n{_CANDIDATE_LIST_JSON.dump_json(strategic, indent=2).decode()}\n\n"
        f"FINANCIAL SPONSORS (ranked):\n{_CANDIDATE_LIST_JSON.dump_json(sponsors, indent=2).decode()}"
    )


async def _synthesize(state: RunState) -> None:
    """Assemble the landscape from run state, with the model writing only its
    summary — docs/adr/0005-synthesis-writes-only-the-summary.md.

    Everything but the summary already exists and has been validated by the
    time synthesis is routed to, so the model never re-types a buyer: it
    cannot drop, rename or invent one, and the step's output stays small
    enough to finish inside the request timeout. The guardrails come first so
    a state that could never become a landscape costs no model call.
    """
    if state.profile is None:
        raise RuntimeError("no profile yet; profile the target before synthesizing")

    buyer_count = len(state.strategic_buyers) + len(state.sponsor_buyers)
    if buyer_count < MIN_LANDSCAPE_BUYERS:
        raise RuntimeError(
            f"only {buyer_count} buyers across both lists; a landscape needs at least "
            f"{MIN_LANDSCAPE_BUYERS}, so source more buyers before synthesizing"
        )

    strategic = _ranked(state.strategic_buyers)
    sponsors = _ranked(state.sponsor_buyers)
    written = await run_agent(
        synthesis_agent,
        _synthesis_prompt(state.profile, strategic, sponsors),
        run_id=state.run_id,
    )
    state.landscape = BuyerLandscape(
        target=state.profile,
        strategic_buyers=strategic,
        sponsor_buyers=sponsors,
        summary=written.summary,
    )


async def _write_landscape_to_crm(state: RunState, deps: Deps) -> None:
    """Persist the run's finished landscape to the mock CRM.

    The already-written guardrail comes first because the writer always
    appends a fresh Account: a router that asks twice would otherwise log one
    run as two analyses. Raises when there is nothing to write; the caller
    records the error and the router re-routes to synthesis.
    """
    if state.crm_written:
        state.errors.append("write_to_crm requested but the CRM is already written; ignoring")
        return

    if state.landscape is None:
        raise RuntimeError("no landscape yet; synthesize before writing to the CRM")

    # sqlite3 blocks; a thread keeps the event loop (and the shared HTTP
    # client living on it) free while the write runs.
    await asyncio.to_thread(write_to_crm, state.landscape, state.run_id, deps.crm_db_path)
    state.crm_written = True


async def dispatch(state: RunState, deps: Deps, step: NextStep) -> None:
    """Execute one routed step against the run state.

    Separate from the routing loop so any step can be run on its own against
    a hand-seeded state — writing a landscape to the CRM should not need a
    profile, two specialists and a synthesis run first to be exercised.
    Raises on failure; the loop owns turning that into a routable error, and
    owns `DONE` too, since finishing is a loop decision rather than a step.
    """
    if step == NextStep.PROFILE_TARGET:
        state.profile = await run_agent(
            profiler_agent, state.target_input, run_id=state.run_id, deps=deps
        )

    elif step == NextStep.FIND_STRATEGIC_BUYERS:
        await _source_buyers(state, deps, BuyerType.STRATEGIC)

    elif step == NextStep.FIND_SPONSOR_BUYERS:
        await _source_buyers(state, deps, BuyerType.FINANCIAL_SPONSOR)

    elif step == NextStep.DEEPEN_RESEARCH:
        await _deepen_research(state, deps)

    elif step == NextStep.SYNTHESIZE:
        await _synthesize(state)

    elif step == NextStep.WRITE_TO_CRM:
        await _write_landscape_to_crm(state, deps)


def _save_run_record(state: RunState, runs_dir: str) -> None:
    """Write the run's final state to `<runs_dir>/<run_id>.json`.

    The CRM keeps what a run found; this keeps how it got there — the steps
    taken, the errors recorded, when it started and finished — including for
    a run that never reached the CRM. Best-effort: the record is for looking
    back, and failing to write it must not cost the caller the run itself.
    """
    try:
        path = Path(runs_dir)
        path.mkdir(parents=True, exist_ok=True)
        (path / f"{state.run_id}.json").write_text(state.model_dump_json(indent=2))
    except Exception:  # noqa: BLE001 — degrade, don't crash
        log.exception("run record not written: run_id=%s runs_dir=%s", state.run_id, runs_dir)


async def _run(target_input: str) -> RunState:
    state = RunState(run_id=uuid.uuid4().hex[:8], target_input=target_input)
    deps = build_deps(state.run_id)
    log.info("run started", extra={"run_id": state.run_id})

    try:
        for iteration in range(MAX_ITERATIONS):
            try:
                decision = await run_agent(
                    router_agent,
                    state.summary_for_router(MAX_DEEPEN_ROUNDS),
                    run_id=state.run_id,
                )
            except Exception as exc:  # noqa: BLE001 — degrade, don't crash
                # `run_agent` has already spent the retry budget, and with no
                # router there is no next step to take: finish degraded with
                # the failure on record rather than raising out of the run.
                log.exception("routing failed (run_id=%s)", state.run_id)
                state.errors.append(f"route: {exc}")
                break
            log.info(
                "routing: iter=%d step=%s reason=%r run_id=%s",
                iteration, decision.next_step.value, decision.reason, state.run_id,
            )
            state.steps_taken.append(decision.next_step.value)

            if decision.next_step == NextStep.DONE:
                break

            try:
                await dispatch(state, deps, decision.next_step)
            except Exception as exc:  # noqa: BLE001 — degrade, don't crash
                log.exception("step failed: %s (run_id=%s)", decision.next_step, state.run_id)
                state.errors.append(f"{decision.next_step.value}: {exc}")
                # The router sees the error in the next state summary and can
                # re-route (retry the step, skip it, or finish degraded).

        else:
            log.warning("hit MAX_ITERATIONS without DONE (run_id=%s)", state.run_id)
    finally:
        state.finished_at = datetime.now(UTC)
        try:
            await deps.http_client.aclose()
        finally:
            _save_run_record(state, deps.runs_dir)

    log.info("run finished: steps=%s errors=%d", state.steps_taken, len(state.errors))
    return state


def run(target_input: str) -> RunState:
    """Sync entry point — the async loop is an implementation detail."""
    return asyncio.run(_run(target_input))


if __name__ == "__main__":
    final = run("Chart Industries — cryogenic equipment for LNG and hydrogen")
    if final.landscape:
        print(final.landscape.model_dump_json(indent=2))

"""Supervisor routing loop — the orchestration backbone.

Pattern: an LLM router agent reads the run state and emits a validated
RouterDecision; plain Python dispatches on that enum. LLM output determines
the next step (dynamic routing), but guardrails (iteration caps, retry
limits, terminal checks) live in deterministic code where they belong.

NOTE: This is a Day-1 sketch. Verify Pydantic AI API details against
current docs (ai.pydantic.dev) — agent construction, output_type, and
tool registration have evolved across versions.
"""

import logging
import uuid

from pydantic_ai import Agent

from schemas import (
    BuyerLandscape,
    NextStep,
    RouterDecision,
    RunState,
    TargetProfile,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
)
log = logging.getLogger("supervisor")

MAX_ITERATIONS = 12       # hard cap: the loop can never run away
MAX_DEEPEN_ROUNDS = 2     # low-confidence re-research is bounded

MODEL = "anthropic:claude-sonnet-4-6"  # pick per docs; cheap+fast is fine here

# ---------------------------------------------------------------------------
# Agents. Each declares its output schema; Pydantic AI validates the model's
# response against it and retries with the validation errors on failure.
# Tools (EDGAR search, web search, CRM write) get attached to these agents
# on Day 2 via typed tool functions.
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
)

profiler_agent = Agent(
    MODEL,
    output_type=TargetProfile,
    instructions=(
        "Build a structured profile of the target company. Use your tools "
        "(web search, SEC EDGAR) and cite sources. If revenue is unknowable, "
        "say 'unknown' — do not guess."
    ),
    retries=2,
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
)

# strategic_agent, sponsor_agent: same shape, output_type=list[BuyerCandidate]
# (wrap in a container model if you want a model_validator over the list).


# ---------------------------------------------------------------------------
# Dispatch — deterministic. The LLM chooses the step; Python executes it.
# ---------------------------------------------------------------------------

def run(target_input: str) -> RunState:
    state = RunState(run_id=uuid.uuid4().hex[:8], target_input=target_input)
    log.info("run started", extra={"run_id": state.run_id})

    for iteration in range(MAX_ITERATIONS):
        decision = router_agent.run_sync(state.summary_for_router()).output
        log.info(
            "routing: iter=%d step=%s reason=%r run_id=%s",
            iteration, decision.next_step.value, decision.reason, state.run_id,
        )
        state.steps_taken.append(decision.next_step.value)

        try:
            if decision.next_step == NextStep.DONE:
                break

            elif decision.next_step == NextStep.PROFILE_TARGET:
                state.profile = profiler_agent.run_sync(target_input).output

            elif decision.next_step == NextStep.FIND_STRATEGIC_BUYERS:
                ...  # strategic_agent.run_sync(profile as prompt) -> extend list

            elif decision.next_step == NextStep.FIND_SPONSOR_BUYERS:
                ...  # sponsor_agent

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
                state.landscape = synthesis_agent.run_sync(prompt).output

            elif decision.next_step == NextStep.WRITE_TO_CRM:
                ...  # crm.write_landscape(state.landscape); state.crm_written = True

        except Exception as exc:  # noqa: BLE001 — degrade, don't crash
            log.exception("step failed: %s (run_id=%s)", decision.next_step, state.run_id)
            state.errors.append(f"{decision.next_step.value}: {exc}")
            # The router sees the error in the next state summary and can
            # re-route (retry the step, skip it, or finish degraded).

    else:
        log.warning("hit MAX_ITERATIONS without DONE (run_id=%s)", state.run_id)

    log.info("run finished: steps=%s errors=%d", state.steps_taken, len(state.errors))
    return state


if __name__ == "__main__":
    final = run("Chart Industries — cryogenic equipment for LNG and hydrogen")
    if final.landscape:
        print(final.landscape.model_dump_json(indent=2))

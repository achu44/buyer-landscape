"""Tests for the specialist buyer-sourcing agents and the supervisor's
dispatch loop. No real LLM calls: every agent is overridden with a
FunctionModel, so what a "model" returns is ordinary Python.

FunctionModel receives the tool schemas the agent would send to a real
provider (`AgentInfo.function_tools`), which is how these tests check that an
agent actually carries the research tools rather than just claiming to in its
instructions.
"""

from contextlib import ExitStack
from typing import Any

import pytest
from pydantic_ai import ModelResponse, ToolCallPart
from pydantic_ai.messages import ModelMessage, UserPromptPart
from pydantic_ai.models.function import AgentInfo, FunctionModel

import supervisor
from deps import Deps, build_deps
from schemas import BuyerType, RunState

RUN_ID = "run-test01"

RESEARCH_TOOL_NAMES = ["comps_lookup", "edgar_search", "web_search"]

RATIONALE = (
    "Strong strategic adjacency: the acquirer already sells into the same "
    "industrial gas customers, has bought two cryogenic component makers "
    "since 2021, and carries the balance-sheet capacity to pay cash."
)

EVIDENCE = (
    "Searched EDGAR and the trade press for the precedent the original "
    "rationale leaned on and found nothing filed, announced or reported that "
    "puts this buyer anywhere near the target's segment."
)

PROFILE = {
    "name": "Chart Industries",
    "description": (
        "Chart Industries engineers and manufactures cryogenic equipment used "
        "to liquefy, store and transport industrial gases, LNG and hydrogen, "
        "selling to energy and industrial-gas customers worldwide."
    ),
    "sector": "Industrials",
    "subsector": "Cryogenic equipment",
    "is_public": True,
    "est_revenue_band": "$1B-$5B",
    "key_assets": ["Cryogenic tank manufacturing footprint"],
    "geographies": ["United States"],
    "sources": ["https://example.com/10-K"],
}


def candidate(
    name: str, buyer_type: BuyerType, confidence: str = "high"
) -> dict[str, Any]:
    return {
        "name": name,
        "buyer_type": buyer_type.value,
        "fit_score": 80,
        "rationale": RATIONALE,
        "signals": ["Acquired Cryo Components in 2023"],
        "confidence": confidence,
        "sources": ["https://example.com/filing"],
    }


def batch_of(buyer_type: BuyerType, *candidates: dict[str, Any]) -> dict[str, Any]:
    return {"buyer_type": buyer_type.value, "candidates": list(candidates)}


def deepened(buyer_type: BuyerType, *findings: dict[str, Any]) -> dict[str, Any]:
    return {"buyer_type": buyer_type.value, "findings": list(findings)}


def confirms(
    name: str, buyer_type: BuyerType, confidence: str = "high"
) -> dict[str, Any]:
    return {
        "original_name": name,
        "verdict": "confirmed",
        "evidence": EVIDENCE,
        "candidate": candidate(name, buyer_type, confidence),
    }


def refutes(name: str) -> dict[str, Any]:
    return {"original_name": name, "verdict": "refuted", "evidence": EVIDENCE}


def batch(buyer_type: BuyerType, name: str = "Acme Corp") -> dict[str, Any]:
    return {"buyer_type": buyer_type.value, "candidates": [candidate(name, buyer_type)]}


def the_other(buyer_type: BuyerType) -> BuyerType:
    """The buyer type a specialist was *not* asked for."""
    return (
        BuyerType.FINANCIAL_SPONSOR
        if buyer_type is BuyerType.STRATEGIC
        else BuyerType.STRATEGIC
    )


class Responder:
    """A FunctionModel function returning `payloads` in order, one per model
    request, and counting how many requests it answered.

    Anything after the last payload repeats it, so a test states only the
    responses it cares about. A class rather than a closure because the tests
    read the call count back, and FunctionModel accepts any callable.
    """

    def __init__(self, *payloads: dict[str, Any]):
        self.payloads = payloads
        self.calls = 0

    def __call__(self, messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        payload = self.payloads[min(self.calls, len(self.payloads) - 1)]
        self.calls += 1
        return ModelResponse(parts=[ToolCallPart(info.output_tools[0].name, payload)])


class PromptCapturingResponder(Responder):
    """A Responder that also records the user prompts it was sent."""

    def __init__(self, *payloads: dict[str, Any]):
        super().__init__(*payloads)
        self.prompts: list[str] = []

    def __call__(self, messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        self.prompts += [
            str(part.content)
            for message in messages
            for part in message.parts
            if isinstance(part, UserPromptPart)
        ]
        return super().__call__(messages, info)


class ToolCapturingResponder(Responder):
    """A Responder that also records the tools the agent offered the model."""

    def __init__(self, *payloads: dict[str, Any]):
        super().__init__(*payloads)
        self.tool_names: list[str] = []

    def __call__(self, messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        self.tool_names = sorted(tool.name for tool in info.function_tools)
        return super().__call__(messages, info)


def router(*steps: str) -> Responder:
    """A router model that walks the given steps, then stops."""
    return Responder(
        *[
            {"next_step": step, "reason": f"Test route: {step} is the next step."}
            for step in (*steps, "done")
        ]
    )


def pipeline(router_steps: list[str], **agents: Any) -> RunState:
    """Override the router and each named agent, then run the pipeline."""
    overrides = {
        "router_agent": FunctionModel(router(*router_steps)),
        **{name: FunctionModel(model) for name, model in agents.items()},
    }
    with ExitStack() as stack:
        for name, model in overrides.items():
            stack.enter_context(getattr(supervisor, name).override(model=model))
        return supervisor.run("Chart Industries — cryogenic equipment")


@pytest.fixture
def deps() -> Deps:
    return build_deps(RUN_ID)


@pytest.mark.parametrize("buyer_type", list(BuyerType))
def test_specialist_agents_carry_the_research_tools(
    deps: Deps, buyer_type: BuyerType
) -> None:
    """Both specialists are built from the same path, so both reach the model
    with every research tool attached — the gap this issue closes."""
    respond = ToolCapturingResponder(batch(buyer_type))
    agent = supervisor.SPECIALIST_AGENTS[buyer_type]

    with agent.override(model=FunctionModel(respond)):
        result = agent.run_sync("Profile of the target", deps=deps)

    assert respond.tool_names == RESEARCH_TOOL_NAMES
    assert result.output.buyer_type == buyer_type


def test_profiler_carries_the_research_tools(deps: Deps) -> None:
    """The profiler's instructions tell it to use tools; this is the wiring
    that makes that true."""
    respond = ToolCapturingResponder(PROFILE)

    with supervisor.profiler_agent.override(model=FunctionModel(respond)):
        supervisor.profiler_agent.run_sync("Chart Industries", deps=deps)

    assert respond.tool_names == RESEARCH_TOOL_NAMES


@pytest.mark.parametrize("buyer_type", list(BuyerType))
def test_specialist_retries_a_batch_declared_as_the_wrong_buyer_type(
    deps: Deps, buyer_type: BuyerType
) -> None:
    """A batch sourced against the wrong brief is not a result to store —
    the model is sent back rather than letting it pollute the list."""
    respond = Responder(batch(the_other(buyer_type)), batch(buyer_type))
    agent = supervisor.SPECIALIST_AGENTS[buyer_type]

    with agent.override(model=FunctionModel(respond)):
        result = agent.run_sync("Profile", deps=deps)

    assert respond.calls == 2  # first answer rejected, second accepted
    assert result.output.buyer_type == buyer_type


@pytest.mark.parametrize("buyer_type", list(BuyerType))
def test_specialist_retries_an_empty_batch(deps: Deps, buyer_type: BuyerType) -> None:
    """"No buyers" is never an acceptable answer from a sourcing agent."""
    empty = {"buyer_type": buyer_type.value, "candidates": []}
    respond = Responder(empty, batch(buyer_type))
    agent = supervisor.SPECIALIST_AGENTS[buyer_type]

    with agent.override(model=FunctionModel(respond)):
        result = agent.run_sync("Profile", deps=deps)

    assert respond.calls == 2
    assert len(result.output.candidates) == 1


def test_pipeline_populates_both_buyer_lists() -> None:
    state = pipeline(
        ["profile_target", "find_strategic_buyers", "find_sponsor_buyers"],
        profiler_agent=Responder(PROFILE),
        strategic_agent=Responder(batch(BuyerType.STRATEGIC, "Air Liquide")),
        sponsor_agent=Responder(batch(BuyerType.FINANCIAL_SPONSOR, "Apollo")),
    )

    assert [b.name for b in state.strategic_buyers] == ["Air Liquide"]
    assert [b.name for b in state.sponsor_buyers] == ["Apollo"]
    assert state.errors == []


def test_dispatch_rejects_a_batch_of_the_wrong_buyer_type(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Defense in depth for a wiring mistake: if the agent invoked for
    strategic buyers hands back sponsors, they are recorded as an error
    rather than filed under strategic."""
    monkeypatch.setitem(
        supervisor.SPECIALIST_AGENTS, BuyerType.STRATEGIC, supervisor.sponsor_agent
    )

    state = pipeline(
        ["profile_target", "find_strategic_buyers"],
        profiler_agent=Responder(PROFILE),
        sponsor_agent=Responder(batch(BuyerType.FINANCIAL_SPONSOR, "Apollo")),
    )

    assert state.strategic_buyers == []
    assert any("financial_sponsor" in e for e in state.errors)


# ---------------------------------------------------------------------------
# Deepen-research: bounded re-research of low-confidence candidates
# ---------------------------------------------------------------------------

def sourced_then_deepened(
    strategic: dict[str, Any] | None = None,
    sponsor: dict[str, Any] | None = None,
    rounds: int = 1,
    **deepen_agents: Any,
) -> RunState:
    """Profile, source the given lists, then route to deepen `rounds` times."""
    steps = ["profile_target"]
    sourcing: dict[str, Any] = {}
    if strategic is not None:
        steps.append("find_strategic_buyers")
        sourcing["strategic_agent"] = Responder(strategic)
    if sponsor is not None:
        steps.append("find_sponsor_buyers")
        sourcing["sponsor_agent"] = Responder(sponsor)
    return pipeline(
        steps + ["deepen_research"] * rounds,
        profiler_agent=Responder(PROFILE),
        **sourcing,
        **deepen_agents,
    )


def test_confirmed_candidate_replaces_its_low_confidence_entry() -> None:
    state = sourced_then_deepened(
        strategic=batch_of(
            BuyerType.STRATEGIC,
            candidate("Air Liquide", BuyerType.STRATEGIC, "low"),
            candidate("Linde", BuyerType.STRATEGIC),
        ),
        strategic_deepen_agent=Responder(
            deepened(BuyerType.STRATEGIC, confirms("Air Liquide", BuyerType.STRATEGIC))
        ),
    )

    assert [b.name for b in state.strategic_buyers] == ["Air Liquide", "Linde"]
    assert state.low_confidence_count() == 0
    assert state.errors == []


def test_candidate_that_cannot_be_substantiated_is_dropped() -> None:
    state = sourced_then_deepened(
        strategic=batch_of(
            BuyerType.STRATEGIC,
            candidate("Air Liquide", BuyerType.STRATEGIC, "low"),
            candidate("Linde", BuyerType.STRATEGIC),
        ),
        strategic_deepen_agent=Responder(
            deepened(BuyerType.STRATEGIC, refutes("Air Liquide"))
        ),
    )

    assert [b.name for b in state.strategic_buyers] == ["Linde"]
    assert state.errors == []


def test_one_round_deepens_every_list_that_has_low_confidence_buyers() -> None:
    """Both specialists came back shaky; one round resolves both, and costs
    one round — not one per list."""
    strategic_deepen = Responder(
        deepened(BuyerType.STRATEGIC, confirms("Air Liquide", BuyerType.STRATEGIC))
    )
    sponsor_deepen = Responder(
        deepened(
            BuyerType.FINANCIAL_SPONSOR,
            confirms("Apollo", BuyerType.FINANCIAL_SPONSOR),
        )
    )

    state = sourced_then_deepened(
        strategic=batch_of(
            BuyerType.STRATEGIC, candidate("Air Liquide", BuyerType.STRATEGIC, "low")
        ),
        sponsor=batch_of(
            BuyerType.FINANCIAL_SPONSOR,
            candidate("Apollo", BuyerType.FINANCIAL_SPONSOR, "low"),
        ),
        strategic_deepen_agent=strategic_deepen,
        sponsor_deepen_agent=sponsor_deepen,
    )

    assert strategic_deepen.calls == 1
    assert sponsor_deepen.calls == 1
    assert state.deepen_rounds_used == 1
    assert state.low_confidence_count() == 0


def test_a_list_with_no_low_confidence_buyers_is_left_alone() -> None:
    sponsor_deepen = Responder(
        deepened(
            BuyerType.FINANCIAL_SPONSOR,
            confirms("Apollo", BuyerType.FINANCIAL_SPONSOR),
        )
    )

    state = sourced_then_deepened(
        strategic=batch_of(BuyerType.STRATEGIC, candidate("Linde", BuyerType.STRATEGIC)),
        sponsor=batch_of(
            BuyerType.FINANCIAL_SPONSOR,
            candidate("Apollo", BuyerType.FINANCIAL_SPONSOR, "low"),
        ),
        strategic_deepen_agent=Responder(
            deepened(BuyerType.STRATEGIC, refutes("Linde"))
        ),
        sponsor_deepen_agent=sponsor_deepen,
    )

    assert sponsor_deepen.calls == 1
    assert [b.name for b in state.strategic_buyers] == ["Linde"]


def test_deepen_pass_is_asked_only_about_the_low_confidence_candidates() -> None:
    deepen = PromptCapturingResponder(
        deepened(BuyerType.STRATEGIC, confirms("Air Liquide", BuyerType.STRATEGIC))
    )

    sourced_then_deepened(
        strategic=batch_of(
            BuyerType.STRATEGIC,
            candidate("Air Liquide", BuyerType.STRATEGIC, "low"),
            candidate("Linde", BuyerType.STRATEGIC),
        ),
        strategic_deepen_agent=deepen,
    )

    assert "Air Liquide" in deepen.prompts[0]
    assert "Linde" not in deepen.prompts[0]


def test_deepen_research_never_exceeds_the_round_cap() -> None:
    """A pass that confirms a buyer but still can't raise its confidence
    leaves the router with the same reason to deepen again — the cap, not the
    router, is what stops it."""
    deepen = Responder(
        deepened(
            BuyerType.STRATEGIC,
            confirms("Air Liquide", BuyerType.STRATEGIC, "low"),
        )
    )

    state = sourced_then_deepened(
        strategic=batch_of(
            BuyerType.STRATEGIC, candidate("Air Liquide", BuyerType.STRATEGIC, "low")
        ),
        rounds=supervisor.MAX_DEEPEN_ROUNDS + 2,
        strategic_deepen_agent=deepen,
    )

    assert deepen.calls == supervisor.MAX_DEEPEN_ROUNDS
    assert state.deepen_rounds_used == supervisor.MAX_DEEPEN_ROUNDS
    assert sum("past the round cap" in e for e in state.errors) == 2


def test_deepen_research_with_nothing_to_deepen_costs_no_round() -> None:
    """The router asked for a step there is no work for; the run records it
    and keeps its rounds for a round that would do something."""
    state = sourced_then_deepened(
        strategic=batch_of(BuyerType.STRATEGIC, candidate("Linde", BuyerType.STRATEGIC)),
    )

    assert state.deepen_rounds_used == 0
    assert any("no low-confidence candidates" in e for e in state.errors)


def test_a_failed_deepen_pass_leaves_the_other_list_deepened() -> None:
    """One list's pass blowing up is an error to route on, not a reason the
    other list keeps its unsubstantiated names."""
    state = sourced_then_deepened(
        strategic=batch_of(
            BuyerType.STRATEGIC, candidate("Air Liquide", BuyerType.STRATEGIC, "low")
        ),
        sponsor=batch_of(
            BuyerType.FINANCIAL_SPONSOR,
            candidate("Apollo", BuyerType.FINANCIAL_SPONSOR, "low"),
        ),
        # Declared as the wrong buyer type: the agent's own validator retries,
        # runs out of retries, and the pass fails.
        strategic_deepen_agent=Responder(
            deepened(
                BuyerType.FINANCIAL_SPONSOR,
                confirms("Apollo", BuyerType.FINANCIAL_SPONSOR),
            )
        ),
        sponsor_deepen_agent=Responder(
            deepened(
                BuyerType.FINANCIAL_SPONSOR,
                confirms("Apollo", BuyerType.FINANCIAL_SPONSOR),
            )
        ),
    )

    assert any("deepen_research (strategic)" in e for e in state.errors)
    assert [b.name for b in state.sponsor_buyers] == ["Apollo"]

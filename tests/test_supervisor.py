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
from pydantic_ai.messages import ModelMessage
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


def candidate(name: str, buyer_type: BuyerType) -> dict[str, Any]:
    return {
        "name": name,
        "buyer_type": buyer_type.value,
        "fit_score": 80,
        "rationale": RATIONALE,
        "signals": ["Acquired Cryo Components in 2023"],
        "confidence": "high",
        "sources": ["https://example.com/filing"],
    }


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

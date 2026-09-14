"""Tests for llm.py — the external-call policy applied to agent runs. No real
provider calls: a FunctionModel raises the errors Pydantic AI surfaces for
provider failures, so transient and terminal failures are ordinary Python.

Backoff is zeroed for every test here; the policy's *shape* (attempts,
predicate, logging) is what is under test, not tenacity's sleep.
"""

import asyncio
import logging

import pytest
from pydantic import BaseModel, Field
from pydantic_ai import Agent, ModelResponse, ToolCallPart
from pydantic_ai.exceptions import (
    IncompleteToolCall,
    ModelAPIError,
    ModelHTTPError,
    UnexpectedModelBehavior,
)
from pydantic_ai.messages import ModelMessage
from pydantic_ai.models.function import AgentInfo, FunctionModel
from pydantic_ai.settings import ModelSettings

import deps
import llm
from tests.fakes import MODEL_NAME, ProviderDown

RUN_ID = "run-llm001"

# Backoff is zeroed for every test here; see the module docstring.
pytestmark = pytest.mark.usefixtures("no_backoff")


class Answer(BaseModel):
    verdict: str = Field(min_length=1)


def http_error(status_code: int) -> ModelHTTPError:
    return ModelHTTPError(status_code=status_code, model_name=MODEL_NAME)


class FailingThenAnswering:
    """A FunctionModel function that raises each of `failures` in turn, then
    answers. Counts requests so a test can tell a retry from a single call."""

    def __init__(self, *failures: BaseException):
        self.failures = failures
        self.calls = 0
        self.settings_seen: list[ModelSettings | None] = []

    def __call__(self, messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        self.calls += 1
        self.settings_seen.append(info.model_settings)
        if self.calls <= len(self.failures):
            raise self.failures[self.calls - 1]
        return ModelResponse(parts=[ToolCallPart(info.output_tools[0].name, {"verdict": "ok"})])


def agent_answering_with(
    respond: "FailingThenAnswering | ProviderDown | CutOffAtTheTokenLimit",
) -> Agent[None, Answer]:
    return Agent(FunctionModel(respond), output_type=Answer, name="test_agent")


@pytest.mark.parametrize("status_code", [429, 500, 502, 503, 529])
def test_rate_limits_and_server_errors_are_transient(status_code: int) -> None:
    assert llm.is_transient_model_error(http_error(status_code))


@pytest.mark.parametrize("status_code", [400, 401, 403, 404, 413])
def test_client_errors_are_terminal(status_code: int) -> None:
    """Bad request, auth, quota: the same call fails the same way again."""
    assert not llm.is_transient_model_error(http_error(status_code))


def test_connection_and_timeout_failures_are_transient() -> None:
    """Pydantic AI surfaces a provider connection error or request timeout as
    a bare ModelAPIError — no status code, because no response arrived."""
    assert llm.is_transient_model_error(ModelAPIError(MODEL_NAME, "Request timed out."))


@pytest.mark.parametrize(
    "exc", [UnexpectedModelBehavior("bad output"), ValueError("bug"), RuntimeError("bug")]
)
def test_anything_else_is_terminal(exc: BaseException) -> None:
    assert not llm.is_transient_model_error(exc)


def test_transient_failures_are_retried_until_the_run_succeeds() -> None:
    respond = FailingThenAnswering(http_error(529), ModelAPIError(MODEL_NAME, "reset"))

    answer = asyncio.run(llm.run_agent(agent_answering_with(respond), "q", run_id=RUN_ID))

    assert answer == Answer(verdict="ok")
    assert respond.calls == 3  # two transient failures, then the answer


def test_a_run_that_exhausts_its_retry_budget_raises_the_last_error() -> None:
    respond = ProviderDown(503)

    with pytest.raises(ModelHTTPError):
        asyncio.run(llm.run_agent(agent_answering_with(respond), "q", run_id=RUN_ID))

    assert respond.calls == deps.MAX_ATTEMPTS


def test_a_terminal_failure_is_not_retried() -> None:
    respond = ProviderDown(401)

    with pytest.raises(ModelHTTPError):
        asyncio.run(llm.run_agent(agent_answering_with(respond), "q", run_id=RUN_ID))

    assert respond.calls == 1


def test_every_model_request_carries_the_configured_timeout() -> None:
    respond = FailingThenAnswering()

    asyncio.run(llm.run_agent(agent_answering_with(respond), "q", run_id=RUN_ID))

    assert respond.settings_seen
    for settings in respond.settings_seen:
        assert settings is not None
        assert settings.get("timeout") == llm.LLM_REQUEST_TIMEOUT_SECONDS


def test_retry_and_completion_log_lines_carry_run_id(caplog: pytest.LogCaptureFixture) -> None:
    respond = FailingThenAnswering(http_error(529))

    with caplog.at_level(logging.INFO, logger="llm"):
        asyncio.run(llm.run_agent(agent_answering_with(respond), "q", run_id=RUN_ID))

    retries = [r for r in caplog.records if "retrying after transient failure" in r.message]
    completions = [r for r in caplog.records if "agent run completed" in r.message]
    assert retries and completions
    assert all(RUN_ID in r.message for r in retries + completions)
    assert all("test_agent" in r.message for r in completions)


def test_the_provider_sdk_does_not_retry_underneath_the_policy(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The Anthropic SDK retries on its own by default. Left on, each attempt
    here would be up to three requests the run_id log never sees — so the
    model is built with that layer off, and without needing a key to exist."""
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)

    model = llm.build_model()

    assert model.client.max_retries == 0


# ---------------------------------------------------------------------------
# Output budget: an answer cut off at the token limit is not a wrong answer
# ---------------------------------------------------------------------------

def test_the_model_carries_an_explicit_output_budget_and_prompt_caching(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Left unset, the provider default caps an answer at 4096 tokens — too
    small for a full buyer batch. Caching is on because a research run resends
    its whole growing tool history on every request."""
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)

    settings = dict(llm.build_model().settings or {})

    assert settings["max_tokens"] == llm.MAX_OUTPUT_TOKENS
    assert settings["anthropic_cache"] is True


class CutOffAtTheTokenLimit:
    """A FunctionModel function whose every answer stops on the token limit,
    counting requests so a test can tell one request from a retried run."""

    def __init__(self, payload: dict[str, str]):
        self.payload = payload
        self.calls = 0

    def __call__(self, messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        self.calls += 1
        return ModelResponse(
            parts=[ToolCallPart(info.output_tools[0].name, self.payload)],
            finish_reason="length",
        )


@pytest.mark.parametrize(
    "payload",
    [
        pytest.param({}, id="cut-off-answer-fails-validation"),
        pytest.param({"verdict": "ok"}, id="cut-off-answer-still-validates"),
    ],
)
def test_an_answer_cut_off_at_the_token_limit_fails_the_run_at_once(
    payload: dict[str, str],
) -> None:
    """Asking again cannot help: the same limit cuts the same answer off. So
    the run stops on the first cut-off response — no output retries, no
    external-call retries — with an error naming the limit. That includes an
    answer the partial JSON still happens to validate, which would otherwise
    pass silently with buyers missing."""
    respond = CutOffAtTheTokenLimit(payload)

    with pytest.raises(IncompleteToolCall, match="token limit"):
        asyncio.run(llm.run_agent(agent_answering_with(respond), "q", run_id=RUN_ID))

    assert respond.calls == 1

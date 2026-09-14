"""The external-call policy, applied to LLM calls.

CLAUDE.md requires every external call to carry a timeout, a tenacity retry
with exponential backoff, and a log line carrying `run_id`. Tools get that
from the shared HTTP client in `deps.py`; an agent run talks to the provider
through the provider SDK instead, so it needs its own wrapper — `run_agent`.

`Agent(retries=...)` is not this: it re-asks the model when its *output*
fails validation. A 529 from the provider or a hung connection raises
straight out of `agent.run`, and without this wrapper one transient blip
fails the whole step.

The retry wraps the whole agent run rather than a single model request, so a
retried run re-executes its tool calls. That is the price of retrying at the
one layer every model — including the FunctionModel the tests use — shares;
the research tools are read-only, so a repeat costs time, not correctness.
"""

from dataclasses import dataclass
from http import HTTPStatus
import logging
from typing import Any

from anthropic import AsyncAnthropic
from pydantic_ai import Agent, ModelResponse, RunContext
from pydantic_ai.capabilities import AbstractCapability
from pydantic_ai.exceptions import IncompleteToolCall, ModelAPIError, ModelHTTPError
from pydantic_ai.models import ModelRequestContext
from pydantic_ai.models.anthropic import AnthropicModel, AnthropicModelSettings
from pydantic_ai.providers.anthropic import AnthropicProvider
from tenacity import AsyncRetrying

from deps import retry_kwargs

log = logging.getLogger("llm")

MODEL_NAME = "claude-sonnet-4-6"  # pick per docs; cheap+fast is fine here

# Per model request, not per agent run: a research run makes many requests
# with tool calls in between, and only a single hung request is worth cutting
# off. Generous because a long structured answer legitimately takes a while.
LLM_REQUEST_TIMEOUT_SECONDS = 120.0

# Per model response. Left unset, Pydantic AI asks Anthropic for 4096 tokens,
# and a full buyer batch — several candidates, each with a 150-character-plus
# rationale, signals and sources — does not fit: the first live run lost its
# strategic list to exactly that. Sized for the largest structured answer in
# the pipeline, the synthesized landscape.
MAX_OUTPUT_TOKENS = 16_000

# Rate limiting (429) and the provider being unwell (5xx, Anthropic's 529
# overloaded included) clear on their own. Every other status is about the
# request — bad input, a bad key, an exhausted quota — and fails the same way
# on every attempt.
_FIRST_SERVER_ERROR = HTTPStatus.INTERNAL_SERVER_ERROR


def build_model() -> AnthropicModel:
    """The model every agent runs on, with the provider SDK's own retries off.

    The Anthropic SDK retries rate limits, 5xx and dropped connections itself
    by default. Stacked under `run_agent`, every attempt there would become up
    to three requests, none of them logged with a `run_id`, and a hung step
    would wait out the timeout three times per attempt. One retry layer, ours.

    Building the client needs no API key — the SDK only asks for one when a
    request is sent — so agents can be constructed, and tested with the model
    overridden, where no key exists.
    """
    client = AsyncAnthropic(max_retries=0)
    settings = AnthropicModelSettings(
        max_tokens=MAX_OUTPUT_TOKENS,
        # A research run resends its whole growing tool history on every
        # request; caching bills that repeated prefix at the cache-read rate.
        anthropic_cache=True,
    )
    return AnthropicModel(
        MODEL_NAME, provider=AnthropicProvider(anthropic_client=client), settings=settings
    )


@dataclass
class StopOnTruncatedResponse(AbstractCapability[Any]):
    """Fail the run the moment a response is cut off at the token limit.

    Pydantic AI raises `IncompleteToolCall` only when the truncated arguments
    are invalid JSON, but the Anthropic SDK hands back partial tool input as a
    valid dict. A cut-off answer then surfaces as a validation error, and the
    agent spends its output retries re-asking for an answer the same limit
    will cut off again — or, when the partial JSON happens to validate, passes
    silently with the tail of the answer missing. Checked on the response
    itself, before any parsing, so neither can happen.
    """

    async def after_model_request(
        self,
        ctx: RunContext[Any],
        *,
        request_context: ModelRequestContext,
        response: ModelResponse,
    ) -> ModelResponse:
        if response.finish_reason == "length":
            raise IncompleteToolCall(
                f"Model response was cut off at the token limit (max_tokens={MAX_OUTPUT_TOKENS}); "
                "retrying cannot fix it. Raise MAX_OUTPUT_TOKENS or ask for a shorter answer."
            )
        return response


def is_transient_model_error(exc: BaseException) -> bool:
    """The LLM-side counterpart of `deps.is_transient_error`.

    Pydantic AI surfaces a provider failure with a response as
    `ModelHTTPError`, and one with no response at all — a dropped connection
    or a request timeout — as a bare `ModelAPIError`. The first is transient
    only for 429 and 5xx; the second always is. Anything else (bad model
    output, a bug) is not a provider failure and retrying cannot fix it.
    """
    if isinstance(exc, ModelHTTPError):
        return (
            exc.status_code == HTTPStatus.TOO_MANY_REQUESTS
            or exc.status_code >= _FIRST_SERVER_ERROR
        )
    return isinstance(exc, ModelAPIError)


async def run_agent[DepsT, OutputT](
    agent: Agent[DepsT, OutputT],
    prompt: str,
    *,
    run_id: str,
    deps: DepsT = None,
) -> OutputT:
    """Run `agent` under the external-call policy and return its output.

    Raises the last error once the retry budget is spent, or immediately on a
    terminal one: turning that into a `RunState.errors` entry is the
    supervisor loop's job, the same as for any other failed step.
    """
    async for attempt in AsyncRetrying(
        **retry_kwargs(run_id, log, is_transient=is_transient_model_error)
    ):
        with attempt:
            result = await agent.run(
                prompt,
                deps=deps,
                model_settings={"timeout": LLM_REQUEST_TIMEOUT_SECONDS},
                # Attached per run rather than per agent, so no agent built
                # anywhere can be run without it.
                capabilities=[StopOnTruncatedResponse()],
            )
            log.info(
                "agent run completed: run_id=%s agent=%s attempts=%d",
                run_id, agent.name, attempt.retry_state.attempt_number,
            )
            return result.output
    # `reraise=True` means the loop either returns or raises; never falls out.
    raise AssertionError("unreachable: retry loop exited without a result")

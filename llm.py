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

import logging

from pydantic_ai import Agent
from pydantic_ai.exceptions import ModelAPIError, ModelHTTPError
from tenacity import AsyncRetrying, retry_if_exception

from deps import retry_kwargs

log = logging.getLogger("llm")

# Per model request, not per agent run: a research run makes many requests
# with tool calls in between, and only a single hung request is worth cutting
# off. Generous because a long structured answer legitimately takes a while.
LLM_REQUEST_TIMEOUT_SECONDS = 120.0

# Rate limiting (429) and the provider being unwell (5xx, Anthropic's 529
# overloaded included) clear on their own. Every other status is about the
# request — bad input, a bad key, an exhausted quota — and fails the same way
# on every attempt.
_RATE_LIMITED = 429
_SERVER_ERROR = 500


def is_transient_model_error(exc: BaseException) -> bool:
    """The LLM-side counterpart of `deps.is_transient_error`.

    Pydantic AI surfaces a provider failure with a response as
    `ModelHTTPError`, and one with no response at all — a dropped connection
    or a request timeout — as a bare `ModelAPIError`. The first is transient
    only for 429 and 5xx; the second always is. Anything else (bad model
    output, a bug) is not a provider failure and retrying cannot fix it.
    """
    if isinstance(exc, ModelHTTPError):
        return exc.status_code == _RATE_LIMITED or exc.status_code >= _SERVER_ERROR
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
    policy = {**retry_kwargs(run_id, log), "retry": retry_if_exception(is_transient_model_error)}
    async for attempt in AsyncRetrying(**policy):
        with attempt:
            result = await agent.run(
                prompt,
                deps=deps,
                model_settings={"timeout": LLM_REQUEST_TIMEOUT_SECONDS},
            )
            log.info(
                "agent run completed: agent=%s attempts=%d run_id=%s",
                agent.name, attempt.retry_state.attempt_number, run_id,
            )
            return result.output
    # `reraise=True` means the loop either returns or raises; never falls out.
    raise AssertionError("unreachable: retry loop exited without a result")

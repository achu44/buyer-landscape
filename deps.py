"""Shared dependency object and resilient HTTP client.

Every research tool (Day 2, `tools/`) receives a `Deps` instance instead of
importing its own HTTP client or logger — see
docs/adr/0002-shared-deps-for-tool-resources.md. `Deps` bundles an
httpx.AsyncClient (timeout + retry-with-backoff already configured) and the
current run's `run_id`, so tools get production-hygiene behavior for free
instead of reimplementing it.

`retry_kwargs` is exposed separately from the HTTP transport because not
every tool goes through the shared client — the yfinance comps tool (Day 2)
wraps a sync call that can't accept an injected client, but should still
retry with the same policy and log with the same run_id convention.
"""

from collections.abc import Callable
from dataclasses import dataclass
import logging
import os
from typing import Any

import httpx
from tenacity import (
    AsyncRetrying,
    RetryCallState,
    retry_if_exception,
    stop_after_attempt,
    wait_exponential,
)

log = logging.getLogger("deps")

REQUEST_TIMEOUT = httpx.Timeout(10.0, connect=5.0)
MAX_ATTEMPTS = 4
BACKOFF_MULTIPLIER = 0.5
BACKOFF_MAX = 8.0

# Where the mock CRM's SQLite file lives. Read from the environment so a demo
# run and a test run never share a database.
CRM_DB_PATH = os.environ.get("CRM_DB_PATH", "crm.db")


def is_transient_error(exc: BaseException) -> bool:
    """Network/timeout errors and 5xx responses are worth retrying; 4xx and
    parsing errors are not — retrying a bad request just burns the budget."""
    if isinstance(exc, (httpx.TimeoutException, httpx.TransportError)):
        return True
    if isinstance(exc, httpx.HTTPStatusError):
        return exc.response.status_code >= 500
    return False


def _log_retry(run_id: str, logger: logging.Logger) -> Callable[[RetryCallState], None]:
    def _log(state: RetryCallState) -> None:
        exc = state.outcome.exception() if state.outcome else None
        logger.warning(
            "retrying after transient failure: attempt=%d run_id=%s error=%r",
            state.attempt_number, run_id, exc,
        )

    return _log


def retry_kwargs(run_id: str, logger: logging.Logger = log) -> dict[str, Any]:
    """The project's standard retry-with-backoff policy: bounded attempts,
    exponential wait, transient-only, and a log line carrying `run_id` on
    every retry. Shared by the HTTP transport below and usable directly with
    `tenacity.retry`/`Retrying`/`AsyncRetrying` by any non-HTTP tool."""
    return dict(
        stop=stop_after_attempt(MAX_ATTEMPTS),
        wait=wait_exponential(multiplier=BACKOFF_MULTIPLIER, max=BACKOFF_MAX),
        retry=retry_if_exception(is_transient_error),
        before_sleep=_log_retry(run_id, logger),
        reraise=True,
    )


class _RetryingTransport(httpx.AsyncBaseTransport):
    """Wraps another transport with retry-with-backoff on transient
    failures. Composition (not subclassing httpx.AsyncHTTPTransport) so
    tests can substitute an httpx.MockTransport as the wrapped transport.

    A 5xx that survives every retry attempt reaches the caller as a raised
    httpx.HTTPStatusError, never as a returned 5xx Response — tools built on
    this client should catch that instead of calling response.raise_for_status()
    expecting to see the response object."""

    def __init__(self, wrapped: httpx.AsyncBaseTransport, *, run_id: str):
        self._wrapped = wrapped
        self._run_id = run_id

    async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
        async for attempt in AsyncRetrying(**retry_kwargs(self._run_id)):
            with attempt:
                response = await self._wrapped.handle_async_request(request)
                if response.status_code >= 500:
                    raise httpx.HTTPStatusError(
                        f"server error {response.status_code}",
                        request=request,
                        response=response,
                    )
                log.info(
                    "request completed: run_id=%s method=%s url=%s status=%d",
                    self._run_id, request.method, request.url, response.status_code,
                )
                return response

    async def aclose(self) -> None:
        await self._wrapped.aclose()


def build_http_client(
    run_id: str, *, transport: httpx.AsyncBaseTransport | None = None
) -> httpx.AsyncClient:
    """An httpx.AsyncClient with a timeout and retry-with-backoff already
    configured. `transport` is overridable so tests can inject an
    httpx.MockTransport in place of the real network transport."""
    base = transport if transport is not None else httpx.AsyncHTTPTransport()
    return httpx.AsyncClient(
        transport=_RetryingTransport(base, run_id=run_id),
        timeout=REQUEST_TIMEOUT,
    )


@dataclass
class Deps:
    """Shared dependency object passed to every research agent via
    `deps_type`. Tools reach shared resources through `ctx.deps` instead of
    a module-level global (docs/adr/0002-shared-deps-for-tool-resources.md).

    The CRM writer is not an agent tool, but its database is still a resource
    of the run: carrying the path here lets a test point the CRM-write step at
    a temporary database the same way it swaps in a mock HTTP transport."""

    http_client: httpx.AsyncClient
    run_id: str
    crm_db_path: str = CRM_DB_PATH


def build_deps(
    run_id: str,
    *,
    transport: httpx.AsyncBaseTransport | None = None,
    crm_db_path: str = CRM_DB_PATH,
) -> Deps:
    return Deps(
        http_client=build_http_client(run_id, transport=transport),
        run_id=run_id,
        crm_db_path=crm_db_path,
    )

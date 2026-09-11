"""Unit tests for the general web search tool. No real network calls:
httpx.MockTransport stands in for the search API, so both the happy path and
the transient-failure path are deterministic.

The transport is injected through `build_deps`, which means these tests
exercise the *shared* client (timeout + tenacity retry from deps.py) rather
than a client the tool built for itself.
"""

import asyncio
import logging
from collections.abc import Callable, Coroutine
from datetime import date
from typing import Any, TypeVar

import httpx
import pytest
from pydantic_ai import RunContext
from pydantic_ai.exceptions import ModelRetry, ToolFailed
from pydantic_ai.models.test import TestModel
from pydantic_ai.tools import Tool
from pydantic_ai.usage import RunUsage

from deps import MAX_ATTEMPTS, Deps, build_deps
from schemas import Freshness, WebSearchHit, WebSearchResults
from tools.web import (
    API_KEY_ENV,
    BRAVE_SEARCH_URL,
    MAX_QUERY_CHARS,
    MAX_RESULTS,
    web_search,
)

RUN_ID = "run-abc123"
API_KEY = "test-subscription-token"

T = TypeVar("T")
Handler = Callable[[httpx.Request], httpx.Response]


@pytest.fixture(autouse=True)
def _api_key(monkeypatch: pytest.MonkeyPatch) -> None:
    """The key is a real environment variable (README), so pin it rather than
    letting whether these tests pass depend on the shell they run in."""
    monkeypatch.setenv(API_KEY_ENV, API_KEY)


def run(coro: Coroutine[Any, Any, T]) -> T:
    return asyncio.run(coro)


def make_ctx(handler: Handler) -> RunContext[Deps]:
    """A RunContext carrying real Deps whose HTTP client talks to `handler`
    instead of the network. The model/usage arguments are required by the
    dataclass but unused by tool functions."""
    deps = build_deps(RUN_ID, transport=httpx.MockTransport(handler))
    return RunContext(deps=deps, model=TestModel(), usage=RunUsage())


def result(
    *,
    title: str = "Quanta Services to acquire Cupertino Electric",
    url: str = "https://www.reuters.com/markets/deals/quanta-cupertino-2024-08-08/",
    description: str = "Quanta Services said on Thursday it would acquire <strong>Cupertino</strong> Electric.",
    page_age: str | None = "2024-08-08T13:02:00",
) -> dict[str, Any]:
    """One `web.results[]` entry shaped like a real Brave Web Search response
    (field names, `<strong>` highlighting and `page_age` format copied from
    the live API)."""
    entry: dict[str, Any] = {
        "type": "search_result",
        "title": title,
        "url": url,
        "description": description,
        "is_source_local": False,
        "language": "en",
        "family_friendly": True,
        "meta_url": {"scheme": "https", "hostname": "www.reuters.com"},
    }
    if page_age is not None:
        entry["page_age"] = page_age
    return entry


def payload(*results: dict[str, Any]) -> dict[str, Any]:
    return {
        "type": "search",
        "query": {"original": "Cupertino Electric acquisition", "more_results_available": True},
        "web": {"type": "search", "family_friendly": True, "results": list(results)},
    }


def ok(body: dict[str, Any]) -> Handler:
    """A handler that always answers with `body`."""

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=body)

    return handler


def capturing(body: dict[str, Any] | None = None) -> tuple[Handler, list[httpx.Request]]:
    """A handler plus the list it records every request into, for the tests
    that assert on what was sent rather than on what came back."""
    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(200, json=body if body is not None else payload())

    return handler, requests


# ---------------------------------------------------------------------------
# Parsing a successful search
# ---------------------------------------------------------------------------

def test_parses_a_successful_search_into_structured_results():
    ctx = make_ctx(ok(payload(result())))

    results = run(web_search(ctx, query="Cupertino Electric acquisition"))

    assert isinstance(results, WebSearchResults)
    assert results.query == "Cupertino Electric acquisition"
    [hit] = results.results
    assert isinstance(hit, WebSearchHit)
    assert hit.title == "Quanta Services to acquire Cupertino Electric"
    assert hit.url == "https://www.reuters.com/markets/deals/quanta-cupertino-2024-08-08/"
    assert hit.host == "www.reuters.com"
    assert hit.published == date(2024, 8, 8)


def test_strips_search_engine_highlighting_from_the_snippet():
    """Snippets arrive with the matched terms wrapped in <strong> and the rest
    HTML-escaped. Agents quote this text into a rationale, so it has to be
    text and not markup."""
    ctx = make_ctx(
        ok(payload(result(description="<strong>Blackstone</strong> &amp; peers bid for the asset")))
    )

    [hit] = run(web_search(ctx, query="Blackstone")).results

    assert hit.snippet == "Blackstone & peers bid for the asset"


def test_a_page_with_no_date_still_comes_back_as_a_source():
    """Most of the web is undated. A missing date is worth recording as
    unknown, not worth discarding an otherwise citeable page over."""
    ctx = make_ctx(ok(payload(result(page_age=None))))

    [hit] = run(web_search(ctx, query="Cupertino Electric")).results

    assert hit.published is None
    assert hit.url.startswith("https://")


def test_an_undateable_page_age_costs_the_date_not_the_hit():
    ctx = make_ctx(ok(payload(result(page_age="a while back"))))

    [hit] = run(web_search(ctx, query="Cupertino Electric")).results

    assert hit.published is None


def test_truncates_to_max_results():
    results_in = [result(url=f"https://example.test/{n}") for n in range(5)]
    ctx = make_ctx(ok(payload(*results_in)))

    results = run(web_search(ctx, query="Cupertino", max_results=2))

    assert len(results.results) == 2


def test_honours_a_max_results_of_a_full_page():
    """The provider serves up to MAX_RESULTS in a single page, so the whole
    range is reachable without paging."""
    results_in = [result(url=f"https://example.test/{n}") for n in range(MAX_RESULTS)]
    ctx = make_ctx(ok(payload(*results_in)))

    results = run(web_search(ctx, query="Cupertino", max_results=MAX_RESULTS))

    assert len(results.results) == MAX_RESULTS == 20


def test_skips_a_result_with_an_unusable_url():
    """The URL is what an agent cites, so a result that isn't a usable link is
    dropped rather than handed over as a dead source."""
    ctx = make_ctx(ok(payload(result(url="javascript:void(0)"), result(url="not a url"))))

    results = run(web_search(ctx, query="Cupertino"))

    assert results.results == []


def test_skips_a_result_with_no_snippet():
    """A hit with no extract is a bare link with nothing to weigh as evidence."""
    ctx = make_ctx(ok(payload(result(description=""))))

    assert run(web_search(ctx, query="Cupertino")).results == []


def test_no_matching_pages_is_an_empty_result_not_a_failure():
    """Zero hits is a fact the agent should be able to report, not an error
    that forces a retry. The provider omits the `web` block entirely when
    nothing matched, so that shape has to read as empty rather than broken."""
    ctx = make_ctx(ok({"type": "search", "query": {"original": "no such company anywhere"}}))

    results = run(web_search(ctx, query="no such company anywhere"))

    assert results.results == []


def test_skips_unparseable_results_and_keeps_the_rest():
    """A malformed result degrades into one fewer source rather than losing
    the whole search (CLAUDE.md: failures degrade, never crash)."""
    broken = result()
    del broken["url"]
    ctx = make_ctx(ok(payload(broken, result(url="https://example.test/good"))))

    results = run(web_search(ctx, query="Cupertino"))

    assert [h.url for h in results.results] == ["https://example.test/good"]


def test_a_skipped_result_does_not_cost_a_result_slot():
    """Skipping a malformed result reaches further down the list rather than
    returning fewer pages than the caller asked for."""
    broken = result(url="https://example.test/broken", description="")
    good = [result(url=f"https://example.test/good{n}") for n in range(2)]
    ctx = make_ctx(ok(payload(broken, *good)))

    results = run(web_search(ctx, query="Cupertino", max_results=2))

    assert [h.url for h in results.results] == [
        "https://example.test/good0",
        "https://example.test/good1",
    ]


# ---------------------------------------------------------------------------
# The request the tool sends
# ---------------------------------------------------------------------------

def test_sends_the_query_to_the_search_api():
    handler, captured = capturing()

    run(web_search(make_ctx(handler), query='"strategic alternatives" Cupertino'))

    [request] = captured
    assert str(request.url).startswith(BRAVE_SEARCH_URL)
    assert request.url.params["q"] == '"strategic alternatives" Cupertino'


def test_always_requests_a_full_page_so_skips_have_headroom():
    """`max_results` is applied after parsing, not pushed down to the API —
    otherwise a skipped result would shrink what the agent gets back."""
    handler, captured = capturing()

    run(web_search(make_ctx(handler), query="Cupertino", max_results=3))

    [request] = captured
    assert request.url.params["count"] == str(MAX_RESULTS)


def test_translates_freshness_into_the_providers_own_code():
    """`Freshness` is the project's vocabulary; the provider's shorthand stays
    inside the tool."""
    handler, captured = capturing()

    run(web_search(make_ctx(handler), query="Cupertino", freshness=Freshness.PAST_WEEK))

    [request] = captured
    assert request.url.params["freshness"] == "pw"


def test_omits_freshness_when_no_window_was_asked_for():
    handler, captured = capturing()

    run(web_search(make_ctx(handler), query="Cupertino"))

    [request] = captured
    assert "freshness" not in request.url.params


def test_authenticates_with_the_key_from_the_environment():
    handler, captured = capturing()

    run(web_search(make_ctx(handler), query="Cupertino"))

    [request] = captured
    assert request.headers["x-subscription-token"] == API_KEY


def test_the_schema_the_model_sees_constrains_every_argument():
    """The provider rejects an over-long query and an out-of-range count, and
    only knows four date windows. Declaring all of that on the arguments means
    the model is corrected by the tool schema rather than by a 4xx."""
    schema = Tool(web_search).function_schema.json_schema

    assert schema["properties"]["query"]["maxLength"] == MAX_QUERY_CHARS == 600
    assert schema["properties"]["max_results"]["maximum"] == MAX_RESULTS
    assert schema["$defs"]["Freshness"]["enum"] == [f.value for f in Freshness]


# ---------------------------------------------------------------------------
# Failure handling — the shared retry policy, then a typed failure
# ---------------------------------------------------------------------------

def test_shared_retry_recovers_from_a_transient_timeout():
    """The tool builds no retry logic of its own: the timeout is retried by
    the shared client from deps.py, and the tool just sees a success."""
    calls = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        if calls["n"] < 3:
            raise httpx.ConnectTimeout("simulated transient timeout", request=request)
        return httpx.Response(200, json=payload(result()))

    results = run(web_search(make_ctx(handler), query="Cupertino"))

    assert calls["n"] == 3  # two failures + one success: the retry happened
    assert len(results.results) == 1


def test_reports_a_terminal_failure_after_the_retries_are_exhausted():
    """Once the shared client gives up, the model should see the failure and
    move on to other sources rather than re-issue the same call."""
    calls = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        raise httpx.ConnectTimeout("always fails", request=request)

    with pytest.raises(ToolFailed, match="Web search"):
        run(web_search(make_ctx(handler), query="Cupertino"))

    assert calls["n"] == MAX_ATTEMPTS


def test_reports_a_terminal_failure_when_the_api_stays_5xx():
    """A 5xx that survives the retries arrives as a raised HTTPStatusError
    rather than a response (see deps._RetryingTransport), so this lands on the
    same path as a timeout — asserted here because it is the behaviour the
    agent sees, whichever branch produces it."""

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(503)

    with pytest.raises(ToolFailed, match="Web search"):
        run(web_search(make_ctx(handler), query="Cupertino"))


@pytest.mark.parametrize("status", [400, 422])
def test_asks_the_model_to_retry_when_the_query_is_rejected(status: int):
    """Every other parameter is set by this tool and always in range, so a
    rejected request means the query — something the model can fix by
    rephrasing, so it gets a retry prompt rather than a dead end."""

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(status, json={"type": "ErrorResponse"})

    with pytest.raises(ModelRetry, match="rejected"):
        run(web_search(make_ctx(handler), query="a " * 300))


@pytest.mark.parametrize("status", [401, 403])
def test_a_rejected_api_key_is_terminal(status: int):
    """Bad credentials are a deployment problem; no rephrasing fixes them."""

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(status, json={"type": "ErrorResponse"})

    with pytest.raises(ToolFailed):
        run(web_search(make_ctx(handler), query="Cupertino"))


def test_being_rate_limited_is_terminal():
    """429 means the quota is spent — re-issuing the call just spends the
    model's retry budget on a source that will keep saying no."""

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(429, json={"type": "ErrorResponse"})

    with pytest.raises(ToolFailed):
        run(web_search(make_ctx(handler), query="Cupertino"))


def test_a_missing_api_key_degrades_instead_of_crashing(monkeypatch: pytest.MonkeyPatch):
    """An unconfigured search provider costs the run some citations, not the
    analysis — the agent is told to source evidence elsewhere."""
    monkeypatch.delenv(API_KEY_ENV, raising=False)
    handler, captured = capturing()

    with pytest.raises(ToolFailed, match=API_KEY_ENV):
        run(web_search(make_ctx(handler), query="Cupertino"))

    assert captured == [], "should not have called the API without a key"


def test_unparseable_response_body_is_terminal():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, text="<html>not json</html>")

    with pytest.raises(ToolFailed):
        run(web_search(make_ctx(handler), query="Cupertino"))


def test_logs_the_search_with_the_run_id(caplog):
    ctx = make_ctx(ok(payload(result())))

    with caplog.at_level(logging.INFO, logger="web"):
        run(web_search(ctx, query="Cupertino"))

    web_logs = [r for r in caplog.records if r.name == "web"]
    assert web_logs, "expected the search to log a line"
    assert all(RUN_ID in r.getMessage() for r in web_logs)


def test_never_logs_the_api_key(caplog):
    """The key is a secret that passes through every call; a log line carrying
    it would leak it into wherever logs land."""
    ctx = make_ctx(ok(payload(result())))

    with caplog.at_level(logging.DEBUG):
        run(web_search(ctx, query="Cupertino"))

    assert all(API_KEY not in r.getMessage() for r in caplog.records)

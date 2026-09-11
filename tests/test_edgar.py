"""Unit tests for the EDGAR full-text search tool. No real network calls:
httpx.MockTransport stands in for EDGAR, so both the happy path and the
transient-failure path are deterministic.

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
from pydantic_ai.usage import RunUsage

from deps import MAX_ATTEMPTS, Deps, build_deps
from schemas import EdgarFiling, EdgarSearchResults
from tools.edgar import EDGAR_COVERAGE_START, EDGAR_FTS_URL, MAX_RESULTS, edgar_search

RUN_ID = "run-abc123"

T = TypeVar("T")
Handler = Callable[[httpx.Request], httpx.Response]


@pytest.fixture(autouse=True)
def _no_ambient_user_agent(monkeypatch: pytest.MonkeyPatch) -> None:
    """EDGAR_USER_AGENT is a real environment variable developers are told to
    set (README), so clear it — otherwise whether these tests pass depends on
    the shell they run in."""
    monkeypatch.delenv("EDGAR_USER_AGENT", raising=False)


def run(coro: Coroutine[Any, Any, T]) -> T:
    return asyncio.run(coro)


def make_ctx(handler: Handler) -> RunContext[Deps]:
    """A RunContext carrying real Deps whose HTTP client talks to `handler`
    instead of the network. The model/usage arguments are required by the
    dataclass but unused by tool functions."""
    deps = build_deps(RUN_ID, transport=httpx.MockTransport(handler))
    return RunContext(deps=deps, model=TestModel(), usage=RunUsage())


def hit(
    *,
    accession: str = "0001193125-24-196237",
    filename: str = "d860554dex991.htm",
    cik: str = "0001050915",
    display_name: str = "QUANTA SERVICES, INC.  (PWR)  (CIK 0001050915)",
    form: str = "8-K",
    file_date: str = "2024-08-08",
    description: str = "EX-99.1",
) -> dict:
    """One `hits.hits[]` entry shaped like a real EDGAR full-text search
    response (field names and `_id` format copied from the live API)."""
    return {
        "_index": "edgar_file",
        "_id": f"{accession}:{filename}",
        "_score": 13.4,
        "_source": {
            "ciks": [cik],
            "display_names": [display_name],
            "root_forms": [form],
            "form": form,
            "adsh": accession,
            "file_date": file_date,
            "file_type": "EX-99.1",
            "file_description": description,
            "biz_locations": ["Houston, TX"],
            "period_ending": "2024-08-07",
            "items": ["1.01"],
        },
    }


def payload(*hits: dict[str, Any], total: int | None = None) -> dict[str, Any]:
    return {
        "took": 12,
        "timed_out": False,
        "hits": {
            "total": {"value": total if total is not None else len(hits), "relation": "eq"},
            "max_score": 13.4,
            "hits": list(hits),
        },
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
    ctx = make_ctx(ok(payload(hit())))

    results = run(edgar_search(ctx, query='"Quanta Services"'))

    assert isinstance(results, EdgarSearchResults)
    assert results.query == '"Quanta Services"'
    assert results.total_hits == 1
    [filing] = results.filings
    assert isinstance(filing, EdgarFiling)
    assert filing.company == "QUANTA SERVICES, INC. (PWR)"
    assert filing.cik == "0001050915"
    assert filing.form == "8-K"
    assert filing.filed_at == date(2024, 8, 8)
    assert filing.description == "EX-99.1"
    assert filing.accession_number == "0001193125-24-196237"


def test_builds_a_citeable_document_url():
    """`sources` fields on TargetProfile/BuyerCandidate hold URLs, so each
    hit has to come back as a link a reader can actually open — the CIK is
    unpadded and the accession number has its dashes stripped in the path."""
    ctx = make_ctx(ok(payload(hit())))

    [filing] = run(edgar_search(ctx, query="acquisition")).filings

    assert filing.url == (
        "https://www.sec.gov/Archives/edgar/data/1050915"
        "/000119312524196237/d860554dex991.htm"
    )


def test_total_hits_reports_edgars_count_not_the_returned_slice():
    ctx = make_ctx(ok(payload(hit(), total=1325)))

    results = run(edgar_search(ctx, query="Cupertino"))

    assert results.total_hits == 1325
    assert len(results.filings) == 1


def test_truncates_to_max_results():
    hits = [hit(accession=f"0001193125-24-19623{n}") for n in range(5)]
    ctx = make_ctx(ok(payload(*hits)))

    results = run(edgar_search(ctx, query="Cupertino", max_results=2))

    assert len(results.filings) == 2
    assert results.total_hits == 5


def test_honours_a_max_results_above_one_page_of_the_default():
    """EDGAR answers with up to 100 hits in a single page, so the whole
    MAX_RESULTS range is reachable without paging."""
    hits = [hit(accession=f"00011931{n:02d}-24-196237") for n in range(40)]
    ctx = make_ctx(ok(payload(*hits)))

    results = run(edgar_search(ctx, query="Cupertino", max_results=MAX_RESULTS))

    assert len(results.filings) == MAX_RESULTS == 25


def test_skips_a_hit_whose_identifiers_are_malformed():
    """cik and accession_number are interpolated into the document URL, so a
    hit with a bad one is dropped rather than cited as a broken source."""
    ctx = make_ctx(ok(payload(hit(cik="12345"), hit(accession="not-an-accession"))))

    results = run(edgar_search(ctx, query="Cupertino"))

    assert results.filings == []
    assert results.total_hits == 2


def test_no_matching_filings_is_an_empty_result_not_a_failure():
    """Zero hits is a fact the agent should be able to report, not an error
    that forces a retry."""
    ctx = make_ctx(ok(payload()))

    results = run(edgar_search(ctx, query="no such company anywhere"))

    assert results.filings == []
    assert results.total_hits == 0


def test_skips_unparseable_hits_and_keeps_the_rest():
    """A malformed hit degrades into one fewer source rather than losing the
    whole search (CLAUDE.md: failures degrade, never crash)."""
    broken = hit()
    del broken["_source"]["file_date"]
    ctx = make_ctx(ok(payload(broken, hit(accession="0000320193-23-000106"))))

    results = run(edgar_search(ctx, query="Cupertino"))

    assert [f.accession_number for f in results.filings] == ["0000320193-23-000106"]


def test_a_skipped_hit_does_not_cost_a_result_slot():
    """Skipping a malformed hit reaches further down the result list rather
    than returning fewer filings than the caller asked for."""
    broken = hit(accession="0001111111-24-000001")
    del broken["_source"]["file_date"]
    good = [hit(accession=f"000222222{n}-24-000001") for n in range(2)]
    ctx = make_ctx(ok(payload(broken, *good)))

    results = run(edgar_search(ctx, query="Cupertino", max_results=2))

    assert [f.accession_number for f in results.filings] == [
        "0002222220-24-000001",
        "0002222221-24-000001",
    ]


# ---------------------------------------------------------------------------
# The request the tool sends
# ---------------------------------------------------------------------------

def test_sends_query_and_filters_as_edgar_search_params():
    handler, captured = capturing()

    run(
        edgar_search(
            make_ctx(handler),
            query='"strategic alternatives"',
            forms=["8-K", "10-K"],
            start_date=date(2023, 1, 1),
            end_date=date(2023, 12, 31),
        )
    )

    [request] = captured
    assert str(request.url).startswith(EDGAR_FTS_URL)
    params = request.url.params
    assert params["q"] == '"strategic alternatives"'
    assert params["forms"] == "8-K,10-K"
    assert params["startdt"] == "2023-01-01"
    assert params["enddt"] == "2023-12-31"


def test_omits_filters_that_were_not_asked_for():
    handler, captured = capturing()

    run(edgar_search(make_ctx(handler), query="Cupertino"))

    [request] = captured
    assert "forms" not in request.url.params
    assert "startdt" not in request.url.params
    assert "enddt" not in request.url.params


def test_fills_in_the_missing_half_of_a_one_sided_date_range():
    """EDGAR applies its date filter only when both bounds are present — a
    lone `startdt` is silently ignored and ancient filings come back — so a
    one-sided range gets the other bound filled in."""
    handler, captured = capturing()

    run(edgar_search(make_ctx(handler), query="Cupertino", start_date=date(2023, 1, 1)))
    run(edgar_search(make_ctx(handler), query="Cupertino", end_date=date(2010, 12, 31)))

    open_ended, open_started = captured
    assert open_ended.url.params["startdt"] == "2023-01-01"
    assert open_ended.url.params["enddt"] == date.today().isoformat()
    assert open_started.url.params["startdt"] == EDGAR_COVERAGE_START.isoformat()
    assert open_started.url.params["enddt"] == "2010-12-31"


def test_declares_a_user_agent_as_sec_requires():
    """SEC blocks or throttles automated traffic that does not identify
    itself, so every EDGAR request carries a descriptive User-Agent."""
    handler, captured = capturing()

    run(edgar_search(make_ctx(handler), query="Cupertino"))

    [request] = captured
    assert "buyer-landscape" in request.headers["user-agent"]


def test_user_agent_is_overridable_from_the_environment(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("EDGAR_USER_AGENT", "acme-research analyst@acme.test")
    handler, captured = capturing()

    run(edgar_search(make_ctx(handler), query="Cupertino"))

    [request] = captured
    assert request.headers["user-agent"] == "acme-research analyst@acme.test"


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
        return httpx.Response(200, json=payload(hit()))

    results = run(edgar_search(make_ctx(handler), query="Cupertino"))

    assert calls["n"] == 3  # two failures + one success: the retry happened
    assert len(results.filings) == 1


def test_reports_a_terminal_failure_after_the_retries_are_exhausted():
    """Once the shared client gives up, the model should see the failure and
    move on to other sources rather than re-issue the same call."""
    calls = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        raise httpx.ConnectTimeout("always fails", request=request)

    with pytest.raises(ToolFailed, match="EDGAR"):
        run(edgar_search(make_ctx(handler), query="Cupertino"))

    assert calls["n"] == MAX_ATTEMPTS


def test_reports_a_terminal_failure_when_edgar_stays_5xx():
    """A 5xx that survives the retries arrives as a raised HTTPStatusError
    rather than a response (see deps._RetryingTransport), so this lands on the
    same path as a timeout — asserted here because it is the behaviour the
    agent sees, whichever branch produces it."""

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(503)

    with pytest.raises(ToolFailed, match="EDGAR"):
        run(edgar_search(make_ctx(handler), query="Cupertino"))


def test_asks_the_model_to_retry_when_edgar_rejects_the_query():
    """A 400 means the query itself was malformed — something the model can
    fix by rephrasing, so it gets a retry prompt rather than a dead end."""

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(400, text="invalid query")

    with pytest.raises(ModelRetry, match="rejected"):
        run(edgar_search(make_ctx(handler), query='"unbalanced'))


def test_other_client_errors_are_terminal():
    """403/404 are not the model's fault and rephrasing will not help."""

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(403, text="blocked")

    with pytest.raises(ToolFailed):
        run(edgar_search(make_ctx(handler), query="Cupertino"))


def test_unparseable_response_body_is_terminal():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, text="<html>not json</html>")

    with pytest.raises(ToolFailed):
        run(edgar_search(make_ctx(handler), query="Cupertino"))


def test_logs_the_search_with_the_run_id(caplog):
    ctx = make_ctx(ok(payload(hit())))

    with caplog.at_level(logging.INFO, logger="edgar"):
        run(edgar_search(ctx, query="Cupertino"))

    edgar_logs = [r for r in caplog.records if r.name == "edgar"]
    assert edgar_logs, "expected the search to log a line"
    assert all(RUN_ID in r.getMessage() for r in edgar_logs)

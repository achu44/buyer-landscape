"""Unit tests for the yfinance comps tool. No real network calls: yfinance's
`Ticker` is replaced at the tool's module boundary with a stub that serves
canned `info` dicts (or raises), so the happy path, the retry path and the
not-found path are all deterministic.

Unlike tests/test_web.py there is no MockTransport to inject — yfinance owns
its HTTP stack — so the stub stands in for the whole library, and the retry
being exercised is the tool's own wrapper around it.
"""

import asyncio
import logging
import re
import time
from collections.abc import Coroutine
from typing import Any, TypeVar

import httpx
import pytest
from pydantic_ai import RunContext
from pydantic_ai.exceptions import ModelRetry, ToolFailed
from pydantic_ai.models.test import TestModel
from pydantic_ai.tools import Tool
from pydantic_ai.usage import RunUsage
from yfinance._http import requests as yf_requests
from yfinance.exceptions import YFDataException, YFRateLimitError

import deps
import tools.comps
from deps import MAX_ATTEMPTS, Deps, build_deps
from schemas import CompanyComps, CompsResults, SkippedTicker, SkipReason
from tools.comps import MAX_TICKERS, comps_lookup

RUN_ID = "run-abc123"

T = TypeVar("T")

# What yfinance hands back for a ticker Yahoo doesn't know: it logs the 404
# and returns a near-empty dict rather than raising (observed live against
# yfinance 1.7.0 with 'ZZZZNOTREAL').
UNKNOWN_TICKER_INFO: dict[str, Any] = {"trailingPegRatio": None}


def msft_info(**overrides: Any) -> dict[str, Any]:
    """A subset of a real `Ticker('MSFT').info` (values copied from the live
    API) — the keys this tool reads, plus a couple it ignores."""
    info: dict[str, Any] = {
        "symbol": "MSFT",
        "quoteType": "EQUITY",
        "longName": "Microsoft Corporation",
        "shortName": "Microsoft Corporation",
        "currency": "USD",
        "financialCurrency": "USD",
        "exchange": "NMS",
        "sector": "Technology",
        "industry": "Software - Infrastructure",
        "country": "United States",
        "marketCap": 3680323239936,
        "enterpriseValue": 3732293025792,
        "totalRevenue": 331839012864,
        "ebitda": 194237005824,
        "enterpriseToRevenue": 11.247,
        "enterpriseToEbitda": 19.215,
        "revenueGrowth": 0.177,
        "ebitdaMargins": 0.58534,
        "fullTimeEmployees": 228000,
    }
    info.update(overrides)
    return info


def run(coro: Coroutine[Any, Any, T]) -> T:
    return asyncio.run(coro)


def make_ctx() -> RunContext[Deps]:
    """A RunContext carrying real Deps. The comps tool never touches the HTTP
    client, so a transport that fails loudly proves it."""

    def no_http(request: httpx.Request) -> httpx.Response:
        raise AssertionError("the comps tool should not use the shared HTTP client")

    return RunContext(
        deps=build_deps(RUN_ID, transport=httpx.MockTransport(no_http)),
        model=TestModel(),
        usage=RunUsage(),
    )


class YahooStub:
    """Stands in for `yfinance.Ticker`. Each symbol maps to a queue of
    outcomes consumed one per `.info` read — a dict is served, an exception is
    raised — so a test can script "fail twice, then succeed"."""

    def __init__(self, outcomes: dict[str, list[dict[str, Any] | BaseException]]):
        self._outcomes = outcomes
        self.calls: dict[str, int] = {}

    def __call__(self, symbol: str) -> "YahooStub._Ticker":
        return YahooStub._Ticker(self, symbol)

    class _Ticker:
        def __init__(self, stub: "YahooStub", symbol: str):
            self._stub = stub
            self._symbol = symbol

        @property
        def info(self) -> dict[str, Any]:
            stub = self._stub
            stub.calls[self._symbol] = stub.calls.get(self._symbol, 0) + 1
            queue = stub._outcomes.get(self._symbol, [UNKNOWN_TICKER_INFO])
            outcome = queue.pop(0) if len(queue) > 1 else queue[0]
            if isinstance(outcome, BaseException):
                raise outcome
            return outcome


@pytest.fixture(autouse=True)
def _no_backoff(monkeypatch: pytest.MonkeyPatch) -> None:
    """Keep the shared backoff policy but without the real sleeps, so the
    retry tests run in milliseconds."""
    monkeypatch.setattr(deps, "BACKOFF_MULTIPLIER", 0)


def stub_yahoo(
    monkeypatch: pytest.MonkeyPatch, outcomes: dict[str, list[dict[str, Any] | BaseException]]
) -> YahooStub:
    stub = YahooStub(outcomes)
    monkeypatch.setattr(tools.comps.yf, "Ticker", stub)
    return stub


# ---------------------------------------------------------------------------
# Parsing a successful lookup
# ---------------------------------------------------------------------------

def test_parses_a_ticker_into_structured_comps(monkeypatch: pytest.MonkeyPatch):
    stub_yahoo(monkeypatch, {"MSFT": [msft_info()]})

    results = run(comps_lookup(make_ctx(), tickers=["MSFT"]))

    assert isinstance(results, CompsResults)
    [msft] = results.companies
    assert isinstance(msft, CompanyComps)
    assert msft.ticker == "MSFT"
    assert msft.name == "Microsoft Corporation"
    assert msft.quote_currency == "USD"
    assert msft.financial_currency == "USD"
    assert msft.sector == "Technology"
    assert msft.industry == "Software - Infrastructure"
    assert msft.market_cap == 3680323239936
    assert msft.enterprise_value == 3732293025792
    assert msft.revenue_ttm == 331839012864
    assert msft.ebitda_ttm == 194237005824
    assert msft.ev_to_revenue == 11.247
    assert msft.ev_to_ebitda == 19.215
    assert msft.revenue_growth == 0.177
    assert msft.ebitda_margin == 0.58534
    assert msft.source_url == "https://finance.yahoo.com/quote/MSFT"
    assert results.skipped == []


def test_normalises_how_the_model_wrote_a_ticker(monkeypatch: pytest.MonkeyPatch):
    """Models write tickers loosely — lowercase, padded, the same peer twice.
    Yahoo symbols are case-insensitive, so that is tidied up here rather than
    costing a retry or a duplicate row in the comp set."""
    stub = stub_yahoo(monkeypatch, {"MSFT": [msft_info()]})

    results = run(comps_lookup(make_ctx(), tickers=[" msft ", "MSFT"]))

    assert [c.ticker for c in results.companies] == ["MSFT"]
    assert results.companies[0].source_url == "https://finance.yahoo.com/quote/MSFT"
    assert stub.calls == {"MSFT": 1}


def test_the_schema_the_model_sees_constrains_the_tickers():
    """A ticker-shaped argument and a bounded comp set are declared on the tool
    itself, so the model is corrected by the schema before any lookup runs."""
    schema = Tool(comps_lookup).function_schema.json_schema

    tickers = schema["properties"]["tickers"]
    assert tickers["minItems"] == 1
    assert tickers["maxItems"] == MAX_TICKERS == 10
    assert tickers["items"]["maxLength"] == 15

    # Loosely written symbols pass, since the tool normalises them; company
    # names and index symbols are refused before any lookup is spent on them.
    ticker_shape = re.compile(tickers["items"]["pattern"])
    for accepted in ["MSFT", " msft ", "SIE.DE", "BRK-B"]:
        assert ticker_shape.search(accepted), accepted
    for refused in ["Microsoft Corp", "^GSPC", "EURUSD=X"]:
        assert not ticker_shape.search(refused), refused


def test_keeps_the_quote_currency_apart_from_the_reporting_currency(
    monkeypatch: pytest.MonkeyPatch,
):
    """An ADR trades in one currency and reports in another: TSM is quoted in
    USD but its revenue and EBITDA come back in TWD (values observed live).
    One currency field would have labelled trillions of TWD as USD."""
    tsm = msft_info(
        symbol="TSM",
        longName="Taiwan Semiconductor Manufacturing Company Limited",
        currency="USD",
        financialCurrency="TWD",
        totalRevenue=4440492343296,
    )
    stub_yahoo(monkeypatch, {"TSM": [tsm]})

    [company] = run(comps_lookup(make_ctx(), tickers=["TSM"])).companies

    assert company.quote_currency == "USD"
    assert company.financial_currency == "TWD"
    assert company.revenue_ttm == 4440492343296


def test_an_unrecognisable_currency_code_is_unknown_not_fatal(monkeypatch: pytest.MonkeyPatch):
    """London listings are quoted in pence ('GBp', observed live for BARC.L),
    which is a real quote unit worth passing on; a reporting currency that is
    not an ISO code is not, and becomes unknown rather than failing the company."""
    barclays = msft_info(symbol="BARC.L", currency="GBp", financialCurrency="pounds")
    stub_yahoo(monkeypatch, {"BARC.L": [barclays]})

    [company] = run(comps_lookup(make_ctx(), tickers=["BARC.L"])).companies

    assert company.quote_currency == "GBp"
    assert company.financial_currency is None


def test_a_company_with_gaps_in_its_data_still_comes_back(monkeypatch: pytest.MonkeyPatch):
    """Yahoo leaves real gaps — no EBITDA for a bank, no EV for a fund — and
    now and then serves a figure as the wrong type ('Infinity', a NaN). Each of
    those costs the one figure, recorded as unknown, never the company."""
    info = msft_info(
        ebitda=None,
        enterpriseToEbitda="Infinity",
        enterpriseToRevenue=float("nan"),
        marketCap=-1,  # impossible, so treated as unknown rather than failing validation
        revenueGrowth=True,
    )
    del info["enterpriseValue"]
    stub_yahoo(monkeypatch, {"JPM": [info]})

    [company] = run(comps_lookup(make_ctx(), tickers=["JPM"])).companies

    assert company.ebitda_ttm is None
    assert company.ev_to_ebitda is None
    assert company.ev_to_revenue is None
    assert company.market_cap is None
    assert company.revenue_growth is None
    assert company.enterprise_value is None
    assert company.revenue_ttm == 331839012864  # the usable figures survive


def test_an_unknown_ticker_is_reported_without_losing_the_rest_of_the_set(
    monkeypatch: pytest.MonkeyPatch,
):
    """yfinance answers an unknown symbol with a near-empty dict, not an error.
    One typo in a comp set should cost that one company — and be named, so the
    agent can tell 'mistyped' from 'no data' — not sink the whole lookup."""
    stub_yahoo(monkeypatch, {"MSFT": [msft_info()], "ZZZZNOTREAL": [UNKNOWN_TICKER_INFO]})

    results = run(comps_lookup(make_ctx(), tickers=["MSFT", "ZZZZNOTREAL"]))

    assert [c.ticker for c in results.companies] == ["MSFT"]
    assert results.skipped == [SkippedTicker(ticker="ZZZZNOTREAL", reason=SkipReason.NOT_FOUND)]


def test_a_quote_that_is_not_a_usable_company_is_skipped_not_fatal(monkeypatch: pytest.MonkeyPatch):
    """Yahoo does recognise symbols that are not a company this schema can
    describe — an index like '^GSPC'. That costs the one entry, reported as
    not found, rather than a validation error crashing the whole lookup."""
    index_info = msft_info(symbol="^GSPC", quoteType="INDEX", longName="S&P 500")
    stub_yahoo(monkeypatch, {"MSFT": [msft_info()], "^GSPC": [index_info]})

    results = run(comps_lookup(make_ctx(), tickers=["MSFT", "^GSPC"]))

    assert [c.ticker for c in results.companies] == ["MSFT"]
    assert results.skipped == [SkippedTicker(ticker="^GSPC", reason=SkipReason.NOT_A_COMPANY)]


def test_asks_the_model_to_retry_when_no_ticker_is_recognised(monkeypatch: pytest.MonkeyPatch):
    """Nothing recognised at all almost always means the model passed company
    names or made-up symbols — something it can fix, so it gets a correction
    prompt rather than an empty result it might read as 'no data exists'."""
    stub_yahoo(monkeypatch, {})

    with pytest.raises(ModelRetry, match="(?i)microsoft corp"):
        run(comps_lookup(make_ctx(), tickers=["Microsoft Corp"]))


# ---------------------------------------------------------------------------
# Failure handling — the shared retry policy, then a typed failure
# ---------------------------------------------------------------------------

# yfinance's HTTP backend is curl_cffi (or requests, as a fallback); either
# way `yfinance._http.requests` is the module whose exceptions reach us.
TRANSIENT_FAILURES = [
    pytest.param(yf_requests.exceptions.ConnectTimeout("simulated timeout"), id="timeout"),
    pytest.param(yf_requests.exceptions.ConnectionError("simulated reset"), id="connection"),
    pytest.param(YFRateLimitError(), id="rate-limited"),
]


@pytest.mark.parametrize("failure", TRANSIENT_FAILURES)
def test_retries_a_transient_failure_with_the_shared_policy(
    monkeypatch: pytest.MonkeyPatch, failure: BaseException
):
    stub = stub_yahoo(monkeypatch, {"MSFT": [failure, failure, msft_info()]})

    results = run(comps_lookup(make_ctx(), tickers=["MSFT"]))

    assert stub.calls["MSFT"] == 3  # two failures + one success: the retry happened
    assert [c.ticker for c in results.companies] == ["MSFT"]


def test_a_read_that_hangs_times_out_and_is_retried(monkeypatch: pytest.MonkeyPatch):
    """yfinance takes no timeout of its own worth waiting on, so the tool
    bounds each read and treats hitting that bound like any other transient
    failure."""
    monkeypatch.setattr(tools.comps, "FETCH_TIMEOUT_SECONDS", 0.05)
    calls = {"n": 0}

    def hang_once_then_answer(symbol: str) -> Any:
        calls["n"] += 1
        if calls["n"] == 1:
            time.sleep(0.5)
        return type("Ticker", (), {"info": msft_info()})()

    monkeypatch.setattr(tools.comps.yf, "Ticker", hang_once_then_answer)

    results = run(comps_lookup(make_ctx(), tickers=["MSFT"]))

    assert calls["n"] == 2  # the hung read timed out, the retry answered
    assert [c.ticker for c in results.companies] == ["MSFT"]


def test_reports_a_terminal_failure_after_the_retries_are_exhausted(monkeypatch: pytest.MonkeyPatch):
    """Once the retries give up Yahoo is genuinely unavailable: the model
    should see that and source its numbers elsewhere, not re-issue the call."""
    stub = stub_yahoo(
        monkeypatch, {"MSFT": [yf_requests.exceptions.ConnectTimeout("always fails")]}
    )

    with pytest.raises(ToolFailed, match="Yahoo Finance"):
        run(comps_lookup(make_ctx(), tickers=["MSFT"]))

    assert stub.calls["MSFT"] == MAX_ATTEMPTS


def test_one_unavailable_ticker_does_not_lose_the_rest_of_the_set(monkeypatch: pytest.MonkeyPatch):
    """A lookup that keeps failing for one ticker costs that ticker, named as
    unavailable, not the companies already fetched — ADR 0003 rejects losing a
    partial answer over one dead source."""
    stub = stub_yahoo(
        monkeypatch,
        {"MSFT": [msft_info()], "ORCL": [yf_requests.exceptions.ConnectTimeout("always fails")]},
    )

    results = run(comps_lookup(make_ctx(), tickers=["MSFT", "ORCL"]))

    assert [c.ticker for c in results.companies] == ["MSFT"]
    assert results.skipped == [SkippedTicker(ticker="ORCL", reason=SkipReason.UNAVAILABLE)]
    assert stub.calls["ORCL"] == MAX_ATTEMPTS


@pytest.mark.parametrize(
    "failure",
    [
        pytest.param(YFDataException("Failed to parse json response"), id="yahoo-data-error"),
        pytest.param(KeyError("quoteSummary"), id="yfinance-internal-error"),
    ],
)
def test_a_non_transient_yfinance_error_is_terminal_without_retrying(
    monkeypatch: pytest.MonkeyPatch, failure: BaseException
):
    """An answer Yahoo gave that yfinance couldn't read won't read any better
    a second time: no retries, and the model is told to fall back rather than
    the exception crashing the agent run (CLAUDE.md: degrade, never crash)."""
    stub = stub_yahoo(monkeypatch, {"MSFT": [failure]})

    with pytest.raises(ToolFailed, match="Yahoo Finance"):
        run(comps_lookup(make_ctx(), tickers=["MSFT"]))

    assert stub.calls["MSFT"] == 1


def test_logs_the_lookup_with_the_run_id(monkeypatch: pytest.MonkeyPatch, caplog):
    stub_yahoo(monkeypatch, {"MSFT": [YFRateLimitError(), msft_info()]})

    with caplog.at_level(logging.INFO, logger="comps"):
        run(comps_lookup(make_ctx(), tickers=["MSFT", "ZZZZNOTREAL"]))

    comps_logs = [r for r in caplog.records if r.name == "comps"]
    assert len(comps_logs) >= 3, "expected retry, not-found and completion lines"
    assert all(RUN_ID in r.getMessage() for r in comps_logs)

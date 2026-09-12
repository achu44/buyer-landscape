"""Comparable-company market data via yfinance — how a specialist grounds a
buyer's ability to pay (market cap, EV) or a comp set's trading multiples in
real numbers rather than recollection.

The one tool in `tools/` that does not go through `deps.http_client`:
yfinance owns its HTTP stack and accepts no injected client. It still follows
the house rules another way — the call is wrapped in the same tenacity policy
the shared client uses (`deps.retry_kwargs`), bounded by a timeout, logged
with the run's `run_id`, and failures reach the model as `ModelRetry` or
`ToolFailed` (docs/adr/0003-tool-failures-as-model-visible-exceptions.md).
"""

from __future__ import annotations

import asyncio
import logging
import math
from typing import Annotated, Any

from pydantic import Field, ValidationError
from pydantic_ai import RunContext
from pydantic_ai.exceptions import ModelRetry, ToolFailed
from tenacity import AsyncRetrying, retry_if_exception
import yfinance as yf
# yfinance re-exports whichever HTTP backend it loaded (curl_cffi, or requests
# as a fallback), so its exception classes are the ones that actually reach us.
from yfinance._http import requests as yf_requests
from yfinance.exceptions import YFRateLimitError

from deps import Deps, retry_kwargs
from schemas import CompanyComps, CompsResults, SkippedTicker, SkipReason

log = logging.getLogger("comps")

# A comp set is rarely wider than this, and each ticker is a sequential
# yfinance read, so the cap also bounds how long one tool call can take.
MAX_TICKERS = 10

# Matches CompanyComps.ticker, so a symbol the schema would reject is refused
# at the tool boundary instead of partway through a lookup.
MAX_TICKER_CHARS = 15

# yfinance's `info` makes two or three requests under the hood, so one read
# gets double the shared HTTP client's 10s per-request timeout. yfinance's own
# 30s-per-request default is too slow to leave an agent waiting on.
FETCH_TIMEOUT_SECONDS = 20.0

YAHOO_QUOTE_URL = "https://finance.yahoo.com/quote/{ticker}"

# Said on every terminal failure, as in tools/web.py: without it the model
# tends to re-issue the same call, which ADR 0003 exists to prevent.
_FALL_BACK = "Do not retry this call; source market data from filings or the web instead."


def _fetch_info(ticker: str) -> dict[str, Any]:
    """The blocking yfinance read, kept to one line so it is the only thing
    that runs on a worker thread."""
    return yf.Ticker(ticker).info


def _is_transient(exc: BaseException) -> bool:
    """The yfinance counterpart of `deps.is_transient_error`: a dropped
    connection, a timeout or Yahoo's rate limit may clear on the next attempt.
    Anything else yfinance raises is about the data, and retrying it just
    burns the budget."""
    return isinstance(
        exc,
        (
            yf_requests.exceptions.ConnectionError,
            yf_requests.exceptions.Timeout,
            YFRateLimitError,
            TimeoutError,
        ),
    )


async def _fetch_info_with_retry(ticker: str, run_id: str) -> dict[str, Any]:
    """Same attempts, backoff and retry log line as the shared HTTP client —
    only the test for what counts as transient differs, because yfinance
    raises its own backend's exceptions rather than httpx's. The timeout
    stops the wait, not the worker thread, which yfinance gives no way to
    cancel."""
    policy: dict[str, Any] = {**retry_kwargs(run_id, log), "retry": retry_if_exception(_is_transient)}
    async for attempt in AsyncRetrying(**policy):
        with attempt:
            return await asyncio.wait_for(
                asyncio.to_thread(_fetch_info, ticker), timeout=FETCH_TIMEOUT_SECONDS
            )
    raise AssertionError("unreachable: tenacity reraises once attempts run out")


def _number(info: dict[str, Any], key: str, *, non_negative: bool = False) -> float | None:
    """One figure from `info`, or None when it is missing or unusable. Yahoo
    occasionally serves 'Infinity' as a string, a NaN, or a figure no real
    company can have; an agent would quote any of those as fact, so each
    becomes unknown rather than failing validation and costing the company.
    (`bool` is excluded explicitly because it subclasses int.)"""
    value = info.get(key)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    if not math.isfinite(value) or (non_negative and value < 0):
        return None
    return float(value)


def _text(info: dict[str, Any], key: str) -> str | None:
    value = info.get(key)
    if not isinstance(value, str):
        return None
    return value.strip() or None


def _comps_from_info(ticker: str, info: dict[str, Any]) -> CompanyComps:
    return CompanyComps(
        ticker=ticker,
        name=_text(info, "longName") or _text(info, "shortName") or ticker,
        currency=_text(info, "currency"),
        sector=_text(info, "sector"),
        industry=_text(info, "industry"),
        market_cap=_number(info, "marketCap", non_negative=True),
        enterprise_value=_number(info, "enterpriseValue"),
        revenue_ttm=_number(info, "totalRevenue", non_negative=True),
        ebitda_ttm=_number(info, "ebitda"),
        ev_to_revenue=_number(info, "enterpriseToRevenue"),
        ev_to_ebitda=_number(info, "enterpriseToEbitda"),
        revenue_growth=_number(info, "revenueGrowth"),
        ebitda_margin=_number(info, "ebitdaMargins"),
        source_url=YAHOO_QUOTE_URL.format(ticker=ticker),
    )


async def comps_lookup(
    ctx: RunContext[Deps],
    tickers: Annotated[
        list[Annotated[str, Field(min_length=1, max_length=MAX_TICKER_CHARS)]],
        Field(min_length=1, max_length=MAX_TICKERS),
    ],
) -> CompsResults:
    """Look up market data for publicly listed companies: market cap,
    enterprise value, trailing revenue and EBITDA, and their EV multiples.

    Use it to judge whether a strategic buyer can afford the target, or to
    see what multiple comparable companies trade at. Private companies and
    financial sponsors have no ticker; use `web_search` for those.

    Args:
        tickers: Exchange ticker symbols, not company names — e.g. ['MSFT',
            'ORCL'], or with a Yahoo suffix for a non-US listing, 'SIE.DE'.
            A ticker that yields no data comes back in `skipped` with the
            reason — not found, not a company, or Yahoo unavailable — while
            the rest of the set is still returned.
    """
    deps = ctx.deps
    companies: list[CompanyComps] = []
    skipped: list[SkippedTicker] = []
    # Yahoo symbols are case-insensitive; normalising here keeps a loosely
    # written or repeated peer from costing a retry or a duplicate row.
    for ticker in dict.fromkeys(t.strip().upper() for t in tickers):
        try:
            info = await _fetch_info_with_retry(ticker, deps.run_id)
        except Exception as exc:
            # Broad on purpose: yfinance scrapes an unofficial API and can raise
            # almost anything when Yahoo changes a response, and none of it
            # should crash the agent run. If the error was transient the retries
            # already ran to exhaustion; if not, it was never going to improve.
            # Either way terminal, and the agent should fall back.
            # Either way it is terminal for this ticker only: the rest of the
            # comp set is still worth returning (ADR 0003).
            log.warning(
                "comps lookup failed: run_id=%s ticker=%s transient=%s error=%r",
                deps.run_id, ticker, _is_transient(exc), exc,
            )
            skipped.append(SkippedTicker(ticker=ticker, reason=SkipReason.UNAVAILABLE))
            continue
        if not info.get("quoteType"):
            # yfinance logs Yahoo's 404 and hands back a near-empty dict rather
            # than raising, so a missing quote type is how "no such symbol"
            # shows up. One typo costs one company, named so the agent can fix it.
            log.info("comps ticker not found: run_id=%s ticker=%s", deps.run_id, ticker)
            skipped.append(SkippedTicker(ticker=ticker, reason=SkipReason.NOT_FOUND))
            continue
        try:
            companies.append(_comps_from_info(ticker, info))
        except ValidationError as exc:
            # A quote Yahoo knows but this schema can't describe — an index
            # like '^GSPC'. As ADR 0003 has it for a malformed record: logged
            # and skipped, costing one entry rather than the lookup.
            log.warning(
                "skipping unusable comps quote: run_id=%s ticker=%s error=%s",
                deps.run_id, ticker, exc.errors(include_url=False),
            )
            skipped.append(SkippedTicker(ticker=ticker, reason=SkipReason.NOT_A_COMPANY))

    if not companies:
        tickers_tried = [s.ticker for s in skipped]
        if any(s.reason == SkipReason.UNAVAILABLE for s in skipped):
            # Yahoo itself failed, so nothing the model rewrites will help.
            raise ToolFailed(
                f"Yahoo Finance could not supply data for {tickers_tried!r}. {_FALL_BACK}"
            )
        # Nothing usable at all is almost always company names or invented
        # symbols rather than a real gap in Yahoo's coverage — fixable by the
        # model, and an empty result would read as "no market data exists".
        raise ModelRetry(
            f"Yahoo Finance had no listed company for any of {tickers_tried!r}. Pass exchange "
            "ticker symbols for companies (e.g. 'MSFT', or 'SIE.DE' for a non-US listing) — "
            "not company names or indices — and try again."
        )
    log.info(
        "comps lookup completed: run_id=%s returned=%d skipped=%s",
        deps.run_id, len(companies), [(s.ticker, s.reason.value) for s in skipped],
    )
    return CompsResults(companies=companies, skipped=skipped)

"""EDGAR full-text search — the profiler's and specialists' primary source of
hard evidence (filings they can cite) rather than model recall.

Attached to agents via constructor `tools=[...]` and reached through
`ctx.deps` so one tool serves `profiler_agent` and both specialists — see
docs/adr/0002-shared-deps-for-tool-resources.md. The shared client brings the
timeout, the tenacity retry policy, and the `run_id` log line with it, so
nothing here reimplements them.

EDGAR's full-text search API is the JSON endpoint behind
https://www.sec.gov/edgar/search/ — an Elasticsearch envelope whose
`hits.hits[]._source` entries this module flattens into `EdgarFiling`s.
"""

from __future__ import annotations

from datetime import date
import logging
import os
import re
from typing import Annotated, Any

import httpx
from pydantic import Field
from pydantic_ai import RunContext
from pydantic_ai.exceptions import ModelRetry, ToolFailed

from deps import Deps
from schemas import EdgarFiling, EdgarSearchResults

log = logging.getLogger("edgar")

EDGAR_FTS_URL = "https://efts.sec.gov/LATEST/search-index"
EDGAR_ARCHIVE_URL = "https://www.sec.gov/Archives/edgar/data"

# SEC requires automated traffic to identify itself and throttles or blocks
# requests that don't (https://www.sec.gov/os/accessing-edgar-data). It is
# configuration rather than a credential, so it lives in the environment with
# a working default instead of being required at import time.
DEFAULT_USER_AGENT = "buyer-landscape/0.1 (set EDGAR_USER_AGENT to your contact)"

# EDGAR's full-text index starts at 2001; it is the floor to use when a
# caller supplies only an end date (see `_date_params`).
EDGAR_COVERAGE_START = date(2001, 1, 1)

# EDGAR answers with up to 100 hits in one page, so this cap is about the
# agent's context rather than the API's limit — a specialist citing 50 filings
# per buyer is noise. Staying under one page is also why there is no paging.
MAX_RESULTS = 25

# Said on every terminal failure: without it the model tends to re-issue the
# same call, which ADR 0003 exists to prevent.
_FALL_BACK = "Do not retry this call; gather evidence from other sources instead."

# display_names look like "QUANTA SERVICES, INC.  (PWR)  (CIK 0001050915)".
# The CIK is returned as its own field, so strip that suffix and keep the
# ticker, which is useful for disambiguating similarly named companies.
_CIK_SUFFIX = re.compile(r"\s*\(CIK\s*\d+\)\s*$", re.IGNORECASE)


def _user_agent() -> str:
    return os.environ.get("EDGAR_USER_AGENT", DEFAULT_USER_AGENT)


def _company_name(source: dict[str, Any]) -> str:
    """First filer on the document; joint filings list co-registrants after
    it, and the first entry is the one the search matched on."""
    display_name = source["display_names"][0]
    return re.sub(r"\s{2,}", " ", _CIK_SUFFIX.sub("", display_name)).strip()


def _date_params(start_date: date | None, end_date: date | None) -> dict[str, str]:
    """EDGAR honors its date filter only when `startdt` and `enddt` are both
    present — send one alone and it is silently dropped, so a caller asking
    for "filings since 2023" would get 2001 filings back. Fill in whichever
    half is missing."""
    if start_date is None and end_date is None:
        return {}
    return {
        "startdt": (start_date or EDGAR_COVERAGE_START).isoformat(),
        "enddt": (end_date or date.today()).isoformat(),
    }


def _filing_from_hit(hit: dict[str, Any]) -> EdgarFiling:
    """Flatten one `hits.hits[]` entry. Raises KeyError/IndexError/TypeError,
    or a ValidationError for a field whose shape EdgarFiling rejects, on
    anything unexpected; the caller logs and skips those hits."""
    source = hit["_source"]
    # `_id` is "{accession-number}:{filename}" — the only place the document's
    # filename appears, and the Archives path needs both halves.
    accession, _, filename = hit["_id"].partition(":")
    cik = source["ciks"][0]

    return EdgarFiling(
        company=_company_name(source),
        cik=cik,
        form=source.get("form") or source["root_forms"][0],
        filed_at=date.fromisoformat(source["file_date"]),
        description=source.get("file_description") or source["file_type"],
        accession_number=accession,
        # Archives paths use the unpadded CIK and a dashless accession number.
        url=f"{EDGAR_ARCHIVE_URL}/{int(cik)}/{accession.replace('-', '')}/{filename}",
    )


async def edgar_search(
    ctx: RunContext[Deps],
    query: str,
    forms: list[str] | None = None,
    start_date: date | None = None,
    end_date: date | None = None,
    max_results: Annotated[int, Field(ge=1, le=MAX_RESULTS)] = 10,
) -> EdgarSearchResults:
    """Search the full text of SEC EDGAR filings and return matching
    documents, each with a URL you can cite as a source.

    Covers filings from 2001 onward. Use it for evidence a buyer actually
    exists in the target's space: past acquisitions, stated strategy,
    segment descriptions, or a competitor naming the target.

    Args:
        query: Words to search for. Wrap a phrase in double quotes for an
            exact-phrase match, e.g. '"strategic alternatives"'; without
            quotes, filings matching the words in any position come back.
        forms: Restrict to these filing types, e.g. ["8-K", "10-K", "DEFM14A"].
            Omit to search every type.
        start_date: Only filings filed on or after this date. Passing just one
            bound is fine; the other defaults to the edge of EDGAR's coverage.
        end_date: Only filings filed on or before this date.
        max_results: How many filings to return, most relevant first.
    """
    deps = ctx.deps
    params: dict[str, str] = {"q": query}
    if forms:
        params["forms"] = ",".join(forms)
    params.update(_date_params(start_date, end_date))

    try:
        response = await deps.http_client.get(
            EDGAR_FTS_URL, params=params, headers={"User-Agent": _user_agent()}
        )
    except httpx.HTTPError as exc:
        # The shared client already retried transient failures to exhaustion
        # (deps.py), so reaching here means EDGAR is genuinely unavailable:
        # terminal, and the agent should fall back to other sources.
        log.warning(
            "edgar search unavailable: run_id=%s query=%r error=%r", deps.run_id, query, exc
        )
        raise ToolFailed(f"EDGAR full-text search is unavailable ({exc!r}). {_FALL_BACK}") from exc

    if response.status_code == httpx.codes.BAD_REQUEST:
        # The query itself was malformed — rephrasing is something the model
        # can actually do, so ask it to.
        log.warning("edgar rejected query: run_id=%s query=%r", deps.run_id, query)
        raise ModelRetry(
            f"EDGAR rejected the search query {query!r}. Simplify it — plain words, "
            "balanced double quotes around any exact phrase — and search again."
        )
    if response.status_code >= httpx.codes.BAD_REQUEST:
        # Only ever a 4xx other than 400: the shared client turns a surviving
        # 5xx into an httpx.HTTPStatusError caught above, never a response.
        # 403 (blocked User-Agent) and 429 (throttled) land here, and neither
        # is something the model can rephrase its way out of.
        log.warning(
            "edgar search failed: run_id=%s query=%r status=%d",
            deps.run_id, query, response.status_code,
        )
        raise ToolFailed(f"EDGAR full-text search returned HTTP {response.status_code}. {_FALL_BACK}")

    try:
        hits = response.json()["hits"]
        total_hits = hits["total"]["value"]
        raw_hits = hits["hits"]
    except (ValueError, KeyError, TypeError) as exc:
        log.warning("edgar response unparseable: run_id=%s error=%r", deps.run_id, exc)
        raise ToolFailed(
            f"EDGAR returned a response this tool could not parse. {_FALL_BACK}"
        ) from exc

    filings: list[EdgarFiling] = []
    for hit in raw_hits:
        if len(filings) == max_results:
            break
        try:
            filings.append(_filing_from_hit(hit))
        except (KeyError, IndexError, TypeError, ValueError) as exc:
            # One odd hit costs one source, not the whole search — and the
            # next hit fills its place rather than the result set shrinking.
            # Narrow on purpose: a bug in the flattener should surface as a
            # crash, not quietly thin out every result set. (ValueError also
            # covers pydantic's ValidationError, which subclasses it.)
            log.warning(
                "skipping unparseable edgar hit: run_id=%s hit=%.200r error=%r",
                deps.run_id, hit, exc,
            )

    log.info(
        "edgar search completed: run_id=%s query=%r total_hits=%d returned=%d",
        deps.run_id, query, total_hits, len(filings),
    )
    return EdgarSearchResults(query=query, total_hits=total_hits, filings=filings)

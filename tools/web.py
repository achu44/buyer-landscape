"""General web search — the profiler's and specialists' source for the
evidence that never reaches an SEC filing: a private buyer's press release, a
sponsor's portfolio page, trade-press coverage of a comparable deal.

Sits alongside `tools/edgar.py` and follows the same two rules: it reaches the
HTTP client through `ctx.deps` rather than building one
(docs/adr/0002-shared-deps-for-tool-resources.md), so the timeout, the tenacity
retry policy and the `run_id` log line come for free; and it reports failure to
the model as `ModelRetry` or `ToolFailed`
(docs/adr/0003-tool-failures-as-model-visible-exceptions.md) rather than
crashing the run.

Backed by the Brave Web Search API (https://api.search.brave.com/res/v1/web/search),
a plain JSON GET endpoint whose `web.results[]` entries this module flattens
into `WebSearchHit`s. The provider's vocabulary stops here: `Freshness` is the
project's own enum, mapped to Brave's shorthand codes below, so swapping
providers is a change to this file and not to `schemas.py` or any agent prompt.
"""

from __future__ import annotations

from datetime import date, datetime
import html
import logging
import os
import re
from typing import Annotated, Any
from urllib.parse import urlsplit

import httpx
from pydantic import Field
from pydantic_ai import RunContext
from pydantic_ai.exceptions import ModelRetry, ToolFailed

from deps import Deps
from schemas import Freshness, WebSearchHit, WebSearchResults

log = logging.getLogger("web")

BRAVE_SEARCH_URL = "https://api.search.brave.com/res/v1/web/search"

# Unlike EDGAR's User-Agent this is a credential, so it has no default — an
# unset key is a configuration failure the tool reports rather than papers over.
API_KEY_ENV = "BRAVE_SEARCH_API_KEY"

# Brave serves at most 20 results in one page, which is also about as many
# links as a specialist can weigh per query — so the cap is the page and there
# is no paging.
MAX_RESULTS = 20

# Brave's GET endpoint rejects a `q` over 600 characters or 75 words. The
# character half is expressible in the tool signature, so the model is told by
# the schema; the word half can only show up as a rejected request, which
# `web_search` turns back into a ModelRetry.
MAX_QUERY_CHARS = 600

# Said on every terminal failure: without it the model tends to re-issue the
# same call, which ADR 0003 exists to prevent.
_FALL_BACK = "Do not retry this call; gather evidence from other sources instead."

# Brave's shorthand for its date filter. Kept as a lookup rather than as the
# enum's values so `Freshness` stays readable to the model choosing it.
_FRESHNESS_CODES = {
    Freshness.PAST_DAY: "pd",
    Freshness.PAST_WEEK: "pw",
    Freshness.PAST_MONTH: "pm",
    Freshness.PAST_YEAR: "py",
}

# Titles and descriptions come back with the matched terms wrapped in
# <strong> and everything else HTML-escaped.
_TAGS = re.compile(r"<[^>]+>")


def _plain_text(markup: str) -> str:
    """Agents quote a snippet verbatim into a rationale, so hand them text
    rather than the search engine's highlighting markup. Tags are stripped
    before entities are unescaped, so an escaped `&lt;b&gt;` in the page's own
    text survives as literal text instead of being mistaken for a tag."""
    return html.unescape(_TAGS.sub("", markup)).strip()


def _published_on(result: dict[str, Any]) -> date | None:
    """`page_age` is an ISO-8601 timestamp when Brave could date the page and
    absent when it could not. Most of the web is undated, and the date is
    context rather than the reason to cite a page — so anything unreadable
    costs the date, not the hit."""
    raw = result.get("page_age")
    if not isinstance(raw, str):
        return None
    try:
        return datetime.fromisoformat(raw.replace("Z", "+00:00")).date()
    except ValueError:
        return None


def _hit_from_result(result: dict[str, Any]) -> WebSearchHit:
    """Flatten one `web.results[]` entry. Raises KeyError/TypeError, or a
    ValidationError for a field whose shape WebSearchHit rejects, on anything
    unexpected; the caller logs and skips those results."""
    url = result["url"]
    if not isinstance(url, str):
        raise TypeError(f"url is {type(url).__name__}, not str")

    return WebSearchHit(
        title=_plain_text(result["title"]),
        url=url,
        # A result with no description is a bare link with nothing to weigh as
        # evidence, so WebSearchHit's min_length rejects it here and the caller
        # reaches further down the page for one that carries text.
        snippet=_plain_text(result.get("description") or ""),
        # Derived rather than read from Brave's `meta_url.hostname`: one code
        # path, and it cannot disagree with the URL actually being cited.
        host=urlsplit(url).hostname or "",
        published=_published_on(result),
    )


async def web_search(
    ctx: RunContext[Deps],
    query: Annotated[str, Field(min_length=1, max_length=MAX_QUERY_CHARS)],
    freshness: Freshness | None = None,
    max_results: Annotated[int, Field(ge=1, le=MAX_RESULTS)] = 10,
) -> WebSearchResults:
    """Search the web and return matching pages, each with a URL you can cite
    as a source.

    Use it for evidence that never reaches an SEC filing: a private company's
    press release, a sponsor's portfolio page, trade-press coverage of a
    comparable deal, or anything about a company that does not file with the
    SEC. For filings by a public company, prefer `edgar_search`.

    Args:
        query: What to search for, in plain words — under 75 words. Wrap a
            phrase in double quotes for an exact-phrase match, e.g.
            '"portfolio company" industrial services'.
        freshness: Only return pages published within this window. Omit for no
            date restriction.
        max_results: How many pages to return, most relevant first.
    """
    deps = ctx.deps
    api_key = os.environ.get(API_KEY_ENV)
    if not api_key:
        # A deployment problem, not a query problem: no rephrasing fixes it, so
        # the run degrades to its other sources instead of crashing.
        log.error("web search unconfigured: run_id=%s missing=%s", deps.run_id, API_KEY_ENV)
        raise ToolFailed(f"Web search is not configured ({API_KEY_ENV} is unset). {_FALL_BACK}")

    params: dict[str, str | int] = {
        "q": query,
        # A full page regardless of `max_results`, so a skipped result costs a
        # slot in the payload rather than a source the agent gets back.
        "count": MAX_RESULTS,
        # Brave otherwise mixes in news/video/discussion blocks this tool does
        # not read; asking for web results only keeps the response to the part
        # that becomes a citation.
        "result_filter": "web",
    }
    if freshness is not None:
        params["freshness"] = _FRESHNESS_CODES[freshness]

    try:
        response = await deps.http_client.get(
            BRAVE_SEARCH_URL,
            params=params,
            headers={"Accept": "application/json", "X-Subscription-Token": api_key},
        )
    except httpx.HTTPError as exc:
        # The shared client already retried transient failures to exhaustion
        # (deps.py), so reaching here means the search API is genuinely
        # unavailable: terminal, and the agent should fall back to other sources.
        log.warning(
            "web search unavailable: run_id=%s query=%r error=%r", deps.run_id, query, exc
        )
        raise ToolFailed(f"Web search is unavailable ({exc!r}). {_FALL_BACK}") from exc

    if response.status_code in (httpx.codes.BAD_REQUEST, httpx.codes.UNPROCESSABLE_ENTITY):
        # Every parameter other than `q` is set by this tool and always in
        # range, so a rejected request means the query — too long, or malformed
        # — which the model can fix by rephrasing.
        log.warning(
            "web search rejected query: run_id=%s query=%r status=%d",
            deps.run_id, query, response.status_code,
        )
        raise ModelRetry(
            f"The web search API rejected the query {query!r}. Shorten it — under 75 words, "
            "plain words, balanced double quotes around any exact phrase — and search again."
        )
    if response.status_code in (httpx.codes.UNAUTHORIZED, httpx.codes.FORBIDDEN):
        log.error(
            "web search credentials rejected: run_id=%s status=%d", deps.run_id, response.status_code
        )
        raise ToolFailed(
            f"Web search rejected this project's API key (HTTP {response.status_code}). {_FALL_BACK}"
        )
    if response.status_code >= httpx.codes.BAD_REQUEST:
        # Only ever a 4xx: the shared client turns a surviving 5xx into an
        # httpx.HTTPStatusError caught above, never a response. 429 (quota
        # spent) lands here, and the model cannot rephrase its way out of it.
        log.warning(
            "web search failed: run_id=%s query=%r status=%d",
            deps.run_id, query, response.status_code,
        )
        raise ToolFailed(f"Web search returned HTTP {response.status_code}. {_FALL_BACK}")

    try:
        body = response.json()
        # Brave omits the `web` block entirely when nothing matched, so a
        # missing key is zero results rather than a malformed response.
        raw_results = (body.get("web") or {}).get("results") or []
    except (ValueError, AttributeError, TypeError) as exc:
        log.warning("web search response unparseable: run_id=%s error=%r", deps.run_id, exc)
        raise ToolFailed(
            f"Web search returned a response this tool could not parse. {_FALL_BACK}"
        ) from exc

    hits: list[WebSearchHit] = []
    for result in raw_results:
        if len(hits) == max_results:
            break
        try:
            hits.append(_hit_from_result(result))
        except (KeyError, IndexError, TypeError, ValueError) as exc:
            # One odd result costs one source, not the whole search — and the
            # next result fills its place rather than the set shrinking. Narrow
            # on purpose: a bug in the flattener should surface as a crash, not
            # quietly thin out every result set. (ValueError also covers
            # pydantic's ValidationError, which subclasses it.)
            log.warning(
                "skipping unparseable web result: run_id=%s result=%.200r error=%r",
                deps.run_id, result, exc,
            )

    log.info(
        "web search completed: run_id=%s query=%r returned=%d", deps.run_id, query, len(hits)
    )
    return WebSearchResults(query=query, results=hits)

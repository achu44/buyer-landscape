"""Data contracts for the buyer-landscape multi-agent system.

Every agent's output must validate against one of these models.
Validation failures are fed back to the model for retry (Pydantic AI
does this automatically when an output fails to validate).
"""

from datetime import date, datetime
from enum import Enum

from pydantic import BaseModel, Field, model_validator


# ---------------------------------------------------------------------------
# Enums — constrain LLM outputs to known values
# ---------------------------------------------------------------------------

class BuyerType(str, Enum):
    STRATEGIC = "strategic"
    FINANCIAL_SPONSOR = "financial_sponsor"


class Confidence(str, Enum):
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


class NextStep(str, Enum):
    """The supervisor's routing vocabulary. LLM output picks one of these,
    and that choice determines the next step — this is the 'dynamic routing'
    requirement from the JD."""

    PROFILE_TARGET = "profile_target"
    FIND_STRATEGIC_BUYERS = "find_strategic_buyers"
    FIND_SPONSOR_BUYERS = "find_sponsor_buyers"
    DEEPEN_RESEARCH = "deepen_research"      # low-confidence loop, capped
    SYNTHESIZE = "synthesize"
    WRITE_TO_CRM = "write_to_crm"
    DONE = "done"


# ---------------------------------------------------------------------------
# Agent outputs
# ---------------------------------------------------------------------------

class TargetProfile(BaseModel):
    """Output of the target profiler agent."""

    name: str
    description: str = Field(min_length=100, description="What the company does, for whom, and how it makes money")
    sector: str
    subsector: str
    is_public: bool
    est_revenue_band: str = Field(description="e.g. '$50M-$100M'; 'unknown' if no basis")
    key_assets: list[str] = Field(min_length=1, max_length=8)
    geographies: list[str] = Field(min_length=1)
    sources: list[str] = Field(min_length=1, description="URLs or filings backing this profile")


class BuyerCandidate(BaseModel):
    """One buyer in the landscape. The Field constraints are the validation
    story: an out-of-range score or a thin rationale fails validation and
    triggers a model retry."""

    name: str
    buyer_type: BuyerType
    fit_score: int = Field(ge=0, le=100)
    rationale: str = Field(
        min_length=150,
        description="Why this buyer fits: strategic adjacency or thesis match, ability to pay, precedent activity",
    )
    signals: list[str] = Field(
        min_length=1,
        description="Concrete evidence: past acquisitions, stated strategy, portfolio companies, filings",
    )
    confidence: Confidence
    sources: list[str] = Field(default_factory=list)


class BuyerCandidateBatch(BaseModel):
    """Output of a specialist research agent (strategic or sponsor) for one
    routing step. Validated as a unit — rather than as a bare
    list[BuyerCandidate] — so a malformed batch (empty, wrong buyer_type,
    duplicate names) fails validation and triggers a Pydantic AI retry
    instead of reaching RunState unchecked."""

    buyer_type: BuyerType
    candidates: list[BuyerCandidate] = Field(min_length=1)

    @model_validator(mode="after")
    def check_types_and_uniqueness(self) -> "BuyerCandidateBatch":
        for c in self.candidates:
            if c.buyer_type != self.buyer_type:
                raise ValueError(
                    f"{c.name} is typed {c.buyer_type.value} but batch is declared {self.buyer_type.value}"
                )
        seen: set[str] = set()
        for c in self.candidates:
            key = c.name.lower()
            if key in seen:
                raise ValueError(f"duplicate candidate name in batch: {c.name}")
            seen.add(key)
        return self


class BuyerLandscape(BaseModel):
    """Output of the synthesis agent — the final deliverable."""

    target: TargetProfile
    strategic_buyers: list[BuyerCandidate]
    sponsor_buyers: list[BuyerCandidate]
    summary: str = Field(min_length=200, description="Banker-readable overview of the landscape")
    generated_at: datetime = Field(default_factory=datetime.utcnow)

    @model_validator(mode="after")
    def check_coverage_and_types(self) -> "BuyerLandscape":
        if len(self.strategic_buyers) + len(self.sponsor_buyers) < 5:
            raise ValueError("Landscape too thin: need at least 5 buyers total. Add more candidates.")
        for b in self.strategic_buyers:
            if b.buyer_type != BuyerType.STRATEGIC:
                raise ValueError(f"{b.name} is in strategic_buyers but typed {b.buyer_type}")
        for b in self.sponsor_buyers:
            if b.buyer_type != BuyerType.FINANCIAL_SPONSOR:
                raise ValueError(f"{b.name} is in sponsor_buyers but typed {b.buyer_type}")
        return self


# ---------------------------------------------------------------------------
# Tool contracts — what research tools accept and hand back
# ---------------------------------------------------------------------------

class EdgarFiling(BaseModel):
    """One hit from an EDGAR full-text search. Flattened out of EDGAR's
    Elasticsearch envelope so an agent sees a filing it can cite, not a
    `_source` blob it has to interpret."""

    company: str = Field(min_length=1)
    # cik and accession_number come from EDGAR and are interpolated into the
    # document URL, so their shape is constrained rather than merely described:
    # a hit that doesn't match is skipped instead of yielding a broken source.
    cik: str = Field(pattern=r"^\d{10}$", description="Zero-padded 10-digit SEC CIK")
    form: str = Field(
        min_length=1,
        description="This document's filing type — may be an amendment ('10-K/A') of the form searched",
    )
    filed_at: date
    description: str = Field(min_length=1, description="EDGAR's label for this document, e.g. 'EX-99.1'")
    accession_number: str = Field(pattern=r"^\d{10}-\d{2}-\d{6}$")
    url: str = Field(description="Direct link to the document — drop this straight into `sources`")


class EdgarSearchResults(BaseModel):
    """Output of the `edgar_search` tool."""

    query: str = Field(description="The query as sent to EDGAR, echoed so the agent can cite it")
    total_hits: int = Field(
        ge=0,
        description="Filings EDGAR matched in total; may exceed len(filings), which is capped",
    )
    filings: list[EdgarFiling] = Field(
        default_factory=list, description="Empty when nothing matched — that is a result, not an error"
    )


class Freshness(str, Enum):
    """How recent a web-search result has to be — an argument the model picks
    when calling `web_search`, as an enum so it cannot invent a window the
    search API would reject. The values are the project's own vocabulary;
    `tools/web.py` maps them to whatever codes its provider uses."""

    PAST_DAY = "past_day"
    PAST_WEEK = "past_week"
    PAST_MONTH = "past_month"
    PAST_YEAR = "past_year"


class WebSearchHit(BaseModel):
    """One result from a general web search, flattened to the parts an agent
    can use as evidence: text to weigh, a link to cite it from, who served it,
    and how old the page is."""

    title: str = Field(min_length=1)
    # The URL is the whole point of a hit — it is what lands in a `sources`
    # list — so its shape is constrained rather than merely described, and a
    # result that doesn't match is skipped instead of cited as a dead link.
    url: str = Field(
        pattern=r"^https?://",
        description="Direct link to the page — drop this straight into `sources`",
    )
    snippet: str = Field(
        min_length=1,
        description="The search engine's extract of the matching text, with markup stripped",
    )
    # Named `host`, not `source`: `sources` elsewhere in this file is a list of
    # cited URLs, and one field meaning two things confuses model and reader alike.
    host: str = Field(min_length=1, description="Host serving the page, e.g. 'www.reuters.com'")
    published: date | None = Field(
        default=None,
        description="When the page was published or last updated; null when the page doesn't say",
    )


class WebSearchResults(BaseModel):
    """Output of the `web_search` tool."""

    query: str = Field(description="The query as sent to the search API, echoed so the agent can cite it")
    results: list[WebSearchHit] = Field(
        default_factory=list, description="Empty when nothing matched — that is a result, not an error"
    )


class CompanyComps(BaseModel):
    """Market data for one public company, as a specialist weighs a buyer's
    ability to pay or the multiple a comparable trades at. Every figure is
    optional: Yahoo leaves gaps (no EBITDA for a bank, no EV for a fund), and
    an honest null beats a guessed number an agent would quote as fact."""

    ticker: str = Field(pattern=r"^[A-Z0-9][A-Z0-9.\-]{0,14}$", description="Yahoo symbol, e.g. 'MSFT' or 'SIE.DE'")
    name: str = Field(min_length=1)
    currency: str | None = Field(default=None, description="ISO code the figures below are quoted in")
    sector: str | None = None
    industry: str | None = None
    market_cap: float | None = Field(default=None, ge=0)
    # Can legitimately be negative for a company holding more cash than its
    # market cap plus debt, so unlike market_cap it carries no lower bound.
    enterprise_value: float | None = None
    revenue_ttm: float | None = Field(default=None, ge=0, description="Trailing-twelve-month revenue")
    ebitda_ttm: float | None = Field(default=None, description="Trailing-twelve-month EBITDA; may be negative")
    ev_to_revenue: float | None = None
    ev_to_ebitda: float | None = None
    revenue_growth: float | None = Field(default=None, description="Year-over-year, as a fraction: 0.18 is 18%")
    ebitda_margin: float | None = Field(default=None, description="As a fraction: 0.58 is 58%")
    source_url: str = Field(
        pattern=r"^https://finance\.yahoo\.com/quote/",
        description="Yahoo Finance quote page — drop this straight into `sources`",
    )


class SkipReason(str, Enum):
    """Why a requested ticker is missing from `CompsResults.companies`. Kept
    distinct because each points the agent somewhere different: fix the
    symbol, drop it from the comp set, or source the numbers elsewhere."""

    NOT_FOUND = "not_found"            # Yahoo has no quote: likely mistyped or not listed
    NOT_A_COMPANY = "not_a_company"    # Yahoo has a quote, but not a company's (e.g. an index)
    UNAVAILABLE = "unavailable"        # Yahoo failed or answered unreadably, after retries


class SkippedTicker(BaseModel):
    """One requested ticker that produced no `CompanyComps`, and why."""

    ticker: str = Field(min_length=1)
    reason: SkipReason


class CompsResults(BaseModel):
    """Output of the `comps_lookup` tool."""

    companies: list[CompanyComps] = Field(default_factory=list)
    skipped: list[SkippedTicker] = Field(
        default_factory=list,
        description="Requested tickers with no entry in `companies`, each with the reason",
    )


class RouterDecision(BaseModel):
    """Output of the supervisor agent. This IS the dynamic routing:
    the validated enum value below is dispatched by plain Python."""

    next_step: NextStep
    reason: str = Field(min_length=20, description="One or two sentences on why this step is next")


# ---------------------------------------------------------------------------
# Run state — passed to the supervisor each iteration
# ---------------------------------------------------------------------------

class RunState(BaseModel):
    """Everything the supervisor can see when deciding the next step.
    Keep it summarizable: the router prompt gets a compact rendering of this."""

    run_id: str
    target_input: str
    profile: TargetProfile | None = None
    strategic_buyers: list[BuyerCandidate] = Field(default_factory=list)
    sponsor_buyers: list[BuyerCandidate] = Field(default_factory=list)
    landscape: BuyerLandscape | None = None
    crm_written: bool = False
    deepen_rounds_used: int = 0
    steps_taken: list[str] = Field(default_factory=list)
    errors: list[str] = Field(default_factory=list)

    def summary_for_router(self) -> str:
        return (
            f"Target input: {self.target_input}\n"
            f"Profile built: {self.profile is not None}\n"
            f"Strategic buyers found: {len(self.strategic_buyers)} "
            f"(low-confidence: {sum(1 for b in self.strategic_buyers if b.confidence == Confidence.LOW)})\n"
            f"Sponsor buyers found: {len(self.sponsor_buyers)} "
            f"(low-confidence: {sum(1 for b in self.sponsor_buyers if b.confidence == Confidence.LOW)})\n"
            f"Landscape synthesized: {self.landscape is not None}\n"
            f"CRM written: {self.crm_written}\n"
            f"Deepen-research rounds used: {self.deepen_rounds_used} of 2\n"
            f"Steps so far: {', '.join(self.steps_taken) or 'none'}\n"
            f"Errors so far: {'; '.join(self.errors) or 'none'}"
        )

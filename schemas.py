"""Data contracts for the buyer-landscape multi-agent system.

Every agent's output must validate against one of these models.
Validation failures are fed back to the model for retry (Pydantic AI
does this automatically when an output fails to validate).
"""

from datetime import UTC, date, datetime
from enum import Enum
from typing import Annotated

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


class TransactionStatus(str, Enum):
    """Whether the target is still available to buy. A company under a signed
    agreement can still draw a topping bid; one whose sale has closed cannot,
    and a landscape of its potential buyers answers a question nobody asked."""

    INDEPENDENT = "independent"
    PENDING = "pending"        # a sale is agreed or announced but not closed
    ACQUIRED = "acquired"      # a sale has closed


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


def name_key(name: str) -> str:
    """How two buyer names are matched.

    Buyer names travel between agents as free text, and the same buyer comes
    back spaced or cased differently often enough that matching on the raw
    string would duplicate a candidate here and fail to replace one there.
    One spelling of the rule so every match in this file agrees.
    """
    return name.strip().lower()


# ---------------------------------------------------------------------------
# Agent outputs
# ---------------------------------------------------------------------------

class TargetDeal(BaseModel):
    """An agreed or completed sale of the target, as the profiler found it."""

    acquirer: str = Field(min_length=1)
    announced_on: date | None = None
    closed_on: date | None = None
    sources: list[str] = Field(min_length=1, description="Press releases or filings reporting the deal")


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
    transaction_status: TransactionStatus = Field(
        description="independent, pending (sale agreed or announced, not closed), or acquired (sale closed)"
    )
    deal: TargetDeal | None = Field(
        default=None, description="The agreed or completed sale. Required unless independent; omit when independent"
    )

    @model_validator(mode="after")
    def check_deal_matches_status(self) -> "TargetProfile":
        if self.transaction_status is TransactionStatus.INDEPENDENT:
            if self.deal is not None:
                raise ValueError("an independent target carries no deal; omit it or change the status")
        elif self.deal is None:
            raise ValueError(
                f"a {self.transaction_status.value} target needs its deal: acquirer and sources"
            )
        return self

    def describe_transaction(self) -> str:
        """One line on the target's availability, for the router prompt."""
        deal = self.deal
        if self.transaction_status is TransactionStatus.INDEPENDENT or deal is None:
            return "independent"
        if self.transaction_status is TransactionStatus.ACQUIRED:
            closed = f" (closed {deal.closed_on.isoformat()})" if deal.closed_on else ""
            return f"acquired by {deal.acquirer}{closed}"
        announced = f" (announced {deal.announced_on.isoformat()})" if deal.announced_on else ""
        return f"pending sale to {deal.acquirer}{announced}"


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
            key = name_key(c.name)
            if key in seen:
                raise ValueError(f"duplicate candidate name in batch: {c.name}")
            seen.add(key)
        return self


class DeepenVerdict(str, Enum):
    """What a deepen-research pass concluded about one low-confidence buyer.

    Deliberately binary: the round exists to decide whether a name stays in
    the landscape, and "maybe" would leave the same unsubstantiated buyer the
    round was called to resolve.
    """

    CONFIRMED = "confirmed"
    REFUTED = "refuted"


class DeepenedCandidate(BaseModel):
    """One low-confidence buyer, re-researched.

    The verdict is stated rather than inferred from what came back, so a
    candidate the pass could not substantiate is distinguishable from one it
    simply forgot to report on.
    """

    original_name: str = Field(
        min_length=1,
        description="The low-confidence candidate this finding is about, named exactly as it was given",
    )
    verdict: DeepenVerdict
    evidence: str = Field(
        min_length=80,
        description="What the pass found: the confirming evidence, or why the name could not be substantiated",
    )
    confirmed_candidate: BuyerCandidate | None = Field(
        default=None,
        description="The re-researched buyer, carrying the new evidence. Required when confirmed; omit when refuted",
    )

    @model_validator(mode="after")
    def check_verdict_matches_candidate(self) -> "DeepenedCandidate":
        if self.verdict is DeepenVerdict.REFUTED:
            if self.confirmed_candidate is not None:
                raise ValueError(
                    f"{self.original_name} is refuted but carries a candidate; a refuted buyer is dropped"
                )
            return self
        if self.confirmed_candidate is None:
            raise ValueError(
                f"{self.original_name} is confirmed but carries no candidate; return the re-researched buyer"
            )
        # The name is what the merge matches on, so a confirmation that renames
        # the buyer would have nothing to replace and would read as a new one.
        if name_key(self.confirmed_candidate.name) != name_key(self.original_name):
            raise ValueError(
                f"confirmed candidate is named {self.confirmed_candidate.name!r} but re-researched "
                f"{self.original_name!r}; keep the original name"
            )
        return self


class DeepenedBatch(BaseModel):
    """Output of a deepen-research pass over one buyer list.

    Unlike `BuyerCandidateBatch`, this may confirm nothing at all — a round
    that refutes every name it was given is a real and useful answer — but it
    must say something about at least one candidate, since it is only ever run
    when there are low-confidence buyers to resolve.
    """

    buyer_type: BuyerType
    findings: list[DeepenedCandidate] = Field(min_length=1)

    @model_validator(mode="after")
    def check_types_and_uniqueness(self) -> "DeepenedBatch":
        seen: set[str] = set()
        for f in self.findings:
            key = name_key(f.original_name)
            if key in seen:
                raise ValueError(f"two findings for the same candidate: {f.original_name}")
            seen.add(key)
            buyer = f.confirmed_candidate
            if buyer is not None and buyer.buyer_type != self.buyer_type:
                raise ValueError(
                    f"{buyer.name} is typed {buyer.buyer_type.value} but batch is "
                    f"declared {self.buyer_type.value}"
                )
        return self


# How long the banker-readable summary may be, in characters. The upper bound
# exists because an unbounded summary grows into a per-buyer essay — live run
# 01100708 wrote 2,334 words in 91 of the request's 120 seconds — and each
# buyer's own rationale already lives on its CRM Opportunity note. Set well
# above the ~400-word target the synthesis prompt asks for, so a compliant
# answer never trips it. One type, so the synthesis agent's output and the
# landscape it becomes can never disagree about what fits.
SUMMARY_MIN_CHARS = 200
SUMMARY_MAX_CHARS = 4000
LandscapeSummaryText = Annotated[
    str,
    Field(
        min_length=SUMMARY_MIN_CHARS,
        max_length=SUMMARY_MAX_CHARS,
        description="Banker-readable overview of the landscape, about 300-400 words",
    ),
]

# A landscape thinner than this is not worth handing a banker. Shared by
# `BuyerLandscape`'s validator and the supervisor's check before synthesis, so
# the step never spends a model call on a state that could not validate.
MIN_LANDSCAPE_BUYERS = 5


class LandscapeSummary(BaseModel):
    """Output of the synthesis agent: the one part of a landscape the run does
    not already hold. The profile and ranked buyers come from `RunState`
    (docs/adr/0005-synthesis-writes-only-the-summary.md)."""

    summary: LandscapeSummaryText


class BuyerLandscape(BaseModel):
    """The final deliverable, assembled by the supervisor from run state and
    the synthesis agent's summary."""

    target: TargetProfile
    strategic_buyers: list[BuyerCandidate]
    sponsor_buyers: list[BuyerCandidate]
    summary: LandscapeSummaryText
    generated_at: datetime = Field(default_factory=datetime.utcnow)

    @model_validator(mode="after")
    def check_coverage_and_types(self) -> "BuyerLandscape":
        if len(self.strategic_buyers) + len(self.sponsor_buyers) < MIN_LANDSCAPE_BUYERS:
            raise ValueError(
                f"Landscape too thin: need at least {MIN_LANDSCAPE_BUYERS} buyers total. "
                "Add more candidates."
            )
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


# Longest Yahoo symbol accepted — shared by CompanyComps.ticker and the
# `comps_lookup` argument so the two limits cannot drift apart.
TICKER_MAX_CHARS = 15


class CompanyComps(BaseModel):
    """Market data for one public company, as a specialist weighs a buyer's
    ability to pay or the multiple a comparable trades at. Every figure is
    optional: Yahoo leaves gaps (no EBITDA for a bank, no EV for a fund), and
    an honest null beats a guessed number an agent would quote as fact."""

    ticker: str = Field(
        pattern=rf"^[A-Z0-9][A-Z0-9.\-]{{0,{TICKER_MAX_CHARS - 1}}}$",
        description="Yahoo symbol, e.g. 'MSFT' or 'SIE.DE'",
    )
    name: str = Field(min_length=1)
    # Two currencies, because a cross-listed company trades in one and reports
    # in another: TSM is quoted in USD but reports in TWD.
    quote_currency: str | None = Field(
        default=None,
        min_length=1,
        description="Unit the share price is quoted in — usually ISO, but e.g. 'GBp' (pence) in London",
    )
    financial_currency: str | None = Field(
        default=None,
        pattern=r"^[A-Z]{3}$",
        description="ISO code of the reported figures: revenue_ttm and ebitda_ttm",
    )
    sector: str | None = Field(default=None, min_length=1)
    industry: str | None = Field(default=None, min_length=1)
    market_cap: float | None = Field(
        default=None,
        ge=0,
        description="As Yahoo reports it; for a cross-listed company check it against financial_currency before comparing",
    )
    # Can legitimately be negative for a company holding more cash than its
    # market cap plus debt, so unlike market_cap it carries no lower bound.
    enterprise_value: float | None = None
    revenue_ttm: float | None = Field(
        default=None, ge=0, description="Trailing-twelve-month revenue, in financial_currency"
    )
    ebitda_ttm: float | None = Field(
        default=None, description="Trailing-twelve-month EBITDA, in financial_currency; may be negative"
    )
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
    # Kept on the state, not only in logs, so a run's record says when it ran
    # and how long it took. `finished_at` is set as the loop exits, however
    # it exits.
    started_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    finished_at: datetime | None = None

    def record_buyers(
        self, buyer_type: BuyerType, candidates: list[BuyerCandidate]
    ) -> None:
        """File a specialist's candidates under the list for their buyer type.

        Replaces rather than appends: a re-run of a sourcing step re-sources
        that list from scratch, so appending would duplicate every buyer found
        twice. Deepening low-confidence buyers is its own step, and updates
        them in place (CONTEXT.md).
        """
        self.buyers_for(buyer_type)[:] = candidates

    def apply_deepening(
        self, buyer_type: BuyerType, findings: list[DeepenedCandidate]
    ) -> None:
        """Fold a deepen-research pass back into one buyer list, in place.

        A confirmed candidate replaces the low-confidence entry it
        re-researched; a refuted one is dropped, and so is a low-confidence
        entry the pass never reported on — silence is not substantiation, and
        the point of the round is that no unevidenced name survives it
        (CONTEXT.md). Entries that were never low-confidence are untouched,
        and order is preserved so the list still reads as it was sourced.
        """
        confirmed = {
            name_key(f.original_name): f.confirmed_candidate
            for f in findings
            if f.verdict is DeepenVerdict.CONFIRMED and f.confirmed_candidate is not None
        }
        kept: list[BuyerCandidate] = []
        for existing in self.buyers_for(buyer_type):
            if existing.confidence is not Confidence.LOW:
                kept.append(existing)
                continue
            replacement = confirmed.get(name_key(existing.name))
            if replacement is not None:
                # Names match case- and space-insensitively, so a pass that
                # re-types one would otherwise rename the buyer in the
                # landscape. Keep the spelling it was sourced under.
                kept.append(replacement.model_copy(update={"name": existing.name}))
        self.record_buyers(buyer_type, kept)

    def buyers_for(self, buyer_type: BuyerType) -> list[BuyerCandidate]:
        """The buyer list for one type — the read side of `record_buyers`."""
        if buyer_type is BuyerType.STRATEGIC:
            return self.strategic_buyers
        return self.sponsor_buyers

    def low_confidence_buyers(self, buyer_type: BuyerType) -> list[BuyerCandidate]:
        """The candidates in one list that a deepen-research round would target."""
        return [b for b in self.buyers_for(buyer_type) if b.confidence is Confidence.LOW]

    def low_confidence_count(self) -> int:
        """Low-confidence candidates across both lists.

        The router decides whether to deepen from this one number: a run with
        one shaky name in each list needs a round as much as one with two in
        the same list, and per-list counts alone hide that.
        """
        return sum(len(self.low_confidence_buyers(bt)) for bt in BuyerType)

    def summary_for_router(self, deepen_round_cap: int) -> str:
        """Render the state for the router prompt.

        The cap is passed in rather than read from here: the supervisor owns
        the guardrail, and taking it as an argument means the number quoted to
        the router cannot drift from the one the loop enforces.
        """
        return (
            f"Target input: {self.target_input}\n"
            f"Profile built: {self.profile is not None}\n"
            f"Target transaction status: "
            f"{self.profile.describe_transaction() if self.profile else 'unknown (no profile yet)'}\n"
            f"Strategic buyers found: {len(self.strategic_buyers)} "
            f"(low-confidence: {len(self.low_confidence_buyers(BuyerType.STRATEGIC))})\n"
            f"Sponsor buyers found: {len(self.sponsor_buyers)} "
            f"(low-confidence: {len(self.low_confidence_buyers(BuyerType.FINANCIAL_SPONSOR))})\n"
            f"Low-confidence buyers across both lists: {self.low_confidence_count()}\n"
            f"Landscape synthesized: {self.landscape is not None}\n"
            f"CRM written: {self.crm_written}\n"
            f"Deepen-research rounds used: {self.deepen_rounds_used} of {deepen_round_cap}\n"
            f"Steps so far: {', '.join(self.steps_taken) or 'none'}\n"
            f"Errors so far: {'; '.join(self.errors) or 'none'}"
        )

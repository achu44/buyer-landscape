"""Data contracts for the buyer-landscape multi-agent system.

Every agent's output must validate against one of these models.
Validation failures are fed back to the model for retry (Pydantic AI
does this automatically when an output fails to validate).
"""

from datetime import datetime
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

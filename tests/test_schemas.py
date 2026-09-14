"""Unit tests for schemas.py. No agent or network calls — pure validation."""

from typing import Any

import pytest
from pydantic import ValidationError

from schemas import (
    MIN_LANDSCAPE_BUYERS,
    BuyerCandidate,
    BuyerCandidateBatch,
    BuyerLandscape,
    BuyerType,
    Confidence,
    DeepenedBatch,
    DeepenedCandidate,
    DeepenVerdict,
    RunState,
)

RATIONALE = (
    "Strong strategic adjacency: the acquirer's existing product line "
    "overlaps directly with the target's core offering, and the company "
    "has a stated M&A thesis around this exact space, with ample balance "
    "sheet capacity to pay a premium."
)


def make_candidate(
    name: str,
    buyer_type: BuyerType = BuyerType.STRATEGIC,
    confidence: Confidence = Confidence.HIGH,
) -> BuyerCandidate:
    return BuyerCandidate(
        name=name,
        buyer_type=buyer_type,
        fit_score=80,
        rationale=RATIONALE,
        signals=["Announced a $500M acquisition budget for this vertical in Q1."],
        confidence=confidence,
    )


def make_state(**kwargs: Any) -> RunState:
    return RunState(run_id="run-test01", target_input="Chart Industries", **kwargs)


def test_happy_path_accepts_valid_batch():
    batch = BuyerCandidateBatch(
        buyer_type=BuyerType.STRATEGIC,
        candidates=[make_candidate("Acme Corp"), make_candidate("Beta Industries")],
    )
    assert len(batch.candidates) == 2


def test_rejects_empty_candidate_list():
    with pytest.raises(ValidationError):
        BuyerCandidateBatch(buyer_type=BuyerType.STRATEGIC, candidates=[])


def test_rejects_candidate_with_mismatched_buyer_type():
    with pytest.raises(ValidationError):
        BuyerCandidateBatch(
            buyer_type=BuyerType.STRATEGIC,
            candidates=[
                make_candidate("Acme Corp", BuyerType.STRATEGIC),
                make_candidate("Sponsor Capital", BuyerType.FINANCIAL_SPONSOR),
            ],
        )


def test_rejects_duplicate_candidate_names_case_insensitive():
    with pytest.raises(ValidationError):
        BuyerCandidateBatch(
            buyer_type=BuyerType.STRATEGIC,
            candidates=[make_candidate("Acme Corp"), make_candidate("acme corp")],
        )


def test_router_summary_reports_low_confidence_across_both_lists():
    """The router decides whether to deepen from one number, not two: a run
    with one shaky name in each list needs a round as much as one with two in
    the same list."""
    state = make_state(
        strategic_buyers=[
            make_candidate("Acme Corp", confidence=Confidence.LOW),
            make_candidate("Beta Industries"),
        ],
        sponsor_buyers=[
            make_candidate("Apollo", BuyerType.FINANCIAL_SPONSOR, Confidence.LOW),
        ],
    )

    assert "Low-confidence buyers across both lists: 2" in state.summary_for_router(2)


def test_router_summary_quotes_the_cap_it_is_given():
    """The router is told the cap its caller enforces, not a literal."""
    assert "Deepen-research rounds used: 0 of 3" in make_state().summary_for_router(3)


# ---------------------------------------------------------------------------
# Deepen-research: what a re-research pass may say, and how it folds back in
# ---------------------------------------------------------------------------

EVIDENCE = (
    "Found the 2024 8-K announcing the bolt-on and two trade-press reports "
    "naming the same acquirer, which together substantiate the appetite."
)


def confirmed(name: str, buyer_type: BuyerType = BuyerType.STRATEGIC,
              confidence: Confidence = Confidence.HIGH) -> DeepenedCandidate:
    return DeepenedCandidate(
        original_name=name,
        verdict=DeepenVerdict.CONFIRMED,
        evidence=EVIDENCE,
        confirmed_candidate=make_candidate(name, buyer_type, confidence),
    )


def refuted(name: str) -> DeepenedCandidate:
    return DeepenedCandidate(
        original_name=name, verdict=DeepenVerdict.REFUTED, evidence=EVIDENCE
    )


def test_confirmed_finding_must_carry_the_re_researched_buyer():
    with pytest.raises(ValidationError):
        DeepenedCandidate(
            original_name="Acme Corp",
            verdict=DeepenVerdict.CONFIRMED,
            evidence=EVIDENCE,
        )


def test_refuted_finding_must_not_carry_a_buyer():
    """A refuted name is dropped; handing one back alongside the verdict is a
    contradiction, not a candidate to keep."""
    with pytest.raises(ValidationError):
        DeepenedCandidate(
            original_name="Acme Corp",
            verdict=DeepenVerdict.REFUTED,
            evidence=EVIDENCE,
            confirmed_candidate=make_candidate("Acme Corp"),
        )


def test_confirmed_finding_must_keep_the_original_name():
    """Renaming the buyer would leave nothing to replace in the list."""
    with pytest.raises(ValidationError):
        DeepenedCandidate(
            original_name="Acme Corp",
            verdict=DeepenVerdict.CONFIRMED,
            evidence=EVIDENCE,
            confirmed_candidate=make_candidate("Acme Holdings"),
        )


def test_deepened_batch_rejects_a_finding_of_the_wrong_buyer_type():
    with pytest.raises(ValidationError):
        DeepenedBatch(
            buyer_type=BuyerType.STRATEGIC,
            findings=[confirmed("Apollo", BuyerType.FINANCIAL_SPONSOR)],
        )


def test_deepened_batch_rejects_two_findings_for_one_candidate():
    with pytest.raises(ValidationError):
        DeepenedBatch(
            buyer_type=BuyerType.STRATEGIC,
            findings=[confirmed("Acme Corp"), refuted("acme corp")],
        )


def test_confirmation_replaces_the_low_confidence_entry_in_place():
    state = make_state(
        strategic_buyers=[
            make_candidate("Acme Corp", confidence=Confidence.LOW),
            make_candidate("Beta Industries"),
        ]
    )

    state.apply_deepening(BuyerType.STRATEGIC, [confirmed("Acme Corp")])

    assert [b.name for b in state.strategic_buyers] == ["Acme Corp", "Beta Industries"]
    assert state.strategic_buyers[0].confidence is Confidence.HIGH


def test_refuted_candidate_is_dropped():
    state = make_state(
        strategic_buyers=[
            make_candidate("Acme Corp", confidence=Confidence.LOW),
            make_candidate("Beta Industries"),
        ]
    )

    state.apply_deepening(BuyerType.STRATEGIC, [refuted("Acme Corp")])

    assert [b.name for b in state.strategic_buyers] == ["Beta Industries"]


def test_low_confidence_candidate_the_pass_ignored_is_dropped():
    """Silence is not substantiation — an unreported name leaves too.

    The supervisor rejects a pass that skips a candidate before it gets here,
    so this is the last line of defense rather than the everyday path.
    """
    state = make_state(
        strategic_buyers=[
            make_candidate("Acme Corp", confidence=Confidence.LOW),
            make_candidate("Gamma Group", confidence=Confidence.LOW),
        ]
    )

    state.apply_deepening(BuyerType.STRATEGIC, [confirmed("Acme Corp")])

    assert [b.name for b in state.strategic_buyers] == ["Acme Corp"]


def test_deepening_one_list_leaves_the_other_alone():
    state = make_state(
        strategic_buyers=[make_candidate("Acme Corp", confidence=Confidence.LOW)],
        sponsor_buyers=[
            make_candidate("Apollo", BuyerType.FINANCIAL_SPONSOR, Confidence.LOW)
        ],
    )

    state.apply_deepening(BuyerType.STRATEGIC, [refuted("Acme Corp")])

    assert state.strategic_buyers == []
    assert [b.name for b in state.sponsor_buyers] == ["Apollo"]


def test_confirmation_keeps_the_name_the_buyer_was_sourced_under():
    """Names match case-insensitively, so a re-typed confirmation must not
    rename the buyer in the landscape."""
    state = make_state(
        strategic_buyers=[make_candidate("Air Liquide", confidence=Confidence.LOW)]
    )
    retyped = DeepenedCandidate(
        original_name="air liquide",
        verdict=DeepenVerdict.CONFIRMED,
        evidence=EVIDENCE,
        confirmed_candidate=make_candidate("air liquide"),
    )

    state.apply_deepening(BuyerType.STRATEGIC, [retyped])

    assert [b.name for b in state.strategic_buyers] == ["Air Liquide"]
    assert state.strategic_buyers[0].confidence is Confidence.HIGH


PROFILE = {
    "name": "Chart Industries",
    "description": (
        "Chart Industries engineers and manufactures cryogenic equipment used "
        "to liquefy, store and transport industrial gases, LNG and hydrogen."
    ),
    "sector": "Industrials",
    "subsector": "Cryogenic equipment",
    "is_public": True,
    "est_revenue_band": "$1B-$5B",
    "key_assets": ["Cryogenic tank manufacturing footprint"],
    "geographies": ["United States"],
    "sources": ["https://example.com/10-K"],
}

SUMMARY = (
    "Chart Industries draws interest from both strategic acquirers and financial "
    "sponsors. Industrial-gas majors see a direct adjacency in cryogenic storage "
    "and transport, while sponsors view it as a hydrogen infrastructure platform."
)


@pytest.mark.parametrize("buyer_count", [MIN_LANDSCAPE_BUYERS - 1, MIN_LANDSCAPE_BUYERS])
def test_landscape_needs_the_minimum_number_of_buyers(buyer_count: int):
    """The supervisor checks the same constant before spending a synthesis
    call, so one definition decides both."""
    strategic = [make_candidate(f"Strategic {i}") for i in range(buyer_count - 1)]
    sponsors = [make_candidate("Sponsor", BuyerType.FINANCIAL_SPONSOR)]

    def build() -> BuyerLandscape:
        return BuyerLandscape.model_validate(
            {"target": PROFILE, "strategic_buyers": strategic, "sponsor_buyers": sponsors, "summary": SUMMARY}
        )

    if buyer_count < MIN_LANDSCAPE_BUYERS:
        with pytest.raises(ValidationError, match="too thin"):
            build()
    else:
        assert len(build().strategic_buyers) == buyer_count - 1

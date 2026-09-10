"""Unit tests for schemas.py. No agent or network calls — pure validation."""

import pytest
from pydantic import ValidationError

from schemas import BuyerCandidate, BuyerCandidateBatch, BuyerType, Confidence

RATIONALE = (
    "Strong strategic adjacency: the acquirer's existing product line "
    "overlaps directly with the target's core offering, and the company "
    "has a stated M&A thesis around this exact space, with ample balance "
    "sheet capacity to pay a premium."
)


def make_candidate(name: str, buyer_type: BuyerType = BuyerType.STRATEGIC) -> BuyerCandidate:
    return BuyerCandidate(
        name=name,
        buyer_type=buyer_type,
        fit_score=80,
        rationale=RATIONALE,
        signals=["Announced a $500M acquisition budget for this vertical in Q1."],
        confidence=Confidence.HIGH,
    )


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

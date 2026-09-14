"""Unit tests for tools/crm.py — the mock CRM writer. Runs against a
temporary on-disk SQLite database; no agent or network calls."""

from datetime import UTC, datetime
import sqlite3

import pytest

from schemas import (
    BuyerCandidate,
    BuyerLandscape,
    BuyerType,
    Confidence,
    TargetProfile,
    TransactionStatus,
)
from tools.crm import CONFIDENCE_TO_STAGE, write_to_crm

RUN_ID = "run-abc123"

DESCRIPTION = (
    "A mid-market industrial manufacturer that designs and produces cryogenic "
    "equipment used in the liquefaction, storage, and transport of natural gas "
    "and hydrogen for energy and industrial customers worldwide."
)

RATIONALE = (
    "Strong strategic adjacency: the acquirer's existing product line overlaps "
    "directly with the target's core offering, and the company has a stated "
    "M&A thesis around this exact space, with ample balance sheet capacity to "
    "pay a premium for a bolt-on of this size."
)

SUMMARY = (
    "This banker-readable summary covers a mid-market cryogenic equipment "
    "manufacturer serving the LNG and hydrogen value chains. Five buyers were "
    "identified across strategic acquirers and financial sponsors, each "
    "substantiated with concrete evidence of fit and appetite. The strongest "
    "candidates show clear thesis alignment and precedent transaction activity."
)


def make_profile() -> TargetProfile:
    return TargetProfile(
        name="Chart Industries",
        description=DESCRIPTION,
        sector="Industrials",
        subsector="Cryogenic Equipment",
        is_public=True,
        transaction_status=TransactionStatus.INDEPENDENT,
        est_revenue_band="$3B-$4B",
        key_assets=["LNG liquefaction IP", "Hydrogen storage patents"],
        geographies=["United States", "Europe"],
        sources=["https://www.sec.gov/cgi-bin/browse-edgar?action=getcompany"],
    )


def make_candidate(
    name: str,
    buyer_type: BuyerType = BuyerType.STRATEGIC,
    confidence: Confidence = Confidence.HIGH,
    fit_score: int = 80,
) -> BuyerCandidate:
    return BuyerCandidate(
        name=name,
        buyer_type=buyer_type,
        fit_score=fit_score,
        rationale=RATIONALE,
        signals=[
            "Announced a $500M acquisition budget for this vertical in Q1.",
            "CEO cited LNG infrastructure as a stated growth priority on the last earnings call.",
        ],
        confidence=confidence,
    )


def make_landscape() -> BuyerLandscape:
    return BuyerLandscape(
        target=make_profile(),
        strategic_buyers=[
            make_candidate("Acme Corp", BuyerType.STRATEGIC, Confidence.HIGH),
            make_candidate("Beta Industries", BuyerType.STRATEGIC, Confidence.MEDIUM),
            make_candidate("Gamma Holdings", BuyerType.STRATEGIC, Confidence.LOW),
        ],
        sponsor_buyers=[
            make_candidate("Sponsor Capital", BuyerType.FINANCIAL_SPONSOR, Confidence.HIGH),
            make_candidate("Growth Equity Partners", BuyerType.FINANCIAL_SPONSOR, Confidence.MEDIUM),
        ],
        summary=SUMMARY,
    )


@pytest.fixture
def db_path(tmp_path) -> str:
    return str(tmp_path / "crm.db")


def _connect(db_path: str) -> sqlite3.Connection:
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    return conn


def test_creates_one_account_row_for_the_target(db_path):
    landscape = make_landscape()

    write_to_crm(landscape, RUN_ID, db_path)

    conn = _connect(db_path)
    rows = conn.execute("SELECT * FROM accounts").fetchall()
    assert len(rows) == 1
    row = rows[0]
    assert row["name"] == "Chart Industries"
    assert row["sector"] == "Industrials"
    assert row["subsector"] == "Cryogenic Equipment"
    assert row["is_public"] == 1
    assert row["est_revenue_band"] == "$3B-$4B"
    assert row["description"] == DESCRIPTION
    assert "United States" in row["geographies"]
    assert row["run_id"] == RUN_ID


def test_creates_one_opportunity_per_buyer_candidate_linked_to_account(db_path):
    landscape = make_landscape()

    write_to_crm(landscape, RUN_ID, db_path)

    conn = _connect(db_path)
    account_id = conn.execute("SELECT id FROM accounts").fetchone()["id"]
    rows = conn.execute("SELECT * FROM opportunities ORDER BY id").fetchall()

    assert len(rows) == 5  # 3 strategic + 2 sponsor
    assert all(r["account_id"] == account_id for r in rows)
    assert all(r["run_id"] == RUN_ID for r in rows)

    by_name = {r["buyer_name"]: r for r in rows}
    assert by_name["Acme Corp"]["buyer_type"] == BuyerType.STRATEGIC.value
    assert by_name["Acme Corp"]["fit_score"] == 80
    assert by_name["Sponsor Capital"]["buyer_type"] == BuyerType.FINANCIAL_SPONSOR.value


def test_confidence_maps_to_opportunity_stage(db_path):
    landscape = make_landscape()

    write_to_crm(landscape, RUN_ID, db_path)

    conn = _connect(db_path)
    rows = conn.execute("SELECT buyer_name, stage FROM opportunities").fetchall()
    by_name = {r["buyer_name"]: r["stage"] for r in rows}

    assert by_name["Acme Corp"] == CONFIDENCE_TO_STAGE[Confidence.HIGH]
    assert by_name["Beta Industries"] == CONFIDENCE_TO_STAGE[Confidence.MEDIUM]
    assert by_name["Gamma Holdings"] == CONFIDENCE_TO_STAGE[Confidence.LOW]


def test_each_opportunity_gets_a_note_with_rationale_and_signals(db_path):
    landscape = make_landscape()

    write_to_crm(landscape, RUN_ID, db_path)

    conn = _connect(db_path)
    opportunities = conn.execute("SELECT id, buyer_name FROM opportunities").fetchall()
    assert len(opportunities) == 5

    for opp in opportunities:
        notes = conn.execute(
            "SELECT * FROM notes WHERE parent_type = 'opportunity' AND opportunity_id = ?",
            (opp["id"],),
        ).fetchall()
        assert len(notes) == 1
        body = notes[0]["body"]
        assert RATIONALE in body
        assert "Announced a $500M acquisition budget for this vertical in Q1." in body
        assert notes[0]["run_id"] == RUN_ID
        assert notes[0]["account_id"] is None


def test_account_gets_a_note_with_the_overall_summary(db_path):
    landscape = make_landscape()

    write_to_crm(landscape, RUN_ID, db_path)

    conn = _connect(db_path)
    account_id = conn.execute("SELECT id FROM accounts").fetchone()["id"]
    notes = conn.execute(
        "SELECT * FROM notes WHERE parent_type = 'account' AND account_id = ?",
        (account_id,),
    ).fetchall()

    assert len(notes) == 1
    assert notes[0]["body"] == SUMMARY
    assert notes[0]["run_id"] == RUN_ID
    assert notes[0]["opportunity_id"] is None


def test_total_note_count_is_one_per_opportunity_plus_one_for_the_account(db_path):
    landscape = make_landscape()

    write_to_crm(landscape, RUN_ID, db_path)

    conn = _connect(db_path)
    count = conn.execute("SELECT COUNT(*) FROM notes").fetchone()[0]
    assert count == 6  # 5 opportunities + 1 account


def test_writing_twice_creates_two_independent_accounts(db_path):
    """write_to_crm always appends a fresh Account — it doesn't upsert by
    run_id, matching how a banker would log two separate analysis runs."""
    landscape = make_landscape()

    write_to_crm(landscape, RUN_ID, db_path)
    write_to_crm(landscape, "run-second", db_path)

    conn = _connect(db_path)
    assert conn.execute("SELECT COUNT(*) FROM accounts").fetchone()[0] == 2
    assert conn.execute("SELECT COUNT(*) FROM opportunities").fetchone()[0] == 10


def test_new_accounts_record_when_they_were_created(db_path):
    """Without a timestamp, past runs can only be ordered, never dated."""
    before = datetime.now(UTC)

    write_to_crm(make_landscape(), RUN_ID, db_path)

    (row,) = _connect(db_path).execute("SELECT created_at FROM accounts").fetchall()
    assert before <= datetime.fromisoformat(row["created_at"]) <= datetime.now(UTC)


def test_a_database_from_before_created_at_is_upgraded_in_place(db_path):
    """A CRM file written before accounts were dated keeps its history: the
    column is added, old accounts read as undated, and new ones are dated."""
    conn = sqlite3.connect(db_path)
    conn.executescript(
        """
        CREATE TABLE accounts (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            run_id TEXT NOT NULL,
            name TEXT NOT NULL,
            sector TEXT NOT NULL,
            subsector TEXT NOT NULL,
            is_public INTEGER NOT NULL,
            est_revenue_band TEXT NOT NULL,
            description TEXT NOT NULL,
            geographies TEXT NOT NULL
        );
        INSERT INTO accounts
            (run_id, name, sector, subsector, is_public, est_revenue_band, description, geographies)
        VALUES ('run-old', 'Old Target', 'Industrials', 'Pumps', 0, 'unknown', 'An older run.', '[]');
        """
    )
    conn.close()

    write_to_crm(make_landscape(), RUN_ID, db_path)

    rows = _connect(db_path).execute("SELECT run_id, created_at FROM accounts ORDER BY id").fetchall()
    assert [row["run_id"] for row in rows] == ["run-old", RUN_ID]
    assert rows[0]["created_at"] is None
    assert rows[1]["created_at"] is not None

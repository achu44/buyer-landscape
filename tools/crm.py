"""Mock CRM writer — persists a finished BuyerLandscape into a local SQLite
database shaped like Salesforce's Account/Opportunity/Note objects.

Called directly from the supervisor dispatch loop, not exposed as an agent
tool: writing an already-synthesized landscape involves no LLM judgment
(see docs/adr/0001-opportunity-per-buyer-candidate.md and CONTEXT.md).
"""

from __future__ import annotations

import json
import logging
import sqlite3

from schemas import BuyerCandidate, BuyerLandscape, Confidence, TargetProfile

log = logging.getLogger("crm")

# A BuyerCandidate has no CRM-native "stage" — confidence is our best proxy
# for how substantiated the opportunity is, so it drives the stage a banker
# would see when opening the record.
CONFIDENCE_TO_STAGE: dict[Confidence, str] = {
    Confidence.HIGH: "Qualification",
    Confidence.MEDIUM: "Prospecting",
    Confidence.LOW: "Unqualified",
}

_SCHEMA = """
CREATE TABLE IF NOT EXISTS accounts (
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

CREATE TABLE IF NOT EXISTS opportunities (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    account_id INTEGER NOT NULL REFERENCES accounts(id),
    run_id TEXT NOT NULL,
    buyer_name TEXT NOT NULL,
    buyer_type TEXT NOT NULL,
    fit_score INTEGER NOT NULL,
    stage TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS notes (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id TEXT NOT NULL,
    parent_type TEXT NOT NULL CHECK (parent_type IN ('account', 'opportunity')),
    account_id INTEGER REFERENCES accounts(id),
    opportunity_id INTEGER REFERENCES opportunities(id),
    body TEXT NOT NULL
);
"""


def init_db(conn: sqlite3.Connection) -> None:
    """Create the accounts/opportunities/notes tables if they don't exist."""
    conn.executescript(_SCHEMA)


def _insert_account(conn: sqlite3.Connection, target: TargetProfile, run_id: str) -> int:
    cursor = conn.execute(
        """
        INSERT INTO accounts
            (run_id, name, sector, subsector, is_public, est_revenue_band, description, geographies)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            run_id,
            target.name,
            target.sector,
            target.subsector,
            int(target.is_public),
            target.est_revenue_band,
            target.description,
            json.dumps(target.geographies),
        ),
    )
    return cursor.lastrowid


def _insert_opportunity(
    conn: sqlite3.Connection, account_id: int, candidate: BuyerCandidate, run_id: str
) -> int:
    cursor = conn.execute(
        """
        INSERT INTO opportunities
            (account_id, run_id, buyer_name, buyer_type, fit_score, stage)
        VALUES (?, ?, ?, ?, ?, ?)
        """,
        (
            account_id,
            run_id,
            candidate.name,
            candidate.buyer_type.value,
            candidate.fit_score,
            CONFIDENCE_TO_STAGE[candidate.confidence],
        ),
    )
    return cursor.lastrowid


def _candidate_note_body(candidate: BuyerCandidate) -> str:
    signals = "\n".join(f"- {s}" for s in candidate.signals)
    return f"{candidate.rationale}\n\nSignals:\n{signals}"


def _insert_note(
    conn: sqlite3.Connection,
    run_id: str,
    body: str,
    *,
    account_id: int | None = None,
    opportunity_id: int | None = None,
) -> None:
    parent_type = "account" if opportunity_id is None else "opportunity"
    conn.execute(
        """
        INSERT INTO notes (run_id, parent_type, account_id, opportunity_id, body)
        VALUES (?, ?, ?, ?, ?)
        """,
        (run_id, parent_type, account_id, opportunity_id, body),
    )


def write_to_crm(landscape: BuyerLandscape, run_id: str, db_path: str) -> None:
    """Persist a finished BuyerLandscape: one Account, one Opportunity per
    buyer across both lists, a Note per Opportunity carrying that buyer's
    rationale and signals, and a Note on the Account carrying the overall
    summary. Always appends a fresh Account rather than upserting by
    run_id — each call represents one logged analysis run."""
    conn = sqlite3.connect(db_path)
    try:
        init_db(conn)
        with conn:
            account_id = _insert_account(conn, landscape.target, run_id)
            _insert_note(conn, run_id, landscape.summary, account_id=account_id)

            candidates = [*landscape.strategic_buyers, *landscape.sponsor_buyers]
            for candidate in candidates:
                opportunity_id = _insert_opportunity(conn, account_id, candidate, run_id)
                _insert_note(
                    conn, run_id, _candidate_note_body(candidate), opportunity_id=opportunity_id
                )

        log.info(
            "crm write completed: run_id=%s account_id=%d opportunities=%d",
            run_id, account_id, len(candidates),
        )
    finally:
        conn.close()

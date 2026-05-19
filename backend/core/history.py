"""
history.py — SQLite-backed analysis history store.

Uses stdlib sqlite3 only — no ORM, no extra deps.
DB file: lexguard.db in the backend working directory.
Each row stores the full JSON report so it can be replayed without re-analyzing.

Schema:
    analyses(
        id         TEXT PRIMARY KEY,   -- UUID
        client_id  TEXT NOT NULL,      -- anonymous browser UUID from localStorage
        filename   TEXT NOT NULL,
        created_at TEXT NOT NULL,      -- ISO-8601 UTC
        doc_type   TEXT NOT NULL,
        score      REAL NOT NULL,
        red_count  INTEGER NOT NULL,
        yellow_count INTEGER NOT NULL,
        green_count  INTEGER NOT NULL,
        report_json  TEXT NOT NULL     -- full AnalyzeResponse JSON
    )
"""
from __future__ import annotations

import json
import logging
import sqlite3
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

DB_PATH = Path(__file__).parent.parent / "lexguard.db"


def _conn() -> sqlite3.Connection:
    conn = sqlite3.connect(str(DB_PATH))
    conn.row_factory = sqlite3.Row
    return conn


def init_db() -> None:
    with _conn() as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS analyses (
                id           TEXT PRIMARY KEY,
                client_id    TEXT NOT NULL,
                filename     TEXT NOT NULL,
                created_at   TEXT NOT NULL,
                doc_type     TEXT NOT NULL,
                score        REAL NOT NULL,
                red_count    INTEGER NOT NULL,
                yellow_count INTEGER NOT NULL,
                green_count  INTEGER NOT NULL,
                report_json  TEXT NOT NULL
            )
        """)
        conn.execute("CREATE INDEX IF NOT EXISTS idx_client ON analyses(client_id, created_at DESC)")
        conn.commit()
    logger.info("History DB initialised", extra={"path": str(DB_PATH)})


def save_analysis(client_id: str, response_dict: dict[str, Any]) -> str:
    """Persist a completed analysis. Returns the new record ID."""
    record_id = str(uuid.uuid4())
    report = response_dict.get("report", {})
    with _conn() as conn:
        conn.execute(
            """INSERT INTO analyses
               (id, client_id, filename, created_at, doc_type, score,
                red_count, yellow_count, green_count, report_json)
               VALUES (?,?,?,?,?,?,?,?,?,?)""",
            (
                record_id,
                client_id,
                response_dict.get("filename", "document"),
                datetime.now(timezone.utc).isoformat(),
                report.get("document_type", "unknown"),
                report.get("overall_score", 0.0),
                report.get("red_count", 0),
                report.get("yellow_count", 0),
                report.get("green_count", 0),
                json.dumps(response_dict),
            ),
        )
        conn.commit()
    return record_id


def list_analyses(client_id: str, limit: int = 20) -> list[dict[str, Any]]:
    """Return summary rows for a client (newest first, no full JSON)."""
    with _conn() as conn:
        rows = conn.execute(
            """SELECT id, filename, created_at, doc_type, score,
                      red_count, yellow_count, green_count
               FROM analyses
               WHERE client_id = ?
               ORDER BY created_at DESC
               LIMIT ?""",
            (client_id, limit),
        ).fetchall()
    return [dict(r) for r in rows]


def get_analysis(record_id: str, client_id: str) -> dict[str, Any] | None:
    """Return the full report for one record, scoped to the client."""
    with _conn() as conn:
        row = conn.execute(
            "SELECT report_json FROM analyses WHERE id = ? AND client_id = ?",
            (record_id, client_id),
        ).fetchone()
    if row is None:
        return None
    return json.loads(row["report_json"])


def delete_analysis(record_id: str, client_id: str) -> bool:
    """Delete a record. Returns True if something was deleted."""
    with _conn() as conn:
        cur = conn.execute(
            "DELETE FROM analyses WHERE id = ? AND client_id = ?",
            (record_id, client_id),
        )
        conn.commit()
    return cur.rowcount > 0

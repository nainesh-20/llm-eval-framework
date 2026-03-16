"""
SQLite-backed results store.

Schema:
  eval_runs        — one row per (model, dataset) run
  eval_results     — one row per (run, metric, sample)
  red_team_results — one row per (run, category, prompt)
"""

from __future__ import annotations

import json
import sqlite3
import uuid
from pathlib import Path
from typing import Optional

CREATE_SCHEMA = """
CREATE TABLE IF NOT EXISTS eval_runs (
    id          TEXT PRIMARY KEY,
    model_name  TEXT NOT NULL,
    dataset     TEXT NOT NULL,
    config_path TEXT,
    created_at  TEXT DEFAULT (datetime('now'))
);

CREATE TABLE IF NOT EXISTS eval_results (
    id         TEXT PRIMARY KEY,
    run_id     TEXT NOT NULL REFERENCES eval_runs(id),
    metric     TEXT NOT NULL,
    score      REAL NOT NULL,
    reason     TEXT,
    passed     INTEGER NOT NULL,
    threshold  REAL,
    sample_id  INTEGER,
    metadata   TEXT,
    created_at TEXT DEFAULT (datetime('now'))
);

CREATE TABLE IF NOT EXISTS red_team_results (
    id         TEXT PRIMARY KEY,
    run_id     TEXT NOT NULL REFERENCES eval_runs(id),
    category   TEXT NOT NULL,
    prompt     TEXT NOT NULL,
    response   TEXT,
    passed     INTEGER NOT NULL,
    score      REAL,
    reason     TEXT,
    created_at TEXT DEFAULT (datetime('now'))
);
"""


class SQLiteStore:
    def __init__(self, db_path: str = "results/evals.db"):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_schema()

    # ── Schema ────────────────────────────────────────────────────────────────

    def _init_schema(self) -> None:
        with self._connect() as conn:
            conn.executescript(CREATE_SCHEMA)

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    # ── Writes ────────────────────────────────────────────────────────────────

    def create_run(
        self,
        model_name: str,
        dataset: str,
        config_path: Optional[str] = None,
    ) -> str:
        run_id = str(uuid.uuid4())
        with self._connect() as conn:
            conn.execute(
                "INSERT INTO eval_runs (id, model_name, dataset, config_path) VALUES (?, ?, ?, ?)",
                (run_id, model_name, dataset, config_path),
            )
        return run_id

    def save_eval_result(
        self,
        run_id: str,
        metric: str,
        score: float,
        reason: str,
        passed: bool,
        threshold: float,
        sample_id: int = 0,
        metadata: Optional[dict] = None,
    ) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO eval_results
                    (id, run_id, metric, score, reason, passed, threshold, sample_id, metadata)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    str(uuid.uuid4()),
                    run_id,
                    metric,
                    score,
                    reason,
                    int(passed),
                    threshold,
                    sample_id,
                    json.dumps(metadata or {}),
                ),
            )

    def save_red_team_result(
        self,
        run_id: str,
        category: str,
        prompt: str,
        response: str,
        passed: bool,
        score: float,
        reason: str,
    ) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO red_team_results
                    (id, run_id, category, prompt, response, passed, score, reason)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    str(uuid.uuid4()),
                    run_id,
                    category,
                    prompt,
                    response,
                    int(passed),
                    score,
                    reason,
                ),
            )

    # ── Reads ─────────────────────────────────────────────────────────────────

    def get_run_summary(self, run_id: str) -> dict:
        with self._connect() as conn:
            run = conn.execute(
                "SELECT * FROM eval_runs WHERE id = ?", (run_id,)
            ).fetchone()
            metrics = conn.execute(
                """
                SELECT
                    metric,
                    AVG(score)         AS avg_score,
                    COUNT(*)           AS count,
                    SUM(passed)        AS passed_count
                FROM eval_results
                WHERE run_id = ?
                GROUP BY metric
                """,
                (run_id,),
            ).fetchall()
        return {
            "run": dict(run) if run else {},
            "metrics": [dict(m) for m in metrics],
        }

    def get_red_team_summary(self, run_id: str) -> list[dict]:
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT
                    category,
                    COUNT(*)    AS total,
                    SUM(passed) AS passed_count,
                    AVG(score)  AS avg_score
                FROM red_team_results
                WHERE run_id = ?
                GROUP BY category
                """,
                (run_id,),
            ).fetchall()
        return [dict(r) for r in rows]

    def get_all_runs_leaderboard(self) -> list[dict]:
        """Returns aggregated scores per (model, metric) for the dashboard."""
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT
                    r.id           AS run_id,
                    r.model_name,
                    r.created_at,
                    e.metric,
                    AVG(e.score)   AS avg_score,
                    COUNT(e.id)    AS sample_count,
                    SUM(e.passed)  AS passed_count
                FROM eval_runs r
                LEFT JOIN eval_results e ON r.id = e.run_id
                GROUP BY r.id, e.metric
                ORDER BY r.created_at DESC
                """
            ).fetchall()
        return [dict(r) for r in rows]

    def get_latest_run_id(self, model_name: Optional[str] = None) -> Optional[str]:
        # Secondary sort by rowid ensures correct ordering within the same second
        with self._connect() as conn:
            if model_name:
                row = conn.execute(
                    "SELECT id FROM eval_runs WHERE model_name = ? ORDER BY created_at DESC, rowid DESC LIMIT 1",
                    (model_name,),
                ).fetchone()
            else:
                row = conn.execute(
                    "SELECT id FROM eval_runs ORDER BY created_at DESC, rowid DESC LIMIT 1"
                ).fetchone()
        return row["id"] if row else None

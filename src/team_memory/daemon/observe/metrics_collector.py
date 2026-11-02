"""MetricsCollector — SQLite-backed metrics storage.

Records system metrics and provides query/summary/cleanup.
Supports cross-DB snapshots for metrics stored in PostgreSQL.
"""

from __future__ import annotations

import json
import logging
import sqlite3
from datetime import datetime, timedelta, timezone

logger = logging.getLogger(__name__)

_CREATE_TABLE = """CREATE TABLE IF NOT EXISTS metrics (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT NOT NULL,
    value REAL NOT NULL,
    labels TEXT DEFAULT '{}',
    recorded_at TEXT DEFAULT (datetime('now'))
)"""

_CREATE_INDEX = """CREATE INDEX IF NOT EXISTS idx_metrics_name ON metrics(name)"""


class MetricsCollector:
    """Collects and queries system metrics from SQLite."""

    MAX_DAYS = 30

    def __init__(self, db_path: str, pg_url: str | None = None) -> None:
        self._db = sqlite3.connect(db_path, check_same_thread=False)
        self._db.execute("PRAGMA journal_mode=WAL")
        self._db.execute(_CREATE_TABLE)
        self._db.execute(_CREATE_INDEX)
        self._db.commit()
        self._pg_url = pg_url

    def record(self, name: str, value: float, labels: dict | None = None) -> None:
        """Record a metric value. Disk full -> log error, no raise."""
        try:
            labels_json = json.dumps(labels or {})
            self._db.execute(
                "INSERT INTO metrics (name, value, labels, recorded_at) VALUES (?, ?, ?, ?)",
                (name, value, labels_json, datetime.now(timezone.utc).isoformat()),
            )
            self._db.commit()
        except sqlite3.Error as e:
            logger.error("metrics_record_failed name=%s error=%s", name, e)

    def record_search_telemetry(
        self,
        *,
        returned_tokens: int | None = None,
        saved_tokens_pct: float | None = None,
        stage_metrics: dict | None = None,
        stage_counts: dict | None = None,
    ) -> None:
        """Record search pipeline telemetry metrics."""
        if returned_tokens is not None:
            self.record("recall_returned_tokens", float(returned_tokens))
        if saved_tokens_pct is not None:
            self.record("recall_saved_tokens_pct", float(saved_tokens_pct))
        if isinstance(stage_metrics, dict) and stage_metrics.get("total_ms") is not None:
            self.record("pipeline_total_ms", float(stage_metrics["total_ms"]))
        if isinstance(stage_counts, dict) and stage_counts.get("graph_unique_added") is not None:
            self.record("graph_unique_added", float(stage_counts["graph_unique_added"]))

    def record_mutation_action(self, action: str, project: str = "default") -> None:
        """Record a memory mutation state machine action (ADD/MERGE/SUPERSEDE/NOOP)."""
        self.record("memory_mutation_action", 1.0, labels={"action": action, "project": project})

    def query(self, name: str | None = None, hours: int = 24) -> list[dict]:
        """Query recent metrics (pinned to MAX_DAYS max)."""
        hours = min(hours, self.MAX_DAYS * 24)
        cutoff = (datetime.now(timezone.utc) - timedelta(hours=hours)).isoformat()
        cols = ["name", "value", "labels", "recorded_at"]
        if name:
            rows = self._db.execute(
                "SELECT name, value, labels, recorded_at FROM metrics "
                "WHERE name = ? AND recorded_at > ? ORDER BY recorded_at DESC",
                (name, cutoff),
            ).fetchall()
        else:
            rows = self._db.execute(
                "SELECT name, value, labels, recorded_at FROM metrics WHERE recorded_at > ? ORDER BY recorded_at DESC",
                (cutoff,),
            ).fetchall()

        results = []
        seen = set()
        for row in rows:
            d = dict(zip(cols, row))
            key = (d["name"], d["labels"])
            if key not in seen:  # dedup: return latest per name+labels
                d["value"] = float(d["value"])
                seen.add(key)
                results.append(d)
        return results

    def summary(self, hours: int = 24) -> dict:
        """Aggregated summary. Returns {metric_name: {label_key: sum_value}}."""
        hours = min(hours, self.MAX_DAYS * 24)
        cutoff = (datetime.now(timezone.utc) - timedelta(hours=hours)).isoformat()
        rows = self._db.execute(
            "SELECT name, labels, SUM(value) as total FROM metrics WHERE recorded_at > ? GROUP BY name, labels",
            (cutoff,),
        ).fetchall()

        result: dict = {}
        for name, labels_json, total in rows:
            labels = json.loads(labels_json) if labels_json else {}
            label_key = "_".join(f"{k}={v}" for k, v in sorted(labels.items())) or "_total"
            if name not in result:
                result[name] = {}
            result[name][label_key] = float(total)
        return result

    def cleanup(self) -> int:
        """Delete records older than MAX_DAYS. Returns count of deleted rows."""
        cutoff = (datetime.now(timezone.utc) - timedelta(days=self.MAX_DAYS)).isoformat()
        cursor = self._db.execute("DELETE FROM metrics WHERE recorded_at < ?", (cutoff,))
        self._db.commit()
        return cursor.rowcount

    async def cross_db_snapshot(self) -> None:
        """Snapshot cross-DB metrics from PostgreSQL. No-op if no pg_url."""
        if not self._pg_url:
            return
        try:
            import asyncpg

            conn = await asyncpg.connect(self._pg_url)
            try:
                # search_hit_rate
                row = await conn.fetchrow(
                    "SELECT COUNT(*) FILTER (WHERE was_used = true) AS used, COUNT(*) AS total FROM search_logs"
                )
                if row and row["total"] > 0:
                    rate = row["used"] / row["total"]
                    self.record("search_hit_rate", rate)

                try:
                    tok_row = await conn.fetchrow(
                        "SELECT AVG(returned_tokens) AS avg_tokens FROM search_logs WHERE returned_tokens IS NOT NULL"
                    )
                    if tok_row and tok_row["avg_tokens"] is not None:
                        self.record("avg_returned_tokens", float(tok_row["avg_tokens"]))

                    sup_row = await conn.fetchrow(
                        "SELECT COUNT(*) FILTER (WHERE valid_until IS NOT NULL) AS superseded_count FROM experiences"
                    )
                    if sup_row and sup_row["superseded_count"] is not None:
                        self.record("superseded_experiences_count", float(sup_row["superseded_count"]))
                except Exception:
                    pass
            finally:
                await conn.close()
        except Exception as e:
            logger.warning("cross_db_snapshot failed (non-critical): %s", e)

"""Payload capture — log raw hook payloads for contract derivation.

数据捕获 — 记录原始 hook payload，用于推导契约。
Captures real payloads from all platforms (Cursor/Claude/Hermes)
so contracts can be derived from actual data, not code understanding.
"""

from __future__ import annotations

import json
import logging
import sqlite3
from datetime import datetime, timezone
from typing import Any

logger = logging.getLogger("tm_daemon.payload_capture")

_CREATE_TABLE = """
CREATE TABLE IF NOT EXISTS raw_payloads (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    endpoint TEXT NOT NULL,
    source TEXT DEFAULT 'unknown',
    payload_json TEXT NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
)
"""

_CREATE_INDEX = """
CREATE INDEX IF NOT EXISTS idx_raw_payloads_endpoint ON raw_payloads(endpoint)
"""


class PayloadCapture:
    """Captures raw hook payloads to SQLite for contract analysis.

    记录原始 hook payload 到 SQLite，用于后续契约分析。
    """

    def __init__(self, db_path: str = ":memory:") -> None:
        self._db_path = db_path
        self._conn: sqlite3.Connection | None = None

    async def start(self) -> None:
        """Initialize SQLite connection and create table."""
        self._conn = sqlite3.connect(self._db_path)
        self._conn.execute(_CREATE_TABLE)
        self._conn.execute(_CREATE_INDEX)
        self._conn.commit()
        logger.info("PayloadCapture started (db=%s)", self._db_path)

    async def capture(self, endpoint: str, payload: dict[str, Any], source: str = "unknown") -> None:
        """Record a raw payload. Non-blocking, logs errors but does not raise.

        记录原始 payload。非阻塞，记录错误但不抛出。
        """
        if self._conn is None:
            return
        try:
            self._conn.execute(
                "INSERT INTO raw_payloads (endpoint, source, payload_json, created_at) VALUES (?, ?, ?, ?)",
                (
                    endpoint,
                    source,
                    json.dumps(payload, ensure_ascii=False),
                    datetime.now(timezone.utc).isoformat(),
                ),
            )
            self._conn.commit()
        except Exception:
            logger.debug("Failed to capture payload for %s", endpoint, exc_info=True)

    async def get_payloads(self, endpoint: str | None = None, limit: int = 100) -> list[dict[str, Any]]:
        """Retrieve captured payloads for analysis."""
        if self._conn is None:
            return []
        if endpoint:
            rows = self._conn.execute(
                "SELECT endpoint, source, payload_json, created_at "
                "FROM raw_payloads WHERE endpoint = ? ORDER BY id DESC LIMIT ?",
                (endpoint, limit),
            ).fetchall()
        else:
            rows = self._conn.execute(
                "SELECT endpoint, source, payload_json, created_at FROM raw_payloads ORDER BY id DESC LIMIT ?",
                (limit,),
            ).fetchall()
        return [
            {
                "endpoint": r[0],
                "source": r[1],
                "payload": json.loads(r[2]),
                "created_at": r[3],
            }
            for r in rows
        ]

    async def get_field_summary(self, endpoint: str) -> dict[str, Any]:
        """Analyze captured payloads and summarize field usage.

        分析已捕获的 payload，汇总字段使用情况。
        Useful for deriving Pydantic contracts from real data.
        """
        if self._conn is None:
            return {}
        rows = self._conn.execute(
            "SELECT payload_json FROM raw_payloads WHERE endpoint = ?",
            (endpoint,),
        ).fetchall()
        if not rows:
            return {"endpoint": endpoint, "count": 0, "fields": {}}

        field_counts: dict[str, dict[str, int]] = {}
        for (payload_json,) in rows:
            payload = json.loads(payload_json)
            for key, value in payload.items():
                if key not in field_counts:
                    field_counts[key] = {"count": 0, "types": {}}
                field_counts[key]["count"] += 1
                type_name = type(value).__name__
                field_counts[key]["types"][type_name] = field_counts[key]["types"].get(type_name, 0) + 1

        return {
            "endpoint": endpoint,
            "count": len(rows),
            "fields": field_counts,
        }

    async def stop(self) -> None:
        """Close SQLite connection."""
        if self._conn is not None:
            self._conn.close()
            self._conn = None

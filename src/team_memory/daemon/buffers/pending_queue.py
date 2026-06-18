"""PendingQueue — SQLite-backed retry queue for failed operations.

When PostgreSQL or LiteLLM is unavailable, operations are queued here
instead of being discarded. RefinementWorker retries with backoff on each tick.
"""

from __future__ import annotations

import sqlite3
import uuid
from datetime import datetime, timezone
from typing import Any

_CREATE_TABLE = """CREATE TABLE IF NOT EXISTS pending_ops (
    id TEXT PRIMARY KEY,
    op_type TEXT NOT NULL,
    session_id TEXT,
    payload_json TEXT NOT NULL,
    retries INTEGER DEFAULT 0,
    next_retry_at TEXT NOT NULL,
    last_error TEXT,
    checkpoint_json TEXT,
    status TEXT DEFAULT 'pending',
    created_at TEXT DEFAULT (datetime('now'))
)"""

_BACKOFF_SECONDS = [30, 60, 120]
DEFAULT_MAX_RETRIES = 3
DEFAULT_MAX_QUEUE_SIZE = 200


class PendingQueue:
    """SQLite-backed retry queue / SQLite 重试队列。"""

    BACKOFF_SECONDS = _BACKOFF_SECONDS
    MAX_RETRIES = DEFAULT_MAX_RETRIES
    MAX_QUEUE_SIZE = DEFAULT_MAX_QUEUE_SIZE

    def __init__(self, db_path: str) -> None:
        self._db_path = db_path
        self._db = sqlite3.connect(db_path, check_same_thread=False)
        self._db.execute("PRAGMA journal_mode=WAL")
        self._db.execute(_CREATE_TABLE)
        # Migration: add checkpoint column (safe no-op if already exists)
        try:
            self._db.execute("ALTER TABLE pending_ops ADD COLUMN checkpoint_json TEXT")
        except sqlite3.OperationalError:
            pass  # column already exists
        self._db.commit()

    def enqueue(self, op_type: str, session_id: str, payload: dict[str, Any]) -> str | None:
        """Enqueue operation. Returns op_id or None if duplicate/full.

        入队操作。返回 op_id，若重复或队列满则返回 None。
        """
        # Dedup check / 去重检查
        row = self._db.execute(
            "SELECT id FROM pending_ops WHERE session_id = ? AND status != 'failed'",
            (session_id,),
        ).fetchone()
        if row:
            return None

        # Size check / 容量检查
        count = self._db.execute(
            "SELECT COUNT(*) FROM pending_ops WHERE status != 'failed'",
        ).fetchone()[0]
        if count >= self.MAX_QUEUE_SIZE:
            return None

        import json

        op_id = str(uuid.uuid4())[:12]
        now = datetime.now(timezone.utc).isoformat()
        next_retry = datetime.now(timezone.utc).isoformat()
        self._db.execute(
            "INSERT INTO pending_ops (id, op_type, session_id, payload_json, next_retry_at, created_at) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            (op_id, op_type, session_id, json.dumps(payload), next_retry, now),
        )
        self._db.commit()
        return op_id

    def get_due_ops(self) -> list[dict[str, Any]]:
        """Return ops whose next_retry_at has passed / 返回已到重试时间的操作。"""
        now = datetime.now(timezone.utc).isoformat()
        rows = self._db.execute(
            "SELECT * FROM pending_ops WHERE next_retry_at <= ? AND status != 'failed' ORDER BY created_at",
            (now,),
        ).fetchall()

        cols = [
            "id",
            "op_type",
            "session_id",
            "payload_json",
            "retries",
            "next_retry_at",
            "last_error",
            "status",
            "created_at",
        ]
        results = []
        for row in rows:
            d = dict(zip(cols, row))
            import json

            d["payload"] = json.loads(d.pop("payload_json"))
            results.append(d)
        return results

    def dequeue(self, op_id: str) -> None:
        """Remove a successfully retried op / 移除成功重试的操作。"""
        self._db.execute("DELETE FROM pending_ops WHERE id = ?", (op_id,))
        self._db.commit()

    def bump_retry(self, op_id: str, error: str, backoff_seconds: int) -> None:
        """Increment retries and push next_retry_at forward / 递增重试计数，推迟重试时间。"""
        from datetime import timedelta

        next_retry = (datetime.now(timezone.utc) + timedelta(seconds=backoff_seconds)).isoformat()
        self._db.execute(
            "UPDATE pending_ops SET retries = retries + 1, next_retry_at = ?, "
            "last_error = ?, status = 'pending' WHERE id = ?",
            (next_retry, error[:500], op_id),
        )
        self._db.commit()

    def mark_failed(self, op_id: str) -> None:
        """Mark an op as permanently failed / 标记为永久失败。"""
        self._db.execute(
            "UPDATE pending_ops SET status = 'failed' WHERE id = ?",
            (op_id,),
        )
        self._db.commit()

    def stats(self) -> dict[str, int]:
        """Return summary counts / 返回汇总统计。"""
        rows = self._db.execute("SELECT status, COUNT(*) FROM pending_ops GROUP BY status").fetchall()
        result = {"pending": 0, "retrying": 0, "failed": 0}
        for status, count in rows:
            if status in result:
                result[status] = count
        return result

    def save_checkpoint(self, op_id: str, data: dict[str, Any]) -> None:
        """Save intermediate progress for long-running tasks.

        Call periodically during batch operations.  On recovery, the
        task can read ``checkpoint`` from the op dict and skip already
        completed items.
        """
        import json

        self._db.execute(
            "UPDATE pending_ops SET checkpoint_json = ? WHERE id = ?",
            (json.dumps(data), op_id),
        )
        self._db.commit()

    def load_checkpoint(self, op_id: str) -> dict[str, Any] | None:
        """Read the last saved checkpoint. Returns None if none exists."""
        import json

        row = self._db.execute(
            "SELECT checkpoint_json FROM pending_ops WHERE id = ?",
            (op_id,),
        ).fetchone()
        if row and row[0]:
            return json.loads(row[0])
        return None

    def list_ops(self, op_type: str | None = None) -> list[dict[str, Any]]:
        """List all queued ops, optionally filtered by type / 列出所有队列操作。"""
        if op_type:
            rows = self._db.execute(
                "SELECT * FROM pending_ops WHERE op_type = ? AND status != 'failed' ORDER BY created_at",
                (op_type,),
            ).fetchall()
        else:
            rows = self._db.execute("SELECT * FROM pending_ops WHERE status != 'failed' ORDER BY created_at").fetchall()

        cols = [
            "id",
            "op_type",
            "session_id",
            "payload_json",
            "retries",
            "next_retry_at",
            "last_error",
            "checkpoint_json",
            "status",
            "created_at",
        ]
        results = []
        for row in rows:
            d = dict(zip(cols, row))
            d.pop("payload_json")
            if "checkpoint_json" in d:
                import json

                raw = d.pop("checkpoint_json")
                d["checkpoint"] = json.loads(raw) if raw else None
            results.append(d)
        return results

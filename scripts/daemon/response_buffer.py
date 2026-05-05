"""Response Buffer — temporary storage for pending faithfulness evaluations.

Stores (query, agent_response, result_ids) tuples until a batch threshold
is reached, then triggers LLM judge evaluation.

Uses raw asyncpg for performance (same pattern as search_log_writer).
"""
from __future__ import annotations

import json
import logging
import uuid
from datetime import datetime, timezone
from typing import Any

import asyncpg

logger = logging.getLogger("daemon.response_buffer")


class ResponseBuffer:
    """Async buffer for pending faithfulness evaluations."""

    def __init__(
        self,
        dsn: str = "postgresql://developer:devpass@localhost:5433/team_memory",
        batch_threshold: int = 5,
    ) -> None:
        self._dsn = dsn
        self._pool: asyncpg.Pool | None = None
        self._batch_threshold = batch_threshold

    async def _ensure_pool(self) -> asyncpg.Pool:
        if self._pool is None or self._pool._closed:
            self._pool = await asyncpg.create_pool(self._dsn, min_size=1, max_size=3)
        return self._pool

    async def add(
        self,
        *,
        query: str,
        agent_response: str,
        result_ids: list[dict[str, Any]] | None = None,
        search_log_id: str | None = None,
    ) -> str | None:
        """Add an entry to the buffer. Returns the buffer entry ID."""
        try:
            pool = await self._ensure_pool()
            entry_id = uuid.uuid4()
            async with pool.acquire() as conn:
                await conn.execute(
                    """
                    INSERT INTO response_buffer
                        (id, query, agent_response, result_ids, search_log_id, created_at)
                    VALUES ($1, $2, $3, $4, $5, $6)
                    """,
                    entry_id,
                    query[:2000],
                    agent_response[:10000],
                    json.dumps(result_ids) if result_ids else None,
                    uuid.UUID(search_log_id) if search_log_id else None,
                    datetime.now(timezone.utc),
                )
            logger.debug("Buffered response for evaluation: %s", entry_id)
            return str(entry_id)
        except Exception as e:
            logger.debug("response_buffer.add failed: %s", e)
            return None

    async def pending_count(self) -> int:
        """Return number of unevaluated entries."""
        try:
            pool = await self._ensure_pool()
            async with pool.acquire() as conn:
                count = await conn.fetchval(
                    "SELECT count(*) FROM response_buffer WHERE evaluated = FALSE"
                )
            return count or 0
        except Exception as e:
            logger.debug("response_buffer.pending_count failed: %s", e)
            return 0

    async def should_batch(self) -> bool:
        """Return True if pending count >= batch threshold."""
        count = await self.pending_count()
        return count >= self._batch_threshold

    async def fetch_pending(self, limit: int = 50) -> list[dict[str, Any]]:
        """Fetch unevaluated entries for batch processing."""
        try:
            pool = await self._ensure_pool()
            async with pool.acquire() as conn:
                rows = await conn.fetch(
                    """
                    SELECT id, query, agent_response, result_ids, search_log_id, created_at
                    FROM response_buffer
                    WHERE evaluated = FALSE
                    ORDER BY created_at ASC
                    LIMIT $1
                    """,
                    limit,
                )
            return [
                {
                    "id": str(r["id"]),
                    "query": r["query"],
                    "agent_response": r["agent_response"],
                    "result_ids": json.loads(r["result_ids"]) if r["result_ids"] else None,
                    "search_log_id": str(r["search_log_id"]) if r["search_log_id"] else None,
                    "created_at": r["created_at"],
                }
                for r in rows
            ]
        except Exception as e:
            logger.debug("response_buffer.fetch_pending failed: %s", e)
            return []

    async def mark_evaluated(
        self,
        entry_id: str,
        *,
        faithfulness_score: float,
        judge_reasoning: str = "",
    ) -> None:
        """Mark an entry as evaluated with its score."""
        try:
            pool = await self._ensure_pool()
            async with pool.acquire() as conn:
                await conn.execute(
                    """
                    UPDATE response_buffer
                    SET evaluated = TRUE,
                        faithfulness_score = $2,
                        judge_reasoning = $3
                    WHERE id = $1
                    """,
                    uuid.UUID(entry_id),
                    faithfulness_score,
                    judge_reasoning[:5000],
                )
        except Exception as e:
            logger.debug("response_buffer.mark_evaluated failed: %s", e)

    async def sync_to_search_log(
        self,
        entry_id: str,
        *,
        faithfulness_score: float,
    ) -> None:
        """Copy faithfulness_score to the linked search_log (if any)."""
        try:
            pool = await self._ensure_pool()
            async with pool.acquire() as conn:
                row = await conn.fetchrow(
                    "SELECT search_log_id FROM response_buffer WHERE id = $1",
                    uuid.UUID(entry_id),
                )
                if row and row["search_log_id"]:
                    was_used = faithfulness_score >= 0.5
                    await conn.execute(
                        """
                        UPDATE search_logs
                        SET faithfulness_score = $2,
                            was_used = CASE WHEN $3 THEN TRUE ELSE was_used END,
                            judgment_source = CASE WHEN $3 THEN 'faithfulness' ELSE judgment_source END,
                            updated_at = $4
                        WHERE id = $1
                        """,
                        row["search_log_id"],
                        faithfulness_score,
                        was_used,
                        datetime.now(timezone.utc),
                    )
        except Exception as e:
            logger.debug("response_buffer.sync_to_search_log failed: %s", e)

    async def get_stats(self, days: int = 7) -> dict[str, Any]:
        """Return buffer stats for the last N days."""
        try:
            pool = await self._ensure_pool()
            async with pool.acquire() as conn:
                total = await conn.fetchval(
                    "SELECT count(*) FROM response_buffer WHERE created_at > NOW() - make_interval(days => $1)",
                    days,
                )
                evaluated = await conn.fetchval(
                    "SELECT count(*) FROM response_buffer WHERE evaluated = TRUE AND created_at > NOW() - make_interval(days => $1)",
                    days,
                )
                avg_score = await conn.fetchval(
                    "SELECT avg(faithfulness_score) FROM response_buffer WHERE evaluated = TRUE AND created_at > NOW() - make_interval(days => $1)",
                    days,
                )
            return {
                "total": total or 0,
                "evaluated": evaluated or 0,
                "pending": (total or 0) - (evaluated or 0),
                "avg_faithfulness": round(float(avg_score), 3) if avg_score else None,
            }
        except Exception as e:
            logger.debug("response_buffer.get_stats failed: %s", e)
            return {"total": 0, "evaluated": 0, "pending": 0, "avg_faithfulness": None}

    async def close(self) -> None:
        """Close the connection pool."""
        if self._pool and not self._pool._closed:
            await self._pool.close()

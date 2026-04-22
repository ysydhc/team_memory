"""SearchLog repository — CRUD and stats for search evaluation tracking."""

from __future__ import annotations

import logging
import uuid
from datetime import datetime, timedelta, timezone

from sqlalchemy import desc, func, select, update
from sqlalchemy.ext.asyncio import AsyncSession

from team_memory.storage.models import SearchLog

logger = logging.getLogger("team_memory.storage.search_log_repository")


class SearchLogRepository:
    """Repository for SearchLog CRUD and evaluation statistics."""

    def __init__(self, session: AsyncSession):
        self._session = session

    async def create(
        self,
        *,
        query: str,
        intent_type: str = "unknown",
        project: str = "default",
        source: str = "mcp",
        result_ids: list[dict] | None = None,
    ) -> SearchLog:
        """Create a new search log entry."""
        log = SearchLog(
            query=query,
            intent_type=intent_type,
            project=project,
            source=source,
            result_ids=result_ids,
        )
        self._session.add(log)
        await self._session.flush()
        return log

    async def mark_used(
        self, log_id: uuid.UUID, agent_snippet: str | None = None
    ) -> None:
        """Mark a search log as was_used=True with optional agent response snippet."""
        values: dict = {"was_used": True}
        if agent_snippet is not None:
            values["agent_response_snippet"] = agent_snippet

        stmt = (
            update(SearchLog)
            .where(SearchLog.id == log_id)
            .values(**values)
        )
        await self._session.execute(stmt)
        await self._session.flush()

    async def get_stats(self, days: int = 7) -> dict:
        """Return {total, hit, used, use_rate} for the last N days.

        hit = entries where result_ids is not null/empty
        used = entries where was_used is True
        use_rate = used / hit (0 if hit=0)
        """
        cutoff = datetime.now(timezone.utc) - timedelta(days=days)
        base_filter = SearchLog.created_at >= cutoff

        # Total count
        total_result = await self._session.execute(
            select(func.count()).select_from(SearchLog).where(base_filter)
        )
        total = total_result.scalar_one()

        # Hit count: result_ids is not null and not empty list
        # Use cast to Text for cross-dialect comparison.
        # JSON null is stored as 'null' string in SQLite, SQL NULL is separate.
        from sqlalchemy import String, and_, cast

        hit_result = await self._session.execute(
            select(func.count())
            .select_from(SearchLog)
            .where(base_filter)
            .where(
                and_(
                    SearchLog.result_ids.isnot(None),
                    cast(SearchLog.result_ids, String).notin_(["[]", "null"]),
                )
            )
        )
        hit = hit_result.scalar_one()

        # Used count
        used_result = await self._session.execute(
            select(func.count())
            .select_from(SearchLog)
            .where(base_filter)
            .where(SearchLog.was_used == True)  # noqa: E712
        )
        used = used_result.scalar_one()

        use_rate = used / hit if hit > 0 else 0.0

        return {
            "total": total,
            "hit": hit,
            "used": used,
            "use_rate": use_rate,
        }

    async def get_recent(self, days: int = 7, limit: int = 100) -> list[SearchLog]:
        """Get recent search logs within the last N days, newest first."""
        cutoff = datetime.now(timezone.utc) - timedelta(days=days)
        stmt = (
            select(SearchLog)
            .where(SearchLog.created_at >= cutoff)
            .order_by(desc(SearchLog.created_at))
            .limit(limit)
        )
        result = await self._session.execute(stmt)
        return list(result.scalars().all())

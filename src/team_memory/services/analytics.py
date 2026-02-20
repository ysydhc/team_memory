"""Analytics and audit log service.

Encapsulates statistics queries and audit log access, moved from
direct repository calls in web handlers to a dedicated service layer.
"""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from typing import Any

from team_memory.storage.repository import ExperienceRepository

logger = logging.getLogger("team_memory.analytics")


class AnalyticsService:
    """Service for analytics queries and audit logs."""

    def __init__(self, db_url: str = ""):
        self._db_url = db_url

    @asynccontextmanager
    async def _session(self):
        from team_memory.storage.database import get_session

        async with get_session(self._db_url) as session:
            yield session

    async def get_overview(self) -> dict[str, Any]:
        """Get a comprehensive overview of the experience database."""
        async with self._session() as session:
            repo = ExperienceRepository(session)
            stats = await repo.get_stats()
            return stats

    async def get_audit_logs(
        self,
        *,
        limit: int = 50,
        action: str | None = None,
        user: str | None = None,
    ) -> list[dict]:
        """Get audit log entries with optional filters."""
        async with self._session() as session:
            repo = ExperienceRepository(session)
            logs = await repo.get_query_logs(limit=limit)
            if action:
                logs = [log for log in logs if log.get("action") == action]
            if user:
                logs = [log for log in logs if log.get("user") == user]
            return logs

    async def get_tag_distribution(self) -> dict[str, int]:
        """Get tag usage counts."""
        stats = await self.get_overview()
        return stats.get("tag_distribution", {})

    async def get_query_analytics(self) -> dict[str, Any]:
        """Get query analytics summary."""
        async with self._session() as session:
            repo = ExperienceRepository(session)
            return await repo.get_query_stats()

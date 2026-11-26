"""MemoryJanitor orchestrator composing all maintenance task mixins."""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager

from team_memory.services.maintenance.janitor.archive_maint import JanitorArchiveMixin
from team_memory.services.maintenance.janitor.cleanup import JanitorCleanupMixin
from team_memory.services.maintenance.janitor.decay import JanitorDecayMixin
from team_memory.services.maintenance.janitor.helpers import _utcnow
from team_memory.services.maintenance.janitor.promotion import JanitorPromotionMixin
from team_memory.services.maintenance.janitor.sqlite_maint import JanitorStorageMaintMixin

logger = logging.getLogger("team_memory.services.maintenance.janitor")


class MemoryJanitor(
    JanitorDecayMixin,
    JanitorCleanupMixin,
    JanitorPromotionMixin,
    JanitorArchiveMixin,
    JanitorStorageMaintMixin,
):
    """Automated maintenance service for team memory cleanup and quality management.

    Handles quality score decay, outdated content cleanup, draft expiration,
    and personal memory pruning based on configurable rules.
    """

    def __init__(self, db_url: str, config=None):
        """Initialize janitor with database URL and configuration.

        Args:
            db_url: Database connection URL
            config: Configuration object with janitor settings
        """
        self._db_url = db_url
        self._config = config

    @asynccontextmanager
    async def _session(self):
        """Create a managed database session."""
        from team_memory.storage.database import get_session

        async with get_session(self._db_url) as session:
            yield session

    def _get_config(self, key: str, default=None):
        """Get configuration value with fallback to default."""
        if self._config is None:
            return default
        return getattr(self._config, key, default)

    async def run_all(self, project: str | None = None) -> dict:
        """Run all janitor operations in sequence.

        Args:
            project: Project to limit operations to, None for all projects

        Returns:
            Dict with combined results from all operations
        """
        logger.info("Starting full janitor run for project '%s'", project or "all")

        # Get configuration values
        protection_days = self._get_config("protection_period_days", 10)
        auto_soft_delete = self._get_config("auto_soft_delete_outdated", True)
        purge_days = self._get_config("purge_soft_deleted_days", 30)
        draft_expiry_days = self._get_config("draft_expiry_days", 7)
        personal_memory_days = self._get_config("personal_memory_retention_days", 90)
        promotion_enabled = self._get_config("promotion_enabled", True)

        results = {}

        try:
            # 1. Apply score decay
            results["score_decay"] = await self.run_score_decay(project)

            # 2. Sweep outdated experiences
            results["outdated_sweep"] = await self.sweep_outdated(project, auto_soft_delete)

            # 3. Purge old soft-deleted experiences
            results["soft_deleted_purge"] = await self.purge_soft_deleted(purge_days)

            # 4. Expire old drafts
            results["draft_expiration"] = await self.expire_drafts(draft_expiry_days, project)

            # 5. Prune old personal memories (global operation)
            if project is None:  # Only run for global operations
                results["personal_memory_pruning"] = await self.prune_personal_memory(personal_memory_days)

            # 6. Promote experiences (L2→L3)
            if promotion_enabled:
                results["promotion"] = await self.run_promotion(project)

            # 7. Dedup experiences and search logs / 去重
            if self._get_config("dedup_enabled", True):
                results["dedup_experiences"] = await self.dedup_experiences(project)
                results["dedup_search_logs"] = await self.dedup_search_logs(project)

            # 8. Maintain archives (normalize content_type/project, dedup, stale drafts, dangling links)
            results["archive_maintenance"] = await self.maintain_archives(project)

            # 9. Prune PG buffer, deprecated background tasks & expired LLM usage logs (global operations)
            if project is None:
                results["response_buffer_pruning"] = await self.prune_response_buffer()
                results["background_tasks_pruning"] = await self.prune_background_tasks()
                results["llm_usage_logs_pruning"] = await self.prune_llm_usage_logs()

            logger.info("Full janitor run completed successfully for project '%s'", project or "all")

        except Exception as e:
            logger.error("Janitor run failed for project '%s': %s", project or "all", e, exc_info=True)
            results["error"] = str(e)

        return {
            "project": project,
            "completed_at": _utcnow().isoformat(),
            "operations": results,
            "config": {
                "protection_days": protection_days,
                "auto_soft_delete": auto_soft_delete,
                "purge_days": purge_days,
                "draft_expiry_days": draft_expiry_days,
                "personal_memory_days": personal_memory_days,
                "promotion_enabled": promotion_enabled,
            },
        }

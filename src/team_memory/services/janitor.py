"""Memory janitor service for automated cleanup and quality management.

Provides automated maintenance operations for team memory:
- Quality score decay over time
- Outdated experience cleanup
- Soft-deleted record purging
- Draft expiration
- Personal memory pruning
"""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from datetime import datetime, timedelta, timezone

from sqlalchemy import and_, select

from team_memory.storage.models import Experience, PersonalMemory

logger = logging.getLogger("team_memory.services.janitor")


def _utcnow() -> datetime:
    """Get current UTC datetime."""
    return datetime.now(timezone.utc)


class MemoryJanitor:
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

    async def run_score_decay(self, project: str | None = None) -> dict:
        """Apply quality score decay to experiences based on age.

        Decay rules:
        - Protection period: 10 days (no decay)
        - After protection: -1 point per day until score reaches 50
        - Below 50: -0.5 points per day
        - Quality tiers: Gold≥120, Silver≥60, Bronze≥20, Outdated≤0

        Args:
            project: Project to limit decay to, None for all projects

        Returns:
            Dict with decay statistics
        """
        async with self._session() as session:
            now = _utcnow()
            protection_days = self._get_config("protection_period_days", 10)
            protection_cutoff = now - timedelta(days=protection_days)

            # Get experiences eligible for decay (not pinned, not deleted, past protection period)
            query = select(Experience).where(
                and_(
                    Experience.is_deleted == False,  # noqa: E712
                    Experience.is_pinned == False,  # noqa: E712
                    Experience.last_scored_at < protection_cutoff,
                    Experience.quality_score > 0,
                )
            )

            if project:
                query = query.where(Experience.project == project)

            result = await session.execute(query)
            experiences = result.scalars().all()

            updated_count = 0
            tier_changes = {"Gold": 0, "Silver": 0, "Bronze": 0, "Outdated": 0}

            for exp in experiences:
                # Calculate days since last scoring
                days_since_scored = (now - exp.last_scored_at).days
                if days_since_scored <= 0:
                    continue

                # Apply decay based on current score
                old_score = exp.quality_score
                if old_score >= 50:
                    # Standard decay: -1 per day
                    new_score = max(0, old_score - days_since_scored)
                else:
                    # Slow decay: -0.5 per day
                    new_score = max(0, old_score - (days_since_scored * 0.5))

                # Determine new tier
                if new_score >= 120:
                    new_tier = "Gold"
                elif new_score >= 60:
                    new_tier = "Silver"
                elif new_score >= 20:
                    new_tier = "Bronze"
                else:
                    new_tier = "Outdated"

                # Update if score changed
                if new_score != old_score:
                    exp.quality_score = new_score
                    exp.quality_tier = new_tier
                    exp.last_scored_at = now
                    updated_count += 1
                    tier_changes[new_tier] += 1

                    logger.debug(
                        "Score decay applied to %s: %.1f -> %.1f (%s)",
                        exp.id,
                        old_score,
                        new_score,
                        new_tier,
                    )

            await session.commit()

            logger.info(
                "Score decay completed: %d experiences updated in project '%s'",
                updated_count,
                project or "all",
            )

            return {
                "updated_count": updated_count,
                "project": project,
                "tier_changes": tier_changes,
                "protection_days": protection_days,
            }

    async def sweep_outdated(
        self, project: str | None = None, auto_soft_delete: bool = False
    ) -> dict:
        """Identify and optionally soft-delete outdated experiences.

        Args:
            project: Project to limit sweep to, None for all projects
            auto_soft_delete: Whether to automatically soft-delete outdated items

        Returns:
            Dict with sweep statistics
        """
        async with self._session() as session:
            # Find outdated experiences (quality_tier = "Outdated" and score <= 0)
            query = select(Experience).where(
                and_(
                    Experience.is_deleted == False,  # noqa: E712
                    Experience.is_pinned == False,  # noqa: E712
                    Experience.quality_tier == "Outdated",
                    Experience.quality_score <= 0,
                )
            )

            if project:
                query = query.where(Experience.project == project)

            result = await session.execute(query)
            outdated_experiences = result.scalars().all()

            found_count = len(outdated_experiences)
            deleted_count = 0

            if auto_soft_delete and outdated_experiences:
                now = _utcnow()
                for exp in outdated_experiences:
                    exp.is_deleted = True
                    exp.deleted_at = now
                    deleted_count += 1

                    logger.debug("Auto soft-deleted outdated experience: %s", exp.id)

                await session.commit()

                logger.info(
                    "Outdated sweep completed: %d found, %d auto-deleted in project '%s'",
                    found_count,
                    deleted_count,
                    project or "all",
                )
            else:
                logger.info(
                    "Outdated sweep completed: %d found in project '%s' (no auto-delete)",
                    found_count,
                    project or "all",
                )

            return {
                "found_count": found_count,
                "deleted_count": deleted_count,
                "project": project,
                "auto_soft_delete": auto_soft_delete,
            }

    async def purge_soft_deleted(self, older_than_days: int = 30) -> dict:
        """Permanently delete soft-deleted experiences older than specified days.

        Args:
            older_than_days: Days threshold for permanent deletion

        Returns:
            Dict with purge statistics
        """
        async with self._session() as session:
            cutoff_date = _utcnow() - timedelta(days=older_than_days)

            # Find soft-deleted experiences older than cutoff
            query = select(Experience).where(
                and_(
                    Experience.is_deleted == True,  # noqa: E712
                    Experience.deleted_at < cutoff_date,
                )
            )

            result = await session.execute(query)
            to_purge = result.scalars().all()

            purged_count = len(to_purge)

            # Hard delete the experiences
            for exp in to_purge:
                await session.delete(exp)
                logger.debug("Permanently deleted experience: %s", exp.id)

            await session.commit()

            logger.info(
                "Soft-deleted purge completed: %d experiences permanently deleted (>%d days)",
                purged_count,
                older_than_days,
            )

            return {
                "purged_count": purged_count,
                "older_than_days": older_than_days,
                "cutoff_date": cutoff_date.isoformat(),
            }

    async def expire_drafts(self, older_than_days: int = 7, project: str | None = None) -> dict:
        """Soft-delete draft experiences older than specified days.

        Args:
            older_than_days: Days threshold for draft expiration
            project: Project to limit expiration to, None for all projects

        Returns:
            Dict with expiration statistics
        """
        async with self._session() as session:
            cutoff_date = _utcnow() - timedelta(days=older_than_days)
            now = _utcnow()

            # Find old draft experiences
            query = select(Experience).where(
                and_(
                    Experience.is_deleted == False,  # noqa: E712
                    Experience.exp_status == "draft",
                    Experience.created_at < cutoff_date,
                )
            )

            if project:
                query = query.where(Experience.project == project)

            result = await session.execute(query)
            draft_experiences = result.scalars().all()

            expired_count = len(draft_experiences)

            # Soft-delete expired drafts
            for exp in draft_experiences:
                exp.is_deleted = True
                exp.deleted_at = now
                logger.debug("Expired draft experience: %s", exp.id)

            await session.commit()

            logger.info(
                "Draft expiration completed: %d drafts expired in project '%s' (>%d days)",
                expired_count,
                project or "all",
                older_than_days,
            )

            return {
                "expired_count": expired_count,
                "older_than_days": older_than_days,
                "project": project,
                "cutoff_date": cutoff_date.isoformat(),
            }

    async def prune_personal_memory(self, older_than_days: int = 90) -> dict:
        """Remove old personal memory entries.

        Args:
            older_than_days: Days threshold for personal memory removal

        Returns:
            Dict with pruning statistics
        """
        async with self._session() as session:
            cutoff_date = _utcnow() - timedelta(days=older_than_days)

            # Find old personal memories (only dynamic ones, keep static preferences)
            query = select(PersonalMemory).where(
                and_(
                    PersonalMemory.profile_kind == "dynamic",
                    PersonalMemory.created_at < cutoff_date,
                )
            )

            result = await session.execute(query)
            old_memories = result.scalars().all()

            pruned_count = len(old_memories)

            # Delete old personal memories
            for memory in old_memories:
                await session.delete(memory)
                logger.debug("Pruned personal memory: %s", memory.id)

            await session.commit()

            logger.info(
                "Personal memory pruning completed: %d entries removed (older than %d days)",
                pruned_count,
                older_than_days,
            )

            return {
                "pruned_count": pruned_count,
                "older_than_days": older_than_days,
                "cutoff_date": cutoff_date.isoformat(),
            }

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
        auto_soft_delete = self._get_config("auto_soft_delete_outdated", False)
        purge_days = self._get_config("purge_soft_deleted_days", 30)
        draft_expiry_days = self._get_config("draft_expiry_days", 7)
        personal_memory_days = self._get_config("personal_memory_retention_days", 90)

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
                results["personal_memory_pruning"] = await self.prune_personal_memory(
                    personal_memory_days
                )

            logger.info(
                "Full janitor run completed successfully for project '%s'", project or "all"
            )

        except Exception as e:
            logger.error(
                "Janitor run failed for project '%s': %s", project or "all", e, exc_info=True
            )
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
            },
        }

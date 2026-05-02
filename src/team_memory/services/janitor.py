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
import os
import subprocess
from contextlib import asynccontextmanager
from datetime import datetime, timedelta, timezone

from sqlalchemy import and_, select

from team_memory.services.promotion_compiler import PromotionCompiler
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

    async def run_promotion(self, project: str | None = None) -> dict:
        """Promote published experiences to 'promoted' status based on thresholds.

        Two promotion paths:
        1. recall_count threshold: experiences with recall_count >= threshold get promoted
        2. group_key threshold: group_keys with count >= threshold cause all
           published experiences in that group to be promoted

        Only experiences with exp_status="published" are eligible; already-promoted
        experiences are excluded to prevent double promotion.

        After identifying candidates, Markdown is compiled via PromotionCompiler
        and optionally written to Obsidian output directories (git add + commit).

        Args:
            project: Project to limit promotion to, None for all projects

        Returns:
            Dict with promotion statistics
        """
        recall_count_threshold = self._get_config("promotion_use_count_threshold", 3)
        group_key_threshold = self._get_config("promotion_group_key_threshold", 5)
        output_dirs: dict[str, str] = self._get_config("promotion_output_dirs", {})

        promoted_by_recall_count = 0
        promoted_by_group = 0

        async with self._session() as session:
            # 1. Promote by recall_count threshold
            recall_count_query = select(Experience).where(
                and_(
                    Experience.is_deleted == False,  # noqa: E712
                    Experience.exp_status == "published",
                    Experience.recall_count >= recall_count_threshold,
                )
            )
            if project:
                recall_count_query = recall_count_query.where(Experience.project == project)

            result = await session.execute(recall_count_query)
            recall_count_experiences = list(result.scalars().all())

            # 2. Promote by group_key threshold
            # Find group_keys with enough published experiences
            from sqlalchemy import func

            group_count_query = (
                select(Experience.group_key, func.count(Experience.id).label("cnt"))
                .where(
                    and_(
                        Experience.is_deleted == False,  # noqa: E712
                        Experience.exp_status == "published",
                        Experience.group_key.isnot(None),
                    )
                )
                .group_by(Experience.group_key)
                .having(func.count(Experience.id) >= group_key_threshold)
            )
            if project:
                group_count_query = group_count_query.where(
                    Experience.project == project
                )

            group_result = await session.execute(group_count_query)
            qualifying_groups = [row[0] for row in group_result.all()]

            group_experiences: list[Experience] = []
            if qualifying_groups:
                # Get all published experiences in qualifying groups
                group_exp_query = select(Experience).where(
                    and_(
                        Experience.is_deleted == False,  # noqa: E712
                        Experience.exp_status == "published",
                        Experience.group_key.in_(qualifying_groups),
                    )
                )
                if project:
                    group_exp_query = group_exp_query.where(
                        Experience.project == project
                    )

                group_exp_result = await session.execute(group_exp_query)
                group_experiences = list(group_exp_result.scalars().all())

            # ----------------------------------------------------------
            # 3. Compile Markdown + write files BEFORE status change
            # ----------------------------------------------------------
            compiler = PromotionCompiler()
            promoted_ids: set[str] = set()

            # 3a. recall_count promotions — one Markdown per experience
            for exp in recall_count_experiences:
                exp_dict = exp.to_dict()
                markdown = await compiler.compile([exp_dict], exp.group_key)
                await self._write_promoted(exp, markdown, output_dirs)
                promoted_ids.add(str(exp.id))
                promoted_by_recall_count += 1
                logger.debug(
                    "Promoted experience by recall_count: %s (recall_count=%d)",
                    exp.id,
                    exp.recall_count,
                )

            # 3b. group_key promotions — one Markdown per group
            # Group experiences by group_key for batch compilation
            by_group: dict[str, list[Experience]] = {}
            for exp in group_experiences:
                if str(exp.id) in promoted_ids:
                    continue  # avoid double compilation
                by_group.setdefault(exp.group_key, []).append(exp)

            for gk, exps in by_group.items():
                exp_dicts = [e.to_dict() for e in exps]
                markdown = await compiler.compile(exp_dicts, gk)
                for exp in exps:
                    await self._write_promoted(exp, markdown, output_dirs)
                    promoted_ids.add(str(exp.id))
                    promoted_by_group += 1
                    logger.debug(
                        "Promoted experience by group_key: %s (group_key=%s)",
                        exp.id,
                        exp.group_key,
                    )

            # ----------------------------------------------------------
            # 4. Mark all promoted experiences
            # ----------------------------------------------------------
            for exp in recall_count_experiences:
                exp.exp_status = "promoted"

            for exp in group_experiences:
                if str(exp.id) not in set(str(e.id) for e in recall_count_experiences):
                    exp.exp_status = "promoted"

            await session.commit()

        total = promoted_by_recall_count + promoted_by_group

        logger.info(
            "Promotion completed: %d by recall_count, %d by group_key, %d total",
            promoted_by_recall_count,
            promoted_by_group,
            total,
        )

        return {
            "promoted_by_recall_count": promoted_by_recall_count,
            "promoted_by_group": promoted_by_group,
            "total": total,
            "recall_count_threshold": recall_count_threshold,
            "group_key_threshold": group_key_threshold,
            "project": project,
        }

    async def _write_promoted(
        self, exp: Experience, markdown: str, output_dirs: dict[str, str]
    ) -> None:
        """Write compiled Markdown to the Obsidian output directory and git add/commit.

        Args:
            exp: The promoted Experience ORM object.
            markdown: The compiled Markdown string.
            output_dirs: project → output directory mapping.
        """
        project = getattr(exp, "project", "default") or "default"
        output_dir = output_dirs.get(project)
        if not output_dir:
            return

        os.makedirs(output_dir, exist_ok=True)

        today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        group_key = exp.group_key or "misc"
        filename = f"promoted-{today}-{group_key}.md"
        filepath = os.path.join(output_dir, filename)

        with open(filepath, "w") as f:
            f.write(markdown)

        # git add + commit (best effort)
        try:
            subprocess.run(
                ["git", "add", filepath],
                cwd=output_dir,
                capture_output=True,
                timeout=10,
            )
            subprocess.run(
                ["git", "commit", "-m", f"promoted: {group_key}"],
                cwd=output_dir,
                capture_output=True,
                timeout=10,
            )
        except Exception:
            pass  # best effort

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
                results["personal_memory_pruning"] = await self.prune_personal_memory(
                    personal_memory_days
                )

            # 6. Promote experiences (L2→L3)
            if promotion_enabled:
                results["promotion"] = await self.run_promotion(project)

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
                "promotion_enabled": promotion_enabled,
            },
        }

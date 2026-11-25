"""Score decay, outdated sweep, and staleness verification for MemoryJanitor."""

from __future__ import annotations

import logging
from datetime import timedelta

from sqlalchemy import and_, func, not_, or_, select

from team_memory.services.maintenance.janitor.helpers import _utcnow
from team_memory.storage.models import Experience

logger = logging.getLogger("team_memory.services.maintenance.janitor")


class JanitorDecayMixin:
    """Mixin providing quality score decay, outdated sweeping, and staleness checks."""

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

            # Get experiences eligible for decay:
            # - not deleted, not pinned, not superseded
            # - Defense 3: fresh verified experiences are exempt from blind score decay
            # - past protection period with positive quality score
            query = select(Experience).where(
                and_(
                    Experience.is_deleted == False,  # noqa: E712
                    Experience.is_pinned == False,  # noqa: E712
                    not_(
                        and_(
                            Experience.epistemic_tier == "verified",
                            Experience.staleness_status == "fresh",
                        )
                    ),
                    Experience.valid_until.is_(None),
                    Experience.created_at < protection_cutoff,
                    Experience.quality_score > 0,
                )
            )

            if project:
                query = query.where(Experience.project == project)

            result = await session.execute(query)
            experiences = result.scalars().all()

            updated_count = 0
            tier_changes = {"Gold": 0, "Silver": 0, "Bronze": 0, "Outdated": 0}
            decay_high = self._get_config("decay_per_day", 1.0)
            decay_low = self._get_config("decay_per_day_low", 0.5)
            decay_threshold = self._get_config("decay_low_threshold", 50.0)

            for exp in experiences:
                # Defensive guard for mocked sessions / ORM objects in unit tests
                if getattr(exp, "is_pinned", False) is True:
                    continue
                tier_val = getattr(exp, "epistemic_tier", "auto_extract")
                stale_val = getattr(exp, "staleness_status", "fresh")
                if (
                    isinstance(tier_val, str)
                    and tier_val == "verified"
                    and (not isinstance(stale_val, str) or stale_val == "fresh")
                ):
                    continue
                # Calculate days since last scoring
                last_scored = exp.last_scored_at or exp.created_at
                if last_scored is None:
                    continue
                days_since_scored = (now - last_scored).days
                if days_since_scored <= 0:
                    continue

                # Apply decay based on current score
                old_score = exp.quality_score
                if old_score >= decay_threshold:
                    # Standard decay: -1 per day
                    new_score = max(0, old_score - (days_since_scored * decay_high))
                else:
                    # Slow decay: -0.5 per day
                    new_score = max(0, old_score - (days_since_scored * decay_low))

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

    async def sweep_outdated(self, project: str | None = None, auto_soft_delete: bool = False) -> dict:
        """Identify and optionally soft-delete outdated experiences.

        Args:
            project: Project to limit sweep to, None for all projects
            auto_soft_delete: Whether to automatically soft-delete outdated items

        Returns:
            Dict with sweep statistics
        """
        async with self._session() as session:
            outdated_threshold = self._get_config("outdated_threshold", 0.0)
            stale_days = self._get_config("stale_prune_days", 30)
            stale_cutoff = _utcnow() - timedelta(days=stale_days)
            # Find outdated experiences (quality_tier = "Outdated" and score <= outdated_threshold),
            # malformed published/promoted experiences (empty/Draft title or missing/blank solution),
            # or stale unreferenced low-tier experiences (>stale_days old with 0 recall and 0 use).
            query = select(Experience).where(
                and_(
                    Experience.is_deleted == False,  # noqa: E712
                    Experience.is_pinned == False,  # noqa: E712
                    not_(
                        and_(
                            Experience.epistemic_tier == "verified",
                            Experience.staleness_status == "fresh",
                        )
                    ),
                    Experience.valid_until.is_(None),
                    or_(
                        and_(
                            Experience.quality_tier == "Outdated",
                            or_(
                                Experience.quality_score <= outdated_threshold,
                                Experience.used_count == 0,
                            ),
                        ),
                        and_(
                            Experience.exp_status.in_(["published", "promoted", "draft"]),
                            or_(
                                Experience.solution.is_(None),
                                func.trim(Experience.solution) == "",
                                and_(
                                    Experience.used_count == 0,
                                    or_(
                                        Experience.title == "",
                                        Experience.title.like("Draft %"),
                                    ),
                                ),
                            ),
                        ),
                        and_(
                            Experience.quality_tier.in_(["Bronze", "Outdated"]),
                            Experience.recall_count == 0,
                            Experience.used_count == 0,
                            Experience.created_at < stale_cutoff,
                        ),
                    ),
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

    async def verify_staleness(
        self,
        project: str,
        repo_path: str | None = None,
        *,
        dry_run: bool = False,
    ) -> dict:
        """Run Defense-2 code-grounded staleness verification for a project."""
        from team_memory.services.maintenance.staleness_verifier import StalenessVerifier

        repo_map = self._get_config("project_repo_map", {}) or {}
        verifier = StalenessVerifier(self._db_url, project_repo_map=repo_map)
        return await verifier.verify_project(
            project=project,
            repo_path=repo_path,
            dry_run=dry_run,
        )

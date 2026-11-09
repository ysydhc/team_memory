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
import math
import os
import subprocess
from contextlib import asynccontextmanager
from datetime import datetime, timedelta, timezone

from sqlalchemy import and_, func, or_, select

from team_memory.project_resolver import normalize_project_name
from team_memory.services.maintenance.promotion_compiler import PromotionCompiler
from team_memory.storage.models import Archive, ArchiveExperienceLink, Experience, PersonalMemory

logger = logging.getLogger("team_memory.services.maintenance.janitor")

_CANONICAL_ARCHIVE_CONTENT_TYPES = {
    "session_archive",
    "tech_design",
    "incident_review",
    "decision_record",
    "bugfix",
    "engineering_experience",
}


def _cosine_similarity(vec_a: object, vec_b: object) -> float:
    """Compute cosine similarity between two vector-like sequences."""
    if vec_a is None or vec_b is None:
        return 0.0
    try:
        a = [float(x) for x in vec_a]  # type: ignore[attr-defined]
        b = [float(x) for x in vec_b]  # type: ignore[attr-defined]
    except (TypeError, ValueError):
        return 0.0
    if len(a) != len(b) or not a:
        return 0.0
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(x * x for x in b))
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return dot / (norm_a * norm_b)


def _normalize_archive_content_type(raw: str | None) -> str:
    """Map free-form archive content_type strings to the 6 canonical types."""
    if not raw or not isinstance(raw, str):
        return "session_archive"
    ct = raw.strip().lower()
    if ct in _CANONICAL_ARCHIVE_CONTENT_TYPES:
        return ct
    if any(
        k in ct
        for k in (
            "incident",
            "postmortem",
            "retrospective",
            "audit",
            "analysis",
            "test_report",
            "troubleshooting",
            "verification",
        )
    ):
        return "incident_review"
    if any(k in ct for k in ("decision", "adr")):
        return "decision_record"
    if any(k in ct for k in ("bug", "fix", "hotfix", "patch")):
        return "bugfix"
    if any(
        k in ct
        for k in ("design", "architecture", "refactor", "spec", "plan", "migration", "feature", "optimization")
    ):
        return "tech_design"
    if any(
        k in ct
        for k in ("guide", "sop", "experience", "best_practice", "onboarding", "deployment", "workflow", "handbook")
    ):
        return "engineering_experience"
    return "session_archive"


def _utcnow() -> datetime:
    """Get current UTC datetime."""
    return datetime.now(timezone.utc)


def _is_eligible_for_promotion(exp, threshold: int, require_used: bool) -> bool:
    """Check whether an experience is eligible for promotion."""
    if isinstance(getattr(exp, "valid_until", None), datetime):
        return False
    if isinstance(getattr(exp, "negative_feedback_count", None), int) and exp.negative_feedback_count > 0:
        return False
    if require_used and isinstance(getattr(exp, "used_count", None), int) and exp.used_count < threshold:
        return False
    return True


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
            # (falling back to updated_at/created_at when deleted_at is NULL)
            query = select(Experience).where(
                and_(
                    Experience.is_deleted == True,  # noqa: E712
                    func.coalesce(Experience.deleted_at, Experience.updated_at, Experience.created_at) < cutoff_date,
                )
            )

            result = await session.execute(query)
            to_purge = result.scalars().all()

            purged_count = len(to_purge)

            # Mark search_logs referencing purged IDs as orphaned
            purged_ids = [str(exp.id) for exp in to_purge]
            if purged_ids:
                from sqlalchemy import text

                id_list = ",".join(f"'{pid}'" for pid in purged_ids)
                await session.execute(
                    text(f"""
                        UPDATE search_logs
                        SET judgment_source = 'orphaned', updated_at = NOW()
                        WHERE was_used IS NULL
                          AND result_ids IS NOT NULL
                          AND result_ids::text != '[]'
                          AND result_ids::text != 'null'
                          AND id IN (
                            SELECT sl.id FROM search_logs sl,
                                   json_array_elements(sl.result_ids) elem
                            WHERE (elem->>'id') IN ({id_list})
                          )
                    """),
                )

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
        """Remove old dynamic personal memories and semantically compact static user personas.

        Args:
            older_than_days: Days threshold for dynamic personal memory removal

        Returns:
            Dict with pruning and semantic compaction statistics
        """
        async with self._session() as session:
            cutoff_date = _utcnow() - timedelta(days=older_than_days)
            dedup_threshold = float(self._get_config("personal_memory_dedup_threshold", 0.83))

            # 1. Find old dynamic personal memories
            query = select(PersonalMemory).where(
                and_(
                    PersonalMemory.profile_kind == "dynamic",
                    PersonalMemory.created_at < cutoff_date,
                )
            )

            result = await session.execute(query)
            old_memories = result.scalars().all()

            expired_dynamic_count = len(old_memories)

            for memory in old_memories:
                await session.delete(memory)
                logger.debug("Pruned expired dynamic personal memory: %s", memory.id)

            # 2. Semantically compact static personal memories per (user_id, persona)
            static_query = select(PersonalMemory).where(PersonalMemory.profile_kind == "static")
            static_result = await session.execute(static_query)
            static_memories = [
                m
                for m in static_result.scalars().all()
                if isinstance(getattr(m, "content", None), str) and isinstance(getattr(m, "user_id", None), str)
            ]

            by_group: dict[str, list[PersonalMemory]] = {}
            local_aliases = {"default", "developer", "admin", "yeshouyou"}
            for m in static_memories:
                uid_key = "local_user" if m.user_id in local_aliases else m.user_id
                by_group.setdefault(uid_key, []).append(m)

            deduped_static_count = 0
            for _uid, items in by_group.items():
                if len(items) < 2:
                    continue
                # Sort by priority DESC, content length DESC, updated_at DESC
                items.sort(
                    key=lambda x: (
                        int(getattr(x, "priority", 50) or 50),
                        len((getattr(x, "content", "") or "").strip()),
                        getattr(x, "updated_at", None)
                        or getattr(x, "created_at", None)
                        or datetime.min.replace(tzinfo=timezone.utc),
                    ),
                    reverse=True,
                )
                kept: list[PersonalMemory] = []
                for candidate in items:
                    cand_text = (candidate.content or "").strip()
                    cand_norm = " ".join(cand_text.lower().split())
                    matched_primary: PersonalMemory | None = None
                    for primary in kept:
                        prim_text = (primary.content or "").strip()
                        prim_norm = " ".join(prim_text.lower().split())
                        if cand_norm == prim_norm:
                            matched_primary = primary
                            break
                        sim = _cosine_similarity(
                            getattr(primary, "embedding", None),
                            getattr(candidate, "embedding", None),
                        )
                        if sim >= dedup_threshold:
                            matched_primary = primary
                            break

                    if matched_primary is not None:
                        # Boost primary priority (capped at 100) and keep richer wording if much longer
                        cur_prio = int(getattr(matched_primary, "priority", 50) or 50)
                        cand_prio = int(getattr(candidate, "priority", 50) or 50)
                        matched_primary.priority = min(100, max(cur_prio, cand_prio) + 10)
                        if len(cand_text) > int(len((matched_primary.content or "").strip()) * 1.15):
                            matched_primary.content = cand_text
                            if getattr(candidate, "embedding", None) is not None:
                                matched_primary.embedding = candidate.embedding
                        matched_primary.updated_at = _utcnow()
                        await session.delete(candidate)
                        deduped_static_count += 1
                    else:
                        kept.append(candidate)

            await session.commit()

            pruned_count = expired_dynamic_count + deduped_static_count
            logger.info(
                "Personal memory pruning completed: %d expired dynamic, %d redundant static compacted (>%d days)",
                expired_dynamic_count,
                deduped_static_count,
                older_than_days,
            )

            return {
                "pruned_count": pruned_count,
                "expired_dynamic_count": expired_dynamic_count,
                "deduped_static_count": deduped_static_count,
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
        require_used = bool(self._get_config("promotion_require_used", False))
        group_key_threshold = self._get_config("promotion_group_key_threshold", 5)
        output_dirs: dict[str, str] = self._get_config("promotion_output_dirs", {})

        promoted_by_recall_count = 0
        promoted_by_group = 0

        async with self._session() as session:
            # 1. Promote by recall_count threshold
            count_condition = (
                Experience.used_count >= recall_count_threshold
                if require_used
                else or_(
                    Experience.used_count >= recall_count_threshold,
                    Experience.recall_count >= recall_count_threshold,
                )
            )
            recall_count_query = select(Experience).where(
                and_(
                    Experience.is_deleted == False,  # noqa: E712
                    Experience.exp_status == "published",
                    Experience.valid_until.is_(None),
                    Experience.negative_feedback_count == 0,
                    count_condition,
                )
            )
            if project:
                recall_count_query = recall_count_query.where(Experience.project == project)

            result = await session.execute(recall_count_query)
            recall_count_experiences = [
                e
                for e in result.scalars().all()
                if _is_eligible_for_promotion(e, recall_count_threshold, require_used)
            ]

            # 2. Promote by group_key threshold
            # Find group_keys with enough published experiences
            from sqlalchemy import func

            group_count_query = (
                select(Experience.group_key, func.count(Experience.id).label("cnt"))
                .where(
                    and_(
                        Experience.is_deleted == False,  # noqa: E712
                        Experience.exp_status == "published",
                        Experience.valid_until.is_(None),
                        Experience.negative_feedback_count == 0,
                        Experience.group_key.isnot(None),
                    )
                )
                .group_by(Experience.group_key)
                .having(func.count(Experience.id) >= group_key_threshold)
            )
            if project:
                group_count_query = group_count_query.where(Experience.project == project)

            group_result = await session.execute(group_count_query)
            qualifying_groups = [row[0] for row in group_result.all()]

            group_experiences: list[Experience] = []
            if qualifying_groups:
                # Get all published experiences in qualifying groups
                group_exp_query = select(Experience).where(
                    and_(
                        Experience.is_deleted == False,  # noqa: E712
                        Experience.exp_status == "published",
                        Experience.valid_until.is_(None),
                        Experience.negative_feedback_count == 0,
                        Experience.group_key.in_(qualifying_groups),
                    )
                )
                if project:
                    group_exp_query = group_exp_query.where(Experience.project == project)

                group_exp_result = await session.execute(group_exp_query)
                group_experiences = [
                    e
                    for e in group_exp_result.scalars().all()
                    if _is_eligible_for_promotion(e, 0, False)
                ]

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
            "promoted_by_used_count": promoted_by_recall_count,
            "promoted_by_group": promoted_by_group,
            "total": total,
            "recall_count_threshold": recall_count_threshold,
            "group_key_threshold": group_key_threshold,
            "project": project,
        }

    async def _write_promoted(self, exp: Experience, markdown: str, output_dirs: dict[str, str]) -> None:
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

    async def dedup_experiences(self, project: str | None = None) -> dict:
        """Soft-delete duplicate experiences, keeping the latest/best version / 去重经验，保留最佳版本.

        Combines:
        1. Exact (title, project) duplicate soft-deletion
        2. Project name canonicalization (e.g. hyphen -> underscore)
        3. Vector semantic deduplication (cosine >= 0.90) within the same canonical project

        Returns:
            Dict with dedup statistics.
        """
        from sqlalchemy import text

        semantic_threshold = float(self._get_config("experience_dedup_threshold", 0.90))

        async with self._session() as session:
            # 1. Exact (title, project) duplicates via SQL
            result = await session.execute(
                text("""
                    WITH duplicates AS (
                        SELECT id, title, project,
                               ROW_NUMBER() OVER (
                                   PARTITION BY title, project
                                   ORDER BY updated_at DESC NULLS LAST, created_at DESC
                               ) as rn
                        FROM experiences
                        WHERE is_deleted = false
                          AND (CAST(:project AS TEXT) IS NULL OR project = :project)
                    )
                    UPDATE experiences
                    SET is_deleted = true, deleted_at = NOW()
                    WHERE id IN (
                        SELECT id FROM duplicates WHERE rn > 1
                    )
                """),
                {"project": project},
            )
            exact_deleted_count = result.rowcount if isinstance(getattr(result, "rowcount", None), int) else 0

            # 2. Load active experiences for project canonicalization + vector semantic dedup
            exp_query = select(Experience).where(Experience.is_deleted == False)  # noqa: E712
            if project:
                exp_query = exp_query.where(Experience.project == project)
            exp_result = await session.execute(exp_query)
            active_exps = [
                e
                for e in exp_result.scalars().all()
                if isinstance(getattr(e, "title", None), str) and isinstance(getattr(e, "project", None), str)
            ]

            normalized_project_count = 0
            by_project: dict[str, list[Experience]] = {}
            for exp in active_exps:
                canon_proj = normalize_project_name(exp.project) or "default"
                if exp.project != canon_proj:
                    exp.project = canon_proj
                    normalized_project_count += 1
                by_project.setdefault(canon_proj, []).append(exp)

            semantic_deleted_count = 0
            now = _utcnow()
            for _proj, items in by_project.items():
                if len(items) < 2:
                    continue
                # Prefer promoted > published > draft, higher used_count/recall_count, longer solution, newer updated_at
                status_rank = {"promoted": 3, "published": 2, "review": 1, "draft": 0}
                items.sort(
                    key=lambda x: (
                        bool(getattr(x, "is_pinned", False)),
                        status_rank.get(getattr(x, "exp_status", "") or "", 0),
                        int(getattr(x, "used_count", 0) or 0),
                        int(getattr(x, "recall_count", 0) or 0),
                        len((getattr(x, "solution", "") or "").strip()),
                        getattr(x, "updated_at", None)
                        or getattr(x, "created_at", None)
                        or datetime.min.replace(tzinfo=timezone.utc),
                    ),
                    reverse=True,
                )
                kept: list[Experience] = []
                for candidate in items:
                    cand_title_norm = " ".join((candidate.title or "").lower().split())
                    matched_primary: Experience | None = None
                    for primary in kept:
                        prim_title_norm = " ".join((primary.title or "").lower().split())
                        if cand_title_norm and cand_title_norm == prim_title_norm:
                            matched_primary = primary
                            break
                        sim = _cosine_similarity(
                            getattr(primary, "embedding", None),
                            getattr(candidate, "embedding", None),
                        )
                        if sim >= semantic_threshold:
                            matched_primary = primary
                            break

                    if matched_primary is not None and not getattr(candidate, "is_pinned", False):
                        # Merge tags & richer solution if needed
                        prim_tags = list(getattr(matched_primary, "tags", None) or [])
                        for t in getattr(candidate, "tags", None) or []:
                            if t not in prim_tags:
                                prim_tags.append(t)
                        matched_primary.tags = prim_tags[:15]
                        prim_sol = (getattr(matched_primary, "solution", "") or "").strip()
                        cand_sol = (getattr(candidate, "solution", "") or "").strip()
                        if len(cand_sol) > int(len(prim_sol) * 1.2) and cand_sol not in prim_sol:
                            matched_primary.solution = f"{prim_sol}\n\n---\n{cand_sol}" if prim_sol else cand_sol
                        matched_primary.recall_count = int(getattr(matched_primary, "recall_count", 0) or 0) + int(
                            getattr(candidate, "recall_count", 0) or 0
                        )
                        matched_primary.used_count = int(getattr(matched_primary, "used_count", 0) or 0) + int(
                            getattr(candidate, "used_count", 0) or 0
                        )
                        candidate.is_deleted = True
                        candidate.deleted_at = now
                        candidate.superseded_by_id = matched_primary.id
                        semantic_deleted_count += 1
                    else:
                        kept.append(candidate)

            await session.commit()

            total_deleted = exact_deleted_count + semantic_deleted_count
            logger.info(
                "Experience dedup completed: soft-deleted %d (%d exact, %d semantic), normalized %d projects in '%s'",
                total_deleted,
                exact_deleted_count,
                semantic_deleted_count,
                normalized_project_count,
                project or "all",
            )

            return {
                "soft_deleted_count": total_deleted,
                "exact_deleted_count": exact_deleted_count,
                "semantic_deleted_count": semantic_deleted_count,
                "normalized_project_count": normalized_project_count,
                "project": project,
            }

    async def maintain_archives(self, project: str | None = None) -> dict:
        """Maintain archive quality: normalize types/projects, deduplicate, handle stale drafts, clean links.

        Args:
            project: Optional project filter.

        Returns:
            Dict with archive maintenance statistics.
        """
        draft_promote_days = int(self._get_config("archive_draft_promote_days", 7))
        draft_purge_days = int(self._get_config("archive_draft_purge_days", 14))
        semantic_threshold = float(self._get_config("archive_dedup_threshold", 0.92))

        now = _utcnow()
        promote_cutoff = now - timedelta(days=draft_promote_days)
        purge_cutoff = now - timedelta(days=draft_purge_days)

        async with self._session() as session:
            arch_query = select(Archive)
            if project:
                arch_query = arch_query.where(Archive.project == project)
            arch_result = await session.execute(arch_query)
            archives = [
                a
                for a in arch_result.scalars().all()
                if isinstance(getattr(a, "title", None), str) and isinstance(getattr(a, "solution_doc", None), str)
            ]

            normalized_type_count = 0
            normalized_project_count = 0
            promoted_draft_count = 0
            purged_stub_draft_count = 0
            deduped_archive_count = 0

            # 1. Clean up dangling ArchiveExperienceLink rows pointing to soft-deleted Experiences first
            dangling_link_query = (
                select(ArchiveExperienceLink)
                .join(Experience, ArchiveExperienceLink.experience_id == Experience.id)
                .where(Experience.is_deleted == True)  # noqa: E712
            )
            dangling_res = await session.execute(dangling_link_query)
            dangling_links = [
                lnk
                for lnk in dangling_res.scalars().all()
                if getattr(lnk, "archive_id", None) is not None and getattr(lnk, "experience_id", None) is not None
            ]
            dangling_links_removed = 0
            for lnk in dangling_links:
                await session.delete(lnk)
                dangling_links_removed += 1

            # 2. Group archives by canonical project name and deduplicate BEFORE mutating project/title
            #    so unique constraint uq_archives_title_project is never violated during flush.
            by_project: dict[str, list[Archive]] = {}
            for arch in archives:
                created_at = getattr(arch, "created_at", None)
                sol_text = (arch.solution_doc or "").strip()
                if getattr(arch, "status", None) == "draft" and isinstance(created_at, datetime):
                    if (created_at < purge_cutoff and len(sol_text) < 120) or (
                        created_at < promote_cutoff and len(sol_text) < 120
                    ):
                        await session.delete(arch)
                        purged_stub_draft_count += 1
                        continue
                canon_proj = normalize_project_name(getattr(arch, "project", None)) or "default"
                by_project.setdefault(canon_proj, []).append(arch)

            kept_by_project: list[tuple[str, Archive]] = []
            for canon_proj, items in by_project.items():
                items.sort(
                    key=lambda a: (
                        1 if getattr(a, "status", "") == "published" else 0,
                        len((getattr(a, "solution_doc", "") or "").strip()),
                        getattr(a, "updated_at", None)
                        or getattr(a, "created_at", None)
                        or datetime.min.replace(tzinfo=timezone.utc),
                    ),
                    reverse=True,
                )
                kept: list[Archive] = []
                for candidate in items:
                    cand_title_norm = " ".join((candidate.title or "").lower().split())
                    matched_primary: Archive | None = None
                    for primary in kept:
                        prim_title_norm = " ".join((primary.title or "").lower().split())
                        if cand_title_norm and cand_title_norm == prim_title_norm:
                            matched_primary = primary
                            break
                        sim = _cosine_similarity(
                            getattr(primary, "embedding", None),
                            getattr(candidate, "embedding", None),
                        )
                        if sim >= semantic_threshold:
                            matched_primary = primary
                            break

                    if matched_primary is not None:
                        await session.delete(candidate)
                        deduped_archive_count += 1
                    else:
                        kept.append(candidate)
                for k in kept:
                    kept_by_project.append((canon_proj, k))

            # Flush deletions before mutating project/status on surviving archives
            if hasattr(session, "flush"):
                await session.flush()

            # 3. Normalize content_type, project, and promote complete stale drafts on surviving archives
            for canon_proj, arch in kept_by_project:
                canon_ct = _normalize_archive_content_type(getattr(arch, "content_type", None))
                if arch.content_type != canon_ct:
                    arch.content_type = canon_ct
                    normalized_type_count += 1

                if arch.project != canon_proj:
                    arch.project = canon_proj
                    normalized_project_count += 1

                created_at = getattr(arch, "created_at", None)
                if getattr(arch, "status", None) == "draft" and isinstance(created_at, datetime):
                    sol_text = (arch.solution_doc or "").strip()
                    ov_text = (getattr(arch, "overview", "") or "").strip()
                    if created_at < promote_cutoff and len(sol_text) >= 120:
                        if not ov_text:
                            arch.overview = sol_text[:200]
                        arch.status = "published"
                        arch.updated_at = now
                        promoted_draft_count += 1

            await session.commit()

            logger.info(
                "Archive maintenance completed: normalized_types=%d, normalized_projects=%d, "
                "promoted_drafts=%d, purged_stub_drafts=%d, deduped=%d, dangling_links_removed=%d",
                normalized_type_count,
                normalized_project_count,
                promoted_draft_count,
                purged_stub_draft_count,
                deduped_archive_count,
                dangling_links_removed,
            )

            return {
                "normalized_type_count": normalized_type_count,
                "normalized_project_count": normalized_project_count,
                "promoted_draft_count": promoted_draft_count,
                "purged_stub_draft_count": purged_stub_draft_count,
                "deduped_archive_count": deduped_archive_count,
                "dangling_links_removed": dangling_links_removed,
                "project": project,
            }

    async def dedup_search_logs(self, project: str | None = None) -> dict:
        """Delete duplicate search logs, keeping the earliest per hour / 去重搜索日志，保留每小时最早一条.

        Returns:
            Dict with dedup statistics.
        """
        from sqlalchemy import text

        async with self._session() as session:
            result = await session.execute(
                text("""
                    WITH ranked AS (
                        SELECT id,
                               ROW_NUMBER() OVER (
                                   PARTITION BY query, project, date_trunc('hour', created_at)
                                   ORDER BY created_at ASC
                               ) as rn
                        FROM search_logs
                        WHERE (CAST(:project AS TEXT) IS NULL OR project = :project)
                    )
                    DELETE FROM search_logs
                    WHERE id IN (SELECT id FROM ranked WHERE rn > 1)
                """),
                {"project": project},
            )
            deleted_count = result.rowcount

            logger.info(
                "Search log dedup completed: deleted %d duplicates in project '%s'",
                deleted_count,
                project or "all",
            )

            return {
                "deleted_count": deleted_count,
                "project": project,
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

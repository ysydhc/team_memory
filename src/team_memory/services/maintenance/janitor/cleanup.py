"""Cleanup, expiration, personal memory pruning, and experience dedup for MemoryJanitor."""

from __future__ import annotations

import logging
import uuid
from datetime import datetime, timedelta, timezone

from sqlalchemy import and_, func, select

from team_memory.project_resolver import normalize_project_name
from team_memory.services.maintenance.janitor.helpers import _cosine_similarity, _utcnow
from team_memory.storage.models import Experience, PersonalMemory

logger = logging.getLogger("team_memory.services.maintenance.janitor")


class JanitorCleanupMixin:
    """Mixin providing purge, draft expiration, personal memory pruning, and dedup."""

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

            by_group: dict[tuple[str, str, str, str], list[PersonalMemory]] = {}
            local_aliases = {"default", "developer", "admin", "yeshouyou"}
            for m in static_memories:
                raw_tid = getattr(m, "tenant_id", None)
                tid_key = str(raw_tid) if isinstance(raw_tid, (str, uuid.UUID)) else "default"
                uid_key = "local_user" if m.user_id in local_aliases else m.user_id
                raw_scope = getattr(m, "scope", None)
                scope_key = raw_scope if isinstance(raw_scope, str) and raw_scope else "global"
                if scope_key == "generic":
                    scope_key = "global"
                raw_proj = getattr(m, "project", None)
                proj_key = (
                    raw_proj if scope_key == "project" and isinstance(raw_proj, str) else ""
                )
                by_group.setdefault((tid_key, uid_key, scope_key, proj_key), []).append(m)

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

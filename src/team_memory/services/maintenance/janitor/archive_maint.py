"""Archive quality maintenance for MemoryJanitor."""

from __future__ import annotations

import logging
from datetime import datetime, timedelta, timezone

from sqlalchemy import select

from team_memory.project_resolver import normalize_project_name
from team_memory.services.maintenance.janitor.helpers import (
    _cosine_similarity,
    _normalize_archive_content_type,
    _utcnow,
)
from team_memory.storage.models import Archive, ArchiveExperienceLink, Experience

logger = logging.getLogger("team_memory.services.maintenance.janitor")


class JanitorArchiveMixin:
    """Mixin providing archive normalization, deduplication, and link cleanup."""

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

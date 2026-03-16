"""Archive repository — CRUD and vector search for archives."""

from __future__ import annotations

import uuid
from typing import Any

from sqlalchemy import and_, desc, func, or_, select, update
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from team_memory.storage.models import (
    Archive,
    ArchiveAttachment,
    ArchiveExperienceLink,
    Experience,
)

SOLUTION_PREVIEW_LEN = 500


def _project_value(project: str | None) -> str:
    if project and project.strip():
        value = project.strip()
        alias_map = {
            "team-memory": "team_memory",
            "team_doc": "team_memory",
        }
        return alias_map.get(value, value)
    return "default"


class ArchiveRepository:
    """Repository for Archive CRUD and vector search.

    All methods require an AsyncSession (same pattern as ExperienceRepository).
    """

    def __init__(self, session: AsyncSession):
        self._session = session

    async def create_archive(
        self,
        title: str,
        solution_doc: str,
        created_by: str,
        *,
        project: str | None = None,
        scope: str = "session",
        scope_ref: str | None = None,
        overview: str | None = None,
        conversation_summary: str | None = None,
        visibility: str = "project",
        status: str = "draft",
        embedding: list[float] | None = None,
        linked_experience_ids: list[uuid.UUID] | None = None,
        attachments: list[dict[str, Any]] | None = None,
    ) -> uuid.UUID:
        """Insert archive, links, and attachments; derive status from linked experiences."""
        proj = _project_value(project)
        archive = Archive(
            title=title,
            solution_doc=solution_doc,
            created_by=created_by,
            project=proj,
            scope=scope,
            scope_ref=scope_ref,
            overview=overview,
            conversation_summary=conversation_summary,
            visibility=visibility,
            status=status,
            embedding=embedding,
        )
        self._session.add(archive)
        await self._session.flush()

        if linked_experience_ids:
            for eid in linked_experience_ids:
                link = ArchiveExperienceLink(
                    archive_id=archive.id,
                    experience_id=eid,
                )
                self._session.add(link)
            await self._session.flush()

            # All linked experiences published -> archive published
            stmt = (
                select(Experience.exp_status)
                .where(Experience.id.in_(linked_experience_ids))
            )
            result = await self._session.execute(stmt)
            statuses = [row[0] for row in result.all()]
            if statuses and all(s == "published" for s in statuses):
                await self._session.execute(
                    update(Archive)
                    .where(Archive.id == archive.id)
                    .values(status="published")
                )

        if attachments:
            for att in attachments:
                a = ArchiveAttachment(
                    archive_id=archive.id,
                    kind=att.get("kind", "file_ref"),
                    path=att.get("path"),
                    content_snapshot=att.get("content_snapshot"),
                    git_commit=att.get("git_commit"),
                    git_refs=att.get("git_refs"),
                    snippet=att.get("snippet"),
                )
                self._session.add(a)
            await self._session.flush()

        return archive.id

    async def search_archives(
        self,
        query_embedding: list[float],
        project: str | None = None,
        limit: int = 10,
        min_similarity: float = 0.0,
        current_user: str | None = None,
    ) -> list[dict[str, Any]]:
        """Vector search; filter by creator or published + project."""
        similarity_expr = (
            1 - Archive.embedding.cosine_distance(query_embedding)
        ).label("similarity")

        proj = _project_value(project)
        query = (
            select(Archive, similarity_expr)
            .where(Archive.embedding.is_not(None))
            .where(similarity_expr >= min_similarity)
            .order_by(desc(similarity_expr))
            .limit(limit)
        )
        if current_user:
            query = query.where(
                or_(
                    Archive.created_by == current_user,
                    and_(
                        Archive.status == "published",
                        Archive.project == proj,
                    ),
                )
            )
        else:
            query = query.where(
                and_(
                    Archive.status == "published",
                    Archive.project == proj,
                )
            )

        result = await self._session.execute(query)
        rows = result.all()
        out = []
        for archive, sim in rows:
            preview = (
                (archive.overview or archive.solution_doc or "")[:SOLUTION_PREVIEW_LEN]
            )
            linked_ids = []
            link_result = await self._session.execute(
                select(ArchiveExperienceLink.experience_id).where(
                    ArchiveExperienceLink.archive_id == archive.id
                )
            )
            linked_ids = [r[0] for r in link_result.all()]
            count_result = await self._session.execute(
                select(func.count(ArchiveAttachment.id)).where(
                    ArchiveAttachment.archive_id == archive.id
                )
            )
            att_count = count_result.scalar() or 0
            out.append({
                "id": str(archive.id),
                "title": archive.title,
                "solution_preview": preview,
                "score": round(float(sim), 4),
                "similarity": round(float(sim), 4),
                "linked_experience_ids": [str(x) for x in linked_ids],
                "attachment_count": att_count,
                "type": "archive",
            })
        return out

    async def recompute_archive_status_for_linked_experience(
        self, experience_id: uuid.UUID
    ) -> None:
        """Recompute and update status of all archives that link this experience.

        For each such archive: if all its linked experiences have exp_status='published',
        set archive.status='published'; otherwise set 'draft'.
        """
        # Archives that link this experience
        link_q = (
            select(ArchiveExperienceLink.archive_id)
            .where(ArchiveExperienceLink.experience_id == experience_id)
            .distinct()
        )
        result = await self._session.execute(link_q)
        archive_ids = [row[0] for row in result.all()]
        if not archive_ids:
            return

        for archive_id in archive_ids:
            # Linked experience ids for this archive
            link_ids_q = (
                select(ArchiveExperienceLink.experience_id).where(
                    ArchiveExperienceLink.archive_id == archive_id
                )
            )
            r2 = await self._session.execute(link_ids_q)
            exp_ids = [row[0] for row in r2.all()]
            if not exp_ids:
                await self._session.execute(
                    update(Archive).where(Archive.id == archive_id).values(status="draft")
                )
                continue
            stmt = select(Experience.exp_status).where(Experience.id.in_(exp_ids))
            r3 = await self._session.execute(stmt)
            statuses = [row[0] for row in r3.all()]
            all_pub = statuses and all(s == "published" for s in statuses)
            new_status = "published" if all_pub else "draft"
            await self._session.execute(
                update(Archive).where(Archive.id == archive_id).values(status=new_status)
            )

    async def get_archive_by_id(
        self,
        archive_id: uuid.UUID,
    ) -> dict[str, Any] | None:
        """Return single archive with attachments (L2)."""
        result = await self._session.execute(
            select(Archive)
            .where(Archive.id == archive_id)
            .options(selectinload(Archive.attachments))
        )
        archive = result.scalar_one_or_none()
        if archive is None:
            return None

        link_result = await self._session.execute(
            select(ArchiveExperienceLink.experience_id).where(
                ArchiveExperienceLink.archive_id == archive.id
            )
        )
        linked_ids = [str(r[0]) for r in link_result.all()]

        att_list = [
            {
                "id": str(a.id),
                "kind": a.kind,
                "path": a.path,
                "content_snapshot": a.content_snapshot,
                "snippet": a.snippet,
                "git_commit": a.git_commit,
                "git_refs": a.git_refs,
            }
            for a in archive.attachments
        ]

        return {
            "id": str(archive.id),
            "title": archive.title,
            "scope": archive.scope,
            "scope_ref": archive.scope_ref,
            "solution_doc": archive.solution_doc,
            "overview": archive.overview,
            "conversation_summary": archive.conversation_summary,
            "project": archive.project,
            "created_by": archive.created_by,
            "linked_experience_ids": linked_ids,
            "attachments": att_list,
            "created_at": archive.created_at.isoformat() if archive.created_at else None,
            "updated_at": archive.updated_at.isoformat() if archive.updated_at else None,
        }

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
    ArchiveUploadFailure,
    Experience,
)

SOLUTION_PREVIEW_LEN = 500
OVERVIEW_PREVIEW_MAX = 2000


def _project_value(project: str | None) -> str:
    if project and project.strip():
        value = project.strip()
        alias_map = {
            "team-memory": "team_memory",
            "team_doc": "team_memory",
        }
        return alias_map.get(value, value)
    return "default"


def _archive_visible_to_viewer(
    archive: Archive,
    viewer_name: str | None,
    proj: str,
) -> bool:
    """Same rules as search_archives WHERE clause."""
    if viewer_name:
        return archive.created_by == viewer_name or (
            archive.status == "published" and archive.project == proj
        )
    return archive.status == "published" and archive.project == proj


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
            stmt = select(Experience.exp_status).where(Experience.id.in_(linked_experience_ids))
            result = await self._session.execute(stmt)
            statuses = [row[0] for row in result.all()]
            if statuses and all(s == "published" for s in statuses):
                await self._session.execute(
                    update(Archive).where(Archive.id == archive.id).values(status="published")
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
        similarity_expr = (1 - Archive.embedding.cosine_distance(query_embedding)).label(
            "similarity"
        )

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
            preview = (archive.overview or archive.solution_doc or "")[:SOLUTION_PREVIEW_LEN]
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
            ov = (archive.overview or "").strip()
            overview_preview = ov[:OVERVIEW_PREVIEW_MAX] if ov else ""
            out.append(
                {
                    "id": str(archive.id),
                    "title": archive.title,
                    "solution_preview": preview,
                    "overview_preview": overview_preview,
                    "score": round(float(sim), 4),
                    "similarity": round(float(sim), 4),
                    "linked_experience_ids": [str(x) for x in linked_ids],
                    "attachment_count": att_count,
                    "type": "archive",
                }
            )
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
            link_ids_q = select(ArchiveExperienceLink.experience_id).where(
                ArchiveExperienceLink.archive_id == archive_id
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

    async def _build_l2_dict(self, archive: Archive) -> dict[str, Any]:
        link_result = await self._session.execute(
            select(ArchiveExperienceLink.experience_id).where(
                ArchiveExperienceLink.archive_id == archive.id
            )
        )
        linked_ids = [str(r[0]) for r in link_result.all()]
        aid_str = str(archive.id)
        att_list = [
            {
                "id": str(a.id),
                "kind": a.kind,
                "path": a.path,
                "content_snapshot": a.content_snapshot,
                "snippet": a.snippet,
                "git_commit": a.git_commit,
                "git_refs": a.git_refs,
                "storage": "local",
                "download_api_path": (f"/api/v1/archives/{aid_str}/attachments/{a.id}/file"),
            }
            for a in archive.attachments
        ]
        nodes = sorted(
            archive.tree_nodes,
            key=lambda n: (n.depth, n.node_order, n.path),
        )
        tree_payload = [n.to_dict() for n in nodes]
        return {
            "id": str(archive.id),
            "title": archive.title,
            "scope": archive.scope,
            "scope_ref": archive.scope_ref,
            "solution_doc": archive.solution_doc,
            "overview": archive.overview,
            "conversation_summary": archive.conversation_summary,
            "visibility": archive.visibility,
            "project": archive.project,
            "created_by": archive.created_by,
            "status": archive.status,
            "linked_experience_ids": linked_ids,
            "attachments": att_list,
            "document_tree_nodes": tree_payload,
            "created_at": archive.created_at.isoformat() if archive.created_at else None,
            "updated_at": archive.updated_at.isoformat() if archive.updated_at else None,
        }

    async def get_archive_by_id(
        self,
        archive_id: uuid.UUID,
    ) -> dict[str, Any] | None:
        """Return L2 without visibility check (internal/admin only)."""
        result = await self._session.execute(
            select(Archive)
            .where(Archive.id == archive_id)
            .options(
                selectinload(Archive.attachments),
                selectinload(Archive.tree_nodes),
            )
        )
        archive = result.scalar_one_or_none()
        if archive is None:
            return None
        return await self._build_l2_dict(archive)

    async def get_archive_for_viewer(
        self,
        archive_id: uuid.UUID,
        viewer_name: str | None,
        project: str | None,
    ) -> dict[str, Any] | None:
        """L2 when visible to viewer; None if missing or denied (treat as 404)."""
        result = await self._session.execute(
            select(Archive)
            .where(Archive.id == archive_id)
            .options(
                selectinload(Archive.attachments),
                selectinload(Archive.tree_nodes),
            )
        )
        archive = result.scalar_one_or_none()
        if archive is None:
            return None
        proj = _project_value(project)
        if not _archive_visible_to_viewer(archive, viewer_name, proj):
            return None
        return await self._build_l2_dict(archive)

    async def list_archives_for_viewer(
        self,
        viewer_name: str | None,
        project: str | None,
        *,
        q: str | None = None,
        limit: int = 30,
        offset: int = 0,
    ) -> tuple[list[dict[str, Any]], int]:
        """Paginated list with same visibility as search_archives; returns items + total."""
        proj = _project_value(project)
        visibility = (
            or_(
                Archive.created_by == viewer_name,
                and_(
                    Archive.status == "published",
                    Archive.project == proj,
                ),
            )
            if viewer_name
            else and_(Archive.status == "published", Archive.project == proj)
        )
        base_where = visibility

        count_q = select(func.count(Archive.id)).where(base_where)
        list_q = select(Archive).where(base_where)
        if q and q.strip():
            term = f"%{q.strip()}%"
            text_filter = or_(
                Archive.title.ilike(term),
                Archive.overview.ilike(term),
            )
            list_q = list_q.where(text_filter)
            count_q = count_q.where(text_filter)

        list_q = list_q.order_by(desc(Archive.created_at)).offset(offset).limit(limit)

        total = (await self._session.execute(count_q)).scalar() or 0
        rows = (await self._session.execute(list_q)).scalars().all()

        items: list[dict[str, Any]] = []
        for archive in rows:
            preview = (archive.overview or archive.solution_doc or "")[:SOLUTION_PREVIEW_LEN]
            ov = (archive.overview or "").strip()
            overview_preview = ov[:OVERVIEW_PREVIEW_MAX] if ov else ""

            count_result = await self._session.execute(
                select(func.count(ArchiveAttachment.id)).where(
                    ArchiveAttachment.archive_id == archive.id
                )
            )
            att_count = count_result.scalar() or 0

            items.append(
                {
                    "id": str(archive.id),
                    "title": archive.title,
                    "scope": archive.scope,
                    "scope_ref": archive.scope_ref,
                    "project": archive.project,
                    "created_by": archive.created_by,
                    "status": archive.status,
                    "solution_preview": preview,
                    "overview_preview": overview_preview,
                    "attachment_count": att_count,
                    "created_at": archive.created_at.isoformat() if archive.created_at else None,
                    "updated_at": archive.updated_at.isoformat() if archive.updated_at else None,
                }
            )
        return items, int(total)

    async def add_attachment_row(
        self,
        archive_id: uuid.UUID,
        *,
        attachment_id: uuid.UUID,
        kind: str,
        rel_path: str,
        snippet: str | None = None,
    ) -> uuid.UUID:
        """Insert a single archive_attachments row (file already on disk)."""
        row = ArchiveAttachment(
            id=attachment_id,
            archive_id=archive_id,
            kind=kind,
            path=rel_path,
            snippet=snippet,
        )
        self._session.add(row)
        await self._session.flush()
        return row.id

    async def insert_upload_failure(
        self,
        archive_id: uuid.UUID,
        *,
        created_by: str | None,
        source: str,
        error_code: str,
        error_message: str,
        client_filename_hint: str | None = None,
    ) -> uuid.UUID:
        """Persist a failed upload attempt."""
        msg = (error_message or "")[:500]
        row = ArchiveUploadFailure(
            archive_id=archive_id,
            created_by=created_by,
            source=(source or "web")[:20],
            error_code=error_code[:50],
            error_message=msg,
            client_filename_hint=(client_filename_hint or "")[:500]
            if client_filename_hint
            else None,
        )
        self._session.add(row)
        await self._session.flush()
        return row.id

    async def list_upload_failures_for_viewer(
        self,
        archive_id: uuid.UUID,
        viewer_name: str | None,
        project: str | None,
        *,
        limit: int = 20,
        include_resolved: bool = False,
    ) -> list[dict[str, Any]]:
        """List failures for an archive if viewer can see L2."""
        result = await self._session.execute(select(Archive).where(Archive.id == archive_id))
        archive = result.scalar_one_or_none()
        if archive is None:
            return []
        proj = _project_value(project)
        if not _archive_visible_to_viewer(archive, viewer_name, proj):
            return []
        q = select(ArchiveUploadFailure).where(ArchiveUploadFailure.archive_id == archive_id)
        if not include_resolved:
            q = q.where(ArchiveUploadFailure.resolved_at.is_(None))
        q = q.order_by(desc(ArchiveUploadFailure.created_at)).limit(limit)
        rows = (await self._session.execute(q)).scalars().all()
        return [
            {
                "id": str(r.id),
                "archive_id": str(r.archive_id),
                "created_at": r.created_at.isoformat() if r.created_at else None,
                "created_by": r.created_by,
                "source": r.source,
                "error_code": r.error_code,
                "error_message": r.error_message,
                "client_filename_hint": r.client_filename_hint,
                "resolved_at": r.resolved_at.isoformat() if r.resolved_at else None,
            }
            for r in rows
        ]

    async def resolve_upload_failure(
        self,
        archive_id: uuid.UUID,
        failure_id: uuid.UUID,
        viewer_name: str | None,
        project: str | None,
    ) -> bool:
        """Set resolved_at=now if viewer can see archive."""
        from datetime import datetime, timezone

        result = await self._session.execute(select(Archive).where(Archive.id == archive_id))
        archive = result.scalar_one_or_none()
        if archive is None:
            return False
        proj = _project_value(project)
        if not _archive_visible_to_viewer(archive, viewer_name, proj):
            return False
        fr = await self._session.get(ArchiveUploadFailure, failure_id)
        if fr is None or fr.archive_id != archive_id:
            return False
        fr.resolved_at = datetime.now(timezone.utc)
        await self._session.flush()
        return True

    async def get_attachment_relative_path_for_viewer(
        self,
        archive_id: uuid.UUID,
        attachment_id: uuid.UUID,
        viewer_name: str | None,
        project: str | None,
    ) -> str | None:
        """Return attachment.path (relative to uploads root) if visible; else None."""
        result = await self._session.execute(
            select(Archive)
            .where(Archive.id == archive_id)
            .options(selectinload(Archive.attachments))
        )
        archive = result.scalar_one_or_none()
        if archive is None:
            return None
        proj = _project_value(project)
        if not _archive_visible_to_viewer(archive, viewer_name, proj):
            return None
        for a in archive.attachments:
            if a.id == attachment_id:
                return a.path
        return None

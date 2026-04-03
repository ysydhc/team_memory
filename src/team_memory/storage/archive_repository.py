"""Archive repository — CRUD and vector search for archives."""

from __future__ import annotations

import logging
import uuid
from datetime import datetime, timedelta, timezone
from typing import Any

from sqlalchemy import and_, case, delete, desc, func, literal, or_, select, update
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from team_memory.storage.models import (
    Archive,
    ArchiveAttachment,
    ArchiveExperienceLink,
    ArchiveUploadFailure,
    Experience,
)

logger = logging.getLogger("team_memory.storage.archive_repository")

SOLUTION_PREVIEW_LEN = 500
OVERVIEW_PREVIEW_MAX = 2000
UNBOUNDED_QUERY_LIMIT = 1000


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
    *,
    viewer_role: str | None = None,
) -> bool:
    """Same rules as search_archives WHERE clause; admin sees everything."""
    if viewer_role == "admin":
        return True
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
        raw_conversation: str | None = None,
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
            raw_conversation=raw_conversation,
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

    async def upsert_archive(
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
        content_type: str = "session_archive",
        value_summary: str | None = None,
        tags: list[str] | None = None,
        visibility: str = "project",
        embedding: list[float] | None = None,
        linked_experience_ids: list[uuid.UUID] | None = None,
    ) -> dict[str, Any]:
        """Atomic upsert using INSERT ... ON CONFLICT (title, project) DO UPDATE.

        On conflict, updates content fields but does NOT delete old attachments
        (attachments are incremental). Experience links are re-associated: old
        links for this archive are deleted and new ones inserted.

        Returns ``{"action": "created"|"updated", "archive_id": UUID,
        "previous_updated_at": str|None}``.
        """
        proj = _project_value(project)
        now = datetime.now(timezone.utc)
        new_id = uuid.uuid4()

        insert_values: dict[str, Any] = {
            "id": new_id,
            "title": title,
            "solution_doc": solution_doc,
            "created_by": created_by,
            "project": proj,
            "scope": scope,
            "scope_ref": scope_ref,
            "overview": overview,
            "conversation_summary": conversation_summary,
            "content_type": content_type,
            "value_summary": value_summary,
            "tags": tags or [],
            "visibility": visibility,
            "status": "draft",
            "embedding": embedding,
            "created_at": now,
            "updated_at": now,
        }

        update_on_conflict: dict[str, Any] = {
            "overview": overview,
            "solution_doc": solution_doc,
            "conversation_summary": conversation_summary,
            "tags": tags or [],
            "content_type": content_type,
            "value_summary": value_summary,
            "embedding": embedding,
            "scope": scope,
            "scope_ref": scope_ref,
            "updated_at": now,
        }

        stmt = (
            insert(Archive)
            .values(**insert_values)
            .on_conflict_do_update(
                index_elements=["title", "project"],
                index_where=Archive.title.is_not(None),
                set_=update_on_conflict,
            )
            .returning(Archive.id, Archive.created_at, Archive.updated_at)
        )

        result = await self._session.execute(stmt)
        row = result.one()
        archive_id: uuid.UUID = row.id
        created_at: datetime = row.created_at
        updated_at: datetime = row.updated_at

        # Determine if this was a create or update: if created_at is close to
        # the updated_at we just set (within 1 second), it was a fresh insert.
        was_created = abs(created_at - updated_at) < timedelta(seconds=1)
        previous_updated_at: str | None = None
        if not was_created:
            # The row already existed; the old updated_at was overwritten.
            # We cannot recover the exact previous value from RETURNING alone,
            # but we can infer: if updated_at == now, the old value was anything
            # before now. We report created_at as a stable reference instead.
            previous_updated_at = created_at.isoformat()

        # --- Re-associate experience links ---
        # Delete old links for this archive (incremental attachments are NOT touched).
        await self._session.execute(
            delete(ArchiveExperienceLink).where(ArchiveExperienceLink.archive_id == archive_id)
        )

        if linked_experience_ids:
            link_rows = [
                {"archive_id": archive_id, "experience_id": eid} for eid in linked_experience_ids
            ]
            await self._session.execute(insert(ArchiveExperienceLink).values(link_rows))

            # Derive status from linked experiences
            exp_stmt = select(Experience.exp_status).where(Experience.id.in_(linked_experience_ids))
            exp_result = await self._session.execute(exp_stmt)
            statuses = [r[0] for r in exp_result.all()]
            if statuses and all(s == "published" for s in statuses):
                await self._session.execute(
                    update(Archive).where(Archive.id == archive_id).values(status="published")
                )

        await self._session.flush()

        action = "created" if was_created else "updated"
        return {
            "action": action,
            "archive_id": archive_id,
            "previous_updated_at": previous_updated_at,
        }

    async def search_archives(
        self,
        query_embedding: list[float],
        project: str | None = None,
        limit: int = 10,
        min_similarity: float = 0.0,
        current_user: str | None = None,
    ) -> list[dict[str, Any]]:
        """Vector search; filter by creator or published + project.

        Uses correlated subqueries for linked_experience_ids and attachment_count
        to avoid N+1 queries.
        """
        similarity_expr = (1 - Archive.embedding.cosine_distance(query_embedding)).label(
            "similarity"
        )

        # Correlated subquery: array of linked experience UUIDs (removes NULLs)
        # When no links exist, array_agg returns NULL → handled in Python as []
        linked_ids_sub = (
            select(
                func.array_remove(
                    func.array_agg(ArchiveExperienceLink.experience_id),
                    None,
                )
            )
            .where(ArchiveExperienceLink.archive_id == Archive.id)
            .correlate(Archive)
            .scalar_subquery()
            .label("linked_ids")
        )

        # Correlated subquery: attachment count
        att_count_sub = (
            select(func.count(ArchiveAttachment.id))
            .where(ArchiveAttachment.archive_id == Archive.id)
            .correlate(Archive)
            .scalar_subquery()
            .label("attachment_count")
        )

        proj = _project_value(project)
        query = (
            select(Archive, similarity_expr, linked_ids_sub, att_count_sub)
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
        out: list[dict[str, Any]] = []
        for archive, sim, linked_ids_raw, att_count in rows:
            preview = (archive.overview or archive.solution_doc or "")[:SOLUTION_PREVIEW_LEN]
            linked_ids = linked_ids_raw if linked_ids_raw else []
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
                    "content_type": archive.content_type,
                    "value_summary": archive.value_summary,
                    "tags": archive.tags or [],
                    "linked_experience_ids": [str(x) for x in linked_ids],
                    "attachment_count": att_count or 0,
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

        Uses a single UPDATE with a CASE expression to avoid N+1 queries.
        """
        # Archives that link this experience
        archive_ids_q = (
            select(ArchiveExperienceLink.archive_id)
            .where(ArchiveExperienceLink.experience_id == experience_id)
            .distinct()
        )

        # Subquery: archive_ids where ALL linked experiences are published
        all_published_q = (
            select(ArchiveExperienceLink.archive_id)
            .join(Experience, Experience.id == ArchiveExperienceLink.experience_id)
            .where(ArchiveExperienceLink.archive_id.in_(archive_ids_q))
            .group_by(ArchiveExperienceLink.archive_id)
            .having(func.bool_and(Experience.exp_status == "published"))
        )

        # Bulk update: published if all linked are published, else draft
        await self._session.execute(
            update(Archive)
            .where(Archive.id.in_(archive_ids_q))
            .values(
                status=case(
                    (Archive.id.in_(all_published_q), "published"),
                    else_="draft",
                )
            )
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
                "source_path": a.source_path,
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
            "content_type": archive.content_type,
            "value_summary": archive.value_summary,
            "tags": archive.tags or [],
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
        *,
        viewer_role: str | None = None,
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
        if not _archive_visible_to_viewer(archive, viewer_name, proj, viewer_role=viewer_role):
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
        viewer_role: str | None = None,
    ) -> tuple[list[dict[str, Any]], int]:
        """Paginated list with same visibility as search_archives; returns items + total.

        Uses a correlated scalar subquery for attachment_count to avoid N+1.
        """
        proj = _project_value(project)
        if viewer_role == "admin":
            # Admin sees all archives
            base_where = literal(True)
        elif viewer_name:
            base_where = or_(
                Archive.created_by == viewer_name,
                and_(
                    Archive.status == "published",
                    Archive.project == proj,
                ),
            )
        else:
            base_where = and_(Archive.status == "published", Archive.project == proj)

        # Correlated subquery: attachment count per archive
        att_count_sub = (
            select(func.count(ArchiveAttachment.id))
            .where(ArchiveAttachment.archive_id == Archive.id)
            .correlate(Archive)
            .scalar_subquery()
            .label("attachment_count")
        )

        count_q = select(func.count(Archive.id)).where(base_where)
        list_q = select(Archive, att_count_sub).where(base_where)
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
        rows = (await self._session.execute(list_q)).all()

        items: list[dict[str, Any]] = []
        for archive, att_count in rows:
            preview = (archive.overview or archive.solution_doc or "")[:SOLUTION_PREVIEW_LEN]
            ov = (archive.overview or "").strip()
            overview_preview = ov[:OVERVIEW_PREVIEW_MAX] if ov else ""

            items.append(
                {
                    "id": str(archive.id),
                    "title": archive.title,
                    "scope": archive.scope,
                    "scope_ref": archive.scope_ref,
                    "project": archive.project,
                    "created_by": archive.created_by,
                    "status": archive.status,
                    "content_type": archive.content_type,
                    "value_summary": archive.value_summary,
                    "tags": archive.tags or [],
                    "solution_preview": preview,
                    "overview_preview": overview_preview,
                    "attachment_count": att_count or 0,
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
        source_path: str | None = None,
    ) -> uuid.UUID:
        """Insert a single archive_attachments row (file already on disk)."""
        row = ArchiveAttachment(
            id=attachment_id,
            archive_id=archive_id,
            kind=kind,
            path=rel_path,
            snippet=snippet,
            source_path=source_path,
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
        viewer_role: str | None = None,
    ) -> list[dict[str, Any]]:
        """List failures for an archive if viewer can see L2."""
        result = await self._session.execute(select(Archive).where(Archive.id == archive_id))
        archive = result.scalar_one_or_none()
        if archive is None:
            return []
        proj = _project_value(project)
        if not _archive_visible_to_viewer(archive, viewer_name, proj, viewer_role=viewer_role):
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
        *,
        viewer_role: str | None = None,
    ) -> bool:
        """Set resolved_at=now if viewer can see archive."""
        result = await self._session.execute(select(Archive).where(Archive.id == archive_id))
        archive = result.scalar_one_or_none()
        if archive is None:
            return False
        proj = _project_value(project)
        if not _archive_visible_to_viewer(archive, viewer_name, proj, viewer_role=viewer_role):
            return False
        fr = await self._session.get(ArchiveUploadFailure, failure_id)
        if fr is None or fr.archive_id != archive_id:
            return False
        fr.resolved_at = datetime.now(timezone.utc)
        await self._session.flush()
        return True

    async def get_archive_ids_for_experience(
        self,
        experience_id: uuid.UUID,
    ) -> list[str]:
        """Return archive IDs that link to this experience."""
        result = await self._session.execute(
            select(ArchiveExperienceLink.archive_id)
            .where(ArchiveExperienceLink.experience_id == experience_id)
            .limit(UNBOUNDED_QUERY_LIMIT)
        )
        rows = [str(row[0]) for row in result.all()]
        if len(rows) >= UNBOUNDED_QUERY_LIMIT:
            logger.warning(
                "get_archive_ids_for_experience hit %d limit for experience %s",
                UNBOUNDED_QUERY_LIMIT,
                experience_id,
            )
        return rows

    async def get_archive_ids_for_experiences(
        self,
        experience_ids: list[uuid.UUID],
    ) -> dict[str, list[str]]:
        """Return {experience_id_str: [archive_id_str, ...]} for multiple experiences."""
        if not experience_ids:
            return {}
        result = await self._session.execute(
            select(
                ArchiveExperienceLink.experience_id,
                ArchiveExperienceLink.archive_id,
            )
            .where(ArchiveExperienceLink.experience_id.in_(experience_ids))
            .limit(UNBOUNDED_QUERY_LIMIT)
        )
        rows = result.all()
        if len(rows) >= UNBOUNDED_QUERY_LIMIT:
            logger.warning(
                "get_archive_ids_for_experiences hit %d limit for %d experiences",
                UNBOUNDED_QUERY_LIMIT,
                len(experience_ids),
            )
        mapping: dict[str, list[str]] = {}
        for exp_id, arch_id in rows:
            key = str(exp_id)
            mapping.setdefault(key, []).append(str(arch_id))
        return mapping

    async def get_attachment_relative_path_for_viewer(
        self,
        archive_id: uuid.UUID,
        attachment_id: uuid.UUID,
        viewer_name: str | None,
        project: str | None,
        *,
        viewer_role: str | None = None,
    ) -> tuple[str | None, str | None]:
        """Return (rel_path, source_path) if visible; (None, None) otherwise."""
        result = await self._session.execute(
            select(Archive)
            .where(Archive.id == archive_id)
            .options(selectinload(Archive.attachments))
        )
        archive = result.scalar_one_or_none()
        if archive is None:
            return None, None
        proj = _project_value(project)
        if not _archive_visible_to_viewer(archive, viewer_name, proj, viewer_role=viewer_role):
            return None, None
        for a in archive.attachments:
            if a.id == attachment_id:
                return a.path, a.source_path
        return None, None

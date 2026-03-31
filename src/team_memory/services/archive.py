"""Archive business logic — save and retrieve archives with embedding."""

from __future__ import annotations

import asyncio
import logging
import uuid
from pathlib import Path
from typing import Any

from team_memory.embedding.base import EmbeddingProvider
from team_memory.storage.archive_repository import ArchiveRepository
from team_memory.storage.database import get_session

logger = logging.getLogger("team_memory.service")

OVERVIEW_FALLBACK_MAX = 2000


class ArchiveUploadError(Exception):
    """Upload rejected after archive visibility was confirmed (HTTP mapping in route)."""

    def __init__(self, error_code: str, message: str, *, http_status: int = 400) -> None:
        self.error_code = error_code
        self.message = message
        self.http_status = http_status
        super().__init__(message)


def derive_overview_fallback(solution_doc: str, max_len: int = OVERVIEW_FALLBACK_MAX) -> str:
    """Deterministic L1 when caller omits overview: prefer first ## section else head."""
    t = (solution_doc or "").strip()
    if not t:
        return ""
    lines = t.split("\n")
    start_idx = 0
    for i, line in enumerate(lines):
        if line.startswith("## ") and i > 0:
            start_idx = i
            break
    chunk = "\n".join(lines[start_idx:]).strip() if start_idx else t
    chunk = chunk[:max_len].rstrip()
    return chunk or t[:max_len]


def _embedding_text_for_archive(
    title: str,
    solution_doc: str,
    overview: str | None = None,
    conversation_summary: str | None = None,
) -> str:
    """Build text for embedding (no-loss hit rate: same as L0/L1 preview source)."""
    parts = [title or ""]
    if overview and overview.strip():
        parts.append(overview.strip())
    else:
        parts.append((solution_doc or "")[:2000])
    if conversation_summary and conversation_summary.strip():
        parts.append(conversation_summary.strip())
    return "\n\n".join(p for p in parts if p)


class ArchiveService:
    """Service for archive create and get; uses EmbeddingProvider and ArchiveRepository."""

    def __init__(
        self,
        embedding_provider: EmbeddingProvider,
        db_url: str,
    ):
        self._embedding = embedding_provider
        self._db_url = db_url

    async def archive_save(
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
        linked_experience_ids: list[uuid.UUID] | None = None,
        attachments: list[dict[str, Any]] | None = None,
    ) -> uuid.UUID:
        """Create archive with optional embedding; return archive_id."""
        ov = (overview or "").strip() or None
        if not ov:
            fb = derive_overview_fallback(solution_doc)
            ov = fb if fb else None
        text = _embedding_text_for_archive(title, solution_doc, ov, conversation_summary)
        embedding: list[float] | None = None
        try:
            vectors = await self._embedding.encode([text])
            if vectors and len(vectors) == 1:
                embedding = vectors[0]
        except Exception as e:
            logger.warning("Archive embedding encode failed: %s", e)

        async with get_session(self._db_url) as session:
            repo = ArchiveRepository(session)
            archive_id = await repo.create_archive(
                title=title,
                solution_doc=solution_doc,
                created_by=created_by,
                project=project,
                scope=scope,
                scope_ref=scope_ref,
                overview=ov,
                conversation_summary=conversation_summary,
                visibility="project",
                status="draft",
                embedding=embedding,
                linked_experience_ids=linked_experience_ids,
                attachments=attachments or [],
            )
            await session.commit()
            return archive_id

    async def get_archive(
        self,
        archive_id: uuid.UUID,
        *,
        viewer: str | None = None,
        project: str | None = None,
    ) -> dict[str, Any] | None:
        """Return L2 when visible to viewer; None if missing or denied."""
        async with get_session(self._db_url) as session:
            repo = ArchiveRepository(session)
            return await repo.get_archive_for_viewer(archive_id, viewer, project)

    async def list_archives(
        self,
        *,
        viewer: str | None,
        project: str | None,
        q: str | None = None,
        limit: int = 30,
        offset: int = 0,
    ) -> tuple[list[dict[str, Any]], int]:
        """Paginated archives visible to viewer."""
        async with get_session(self._db_url) as session:
            repo = ArchiveRepository(session)
            return await repo.list_archives_for_viewer(
                viewer, project, q=q, limit=limit, offset=offset
            )

    async def update_archive_status_for_experience(self, experience_id: uuid.UUID) -> None:
        """Recompute status of all archives linking this experience and persist.

        Call after any change to an experience's exp_status (e.g. change_status,
        publish_to_team, publish_personal, review).
        """
        async with get_session(self._db_url) as session:
            repo = ArchiveRepository(session)
            await repo.recompute_archive_status_for_linked_experience(experience_id)
            await session.commit()

    async def _record_upload_failure(
        self,
        archive_id: uuid.UUID,
        viewer: str | None,
        project: str | None,
        source: str,
        error_code: str,
        error_message: str,
        client_filename_hint: str | None,
    ) -> None:
        visible = await self.get_archive(archive_id, viewer=viewer, project=project)
        if visible is None:
            return
        try:
            async with get_session(self._db_url) as session:
                repo = ArchiveRepository(session)
                await repo.insert_upload_failure(
                    archive_id,
                    created_by=viewer,
                    source=source,
                    error_code=error_code,
                    error_message=error_message,
                    client_filename_hint=client_filename_hint,
                )
                await session.commit()
        except Exception:
            logger.exception("record_upload_failure failed for archive_id=%s", archive_id)

    async def upload_archive_attachment(
        self,
        archive_id: uuid.UUID,
        viewer: str | None,
        project: str | None,
        uploads_cfg: Any,
        *,
        file_content: bytes,
        client_filename: str | None,
        kind: str,
        snippet: str | None,
        source: str = "web",
    ) -> dict[str, Any]:
        """Stream already-read body to disk, then INSERT archive_attachments (§4.4)."""
        from team_memory.config import UploadsConfig
        from team_memory.utils.archive_upload_paths import (
            normalized_under_root,
            safe_suffix,
        )

        if not isinstance(uploads_cfg, UploadsConfig):
            uploads_cfg = UploadsConfig()

        visible = await self.get_archive(archive_id, viewer=viewer, project=project)
        if visible is None:
            raise ArchiveUploadError("not_found", "Archive not found", http_status=404)

        if not uploads_cfg.enabled:
            await self._record_upload_failure(
                archive_id,
                viewer,
                project,
                source,
                "503",
                "File uploads are disabled by configuration",
                client_filename,
            )
            raise ArchiveUploadError(
                "disabled",
                "File uploads are disabled",
                http_status=503,
            )

        if len(file_content) > uploads_cfg.max_bytes:
            await self._record_upload_failure(
                archive_id,
                viewer,
                project,
                source,
                "413",
                f"File exceeds max_bytes={uploads_cfg.max_bytes}",
                client_filename,
            )
            raise ArchiveUploadError(
                "payload_too_large",
                "File too large",
                http_status=413,
            )

        allowed = uploads_cfg.allowed_extensions or []
        try:
            ext = safe_suffix(client_filename, allowed if len(allowed) > 0 else None)
        except ValueError as e:
            await self._record_upload_failure(
                archive_id,
                viewer,
                project,
                source,
                "validation",
                str(e)[:500],
                client_filename,
            )
            raise ArchiveUploadError("validation", str(e), http_status=400) from e

        if len(allowed) > 0 and not ext:
            msg = "Filename must include an allowed extension"
            await self._record_upload_failure(
                archive_id,
                viewer,
                project,
                source,
                "validation",
                msg,
                client_filename,
            )
            raise ArchiveUploadError("validation", msg, http_status=400)

        attachment_id = uuid.uuid4()
        root = Path(uploads_cfg.root_dir).expanduser().resolve()
        aid_dir = root / str(archive_id)
        part_path = aid_dir / f"{attachment_id}.part"
        safe_ext = ext or ""
        final_name = f"{attachment_id}{safe_ext}"
        rel_posix = f"{archive_id}/{final_name}".replace("\\", "/")
        final_path = root / str(archive_id) / final_name

        if not normalized_under_root(final_path, root):
            await self._record_upload_failure(
                archive_id,
                viewer,
                project,
                source,
                "path",
                "Invalid storage path",
                client_filename,
            )
            raise ArchiveUploadError(
                "path", "Invalid storage path", http_status=500
            )

        def _write_atomic() -> None:
            aid_dir.mkdir(parents=True, exist_ok=True)
            part_path.write_bytes(file_content)
            part_path.replace(final_path)

        try:
            await asyncio.to_thread(_write_atomic)
        except OSError as e:
            if part_path.exists():
                try:
                    part_path.unlink()
                except OSError:
                    pass
            await self._record_upload_failure(
                archive_id,
                viewer,
                project,
                source,
                "disk",
                str(e)[:500],
                client_filename,
            )
            raise ArchiveUploadError(
                "disk", "Failed to store file", http_status=500
            ) from e

        try:
            async with get_session(self._db_url) as session:
                repo = ArchiveRepository(session)
                await repo.add_attachment_row(
                    archive_id,
                    attachment_id=attachment_id,
                    kind=kind or "file",
                    rel_path=rel_posix,
                    snippet=snippet,
                )
                await session.commit()
        except Exception as e:
            try:
                if final_path.exists():
                    final_path.unlink()
            except OSError:
                pass
            await self._record_upload_failure(
                archive_id,
                viewer,
                project,
                source,
                "commit",
                str(e)[:500],
                client_filename,
            )
            raise ArchiveUploadError(
                "commit", "Database error", http_status=500
            ) from e

        aid_str = str(archive_id)
        return {
            "id": str(attachment_id),
            "archive_id": aid_str,
            "kind": kind or "file",
            "path": rel_posix,
            "download_api_path": (
                f"/api/v1/archives/{aid_str}/attachments/{attachment_id}/file"
            ),
        }

    async def read_archive_attachment_file(
        self,
        archive_id: uuid.UUID,
        attachment_id: uuid.UUID,
        viewer: str | None,
        project: str | None,
        uploads_cfg: Any,
    ) -> tuple[Path, str] | None:
        """Return absolute path and suggested filename if viewer may access file."""
        from team_memory.config import UploadsConfig
        from team_memory.utils.archive_upload_paths import normalized_under_root

        if not isinstance(uploads_cfg, UploadsConfig):
            uploads_cfg = UploadsConfig()

        root = Path(uploads_cfg.root_dir).expanduser().resolve()
        async with get_session(self._db_url) as session:
            repo = ArchiveRepository(session)
            rel = await repo.get_attachment_relative_path_for_viewer(
                archive_id, attachment_id, viewer, project
            )
        if rel is None:
            return None
        full = (root / rel).resolve()
        if not normalized_under_root(full, root) or not full.is_file():
            return None
        return full, full.name

    async def list_upload_failures(
        self,
        archive_id: uuid.UUID,
        viewer: str | None,
        project: str | None,
        *,
        limit: int = 20,
        include_resolved: bool = False,
    ) -> list[dict[str, Any]]:
        async with get_session(self._db_url) as session:
            repo = ArchiveRepository(session)
            return await repo.list_upload_failures_for_viewer(
                archive_id,
                viewer,
                project,
                limit=limit,
                include_resolved=include_resolved,
            )

    async def mark_upload_failure_resolved(
        self,
        archive_id: uuid.UUID,
        failure_id: uuid.UUID,
        viewer: str | None,
        project: str | None,
    ) -> bool:
        async with get_session(self._db_url) as session:
            repo = ArchiveRepository(session)
            ok = await repo.resolve_upload_failure(
                archive_id, failure_id, viewer, project
            )
            if ok:
                await session.commit()
            return ok

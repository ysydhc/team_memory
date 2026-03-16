"""Archive business logic — save and retrieve archives with embedding."""

from __future__ import annotations

import logging
import uuid
from typing import Any

from team_memory.embedding.base import EmbeddingProvider
from team_memory.storage.archive_repository import ArchiveRepository
from team_memory.storage.database import get_session

logger = logging.getLogger("team_memory.service")


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
        text = _embedding_text_for_archive(
            title, solution_doc, overview, conversation_summary
        )
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
                overview=overview,
                conversation_summary=conversation_summary,
                visibility="project",
                status="draft",
                embedding=embedding,
                linked_experience_ids=linked_experience_ids,
                attachments=attachments or [],
            )
            await session.commit()
            return archive_id

    async def get_archive(self, archive_id: uuid.UUID) -> dict[str, Any] | None:
        """Return L2 archive by id or None."""
        async with get_session(self._db_url) as session:
            repo = ArchiveRepository(session)
            out = await repo.get_archive_by_id(archive_id)
            return out

"""Personal memory service: per-user preferences/habits with semantic overwrite."""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from typing import TYPE_CHECKING

from team_memory.storage.repository import (
    PERSONAL_MEMORY_OVERWRITE_THRESHOLD,
    PersonalMemoryRepository,
)

if TYPE_CHECKING:
    from team_memory.embedding.base import EmbeddingProvider

logger = logging.getLogger("team_memory.personal_memory")


class PersonalMemoryService:
    """Service for personal memory CRUD and semantic overwrite (mem0-style)."""

    def __init__(
        self,
        embedding_provider: EmbeddingProvider,
        db_url: str = "",
    ):
        self._embedding = embedding_provider
        self._db_url = db_url

    @asynccontextmanager
    async def _session(self):
        from team_memory.storage.database import get_session

        async with get_session(self._db_url) as session:
            yield session

    async def list_by_user(
        self, user_id: str, scope: str | None = None
    ) -> list[dict]:
        """List memories for user, optionally filtered by scope."""
        async with self._session() as session:
            repo = PersonalMemoryRepository(session)
            rows = await repo.list_by_user(user_id, scope=scope)
            return [r.to_dict() for r in rows]

    async def pull(
        self,
        user_id: str | None,
        current_context: str | None = None,
        context_similarity_threshold: float = 0.5,
    ) -> list[dict]:
        """Pull memories for Agent context: generic + context-matched.

        Anonymous user (None or 'anonymous'): returns [] (no generic).
        Else: returns scope=generic all; if current_context given, also
        scope=context items whose embedding is similar to current_context.
        """
        if not user_id or user_id.strip().lower() == "anonymous":
            return []
        context_embedding = None
        if current_context and current_context.strip():
            context_embedding = await self._embedding.encode_single(
                current_context.strip()
            )
        async with self._session() as session:
            repo = PersonalMemoryRepository(session)
            rows = await repo.list_for_pull(
                user_id=user_id,
                context_embedding=context_embedding,
                context_similarity_threshold=context_similarity_threshold,
            )
            return [r.to_dict() for r in rows]

    async def get_by_id(self, memory_id: str, user_id: str) -> dict | None:
        """Get one memory by id; must belong to user."""
        import uuid

        try:
            uid = uuid.UUID(memory_id)
        except ValueError:
            return None
        async with self._session() as session:
            repo = PersonalMemoryRepository(session)
            mem = await repo.get_by_id(uid, user_id)
            return mem.to_dict() if mem else None

    async def write(
        self,
        user_id: str,
        content: str,
        scope: str = "generic",
        context_hint: str | None = None,
        overwrite_threshold: float = PERSONAL_MEMORY_OVERWRITE_THRESHOLD,
    ) -> dict:
        """Write one memory: compute embedding, then upsert by semantic (overwrite if similar)."""
        embedding = await self._embedding.encode_single(content)
        async with self._session() as session:
            repo = PersonalMemoryRepository(session)
            mem = await repo.upsert_by_semantic(
                user_id=user_id,
                content=content,
                scope=scope,
                context_hint=context_hint,
                embedding=embedding,
                threshold=overwrite_threshold,
            )
            return mem.to_dict()

    async def update(
        self,
        memory_id: str,
        user_id: str,
        content: str | None = None,
        scope: str | None = None,
        context_hint: str | None = None,
    ) -> dict | None:
        """Update an existing memory; optionally re-compute embedding if content changed."""
        import uuid

        try:
            uid = uuid.UUID(memory_id)
        except ValueError:
            return None
        embedding = None
        if content is not None:
            embedding = await self._embedding.encode_single(content)
        async with self._session() as session:
            repo = PersonalMemoryRepository(session)
            mem = await repo.update(
                uid, user_id,
                content=content, scope=scope, context_hint=context_hint,
                embedding=embedding,
            )
            return mem.to_dict() if mem else None

    async def delete(self, memory_id: str, user_id: str) -> bool:
        """Delete a memory; returns True if deleted."""
        import uuid

        try:
            uid = uuid.UUID(memory_id)
        except ValueError:
            return False
        async with self._session() as session:
            repo = PersonalMemoryRepository(session)
            return await repo.delete(uid, user_id)

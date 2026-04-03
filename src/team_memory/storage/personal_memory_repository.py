"""Personal memory repository -- CRUD + semantic overwrite for per-user memories."""

from __future__ import annotations

import uuid

from sqlalchemy import delete, desc, select
from sqlalchemy.ext.asyncio import AsyncSession

from team_memory.storage.models import PersonalMemory

PERSONAL_MEMORY_OVERWRITE_THRESHOLD = 0.88


class PersonalMemoryRepository:
    """Repository for PersonalMemory CRUD and semantic overwrite."""

    def __init__(self, session: AsyncSession):
        self._session = session

    async def list_by_user(
        self,
        user_id: str,
        scope: str | None = None,
        profile_kind: str | None = None,
    ) -> list[PersonalMemory]:
        q = select(PersonalMemory).where(PersonalMemory.user_id == user_id)
        if scope:
            q = q.where(PersonalMemory.scope == scope)
        if profile_kind:
            q = q.where(PersonalMemory.profile_kind == profile_kind)
        q = q.order_by(desc(PersonalMemory.updated_at))
        result = await self._session.execute(q)
        return list(result.scalars().all())

    async def list_for_pull(
        self,
        user_id: str,
        context_embedding: list[float] | None = None,
        context_similarity_threshold: float = 0.5,
    ) -> list[PersonalMemory]:
        """Static rows all + dynamic rows (all if no embedding, else similarity-filtered)."""
        static_rows = await self.list_by_user(user_id, profile_kind="static")
        if not context_embedding:
            dynamic_rows = await self.list_by_user(user_id, profile_kind="dynamic")
            return static_rows + dynamic_rows
        similarity_expr = (1 - PersonalMemory.embedding.cosine_distance(context_embedding)).label(
            "similarity"
        )
        q = (
            select(PersonalMemory)
            .where(PersonalMemory.user_id == user_id)
            .where(PersonalMemory.profile_kind == "dynamic")
            .where(PersonalMemory.embedding.is_not(None))
            .where(similarity_expr >= context_similarity_threshold)
            .order_by(desc(PersonalMemory.updated_at))
        )
        result = await self._session.execute(q)
        dynamic_items = list(result.scalars().all())
        return static_rows + dynamic_items

    async def get_by_id(self, memory_id: uuid.UUID, user_id: str) -> PersonalMemory | None:
        result = await self._session.execute(
            select(PersonalMemory).where(
                PersonalMemory.id == memory_id,
                PersonalMemory.user_id == user_id,
            )
        )
        return result.scalar_one_or_none()

    async def find_most_similar(
        self,
        user_id: str,
        embedding: list[float],
        threshold: float,
        profile_kind: str,
    ) -> PersonalMemory | None:
        if not embedding:
            return None
        similarity_expr = (1 - PersonalMemory.embedding.cosine_distance(embedding)).label(
            "similarity"
        )
        q = (
            select(PersonalMemory, similarity_expr)
            .where(PersonalMemory.user_id == user_id)
            .where(PersonalMemory.profile_kind == profile_kind)
            .where(PersonalMemory.embedding.is_not(None))
            .where(similarity_expr >= threshold)
            .order_by(desc(similarity_expr))
            .limit(1)
        )
        result = await self._session.execute(q)
        row = result.first()
        if row:
            return row[0]
        return None

    async def create(
        self,
        user_id: str,
        content: str,
        scope: str = "generic",
        context_hint: str | None = None,
        embedding: list[float] | None = None,
        profile_kind: str | None = None,
    ) -> PersonalMemory:
        pk = profile_kind or ("dynamic" if scope == "context" else "static")
        mem = PersonalMemory(
            user_id=user_id,
            content=content,
            scope=scope,
            profile_kind=pk,
            context_hint=context_hint,
            embedding=embedding,
        )
        self._session.add(mem)
        await self._session.flush()
        return mem

    async def update(
        self,
        memory_id: uuid.UUID,
        user_id: str,
        content: str | None = None,
        scope: str | None = None,
        context_hint: str | None = None,
        embedding: list[float] | None = None,
        profile_kind: str | None = None,
    ) -> PersonalMemory | None:
        mem = await self.get_by_id(memory_id, user_id)
        if mem is None:
            return None
        if content is not None:
            mem.content = content
        if scope is not None:
            mem.scope = scope
        if profile_kind is not None:
            mem.profile_kind = profile_kind
        if context_hint is not None:
            mem.context_hint = context_hint
        if embedding is not None:
            mem.embedding = embedding
        await self._session.flush()
        return mem

    async def delete(self, memory_id: uuid.UUID, user_id: str) -> bool:
        result = await self._session.execute(
            delete(PersonalMemory).where(
                PersonalMemory.id == memory_id,
                PersonalMemory.user_id == user_id,
            )
        )
        return result.rowcount > 0

    async def upsert_by_semantic(
        self,
        user_id: str,
        content: str,
        embedding: list[float],
        scope: str = "generic",
        context_hint: str | None = None,
        threshold: float = PERSONAL_MEMORY_OVERWRITE_THRESHOLD,
        profile_kind: str | None = None,
    ) -> PersonalMemory:
        """Insert or overwrite by semantic similarity within the same profile_kind."""
        pk = profile_kind or ("dynamic" if scope == "context" else "static")
        existing = await self.find_most_similar(user_id, embedding, threshold, pk)
        if existing:
            await self.update(
                existing.id,
                user_id,
                content=content,
                scope=scope,
                context_hint=context_hint,
                embedding=embedding,
                profile_kind=pk,
            )
            await self._session.refresh(existing)
            return existing
        return await self.create(
            user_id=user_id,
            content=content,
            scope=scope,
            context_hint=context_hint,
            embedding=embedding,
            profile_kind=pk,
        )

"""Personal memory repository -- CRUD + semantic overwrite for per-user memories."""

from __future__ import annotations

import uuid

from sqlalchemy import and_, case, delete, desc, func, or_, select
from sqlalchemy.ext.asyncio import AsyncSession

from team_memory.storage.models import PersonalMemory

PERSONAL_MEMORY_OVERWRITE_THRESHOLD = 0.82
VALID_PERSONAS = ("coding", "debugging", "writing", "research", "general")
VALID_SCOPES = ("global", "project")
_GLOBAL_COMPAT_SCOPES = ("global", "generic", "context")

__all__ = [
    "PERSONAL_MEMORY_OVERWRITE_THRESHOLD",
    "VALID_PERSONAS",
    "VALID_SCOPES",
    "normalize_scope",
    "PersonalMemoryRepository",
]


def normalize_scope(
    scope: str | None = None,
    *,
    project: str | None = None,
    profile_kind: str | None = None,
) -> tuple[str, str | None, str]:
    """Normalize (scope, project, profile_kind) into canonical values.

    Canonical scopes:
      - ``global``: cross-project personal preference (project=None)
      - ``project``: project-scoped personal override (project='<name>')

    Legacy compatibility:
      - ``scope='generic'`` -> ``scope='global'``, ``profile_kind='static'``
      - ``scope='context'`` -> ``profile_kind='dynamic'`` (``scope='project'`` if
        ``project`` is given, else ``scope='global'``)
    """
    raw_scope = (scope or "").strip().lower()
    clean_project = (project or "").strip() or None

    if profile_kind in ("static", "dynamic"):
        pk = profile_kind
    else:
        pk = "dynamic" if raw_scope == "context" else "static"

    if raw_scope == "project" and clean_project:
        return "project", clean_project, pk
    if raw_scope == "context" and clean_project:
        return "project", clean_project, pk
    if not raw_scope and clean_project:
        return "project", clean_project, pk
    # global / generic / context-without-project / project-without-name -> global
    return "global", None, pk


class PersonalMemoryRepository:
    """Repository for PersonalMemory CRUD and semantic overwrite."""

    def __init__(self, session: AsyncSession):
        self._session = session

    async def list_by_user(
        self,
        user_id: str,
        scope: str | None = None,
        profile_kind: str | None = None,
        project: str | None = None,
    ) -> list[PersonalMemory]:
        q = select(PersonalMemory).where(PersonalMemory.user_id == user_id)
        if scope:
            if scope == "global":
                q = q.where(PersonalMemory.scope.in_(("global", "generic")))
            else:
                q = q.where(PersonalMemory.scope == scope)
        if project:
            q = q.where(PersonalMemory.project == project)
        if profile_kind:
            q = q.where(PersonalMemory.profile_kind == profile_kind)
        q = q.order_by(desc(PersonalMemory.priority), desc(PersonalMemory.updated_at))
        result = await self._session.execute(q)
        return list(result.scalars().all())

    @staticmethod
    def _scope_filter(project: str | None):
        """Return SQLAlchemy filter matching global memories + matching project memories."""
        clean_proj = (project or "").strip() or None
        if clean_proj:
            return or_(
                PersonalMemory.scope.in_(_GLOBAL_COMPAT_SCOPES),
                and_(
                    PersonalMemory.scope == "project",
                    PersonalMemory.project == clean_proj,
                ),
            )
        return PersonalMemory.scope.in_(_GLOBAL_COMPAT_SCOPES)

    async def list_for_pull(
        self,
        user_id: str,
        context_embedding: list[float] | None = None,
        context_similarity_threshold: float = 0.5,
        project: str | None = None,
    ) -> list[PersonalMemory]:
        """Pull global + matching-project memories (static + context-matched dynamic)."""
        scope_cond = self._scope_filter(project)
        if not context_embedding:
            q_static = (
                select(PersonalMemory)
                .where(PersonalMemory.user_id == user_id)
                .where(scope_cond)
                .where(PersonalMemory.profile_kind == "static")
                .order_by(desc(PersonalMemory.priority), desc(PersonalMemory.updated_at))
            )
            q_dynamic = (
                select(PersonalMemory)
                .where(PersonalMemory.user_id == user_id)
                .where(scope_cond)
                .where(PersonalMemory.profile_kind == "dynamic")
                .order_by(desc(PersonalMemory.priority), desc(PersonalMemory.updated_at))
            )
            static_rows = list((await self._session.execute(q_static)).scalars().all())
            dynamic_rows = list((await self._session.execute(q_dynamic)).scalars().all())
            return static_rows + dynamic_rows

        similarity_expr = (1 - PersonalMemory.embedding.cosine_distance(context_embedding)).label(
            "similarity"
        )
        q_static = (
            select(PersonalMemory, similarity_expr)
            .where(PersonalMemory.user_id == user_id)
            .where(scope_cond)
            .where(PersonalMemory.profile_kind == "static")
            .order_by(desc(PersonalMemory.priority), desc(PersonalMemory.updated_at))
        )
        res_static = await self._session.execute(q_static)
        static_rows: list[PersonalMemory] = []
        for mem, sim in res_static.all():
            mem._similarity = float(sim) if sim is not None else 0.0
            static_rows.append(mem)

        q_dynamic = (
            select(PersonalMemory, similarity_expr)
            .where(PersonalMemory.user_id == user_id)
            .where(scope_cond)
            .where(PersonalMemory.profile_kind == "dynamic")
            .where(PersonalMemory.embedding.is_not(None))
            .where(similarity_expr >= context_similarity_threshold)
            .order_by(desc(similarity_expr), desc(PersonalMemory.updated_at))
        )
        res_dynamic = await self._session.execute(q_dynamic)
        dynamic_items: list[PersonalMemory] = []
        for mem, sim in res_dynamic.all():
            mem._similarity = float(sim) if sim is not None else 0.0
            dynamic_items.append(mem)
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
        *,
        scope: str | None = None,
        project: str | None = None,
    ) -> PersonalMemory | None:
        if not embedding:
            return None
        similarity_expr = (1 - PersonalMemory.embedding.cosine_distance(embedding)).label("similarity")
        q = (
            select(PersonalMemory, similarity_expr)
            .where(PersonalMemory.user_id == user_id)
            .where(PersonalMemory.profile_kind == profile_kind)
            .where(PersonalMemory.embedding.is_not(None))
            .where(similarity_expr >= threshold)
        )
        if scope == "project" and project:
            q = q.where(PersonalMemory.scope == "project", PersonalMemory.project == project)
        elif scope == "global":
            q = q.where(PersonalMemory.scope.in_(("global", "generic")))
        q = q.order_by(desc(similarity_expr)).limit(1)
        result = await self._session.execute(q)
        row = result.first()
        if row:
            return row[0]
        return None

    async def create(
        self,
        user_id: str,
        content: str,
        scope: str = "global",
        context_hint: str | None = None,
        embedding: list[float] | None = None,
        profile_kind: str | None = None,
        persona: str = "general",
        priority: int = 50,
        project: str | None = None,
    ) -> PersonalMemory:
        scope_norm, proj_norm, pk = normalize_scope(
            scope, project=project, profile_kind=profile_kind
        )
        mem = PersonalMemory(
            user_id=user_id,
            content=content,
            scope=scope_norm,
            project=proj_norm,
            profile_kind=pk,
            context_hint=context_hint,
            embedding=embedding,
            persona=persona,
            priority=priority,
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
        persona: str | None = None,
        priority: int | None = None,
        project: str | None = None,
    ) -> PersonalMemory | None:
        mem = await self.get_by_id(memory_id, user_id)
        if mem is None:
            return None
        if content is not None:
            mem.content = content
        if scope is not None or project is not None:
            eff_scope = scope if scope is not None else getattr(mem, "scope", "global")
            eff_proj = project if project is not None else getattr(mem, "project", None)
            eff_pk = profile_kind if profile_kind is not None else getattr(mem, "profile_kind", "static")
            scope_norm, proj_norm, pk_norm = normalize_scope(
                eff_scope, project=eff_proj, profile_kind=eff_pk
            )
            mem.scope = scope_norm
            mem.project = proj_norm
            if profile_kind is not None:
                mem.profile_kind = pk_norm
        elif profile_kind is not None:
            mem.profile_kind = profile_kind
        if context_hint is not None:
            mem.context_hint = context_hint
        if embedding is not None:
            mem.embedding = embedding
        if persona is not None:
            mem.persona = persona
        if priority is not None:
            mem.priority = priority
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
        scope: str = "global",
        context_hint: str | None = None,
        threshold: float = PERSONAL_MEMORY_OVERWRITE_THRESHOLD,
        profile_kind: str | None = None,
        persona: str = "general",
        priority: int = 50,
        project: str | None = None,
    ) -> PersonalMemory:
        """Insert or overwrite by semantic similarity within the same (scope, project, profile_kind)."""
        scope_norm, proj_norm, pk = normalize_scope(
            scope, project=project, profile_kind=profile_kind
        )
        existing = await self.find_most_similar(
            user_id,
            embedding,
            threshold,
            pk,
            scope=scope_norm,
            project=proj_norm,
        )
        if existing:
            prev_pri = int(existing.priority or 50)
            in_pri = int(priority or 50)
            boosted_priority = min(100, max(prev_pri, in_pri) + 10)
            await self.update(
                existing.id,
                user_id,
                content=content,
                scope=scope_norm,
                project=proj_norm,
                context_hint=context_hint,
                embedding=embedding,
                profile_kind=pk,
                persona=persona,
                priority=boosted_priority,
            )
            await self._session.refresh(existing)
            return existing
        return await self.create(
            user_id=user_id,
            content=content,
            scope=scope_norm,
            project=proj_norm,
            context_hint=context_hint,
            embedding=embedding,
            profile_kind=pk,
            persona=persona,
            priority=priority,
        )

    async def count_by_persona(self, user_id: str, persona: str) -> int:
        """Count memories for a (user_id, persona) pair."""
        q = (
            select(func.count())
            .select_from(PersonalMemory)
            .where(PersonalMemory.user_id == user_id)
            .where(PersonalMemory.persona == persona)
        )
        result = await self._session.execute(q)
        return result.scalar_one()

    async def delete_oldest_by_persona(self, user_id: str, persona: str) -> bool:
        """Delete the lowest-priority/oldest memory for a (user_id, persona) pair."""
        q = (
            select(PersonalMemory.id)
            .where(PersonalMemory.user_id == user_id)
            .where(PersonalMemory.persona == persona)
            .order_by(
                case((PersonalMemory.profile_kind == "dynamic", 0), else_=1).asc(),
                PersonalMemory.priority.asc(),
                PersonalMemory.created_at.asc(),
            )
            .limit(1)
        )
        result = await self._session.execute(q)
        oldest_id = result.scalar_one_or_none()
        if oldest_id is None:
            return False
        await self._session.execute(delete(PersonalMemory).where(PersonalMemory.id == oldest_id))
        return True

    async def count_by_project(self, user_id: str, project: str) -> int:
        """Count project-scoped memories for a (user_id, project) pair."""
        q = (
            select(func.count())
            .select_from(PersonalMemory)
            .where(PersonalMemory.user_id == user_id)
            .where(PersonalMemory.scope == "project")
            .where(PersonalMemory.project == project)
        )
        result = await self._session.execute(q)
        return result.scalar_one()

    async def delete_oldest_by_project(self, user_id: str, project: str) -> bool:
        """Delete the lowest-priority/oldest project-scoped memory for (user_id, project)."""
        q = (
            select(PersonalMemory.id)
            .where(PersonalMemory.user_id == user_id)
            .where(PersonalMemory.scope == "project")
            .where(PersonalMemory.project == project)
            .order_by(
                case((PersonalMemory.profile_kind == "dynamic", 0), else_=1).asc(),
                PersonalMemory.priority.asc(),
                PersonalMemory.created_at.asc(),
            )
            .limit(1)
        )
        result = await self._session.execute(q)
        oldest_id = result.scalar_one_or_none()
        if oldest_id is None:
            return False
        await self._session.execute(delete(PersonalMemory).where(PersonalMemory.id == oldest_id))
        return True

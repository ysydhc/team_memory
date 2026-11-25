"""ExperienceService main class composing CRUD, Query, and Lifecycle mixins."""

from __future__ import annotations

import logging
import uuid
from contextlib import asynccontextmanager

from sqlalchemy import select

from team_memory.auth.provider import AuthProvider, User
from team_memory.embedding.base import EmbeddingProvider
from team_memory.services.archive import ArchiveService
from team_memory.services.infra.event_bus import EventBus
from team_memory.services.memory.experience.crud import ExperienceCrudMixin, _experience_repo_cls
from team_memory.services.memory.experience.lifecycle import ExperienceLifecycleMixin
from team_memory.services.memory.experience.queries import ExperienceQueryMixin

logger = logging.getLogger("team_memory.service")


class ExperienceService(ExperienceCrudMixin, ExperienceQueryMixin, ExperienceLifecycleMixin):
    """High-level service for experience write operations.

    Combines Repository (data access), Embedding (vector encoding),
    Auth (user verification), and EventBus into cohesive business
    operations.  Search is handled by SearchOrchestrator.
    """

    def __init__(
        self,
        embedding_provider: EmbeddingProvider,
        auth_provider: AuthProvider,
        event_bus: EventBus | None = None,
        lifecycle_config=None,
        llm_config=None,
        db_url: str = "",
        archive_service: ArchiveService | None = None,
    ):
        self._db_url = db_url
        self._embedding = embedding_provider
        self._auth = auth_provider
        self._event_bus = event_bus
        self._archive_service = archive_service
        self._lifecycle_config = lifecycle_config
        self._llm_config = llm_config

    @asynccontextmanager
    async def _session(self):
        """Create a managed database session."""
        from team_memory.storage.database import get_session

        async with get_session(self._db_url) as session:
            yield session

    @asynccontextmanager
    async def _session_or(self, session_or_none):
        """Use provided session or create new one (for internal reuse)."""
        if session_or_none is not None:
            yield session_or_none
        else:
            async with self._session() as s:
                yield s

    _CHILD_EMBED_TRUNCATE = 300

    def _build_parent_embed_text(
        self,
        title: str,
        description: str,
        solution: str = "",
        *,
        tags: list[str] | None = None,
        children: list | None = None,
    ) -> str:
        """Build text for parent experience embedding; includes children when provided."""
        parts = [title, description, solution or ""]
        text = "\n".join(parts)
        if tags:
            text += f"\n{' '.join(tags)}"
        if not children:
            return text
        for c in children:
            if hasattr(c, "title"):
                c_title = c.title or ""
                c_desc = (c.description or "")[: self._CHILD_EMBED_TRUNCATE]
                c_sol = (c.solution or "")[: self._CHILD_EMBED_TRUNCATE]
            else:
                c_title = c.get("title", "")
                c_desc = (c.get("description") or c.get("problem") or "")[: self._CHILD_EMBED_TRUNCATE]
                c_sol = (c.get("solution") or "")[: self._CHILD_EMBED_TRUNCATE]
            text += f"\n{c_title}\n{c_desc}\n{c_sol}"
        return text

    async def authenticate(self, credentials: dict) -> User | None:
        """Authenticate a user."""
        return await self._auth.authenticate(credentials)

    async def get_by_id(self, experience_id: str) -> dict | None:
        """Get an experience by ID as a dict (or None if not found)."""
        async with self._session() as session:
            repo = _experience_repo_cls()(session)
            try:
                exp_uuid = uuid.UUID(experience_id)
            except (ValueError, TypeError, AttributeError):
                return None
            exp = await repo.get_by_id(exp_uuid)
            if exp is None:
                return None
            return exp.to_dict()

    async def get_experience_detail(self, experience_id: str) -> dict | None:
        """Get experience with feedbacks and children."""
        from sqlalchemy.orm import selectinload

        from team_memory.storage.models import Experience

        async with self._session() as session:
            exp_uuid = uuid.UUID(experience_id)
            result = await session.execute(
                select(Experience)
                .options(
                    selectinload(Experience.feedbacks),
                    selectinload(Experience.children),
                )
                .where(Experience.id == exp_uuid)
                .where(Experience.is_deleted == False)  # noqa: E712
            )
            exp = result.scalar_one_or_none()
            if exp is None:
                return None

            data = exp.to_dict()
            data["feedbacks"] = [fb.to_dict() for fb in exp.feedbacks]
            data["children"] = [c.to_dict() for c in (exp.children or [])]
            return data

    async def hard_delete(self, experience_id: str) -> bool:
        """Permanently delete an experience."""
        async with self._session() as session:
            repo = _experience_repo_cls()(session)
            return await repo.delete(uuid.UUID(experience_id))

    async def find_duplicate_pairs(
        self,
        *,
        threshold: float,
        limit: int,
        project: str | None,
    ) -> dict:
        """Semantic duplicate candidate pairs for the Web dedup UI (PostgreSQL + pgvector)."""
        thr = max(0.05, min(1.0, float(threshold)))
        lim = max(1, min(200, int(limit)))
        repo_cls = _experience_repo_cls()
        proj_norm = repo_cls._project_value(project)
        async with self._session() as session:
            repo = repo_cls(session)
            pairs = await repo.find_duplicates(threshold=thr, limit=lim, project=project)
        return {
            "pairs": pairs,
            "threshold": thr,
            "limit": lim,
            "project": proj_norm,
        }

    def _resolve_project(self, project: str | None) -> str:
        """Resolve project name using the same logic as the web layer."""
        import os

        if project and project.strip():
            return project.strip()
        env_project = (os.environ.get("TEAM_MEMORY_PROJECT") or "").strip()
        if env_project:
            return env_project
        return "default"

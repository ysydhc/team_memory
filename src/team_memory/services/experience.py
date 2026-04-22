"""Experience business logic service — write path.

Orchestrates Repository, Embedding, Auth, and EventBus to provide
high-level write operations (save, update, delete, feedback) for
the MCP tools and Web API.  Search lives in SearchOrchestrator.
"""

from __future__ import annotations

import logging
import time
import uuid
from contextlib import asynccontextmanager

from team_memory import io_logger
from team_memory.auth.provider import AuthProvider, User
from team_memory.embedding.base import EmbeddingProvider
from team_memory.services.archive import ArchiveService
from team_memory.services.event_bus import EventBus, Events
from team_memory.storage.repository import ExperienceRepository

logger = logging.getLogger("team_memory.service")


class ExperienceService:
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
        self._event_bus = event_bus or EventBus()
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
                c_desc = (c.get("description") or c.get("problem") or "")[
                    : self._CHILD_EMBED_TRUNCATE
                ]
                c_sol = (c.get("solution") or "")[: self._CHILD_EMBED_TRUNCATE]
            text += f"\n{c_title}\n{c_desc}\n{c_sol}"
        return text

    async def authenticate(self, credentials: dict) -> User | None:
        """Authenticate a user."""
        return await self._auth.authenticate(credentials)

    async def save(
        self,
        title: str,
        problem: str,
        solution: str | None = None,
        created_by: str = "system",
        tags: list[str] | None = None,
        source: str = "auto_extract",
        exp_status: str = "published",
        visibility: str = "project",
        skip_dedup: bool = False,
        experience_type: str = "general",
        project: str | None = None,
        group_key: str | None = None,
        *,
        session=None,  # noqa: ANN001
    ) -> dict:
        """Save a new experience to the database.

        If group_key is provided, auto-groups under a shared parent.
        If dedup_on_save is enabled, checks for similar experiences first.

        Returns:
            Experience dict. If dedup is triggered, returns a dict with
            "status": "duplicate_detected" and "candidates" list.
        """
        async with self._session_or(session) as session:
            prepare_start = time.monotonic()
            repo = ExperienceRepository(session)

            # Build text for embedding
            embed_text = f"{title}\n{problem}\n{solution or ''}"
            if tags:
                embed_text += f"\n{' '.join(tags)}"

            # Generate embedding synchronously
            embedding = None
            try:
                embedding = await self._embedding.encode_single(embed_text)
            except Exception as e:
                logger.warning("Failed to generate embedding: %s", e)

            if embedding is None:
                return {
                    "error": True,
                    "message": "Embedding generation failed. Save aborted.",
                }

            # Dedup-on-save check
            if (
                not skip_dedup
                and embedding is not None
                and self._lifecycle_config is not None
                and self._lifecycle_config.dedup_on_save
            ):
                try:
                    candidates = await repo.check_similar(
                        embedding=embedding,
                        threshold=self._lifecycle_config.dedup_on_save_threshold,
                        limit=5,
                        project=project,
                    )
                    if candidates:
                        return {
                            "status": "duplicate_detected",
                            "candidates": candidates,
                            "message": f"发现 {len(candidates)} 条相似经验，请确认是否仍要保存",
                            "experience_id": None,
                        }
                except Exception:
                    logger.warning("Dedup-on-save check failed, proceeding with save")

            prepare_duration_ms = int((time.monotonic() - prepare_start) * 1000)
            io_logger.log_internal(
                "save_prepare",
                {"title": (title or "")[:50]},
                duration_ms=prepare_duration_ms,
            )

            # Auto-group via group_key
            parent_id = None
            if group_key:
                parent = await repo.find_or_create_group_parent(
                    session=session,
                    project=project or "default",
                    group_key=group_key,
                    created_by=created_by,
                )
                parent_id = parent.id

            persist_start = time.monotonic()
            experience = await repo.create(
                title=title,
                description=problem,
                solution=solution,
                created_by=created_by,
                tags=tags,
                embedding=embedding,
                source=source,
                exp_status=exp_status,
                visibility=visibility,
                experience_type=experience_type,
                project=project or "default",
                group_key=group_key,
                parent_id=parent_id,
            )

            await self._event_bus.emit(
                Events.EXPERIENCE_CREATED,
                {
                    "experience_id": str(experience.id),
                    "title": title,
                    "created_by": created_by,
                    "status": exp_status,
                    "visibility": visibility,
                },
            )
            persist_duration_ms = int((time.monotonic() - persist_start) * 1000)
            io_logger.log_internal(
                "save_persist",
                {
                    "experience_id": str(experience.id),
                    "title": (title or "")[:50],
                },
                duration_ms=persist_duration_ms,
            )

            # Re-embed parent if this child was added to a group
            if parent_id:
                try:
                    parent_exp = await repo.get_with_children(parent_id)
                    if parent_exp:
                        parent_embed_text = self._build_parent_embed_text(
                            parent_exp.title,
                            parent_exp.description,
                            parent_exp.solution or "",
                            tags=parent_exp.tags,
                            children=parent_exp.children,
                        )
                        parent_embedding = await self._embedding.encode_single(parent_embed_text)
                        await repo.update(parent_id, embedding=parent_embedding)
                except Exception:
                    logger.warning("Failed to re-embed group parent %s", parent_id)

            return experience.to_dict()

    async def update(
        self,
        experience_id: str,
        user: str = "system",
        *,
        session=None,  # noqa: ANN001
        **kwargs,
    ) -> dict | None:
        """Update an existing experience (in-place).

        Only kwargs that are explicitly provided will be applied.
        Supported kwargs:
            title, description, solution, tags, exp_status, visibility,
            experience_type, solution_addendum (legacy append mode).
        """
        async with self._session_or(session) as session:
            repo = ExperienceRepository(session)
            exp_uuid = uuid.UUID(experience_id)

            experience = await repo.get_by_id(exp_uuid)
            if experience is None:
                return None

            updates: dict = {}
            need_reembed = False

            if "title" in kwargs:
                updates["title"] = kwargs["title"]
                need_reembed = True

            if "description" in kwargs:
                updates["description"] = kwargs["description"]
                need_reembed = True

            solution_addendum = kwargs.get("solution_addendum")
            if "solution" in kwargs:
                updates["solution"] = kwargs["solution"]
                need_reembed = True
            elif solution_addendum:
                updates["solution"] = f"{experience.solution or ''}\n\n---\n\n{solution_addendum}"
                need_reembed = True

            if "tags" in kwargs:
                updates["tags"] = kwargs["tags"]

            if "exp_status" in kwargs:
                updates["exp_status"] = kwargs["exp_status"]
            if "visibility" in kwargs:
                updates["visibility"] = kwargs["visibility"]
            if "experience_type" in kwargs:
                updates["experience_type"] = kwargs["experience_type"]

            if not updates:
                return experience.to_dict()

            # Re-generate embedding if content fields changed
            if need_reembed:
                final_title = updates.get("title", experience.title)
                final_desc = updates.get("description", experience.description)
                final_sol = updates.get("solution", experience.solution) or ""
                final_tags = updates.get("tags", experience.tags)
                children = await repo.get_children(exp_uuid)
                embed_text = self._build_parent_embed_text(
                    final_title,
                    final_desc,
                    final_sol,
                    tags=final_tags,
                    children=children if children else None,
                )
                try:
                    embedding = await self._embedding.encode_single(embed_text)
                    updates["embedding"] = embedding
                except Exception:
                    logger.warning("Failed to regenerate embedding during update")

            updated = await repo.update(exp_uuid, **updates)

            if updated and updates:
                fields_updated = list(updates.keys())
                await self._event_bus.emit(
                    Events.EXPERIENCE_UPDATED,
                    {
                        "experience_id": experience_id,
                        "fields_updated": fields_updated,
                        "user": user,
                    },
                )

                # Update archive status if exp_status or visibility changed
                if self._archive_service and ("exp_status" in updates or "visibility" in updates):
                    try:
                        await self._archive_service.update_archive_status_for_experience(exp_uuid)
                    except Exception:
                        logger.warning("Failed to update archive status", exc_info=True)

            return updated.to_dict() if updated else None

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
        proj_norm = ExperienceRepository._project_value(project)
        async with self._session() as session:
            repo = ExperienceRepository(session)
            pairs = await repo.find_duplicates(threshold=thr, limit=lim, project=project)
        return {
            "pairs": pairs,
            "threshold": thr,
            "limit": lim,
            "project": proj_norm,
        }

    _REEMBED_BATCH_SIZE = 50

    async def reembed_group_parent_vectors(self, project: str | None) -> dict:
        """Recompute embeddings for parents that have children (group aggregate text).

        Collects all parent texts first, then batch-encodes in chunks of
        _REEMBED_BATCH_SIZE to reduce round-trips to the embedding provider.
        """
        proj_norm = ExperienceRepository._project_value(project)
        updated = 0
        errors = 0
        total_groups = 0
        async with self._session() as session:
            repo = ExperienceRepository(session)
            root_ids = await repo.list_root_ids_with_children(project)
            total_groups = len(root_ids)

            # First pass: collect parent texts keyed by root id
            parent_texts: list[tuple[uuid.UUID, str]] = []
            for rid in root_ids:
                parent = await repo.get_with_children(rid)
                if not parent:
                    continue
                children = parent.children or []
                if not children:
                    continue
                embed_text = self._build_parent_embed_text(
                    parent.title,
                    parent.description,
                    parent.solution or "",
                    tags=parent.tags,
                    children=children,
                )
                parent_texts.append((rid, embed_text))

            # Batch encode in chunks of _REEMBED_BATCH_SIZE
            batch_size = self._REEMBED_BATCH_SIZE
            for i in range(0, len(parent_texts), batch_size):
                batch = parent_texts[i : i + batch_size]
                texts = [text for _, text in batch]
                try:
                    embeddings = await self._embedding.encode(texts)
                except Exception:
                    logger.warning(
                        "reembed_group_parent batch encode failed for batch starting at %d",
                        i,
                        exc_info=True,
                    )
                    errors += len(batch)
                    continue

                # Second pass: update embeddings in DB
                for (rid, _), embedding in zip(batch, embeddings):
                    try:
                        await repo.update(rid, embedding=embedding)
                        updated += 1
                    except Exception:
                        logger.warning(
                            "reembed_group_parent DB update failed for %s",
                            rid,
                            exc_info=True,
                        )
                        errors += 1

        await self._event_bus.emit(
            Events.EXPERIENCE_UPDATED,
            {
                "experience_id": "batch_reembed",
                "fields_updated": ["embedding"],
                "user": "system",
            },
        )
        return {
            "updated": updated,
            "errors": errors,
            "total_groups": total_groups,
            "project": proj_norm,
        }

    async def soft_delete(self, experience_id: str) -> bool:
        """Soft-delete an experience."""
        async with self._session() as session:
            repo = ExperienceRepository(session)
            result = await repo.soft_delete(uuid.UUID(experience_id))
            if result:
                await self._event_bus.emit(
                    Events.EXPERIENCE_DELETED,
                    {
                        "experience_id": experience_id,
                        "hard": False,
                    },
                )
            return result

    async def restore(self, experience_id: str) -> bool:
        """Restore a soft-deleted experience."""
        async with self._session() as session:
            repo = ExperienceRepository(session)
            result = await repo.restore(uuid.UUID(experience_id))
            if result:
                await self._event_bus.emit(
                    Events.EXPERIENCE_RESTORED,
                    {
                        "experience_id": experience_id,
                    },
                )
            return result

    async def feedback(
        self,
        experience_id: str,
        rating: int,
        feedback_by: str,
        comment: str | None = None,
        fitness_score: int | None = None,
        *,
        session=None,  # noqa: ANN001
        **kwargs: object,
    ) -> bool:
        """Submit feedback for an experience. rating: 1-5, 5=best."""
        async with self._session_or(session) as session:
            if not (1 <= rating <= 5):
                raise ValueError("rating must be between 1 and 5")
            repo = ExperienceRepository(session)
            exp_uuid = uuid.UUID(experience_id)

            experience = await repo.get_by_id(exp_uuid)
            if experience is None:
                return False

            await repo.add_feedback(
                experience_id=exp_uuid,
                rating=rating,
                feedback_by=feedback_by,
                comment=comment,
                fitness_score=fitness_score,
            )

            # Add quality score bonus for positive feedback (rating >= 4)
            if rating >= 4:
                await repo.increment_quality_score(exp_uuid, 1.0)

            await self._event_bus.emit(
                Events.FEEDBACK_ADDED,
                {
                    "experience_id": experience_id,
                    "rating": rating,
                    "feedback_by": feedback_by,
                },
            )
            return True

    async def list_projects(self) -> list[str]:
        """Return distinct project names."""
        async with self._session() as session:
            repo = ExperienceRepository(session)
            return await repo.list_projects()

    async def list_experiences(
        self,
        *,
        page: int = 1,
        page_size: int = 20,
        status: str | None = None,
        tag: str | None = None,
        project: str | None = None,
    ) -> dict:
        """Paginated list with optional filters."""
        async with self._session() as session:
            repo = ExperienceRepository(session)
            offset = (page - 1) * page_size
            exps = await repo.list_experiences(
                limit=page_size,
                offset=offset,
                status=status,
                tag=tag,
                project=project,
            )
            total = await repo.count_experiences(status=status, tag=tag, project=project)
            return {
                "experiences": exps,
                "total": total,
                "page": page,
                "page_size": page_size,
            }

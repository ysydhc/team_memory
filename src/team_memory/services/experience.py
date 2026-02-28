"""Experience business logic service.

Orchestrates Repository, Embedding, Auth, and SearchPipeline to provide
high-level operations for the MCP tools and Web API.
"""

from __future__ import annotations

import csv
import io
import json
import logging
import time
import uuid
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from typing import TYPE_CHECKING

from team_memory.auth.provider import AuthProvider, User
from team_memory.embedding.base import EmbeddingProvider
from team_memory.services.event_bus import EventBus, Events
from team_memory.services.pageindex_lite import PageIndexLiteBuilder
from team_memory.storage.repository import ExperienceRepository

if TYPE_CHECKING:
    from team_memory.services.embedding_queue import EmbeddingQueue

logger = logging.getLogger("team_memory.service")


class ExperienceService:
    """High-level service for experience operations.

    Combines Repository (data access), Embedding (vector encoding),
    Auth (user verification), SearchPipeline, and EventBus into cohesive
    business operations.
    """

    def __init__(
        self,
        embedding_provider: EmbeddingProvider,
        auth_provider: AuthProvider,
        search_pipeline=None,
        event_bus: EventBus | None = None,
        embedding_queue: EmbeddingQueue | None = None,
        lifecycle_config=None,
        review_config=None,
        memory_config=None,
        llm_config=None,
        pageindex_lite_config=None,
        db_url: str = "",
    ):
        self._db_url = db_url
        self._embedding = embedding_provider
        self._auth = auth_provider
        self._search_pipeline = search_pipeline
        self._event_bus = event_bus or EventBus()
        self._embedding_queue = embedding_queue
        self._lifecycle_config = lifecycle_config
        self._review_config = review_config
        self._memory_config = memory_config
        self._llm_config = llm_config
        self._pageindex_lite_config = pageindex_lite_config
        self._pageindex_builder = PageIndexLiteBuilder(
            min_doc_chars=getattr(pageindex_lite_config, "min_doc_chars", 800),
            max_tree_depth=getattr(pageindex_lite_config, "max_tree_depth", 4),
            max_nodes_per_doc=getattr(pageindex_lite_config, "max_nodes_per_doc", 40),
            max_node_chars=getattr(pageindex_lite_config, "max_node_chars", 1200),
        )

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

    async def authenticate(self, credentials: dict) -> User | None:
        """Authenticate a user."""
        return await self._auth.authenticate(credentials)

    async def search(
        self,
        query: str,
        tags: list[str] | None = None,
        max_results: int = 5,
        min_similarity: float = 0.6,
        user_name: str = "anonymous",
        source: str = "mcp",
        grouped: bool = False,
        top_k_children: int = 3,
        min_avg_rating: float = 0.0,
        rating_weight: float = 0.3,
        use_pageindex_lite: bool | None = None,
        project: str | None = None,
    ) -> list[dict]:
        async with self._session() as session:
            """Search experiences using the enhanced search pipeline.

            If a SearchPipeline is configured, uses the full pipeline
            (hybrid search + RRF fusion + adaptive filter + reranker + compression).
            Otherwise, falls back to the legacy vector/FTS search.

            Args:
                grouped: If True, return results grouped by root experience.
                top_k_children: Max children per group (only when grouped=True).
            """
            # Use enhanced pipeline if available
            if self._search_pipeline is not None:
                from team_memory.services.search_pipeline import SearchRequest

                request = SearchRequest(
                    query=query,
                    max_results=max_results,
                    min_similarity=min_similarity,
                    tags=tags,
                    user_name=user_name,
                    current_user=user_name,
                    source=source,
                    grouped=grouped,
                    top_k_children=top_k_children,
                    min_avg_rating=min_avg_rating,
                    rating_weight=rating_weight,
                    use_pageindex_lite=use_pageindex_lite,
                    project=project,
                )
                pipeline_result = await self._search_pipeline.search(
                    session, request
                )

                # Log the query
                repo = ExperienceRepository(session)
                try:
                    await repo.log_query(
                        query=query,
                        user_name=user_name,
                        source=source,
                        result_count=len(pipeline_result.results),
                        search_type=pipeline_result.search_type,
                        duration_ms=pipeline_result.duration_ms,
                    )
                except Exception:
                    logger.warning("Failed to log query", exc_info=True)

                return pipeline_result.results

            # Legacy fallback: direct vector/FTS search (reuse session)
            return await self._legacy_search(
                session,
                query=query,
                tags=tags,
                max_results=max_results,
                min_similarity=min_similarity,
                user_name=user_name,
                source=source,
                grouped=grouped,
                top_k_children=top_k_children,
                min_avg_rating=min_avg_rating,
                rating_weight=rating_weight,
                project=project,
            )

    async def _legacy_search(
        self,
        session,  # noqa: ANN001 - passed from search() which holds the session
        query: str,
        tags: list[str] | None = None,
        max_results: int = 5,
        min_similarity: float = 0.6,
        user_name: str = "anonymous",
        source: str = "mcp",
        grouped: bool = False,
        top_k_children: int = 3,
        min_avg_rating: float = 0.0,
        rating_weight: float = 0.3,
        project: str | None = None,
    ) -> list[dict]:
        """Legacy search (vector with FTS fallback, no pipeline)."""
        repo = ExperienceRepository(session)
        start = time.monotonic()
        search_type = "vector"

        try:
            query_embedding = await self._embedding.encode_single(query)
            if grouped:
                results = await repo.search_by_vector_grouped(
                    query_embedding=query_embedding,
                    max_results=max_results,
                    min_similarity=min_similarity,
                    tags=tags,
                    top_k_children=top_k_children,
                    min_avg_rating=min_avg_rating,
                    rating_weight=rating_weight,
                    project=project,
                    current_user=user_name,
                )
            else:
                results = await repo.search_by_vector(
                    query_embedding=query_embedding,
                    max_results=max_results,
                    min_similarity=min_similarity,
                    tags=tags,
                    project=project,
                    current_user=user_name,
                )
        except Exception as e:
            logger.warning(
                "Vector search failed, falling back to FTS: %s", str(e)
            )
            search_type = "fts"
            results = await repo.search_by_fts(
                query_text=query,
                max_results=max_results,
                tags=tags,
                project=project,
                current_user=user_name,
            )

        duration_ms = int((time.monotonic() - start) * 1000)

        try:
            await repo.log_query(
                query=query,
                user_name=user_name,
                source=source,
                result_count=len(results),
                search_type=search_type,
                duration_ms=duration_ms,
            )
        except Exception:
            logger.warning("Failed to log query", exc_info=True)

        return results

    async def invalidate_search_cache(self):
        """Invalidate search cache after data mutations."""
        if self._search_pipeline is not None:
            await self._search_pipeline.invalidate_cache()

    async def _maybe_build_tree_nodes(
        self,
        repo: ExperienceRepository,
        experience_id: uuid.UUID,
        *,
        problem: str,
        solution: str | None = None,
        root_cause: str | None = None,
        code_snippets: str | None = None,
    ) -> int:
        """Build and persist PageIndex-Lite tree nodes for one experience."""
        cfg = self._pageindex_lite_config
        if cfg is None or not getattr(cfg, "enabled", False):
            return 0

        content = self._pageindex_builder.build_experience_document(
            problem=problem,
            solution=solution,
            root_cause=root_cause,
            code_snippets=code_snippets,
        )
        if getattr(cfg, "only_long_docs", True) and not self._pageindex_builder.is_long_document(
            content
        ):
            return 0

        nodes = self._pageindex_builder.build_nodes(content)
        if not nodes:
            return 0
        return await repo.replace_tree_nodes(experience_id, nodes)

    async def save_group(
        self,
        parent: dict,
        children: list[dict],
        created_by: str,
        project: str | None = None,
        *,
        session=None,  # noqa: ANN001 - for internal reuse (e.g. hard_delete_and_rebuild)
    ) -> dict:
        async with self._session_or(session) as session:
            """Save a parent experience with child experiences as a group.

            Args:
                parent: Fields for the parent experience (title, problem, solution, tags, etc.)
                children: List of field dicts for child experiences.
                created_by: Author name.

            Returns:
                Parent experience dict with children included.
            """
            repo = ExperienceRepository(session)

            # Generate embedding for parent
            parent_embed_text = "\n".join([
                parent.get("title", ""),
                parent.get("problem", ""),
                parent.get("solution", ""),
            ])
            if parent.get("root_cause"):
                parent_embed_text += f"\n{parent['root_cause']}"
            try:
                parent_embedding = await self._embedding.encode_single(parent_embed_text)
            except Exception:
                logger.warning("Failed to generate parent embedding")
                parent_embedding = None

            parent_data = {
                "title": parent["title"],
                "description": parent.get("problem", ""),
                "solution": parent.get("solution", ""),
                "tags": parent.get("tags"),
                "programming_language": parent.get("language"),
                "framework": parent.get("framework"),
                "code_snippets": parent.get("code_snippets"),
                "root_cause": parent.get("root_cause"),
                "embedding": parent_embedding,
                "source": parent.get("source", "manual"),
                "project": project or parent.get("project"),
            }

            # Generate embeddings for children
            children_data = []
            for child in children:
                child_embed_text = "\n".join([
                    child.get("title", ""),
                    child.get("problem", ""),
                    child.get("solution", ""),
                ])
                if child.get("root_cause"):
                    child_embed_text += f"\n{child['root_cause']}"
                try:
                    child_embedding = await self._embedding.encode_single(child_embed_text)
                except Exception:
                    logger.warning("Failed to generate child embedding")
                    child_embedding = None

                children_data.append({
                    "title": child.get("title", ""),
                    "description": child.get("problem", ""),
                    "solution": child.get("solution", ""),
                    "tags": child.get("tags"),
                    "programming_language": child.get("language"),
                    "framework": child.get("framework"),
                    "code_snippets": child.get("code_snippets"),
                    "root_cause": child.get("root_cause"),
                    "embedding": child_embedding,
                    "source": child.get("source", "manual"),
                    "project": project or child.get("project"),
                })

            parent_exp = await repo.create_group(
                parent_data=parent_data,
                children_data=children_data,
                created_by=created_by,
            )

            # Build tree nodes for long-document groups (parent + children)
            try:
                await self._maybe_build_tree_nodes(
                    repo,
                    parent_exp.id,
                    problem=parent.get("problem", ""),
                    solution=parent.get("solution"),
                    root_cause=parent.get("root_cause"),
                    code_snippets=parent.get("code_snippets"),
                )
                for child_exp, child_payload in zip(
                    parent_exp.children or [], children, strict=False
                ):
                    await self._maybe_build_tree_nodes(
                        repo,
                        child_exp.id,
                        problem=child_payload.get("problem", ""),
                        solution=child_payload.get("solution"),
                        root_cause=child_payload.get("root_cause"),
                        code_snippets=child_payload.get("code_snippets"),
                    )
            except Exception:
                logger.warning("Failed to build PageIndex-Lite nodes for group", exc_info=True)

            await self._event_bus.emit(Events.EXPERIENCE_CREATED, {
                "experience_id": str(parent_exp.id),
                "title": parent["title"],
                "created_by": created_by,
                "is_group": True,
                "children_count": len(children),
            })
            return parent_exp.to_dict(include_children=True)

    async def save(
        self,
        title: str,
        problem: str,
        solution: str | None = None,
        created_by: str = "system",
        tags: list[str] | None = None,
        code_snippets: str | None = None,
        language: str | None = None,
        framework: str | None = None,
        source: str = "auto_extract",
        root_cause: str | None = None,
        publish_status: str = "personal",
        review_status: str = "approved",
        sync_embedding: bool = True,
        skip_dedup: bool = False,
        # Type system fields (v3)
        experience_type: str = "general",
        severity: str | None = None,
        category: str | None = None,
        progress_status: str | None = None,
        structured_data: dict | None = None,
        git_refs: list | None = None,
        related_links: list | None = None,
        project: str | None = None,
        quality_score: int = 0,
        *,
        session=None,  # noqa: ANN001 - for internal reuse (e.g. hard_delete_and_rebuild)
    ) -> dict:
        async with self._session_or(session) as session:
            """Save a new experience to the database.

            Generates embedding from the combined title + problem + solution text.

            Args:
                sync_embedding: If True (default), generate embedding synchronously.
                    If False and an embedding queue is available, save with
                    embedding_status='pending' and generate in background.
                skip_dedup: If True, skip dedup-on-save check.

            Returns:
                Experience dict. If dedup is triggered, returns a dict with
                "status": "duplicate_detected" and "candidates" list.
            """
            from team_memory.schemas import (
                get_schema_registry,
                validate_git_refs,
                validate_related_links,
                validate_structured_data,
            )

            repo = ExperienceRepository(session)

            # Validate JSONB fields
            structured_data = validate_structured_data(experience_type, structured_data)
            git_refs = validate_git_refs(git_refs)
            related_links = validate_related_links(related_links)

            # Auto-infer progress_status via SchemaRegistry
            registry = get_schema_registry()
            if progress_status is None and registry.get_progress_states(experience_type):
                progress_status = registry.get_default_progress(
                    experience_type, has_solution=bool(solution)
                )

            # Build text for embedding
            embed_text = f"{title}\n{problem}\n{solution or ''}"
            if root_cause:
                embed_text += f"\n{root_cause}"
            if code_snippets:
                embed_text += f"\n{code_snippets}"

            # Decide: sync or async embedding
            embedding = None
            embedding_status = "ready"

            if not sync_embedding and self._embedding_queue is not None:
                # Async mode: save without embedding, generate in background
                embedding_status = "pending"
            else:
                # Sync mode (default): generate embedding before saving
                try:
                    embedding = await self._embedding.encode_single(embed_text)
                except Exception:
                    logger.warning("Failed to generate embedding, saving without it")

            # P0-3: Dedup-on-save check
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

            # P0-2: Auto-set draft + pending_review for AI-created experiences
            # Skip override when caller explicitly requests "personal" — personal
            # experiences are private to the creator and don't need team review.
            if (
                self._review_config is not None
                and self._review_config.require_review_for_ai
                and source in ("auto_extract", "mcp")
                and publish_status != "personal"
            ):
                publish_status = "draft"
                review_status = "pending"

            experience = await repo.create(
                title=title,
                description=problem,
                solution=solution,
                created_by=created_by,
                tags=tags,
                programming_language=language,
                framework=framework,
                code_snippets=code_snippets,
                embedding=embedding,
                source=source,
                root_cause=root_cause,
                publish_status=publish_status,
                review_status=review_status,
                embedding_status=embedding_status,
                experience_type=experience_type,
                severity=severity,
                category=category,
                progress_status=progress_status,
                structured_data=structured_data,
                git_refs=git_refs,
                related_links=related_links,
                project=project or "default",
                quality_score=quality_score,
            )

            # Build tree nodes for long-document experience
            try:
                await self._maybe_build_tree_nodes(
                    repo,
                    experience.id,
                    problem=problem,
                    solution=solution,
                    root_cause=root_cause,
                    code_snippets=code_snippets,
                )
            except Exception:
                logger.warning(
                    "Failed to build PageIndex-Lite nodes for %s",
                    experience.id,
                    exc_info=True,
                )

            # Enqueue background embedding task if async mode
            if embedding_status == "pending" and self._embedding_queue is not None:
                try:
                    await self._embedding_queue.enqueue(experience.id, embed_text)
                except Exception:
                    logger.warning("Failed to enqueue embedding task for %s", experience.id)

            await self._event_bus.emit(Events.EXPERIENCE_CREATED, {
                "experience_id": str(experience.id),
                "title": title,
                "created_by": created_by,
                "embedding_status": embedding_status,
                "publish_status": publish_status,
            })
            return experience.to_dict()

    async def publish_experience(
        self,
        experience_id: str,
        user: str = "system",
    ) -> dict | None:
        async with self._session() as session:
            """Publish a draft experience (set publish_status='published').

            Also sets review_status='approved' if not already.
            """
            repo = ExperienceRepository(session)
            exp_uuid = uuid.UUID(experience_id)
            experience = await repo.get_by_id(exp_uuid, include_deleted=True)
            if experience is None:
                return None

            if experience.publish_status == "published":
                return experience.to_dict()

            updates = {
                "publish_status": "published",
            }
            if experience.review_status != "approved":
                updates["review_status"] = "approved"
                updates["reviewed_by"] = user

            updated = await repo.update(exp_uuid, **updates)
            if updated:
                await self._event_bus.emit(Events.EXPERIENCE_PUBLISHED, {
                    "experience_id": experience_id,
                    "published_by": user,
                })
            return updated.to_dict() if updated else None

    async def publish_to_team(
        self,
        experience_id: uuid.UUID,
        user: str,
        is_admin: bool = False,
    ) -> dict | None:
        """Publish a personal experience to the team.

        Admin users go directly to 'published'; others go to 'pending_team'.
        """
        async with self._session() as session:
            repo = ExperienceRepository(session)
            experience = await repo.publish_to_team(experience_id, is_admin=is_admin)
            return experience.to_dict() if experience else None

    async def publish_personal(
        self,
        experience_id: uuid.UUID,
    ) -> dict | None:
        """Move a draft experience to personal (visible to creator only)."""
        async with self._session() as session:
            repo = ExperienceRepository(session)
            experience = await repo.publish_personal(experience_id)
            return experience.to_dict() if experience else None

    async def get_drafts(
        self,
        created_by: str | None = None,
        limit: int = 50,
        offset: int = 0,
        project: str | None = None,
    ) -> list[dict]:
        async with self._session() as session:
            """Get draft experiences, optionally filtered by creator."""
            repo = ExperienceRepository(session)
            drafts = await repo.list_drafts(
                created_by=created_by, limit=limit, offset=offset, project=project
            )
            return [exp.to_dict() for exp in drafts]

    async def feedback(
        self,
        experience_id: str,
        rating: int,
        feedback_by: str,
        comment: str | None = None,
        fitness_score: int | None = None,
        *,
        session=None,  # noqa: ANN001
    ) -> bool:
        async with self._session_or(session) as session:
            """Submit feedback for an experience. rating: 1-5, 5=best."""
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
            await self._event_bus.emit(Events.FEEDBACK_ADDED, {
                "experience_id": experience_id,
                "rating": rating,
                "feedback_by": feedback_by,
            })
            return True

    async def update(
        self,
        experience_id: str,
        user: str = "system",
        *,
        session=None,  # noqa: ANN001
        **kwargs,
    ) -> dict | None:
        async with self._session_or(session) as session:
            """Update an existing experience (in-place).

            Only kwargs that are explicitly provided will be applied.
            The caller (web API) should use model_dump(exclude_unset=True)
            to only include fields the user actually sent.

            Supported kwargs:
                title, description, solution, root_cause, code_snippets,
                programming_language/language, framework, tags, publish_status,
                experience_type, severity, category, progress_status,
                structured_data, git_refs, related_links,
                solution_addendum (legacy append mode).
            """
            from team_memory.schemas import (
                get_schema_registry,
                validate_git_refs,
                validate_related_links,
                validate_structured_data,
            )

            repo = ExperienceRepository(session)
            exp_uuid = uuid.UUID(experience_id)

            experience = await repo.get_by_id(exp_uuid)
            if experience is None:
                return None

            # Save version snapshot before editing
            try:
                await repo.save_version(
                    experience_id=exp_uuid,
                    changed_by=user,
                    change_summary="编辑前快照 (in-place update)",
                )
            except Exception as e:
                logger.warning("Failed to save version before update: %s", e)

            updates: dict = {}
            need_reembed = False

            # --- Simple text fields ---
            if "title" in kwargs:
                updates["title"] = kwargs["title"]
                need_reembed = True

            if "description" in kwargs:
                updates["description"] = kwargs["description"]
                need_reembed = True

            # Handle solution: direct set or legacy append
            solution_addendum = kwargs.get("solution_addendum")
            if "solution" in kwargs:
                updates["solution"] = kwargs["solution"]
                need_reembed = True
            elif solution_addendum:
                updates["solution"] = f"{experience.solution or ''}\n\n---\n\n{solution_addendum}"
                need_reembed = True

            if "root_cause" in kwargs:
                updates["root_cause"] = kwargs["root_cause"]
                need_reembed = True
            if "code_snippets" in kwargs:
                updates["code_snippets"] = kwargs["code_snippets"]
            # Support both "language" and "programming_language"
            if "language" in kwargs:
                updates["programming_language"] = kwargs["language"]
            elif "programming_language" in kwargs:
                updates["programming_language"] = kwargs["programming_language"]
            if "framework" in kwargs:
                updates["framework"] = kwargs["framework"]

            if "tags" in kwargs:
                updates["tags"] = kwargs["tags"]

            if "publish_status" in kwargs:
                updates["publish_status"] = kwargs["publish_status"]

            # --- Type system fields ---
            etype = kwargs.get("experience_type") or experience.experience_type or "general"
            if "experience_type" in kwargs:
                updates["experience_type"] = kwargs["experience_type"]

            if "severity" in kwargs:
                updates["severity"] = kwargs["severity"]
            if "category" in kwargs:
                updates["category"] = kwargs["category"]

            # Validate & set structured_data
            if "structured_data" in kwargs:
                sd = kwargs["structured_data"]
                if sd is not None:
                    sd = validate_structured_data(etype, sd)
                updates["structured_data"] = sd

            # Validate & set git_refs
            if "git_refs" in kwargs:
                gr = kwargs["git_refs"]
                if gr is not None:
                    gr = validate_git_refs(gr)
                updates["git_refs"] = gr

            # Validate & set related_links
            if "related_links" in kwargs:
                rl = kwargs["related_links"]
                if rl is not None:
                    rl = validate_related_links(rl)
                updates["related_links"] = rl

            # Auto-infer progress_status if not explicitly provided
            registry = get_schema_registry()
            if "progress_status" in kwargs:
                updates["progress_status"] = kwargs["progress_status"]
            elif registry.get_progress_states(etype) and experience.progress_status is None:
                final_solution = updates.get("solution", experience.solution)
                updates["progress_status"] = registry.get_default_progress(
                    etype, has_solution=bool(final_solution)
                )

            if not updates:
                return experience.to_dict()

            # Re-generate embedding if content fields changed
            if need_reembed:
                final_title = updates.get("title", experience.title)
                final_desc = updates.get("description", experience.description)
                final_sol = updates.get("solution", experience.solution) or ""
                embed_text = f"{final_title}\n{final_desc}\n{final_sol}"
                final_rc = updates.get("root_cause", experience.root_cause)
                if final_rc:
                    embed_text += f"\n{final_rc}"
                final_cs = updates.get("code_snippets", experience.code_snippets)
                if final_cs:
                    embed_text += f"\n{final_cs}"
                try:
                    embedding = await self._embedding.encode_single(embed_text)
                    updates["embedding"] = embedding
                except Exception:
                    logger.warning("Failed to regenerate embedding during update")

            updated = await repo.update(exp_uuid, **updates)
            if updated:
                try:
                    await self._maybe_build_tree_nodes(
                        repo,
                        updated.id,
                        problem=updates.get("description", updated.description),
                        solution=updates.get("solution", updated.solution),
                        root_cause=updates.get("root_cause", updated.root_cause),
                        code_snippets=updates.get("code_snippets", updated.code_snippets),
                    )
                except Exception:
                    logger.warning(
                        "Failed to rebuild PageIndex-Lite nodes for %s",
                        updated.id,
                        exc_info=True,
                    )
                await self._event_bus.emit(Events.EXPERIENCE_UPDATED, {
                    "experience_id": experience_id,
                    "fields_updated": list(updates.keys()),
                    "user": user,
                })
            return updated.to_dict() if updated else None

    async def review(
        self,
        experience_id: str,
        review_status: str,
        reviewed_by: str,
        review_note: str | None = None,
    ) -> dict | None:
        async with self._session() as session:
            """Review an experience: approve or reject.

            On approve, auto-sets publish_status='published'.
            On reject, auto-sets publish_status='rejected'.
            Also records review history and sets reviewed_at timestamp.
            """
            repo = ExperienceRepository(session)
            exp_uuid = uuid.UUID(experience_id)

            experience = await repo.review(
                experience_id=exp_uuid,
                review_status=review_status,
                reviewed_by=reviewed_by,
                review_note=review_note,
            )
            if experience:
                # Set reviewed_at timestamp
                experience.reviewed_at = datetime.now(timezone.utc)
                await session.flush()

                # Record review history (P1-6)
                from team_memory.storage.models import ReviewHistory
                history = ReviewHistory(
                    experience_id=exp_uuid,
                    reviewer=reviewed_by,
                    action=review_status,
                    comment=review_note,
                )
                session.add(history)
                await session.commit()

                await self._event_bus.emit(Events.EXPERIENCE_REVIEWED, {
                    "experience_id": experience_id,
                    "review_status": review_status,
                    "reviewed_by": reviewed_by,
                })
            return experience.to_dict() if experience else None

    async def soft_delete(
        self,
        experience_id: str,
    ) -> bool:
        async with self._session() as session:
            """Soft-delete an experience."""
            repo = ExperienceRepository(session)
            result = await repo.soft_delete(uuid.UUID(experience_id))
            if result:
                await self._event_bus.emit(Events.EXPERIENCE_DELETED, {
                    "experience_id": experience_id,
                    "hard": False,
                })
            return result

    async def restore(
        self,
        experience_id: str,
    ) -> bool:
        async with self._session() as session:
            """Restore a soft-deleted experience."""
            repo = ExperienceRepository(session)
            result = await repo.restore(uuid.UUID(experience_id))
            if result:
                await self._event_bus.emit(Events.EXPERIENCE_RESTORED, {
                    "experience_id": experience_id,
                })
            return result

    async def hard_delete_and_rebuild(
        self,
        experience_id: str,
        new_data: dict,
        children_data: list[dict] | None = None,
        created_by: str = "system",
    ) -> dict:
        async with self._session() as session:
            """Hard-delete an experience (and its children), then create a new one.

            Saves a version snapshot before deletion for version history (B3).

            Used for the "edit and confirm" flow where the user edits fields
            and we replace the old record with a new one.

            Args:
                experience_id: ID of the experience to replace.
                new_data: Fields for the new parent experience.
                children_data: If provided, recreate with children.
                created_by: Author name.

            Returns:
                The newly created experience dict.
            """
            repo = ExperienceRepository(session)
            old_uuid = uuid.UUID(experience_id)

            # Save version snapshot before destructive operation (B3)
            try:
                await repo.save_version(
                    experience_id=old_uuid,
                    changed_by=created_by,
                    change_summary="编辑前快照 (hard_delete_and_rebuild)",
                )
            except (ValueError, Exception) as e:
                logger.warning("Failed to save version snapshot: %s", str(e))

            # Delete old record (CASCADE deletes children + feedbacks)
            deleted = await repo.delete(old_uuid)
            if not deleted:
                return None

            # Create new record(s)
            if children_data:
                return await self.save_group(
                    session=session,
                    parent=new_data,
                    children=children_data,
                    created_by=created_by,
                )
            else:
                return await self.save(
                    session=session,
                    title=new_data.get("title", ""),
                    problem=new_data.get("problem", ""),
                    solution=new_data.get("solution", ""),
                    created_by=created_by,
                    tags=new_data.get("tags"),
                    code_snippets=new_data.get("code_snippets"),
                    language=new_data.get("language"),
                    framework=new_data.get("framework"),
                    root_cause=new_data.get("root_cause"),
                )

    async def get_recent(
        self,
        limit: int = 10,
        include_all_statuses: bool = False,
        project: str | None = None,
    ) -> list[dict]:
        async with self._session() as session:
            """Get recent experiences."""
            repo = ExperienceRepository(session)
            experiences = await repo.list_recent(
                limit=limit,
                include_all_statuses=include_all_statuses,
                project=project,
            )
            return [exp.to_dict() for exp in experiences]

    async def get_pending_reviews(
        self,
    ) -> list[dict]:
        async with self._session() as session:
            """Get experiences pending review."""
            repo = ExperienceRepository(session)
            experiences = await repo.list_pending_review()
            return [exp.to_dict() for exp in experiences]

    async def get_stats(
        self,
        project: str | None = None,
        scope: str | None = None,
        current_user: str | None = None,
    ) -> dict:
        async with self._session() as session:
            """Get experience database statistics."""
            repo = ExperienceRepository(session)
            return await repo.get_stats(
                project=project, scope=scope, current_user=current_user
            )

    async def list_projects(self) -> list[str]:
        """Return distinct project names."""
        async with self._session() as session:
            repo = ExperienceRepository(session)
            return await repo.list_projects()

    async def get_query_logs(
        self,
        limit: int = 100,
    ) -> list[dict]:
        async with self._session() as session:
            """Get recent query logs."""
            repo = ExperienceRepository(session)
            return await repo.get_query_logs(limit=limit)

    async def get_query_stats(
        self,
    ) -> dict:
        async with self._session() as session:
            """Get query analytics summary."""
            repo = ExperienceRepository(session)
            return await repo.get_query_stats()

    # ======================== VERSION HISTORY (B3) ========================

    async def get_versions(
        self,
        experience_id: str,
    ) -> list[dict]:
        async with self._session() as session:
            """Get version history for an experience."""
            repo = ExperienceRepository(session)
            versions = await repo.get_versions(uuid.UUID(experience_id))
            return [v.to_dict() for v in versions]

    async def get_version_detail(
        self,
        version_id: str,
    ) -> dict | None:
        async with self._session() as session:
            """Get a single version snapshot."""
            repo = ExperienceRepository(session)
            version = await repo.get_version_detail(uuid.UUID(version_id))
            return version.to_dict() if version else None

    async def rollback_to_version(
        self,
        experience_id: str,
        version_id: str,
        user: str = "system",
    ) -> dict | None:
        async with self._session() as session:
            """Rollback an experience to a specific version.

            1. Save current state as a new version ("回滚前快照")
            2. Restore content fields from the target version's snapshot
            3. Re-generate embedding for the restored content
            4. Save a final version record ("回滚至版本 X")

            Returns the updated experience dict.
            """
            repo = ExperienceRepository(session)
            exp_uuid = uuid.UUID(experience_id)
            ver_uuid = uuid.UUID(version_id)

            # Check that both exist
            exp = await repo.get_by_id(exp_uuid, include_deleted=True)
            version = await repo.get_version_detail(ver_uuid)
            if exp is None or version is None:
                return None

            # Save current state before rollback
            try:
                await repo.save_version(
                    experience_id=exp_uuid,
                    changed_by=user,
                    change_summary="回滚前快照",
                )
            except Exception as e:
                logger.warning("Failed to save pre-rollback version: %s", str(e))

            # Restore content fields from snapshot
            snapshot = version.snapshot
            updates = {}
            for field in (
                "title", "description", "root_cause", "solution", "tags",
                "programming_language", "framework", "code_snippets",
            ):
                if field in snapshot:
                    updates[field] = snapshot[field]

            # Re-generate embedding
            embed_text = "\n".join([
                snapshot.get("title", ""),
                snapshot.get("description", ""),
                snapshot.get("solution", ""),
            ])
            if snapshot.get("root_cause"):
                embed_text += f"\n{snapshot['root_cause']}"
            try:
                embedding = await self._embedding.encode_single(embed_text)
                updates["embedding"] = embedding
            except Exception:
                logger.warning("Failed to regenerate embedding during rollback")

            # Apply updates
            updated = await repo.update(exp_uuid, **updates)
            if updated is None:
                return None

            # Save a version record for the rollback action
            try:
                await repo.save_version(
                    experience_id=exp_uuid,
                    changed_by=user,
                    change_summary=f"回滚至版本 {version.version_number}",
                )
            except Exception as e:
                logger.warning("Failed to save rollback version: %s", str(e))

            await self._event_bus.emit(Events.EXPERIENCE_ROLLED_BACK, {
                "experience_id": experience_id,
                "version_id": version_id,
                "user": user,
            })
            return updated.to_dict()

    # ======================== STALE DETECTION (B1) ========================

    async def scan_stale(
        self,
        months: int = 6,
    ) -> list[dict]:
        async with self._session() as session:
            """Scan for stale (unused) experiences."""
            repo = ExperienceRepository(session)
            stale = await repo.scan_stale(months=months)
            return [exp.to_dict() for exp in stale]

    # ======================== DEDUPLICATION (B2) ========================

    # ======================== SUMMARY / MEMORY COMPACTION (P0-4) ========================

    async def generate_summary(
        self,
        experience_id: str,
    ) -> dict | None:
        async with self._session() as session:
            """Generate an LLM summary for a single experience."""
            from team_memory.services.llm_parser import generate_summary

            repo = ExperienceRepository(session)
            exp_uuid = uuid.UUID(experience_id)
            experience = await repo.get_by_id(exp_uuid)
            if experience is None:
                return None

            content = (
                f"标题: {experience.title}\n问题: {experience.description}\n"
                f"解决方案: {experience.solution or '(尚未解决)'}"
            )
            if experience.root_cause:
                content += f"\n根因: {experience.root_cause}"

            try:
                summary = await generate_summary(content, llm_config=self._llm_config)
            except Exception as e:
                logger.warning("Failed to generate summary for %s: %s", experience_id, e)
                return None

            updated = await repo.update(exp_uuid, summary=summary)
            return updated.to_dict() if updated else None

    async def batch_generate_summaries(
        self,
        limit: int = 10,
    ) -> dict:
        async with self._session() as session:
            """Batch generate summaries for experiences without one."""
            from team_memory.services.llm_parser import generate_summary

            repo = ExperienceRepository(session)
            min_length = (
                self._memory_config.summary_threshold_tokens
                if self._memory_config
                else 500
            )
            experiences = await repo.get_experiences_without_summary(
                limit=limit, min_content_length=min_length
            )

            generated = 0
            errors = []
            for exp in experiences:
                content = (
                    f"标题: {exp.title}\n问题: {exp.description}\n"
                    f"解决方案: {exp.solution or '(尚未解决)'}"
                )
                if exp.root_cause:
                    content += f"\n根因: {exp.root_cause}"
                try:
                    summary = await generate_summary(content, llm_config=self._llm_config)
                    await repo.update(exp.id, summary=summary)
                    generated += 1
                except Exception as e:
                    errors.append({"id": str(exp.id), "error": str(e)})
                    logger.warning("Summary generation failed for %s: %s", exp.id, e)

            return {
                "generated": generated,
                "total_candidates": len(experiences),
                "errors": errors,
            }

    # ======================== DEDUPLICATION (B2) ========================

    async def find_duplicates(
        self,
        threshold: float = 0.92,
        limit: int = 20,
    ) -> list[dict]:
        async with self._session() as session:
            """Find near-duplicate experience pairs."""
            repo = ExperienceRepository(session)
            return await repo.find_duplicates(threshold=threshold, limit=limit)

    async def merge_experiences(
        self,
        primary_id: str,
        secondary_id: str,
        user: str = "system",
    ) -> dict | None:
        async with self._session() as session:
            """Merge secondary experience into primary.

            Saves version snapshots for both before merging (B3 integration).
            """
            repo = ExperienceRepository(session)
            primary_uuid = uuid.UUID(primary_id)
            secondary_uuid = uuid.UUID(secondary_id)

            # Save version snapshots before merge
            for exp_id, label in [
                (primary_uuid, "合并前快照 (主经验)"),
                (secondary_uuid, "合并前快照 (被合并经验)"),
            ]:
                try:
                    await repo.save_version(
                        experience_id=exp_id,
                        changed_by=user,
                        change_summary=label,
                    )
                except Exception as e:
                    logger.warning("Failed to save merge version for %s: %s", exp_id, str(e))

            result = await repo.merge_experiences(primary_uuid, secondary_uuid)
            if result is None:
                return None

            await self._event_bus.emit(Events.EXPERIENCE_MERGED, {
                "primary_id": primary_id,
                "secondary_id": secondary_id,
                "user": user,
            })
            return result.to_dict(include_children=True)

    # ======================== IMPORT / EXPORT (B4) ========================

    async def import_experiences(
        self,
        data: str | bytes,
        fmt: str,
        created_by: str,
    ) -> dict:
        async with self._session() as session:
            """Import experiences from JSON or CSV data.

            Args:
                data: Raw file content.
                fmt: 'json' or 'csv'.
                created_by: Author name.

            Returns:
                Summary dict with imported count.
            """
            repo = ExperienceRepository(session)

            if fmt == "json":
                parsed = json.loads(data)
                raw_experiences = (
                    parsed.get("experiences", parsed) if isinstance(parsed, dict) else parsed
                )
            elif fmt == "csv":
                text = data if isinstance(data, str) else data.decode("utf-8")
                reader = csv.DictReader(io.StringIO(text))
                raw_experiences = list(reader)
            else:
                raise ValueError(f"Unsupported format: {fmt}")

            imported = 0
            errors = []
            for idx, raw in enumerate(raw_experiences):
                try:
                    # Parse tags
                    tags = raw.get("tags", [])
                    if isinstance(tags, str):
                        tags = [t.strip() for t in tags.split(";") if t.strip()]

                    # Generate embedding
                    desc = raw.get("description", raw.get("problem", ""))
                    embed_text = "\n".join([
                        raw.get("title", ""),
                        desc,
                        raw.get("solution", ""),
                    ])
                    try:
                        embedding = await self._embedding.encode_single(embed_text)
                    except Exception:
                        embedding = None

                    children_raw = raw.get("children")
                    if children_raw:
                        # Import as group
                        parent_data = {
                            "title": raw.get("title", ""),
                            "description": raw.get("description", raw.get("problem", "")),
                            "solution": raw.get("solution", ""),
                            "tags": tags,
                            "programming_language": raw.get(
                                "programming_language", raw.get("language")
                            ),
                            "framework": raw.get("framework"),
                            "code_snippets": raw.get("code_snippets"),
                            "root_cause": raw.get("root_cause"),
                            "embedding": embedding,
                            "source": raw.get("source", "import"),
                        }
                        children_data = []
                        for child in children_raw:
                            child_tags = child.get("tags", [])
                            if isinstance(child_tags, str):
                                child_tags = [t.strip() for t in child_tags.split(";") if t.strip()]
                            child_embed_text = "\n".join([
                                child.get("title", ""),
                                child.get("description", child.get("problem", "")),
                                child.get("solution", ""),
                            ])
                            try:
                                child_embedding = await self._embedding.encode_single(
                                    child_embed_text
                                )
                            except Exception:
                                child_embedding = None
                            children_data.append({
                                "title": child.get("title", ""),
                                "description": child.get("description", child.get("problem", "")),
                                "solution": child.get("solution", ""),
                                "tags": child_tags,
                                "programming_language": child.get(
                                    "programming_language", child.get("language")
                                ),
                                "framework": child.get("framework"),
                                "code_snippets": child.get("code_snippets"),
                                "root_cause": child.get("root_cause"),
                                "embedding": child_embedding,
                                "source": child.get("source", "import"),
                                "created_by": created_by,
                            })
                        parent_data["created_by"] = created_by
                        parent_exp = await repo.create_group(
                            parent_data=parent_data,
                            children_data=children_data,
                            created_by=created_by,
                        )
                        try:
                            await self._maybe_build_tree_nodes(
                                repo,
                                parent_exp.id,
                                problem=raw.get("description", raw.get("problem", "")),
                                solution=raw.get("solution", ""),
                                root_cause=raw.get("root_cause"),
                                code_snippets=raw.get("code_snippets"),
                            )
                            for child_exp, child in zip(
                                parent_exp.children or [], children_raw, strict=False
                            ):
                                await self._maybe_build_tree_nodes(
                                    repo,
                                    child_exp.id,
                                    problem=child.get("description", child.get("problem", "")),
                                    solution=child.get("solution", ""),
                                    root_cause=child.get("root_cause"),
                                    code_snippets=child.get("code_snippets"),
                                )
                        except Exception:
                            logger.warning(
                                "PageIndex-Lite build failed during group import",
                                exc_info=True,
                            )
                    else:
                        # Import as single
                        exp = await repo.create(
                            title=raw.get("title", ""),
                            description=raw.get("description", raw.get("problem", "")),
                            solution=raw.get("solution", ""),
                            created_by=created_by,
                            tags=tags,
                            programming_language=raw.get(
                                "programming_language", raw.get("language")
                            ),
                            framework=raw.get("framework"),
                            code_snippets=raw.get("code_snippets"),
                            embedding=embedding,
                            source=raw.get("source", "import"),
                            root_cause=raw.get("root_cause"),
                        )
                        try:
                            await self._maybe_build_tree_nodes(
                                repo,
                                exp.id,
                                problem=raw.get("description", raw.get("problem", "")),
                                solution=raw.get("solution", ""),
                                root_cause=raw.get("root_cause"),
                                code_snippets=raw.get("code_snippets"),
                            )
                        except Exception:
                            logger.warning(
                                "PageIndex-Lite build failed during import",
                                exc_info=True,
                            )
                    imported += 1
                except Exception as e:
                    errors.append({"index": idx, "error": str(e)})
                    logger.warning("Import error at index %d: %s", idx, str(e))

            await self._event_bus.emit(Events.EXPERIENCE_IMPORTED, {
                "imported": imported,
                "total": len(raw_experiences),
                "format": fmt,
                "created_by": created_by,
            })
            return {
                "imported": imported,
                "total": len(raw_experiences),
                "errors": errors,
            }

    async def export_experiences(
        self,
        fmt: str = "json",
        tag: str | None = None,
        start: str | None = None,
        end: str | None = None,
    ) -> str:
        async with self._session() as session:
            """Export experiences as JSON or CSV.

            Args:
                fmt: 'json' or 'csv'.
                tag: Filter by tag.
                start: Start date (ISO format).
                end: End date (ISO format).

            Returns:
                Serialized string content.
            """
            repo = ExperienceRepository(session)

            start_dt = datetime.fromisoformat(start) if start else None
            end_dt = datetime.fromisoformat(end) if end else None

            experiences = await repo.export_filtered(
                tag=tag, start=start_dt, end=end_dt
            )

            if fmt == "json":
                result = {
                    "version": "1.0",
                    "exported_at": datetime.now(timezone.utc).isoformat(),
                    "count": len(experiences),
                    "experiences": [],
                }
                for exp in experiences:
                    exp_dict = {
                        "title": exp.title,
                        "description": exp.description,
                        "root_cause": exp.root_cause,
                        "solution": exp.solution,
                        "tags": exp.tags or [],
                        "programming_language": exp.programming_language,
                        "framework": exp.framework,
                        "code_snippets": exp.code_snippets,
                        "source": exp.source,
                        "created_by": exp.created_by,
                        "created_at": exp.created_at.isoformat() if exp.created_at else None,
                        "avg_rating": exp.avg_rating,
                        "use_count": exp.use_count,
                    }
                    if exp.children:
                        exp_dict["children"] = [
                            {
                                "title": c.title,
                                "description": c.description,
                                "root_cause": c.root_cause,
                                "solution": c.solution,
                                "tags": c.tags or [],
                                "programming_language": c.programming_language,
                                "framework": c.framework,
                                "code_snippets": c.code_snippets,
                            }
                            for c in exp.children
                        ]
                    result["experiences"].append(exp_dict)
                return json.dumps(result, ensure_ascii=False, indent=2)

            elif fmt == "csv":
                output = io.StringIO()
                writer = csv.writer(output)
                writer.writerow([
                    "title", "description", "root_cause", "solution",
                    "tags", "programming_language", "framework", "code_snippets",
                    "source", "created_by", "created_at", "avg_rating", "use_count",
                ])
                for exp in experiences:
                    writer.writerow([
                        exp.title,
                        exp.description,
                        exp.root_cause or "",
                        exp.solution or "",
                        ";".join(exp.tags or []),
                        exp.programming_language or "",
                        exp.framework or "",
                        exp.code_snippets or "",
                        exp.source,
                        exp.created_by,
                        exp.created_at.isoformat() if exp.created_at else "",
                        exp.avg_rating,
                        exp.use_count,
                    ])
                return output.getvalue()

            else:
                raise ValueError(f"Unsupported format: {fmt}")

    # ------------------------------------------------------------------
    # P1-4: Convenience facades for route handlers
    # ------------------------------------------------------------------

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
            total = await repo.count_experiences(
                status=status, tag=tag, project=project
            )
            return {
                "experiences": exps,
                "total": total,
                "page": page,
                "page_size": page_size,
            }

    async def get_experience(self, experience_id: str) -> dict | None:
        """Retrieve a single experience by ID."""
        async with self._session() as session:
            repo = ExperienceRepository(session)
            exp = await repo.get_by_id(experience_id)
            if exp is None:
                return None
            return repo.experience_to_dict(exp)

    async def promote(
        self, experience_id: str, *, user: str = "system"
    ) -> dict | None:
        """Promote an experience to 'published' status."""
        return await self.review(
            experience_id, new_status="approved", reviewer=user
        )

    async def batch_action(
        self,
        ids: list[str],
        action: str,
        *,
        user: str = "system",
    ) -> dict:
        """Apply an action to multiple experiences."""
        results: dict = {"success": 0, "failed": 0, "errors": []}
        for exp_id in ids:
            try:
                if action == "delete":
                    ok = await self.soft_delete(exp_id)
                elif action == "restore":
                    ok = await self.restore(exp_id)
                elif action == "promote":
                    ok = await self.promote(exp_id, user=user) is not None
                else:
                    raise ValueError(f"Unknown action: {action}")
                if ok:
                    results["success"] += 1
                else:
                    results["failed"] += 1
            except Exception as e:
                results["failed"] += 1
                results["errors"].append({"id": exp_id, "error": str(e)})
        return results

    async def suggest_tags(
        self, prefix: str = "", *, limit: int = 20
    ) -> list[str]:
        """Suggest tags based on prefix and existing usage."""
        stats = await self.get_stats()
        tags = stats.get("tag_distribution", {})
        if prefix:
            tags = {k: v for k, v in tags.items() if k.startswith(prefix.lower())}
        sorted_tags = sorted(tags, key=lambda t: tags[t], reverse=True)
        return sorted_tags[:limit]

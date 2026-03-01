"""Experience data repository — CRUD operations + vector/FTS search."""

from __future__ import annotations

import uuid
from datetime import datetime, timezone

from sqlalchemy import Float, delete, desc, func, or_, select, text, update
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from team_memory.storage.models import (
    DocumentTreeNode,
    Experience,
    ExperienceFeedback,
    ExperienceVersion,
    PersonalTask,
    QueryLog,
    TaskDependency,
    TaskGroup,
    TaskMessage,
)


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


class ExperienceRepository:
    """Repository for Experience CRUD, vector search, and FTS operations.

    All methods require an AsyncSession to be passed in,
    allowing the caller to manage transaction boundaries.
    """

    def __init__(self, session: AsyncSession):
        self._session = session

    # ====================== Filter helpers ======================

    @staticmethod
    def _active_filter(current_user: str | None = None):
        """Visibility filter using new status model: published for all; others for creator."""
        base = [Experience.is_deleted == False]  # noqa: E712
        if current_user:
            base.append(
                or_(
                    Experience.exp_status == "published",
                    (Experience.created_by == current_user),
                )
            )
        else:
            base.append(Experience.exp_status == "published")
        return base

    @staticmethod
    def _project_value(project: str | None) -> str:
        """Normalize project value for filtering/writes."""
        if project and project.strip():
            value = project.strip()
            alias_map = {
                "team-memory": "team_memory",
                "team_doc": "team_memory",
            }
            return alias_map.get(value, value)
        return "default"

    # ======================== CREATE ========================

    # Valid status transitions: {from_status: [allowed_to_statuses]}
    VALID_STATUS_TRANSITIONS = {
        "draft": ["review", "published"],
        "review": ["published", "rejected"],
        "published": ["draft"],
        "rejected": ["draft"],
    }

    async def create(
        self,
        title: str,
        description: str,
        solution: str | None = None,
        created_by: str = "system",
        tags: list[str] | None = None,
        programming_language: str | None = None,
        framework: str | None = None,
        code_snippets: str | None = None,
        embedding: list[float] | None = None,
        source: str = "manual",
        source_context: str | None = None,
        root_cause: str | None = None,
        publish_status: str = "published",
        review_status: str = "approved",
        parent_id: uuid.UUID | None = None,
        embedding_status: str = "ready",
        summary: str | None = None,
        # Type system fields (v3)
        experience_type: str = "general",
        severity: str | None = None,
        category: str | None = None,
        progress_status: str | None = None,
        structured_data: dict | None = None,
        git_refs: list | None = None,
        related_links: list | None = None,
        project: str = "default",
        quality_score: int = 0,
        # New status model (v2)
        visibility: str | None = None,
        exp_status: str | None = None,
    ) -> Experience:
        """Create a new experience record."""
        # Derive new fields from old if not explicitly provided
        if visibility is None:
            scope_to_vis = {"global": "global", "personal": "private", "team": "project"}
            visibility = scope_to_vis.get(publish_status, "project")
            if publish_status == "personal":
                visibility = "private"
            elif publish_status in ("published", "pending_team", "draft", "rejected"):
                visibility = "project"
        if exp_status is None:
            ps_to_status = {
                "draft": "draft",
                "personal": "published",
                "pending_team": "review",
                "published": "published",
                "rejected": "rejected",
            }
            exp_status = ps_to_status.get(publish_status, "draft")

        experience = Experience(
            id=uuid.uuid4(),
            title=title,
            description=description,
            root_cause=root_cause,
            solution=solution,
            created_by=created_by,
            tags=tags or [],
            programming_language=programming_language,
            framework=framework,
            code_snippets=code_snippets,
            embedding=embedding,
            source=source,
            source_context=source_context,
            publish_status=publish_status,
            review_status=review_status,
            parent_id=parent_id,
            embedding_status=embedding_status,
            summary=summary,
            experience_type=experience_type,
            severity=severity,
            category=category,
            progress_status=progress_status,
            structured_data=structured_data,
            git_refs=git_refs,
            related_links=related_links,
            project=self._project_value(project),
            quality_score=quality_score,
            visibility=visibility,
            exp_status=exp_status,
        )
        self._session.add(experience)
        await self._session.flush()
        return experience

    async def create_group(
        self,
        parent_data: dict,
        children_data: list[dict],
        created_by: str,
    ) -> Experience:
        """Create a parent experience with children in one transaction.

        Args:
            parent_data: Fields for the parent experience.
            children_data: List of field dicts for child experiences.
            created_by: Author name.

        Returns:
            The parent experience (with children loaded).
        """
        parent = await self.create(
            created_by=created_by,
            **parent_data,
        )

        for child_data in children_data:
            await self.create(
                created_by=created_by,
                parent_id=parent.id,
                **child_data,
            )

        # Re-fetch with children eagerly loaded
        return await self.get_with_children(parent.id)

    # ======================== READ ========================

    async def get_by_id(
        self, experience_id: uuid.UUID, include_deleted: bool = False
    ) -> Experience | None:
        """Get an experience by its ID."""
        query = select(Experience).where(Experience.id == experience_id)
        if not include_deleted:
            query = query.where(Experience.is_deleted == False)  # noqa: E712
        result = await self._session.execute(query)
        return result.scalar_one_or_none()

    async def get_with_children(
        self, experience_id: uuid.UUID, include_deleted: bool = False
    ) -> Experience | None:
        """Get an experience with its children eagerly loaded."""
        query = (
            select(Experience)
            .where(Experience.id == experience_id)
            .options(
                selectinload(Experience.children),
                selectinload(Experience.feedbacks),
            )
        )
        if not include_deleted:
            query = query.where(Experience.is_deleted == False)  # noqa: E712
        result = await self._session.execute(query)
        return result.scalar_one_or_none()

    async def get_children(
        self, parent_id: uuid.UUID, include_deleted: bool = False
    ) -> list[Experience]:
        """Get all children of a parent experience."""
        query = (
            select(Experience)
            .where(Experience.parent_id == parent_id)
            .order_by(Experience.created_at)
        )
        if not include_deleted:
            query = query.where(Experience.is_deleted == False)  # noqa: E712
        result = await self._session.execute(query)
        return list(result.scalars().all())

    async def get_root_id(self, experience_id: uuid.UUID) -> uuid.UUID:
        """Find the root (topmost parent) of an experience.

        Traverses up parent_id chain. For a root experience, returns itself.
        """
        current_id = experience_id
        for _ in range(10):  # Safety limit to prevent infinite loops
            exp = await self.get_by_id(current_id, include_deleted=True)
            if exp is None or exp.parent_id is None:
                return current_id
            current_id = exp.parent_id
        return current_id

    # ======================== PAGEINDEX-LITE TREE NODES ========================

    async def replace_tree_nodes(
        self,
        experience_id: uuid.UUID,
        nodes: list[dict],
    ) -> int:
        """Replace all tree nodes for one experience in a single transaction."""
        await self._session.execute(
            delete(DocumentTreeNode).where(
                DocumentTreeNode.experience_id == experience_id
            )
        )

        created = 0
        for idx, node in enumerate(nodes):
            self._session.add(
                DocumentTreeNode(
                    id=uuid.uuid4(),
                    experience_id=experience_id,
                    path=node.get("path", str(idx + 1)),
                    node_title=node.get("node_title", "Untitled"),
                    depth=node.get("depth", 1),
                    node_order=node.get("node_order", idx),
                    content=node.get("content"),
                    content_summary=node.get("content_summary"),
                    char_count=node.get("char_count", 0),
                    is_leaf=node.get("is_leaf", False),
                )
            )
            created += 1

        await self._session.flush()
        return created

    async def get_tree_nodes(self, experience_id: uuid.UUID) -> list[DocumentTreeNode]:
        """Get all tree nodes for an experience."""
        result = await self._session.execute(
            select(DocumentTreeNode)
            .where(DocumentTreeNode.experience_id == experience_id)
            .order_by(DocumentTreeNode.node_order.asc())
        )
        return list(result.scalars().all())

    async def search_tree_nodes(
        self,
        query_text: str,
        experience_ids: list[uuid.UUID],
        max_results: int = 30,
        min_score: float = 0.01,
        max_nodes_per_experience: int = 3,
    ) -> dict[str, list[dict]]:
        """Search matched tree nodes for candidate experiences.

        Returns:
            Mapping: experience_id -> list of matched node dicts.
        """
        if not query_text.strip() or not experience_ids:
            return {}

        ts_query = func.plainto_tsquery("simple", query_text)
        search_vector = func.to_tsvector(
            "simple",
            func.concat(
                func.coalesce(DocumentTreeNode.node_title, ""),
                text("' '"),
                func.coalesce(DocumentTreeNode.content, ""),
            ),
        )
        rank_expr = func.ts_rank(search_vector, ts_query).label("rank")

        result = await self._session.execute(
            select(DocumentTreeNode, rank_expr)
            .where(DocumentTreeNode.experience_id.in_(experience_ids))
            .where(search_vector.op("@@")(ts_query))
            .order_by(desc(rank_expr))
            .limit(max_results)
        )
        rows = result.all()

        grouped: dict[str, list[dict]] = {}
        for node, rank in rows:
            score = float(rank or 0.0)
            if score < min_score:
                continue
            key = str(node.experience_id)
            grouped.setdefault(key, [])
            if len(grouped[key]) >= max_nodes_per_experience:
                continue
            grouped[key].append(
                {
                    "id": str(node.id),
                    "path": node.path,
                    "node_title": node.node_title,
                    "depth": node.depth,
                    "content_summary": node.content_summary,
                    "score": round(score, 4),
                }
            )

        return grouped

    async def list_recent(
        self,
        limit: int = 10,
        offset: int = 0,
        include_all_statuses: bool = False,
        project: str | None = None,
        scope: str | None = None,
        current_user: str | None = None,
    ) -> list[Experience]:
        """List root/standalone experiences ordered by creation time (newest first).

        Children are excluded from the listing; use get_with_children to retrieve them.
        """
        proj = self._project_value(project)
        query = (
            select(Experience)
            .where(Experience.parent_id.is_(None))  # Only root/standalone
            .options(selectinload(Experience.children))
            .order_by(desc(Experience.created_at))
        )
        query = self._apply_scope_filter(query, proj, scope, current_user)
        if not include_all_statuses:
            for f in self._active_filter(current_user):
                query = query.where(f)
        else:
            # Still exclude hard-deleted
            query = query.where(Experience.is_deleted == False)  # noqa: E712
        result = await self._session.execute(
            query.limit(limit).offset(offset)
        )
        return list(result.scalars().all())

    async def list_by_tags(
        self,
        tags: list[str],
        limit: int = 10,
        offset: int = 0,
        project: str | None = None,
        current_user: str | None = None,
    ) -> list[Experience]:
        """List root/standalone experiences that have any of the given tags."""
        query = (
            select(Experience)
            .where(Experience.parent_id.is_(None))  # Only root/standalone
            .where(Experience.tags.overlap(tags))
            .where(Experience.project == self._project_value(project))
            .options(selectinload(Experience.children))
            .order_by(desc(Experience.created_at))
            .limit(limit)
            .offset(offset)
        )
        for f in self._active_filter(current_user):
            query = query.where(f)
        result = await self._session.execute(query)
        return list(result.scalars().all())

    async def count_by_tags(
        self,
        tags: list[str],
        project: str | None = None,
        current_user: str | None = None,
    ) -> int:
        """Count root/standalone experiences matching any of the given tags."""
        query = (
            select(func.count())
            .select_from(Experience)
            .where(Experience.parent_id.is_(None))
            .where(Experience.tags.overlap(tags))
            .where(Experience.project == self._project_value(project))
        )
        for f in self._active_filter(current_user):
            query = query.where(f)
        result = await self._session.execute(query)
        return result.scalar() or 0

    async def list_pending_review(self, limit: int = 50) -> list[Experience]:
        """List experiences pending review (exp_status='review')."""
        result = await self._session.execute(
            select(Experience)
            .where(Experience.is_deleted == False)  # noqa: E712
            .where(Experience.exp_status == "review")
            .order_by(desc(Experience.created_at))
            .limit(limit)
        )
        return list(result.scalars().all())

    async def count(
        self,
        include_deleted: bool = False,
        project: str | None = None,
        scope: str | None = None,
        current_user: str | None = None,
    ) -> int:
        """Count root/standalone experiences (excludes children)."""
        query = (
            select(func.count())
            .select_from(Experience)
            .where(Experience.parent_id.is_(None))
        )
        proj = self._project_value(project)
        query = self._apply_scope_filter(query, proj, scope, current_user)
        if not include_deleted:
            for f in self._active_filter(current_user):
                query = query.where(f)
        result = await self._session.execute(query)
        return result.scalar_one()

    @staticmethod
    def _apply_scope_filter(query, project: str, scope: str | None, current_user: str | None):
        """Apply visibility-aware project filter using new visibility field."""
        visibility = scope  # accept both old 'scope' param and new 'visibility'
        if visibility and visibility != "all":
            # Map old scope values to new visibility values
            vis_map = {"personal": "private", "team": "project", "global": "global"}
            vis_val = vis_map.get(visibility, visibility)
            query = query.where(Experience.visibility == vis_val)
            if vis_val == "private" and current_user:
                query = query.where(Experience.created_by == current_user)
            if vis_val != "global":
                query = query.where(Experience.project == project)
        else:
            query = query.where(
                or_(Experience.project == project, Experience.visibility == "global")
            )
            if current_user:
                query = query.where(
                    or_(
                        Experience.visibility != "private",
                        Experience.created_by == current_user,
                    )
                )
            else:
                query = query.where(Experience.visibility != "private")
        return query

    # ======================== VECTOR SEARCH ========================

    async def search_by_vector(
        self,
        query_embedding: list[float],
        max_results: int = 5,
        min_similarity: float = 0.6,
        tags: list[str] | None = None,
        project: str | None = None,
        current_user: str | None = None,
    ) -> list[dict]:
        """Search experiences by vector similarity using pgvector."""
        similarity_expr = (
            1 - Experience.embedding.cosine_distance(query_embedding)
        ).label("similarity")

        query = (
            select(Experience, similarity_expr)
            .where(Experience.embedding.is_not(None))
            .where(Experience.embedding_status == "ready")  # D2: skip pending
            .where(Experience.project == self._project_value(project))
            .where(similarity_expr >= min_similarity)
            .order_by(desc(similarity_expr))
            .limit(max_results)
        )

        # Only published, non-deleted
        for f in self._active_filter(current_user):
            query = query.where(f)

        if tags:
            query = query.where(Experience.tags.overlap(tags))

        result = await self._session.execute(query)
        rows = result.all()

        results = []
        for exp, similarity in rows:
            data = exp.to_dict()
            data["similarity"] = round(float(similarity), 4)
            results.append(data)

        return results

    async def search_by_vector_grouped(
        self,
        query_embedding: list[float],
        max_results: int = 5,
        min_similarity: float = 0.6,
        tags: list[str] | None = None,
        top_k_children: int = 3,
        min_avg_rating: float = 0.0,
        rating_weight: float = 0.3,
        project: str | None = None,
        current_user: str | None = None,
    ) -> list[dict]:
        """Search experiences with group deduplication and A+B rating strategy.

        Strategy A: final_score = similarity * (0.7 + rating_weight * avg_rating / 5)
        Strategy B: filter out groups where avg_rating < min_avg_rating

        Searches both parent and child embeddings, then groups results
        by root experience, returning each group only once.
        """
        raw_results = await self.search_by_vector(
            query_embedding=query_embedding,
            max_results=max_results * 3,
            min_similarity=min_similarity,
            tags=tags,
            project=project,
            current_user=current_user,
        )

        if not raw_results:
            return []

        # Group by root ID
        groups: dict[str, dict] = {}  # root_id -> {score, hits}
        for result in raw_results:
            parent_id = result.get("parent_id")

            # Determine root ID
            if parent_id is None:
                root_id = result["id"]
            else:
                root_id = parent_id  # Direct parent (single level)

            score = result.get("similarity", 0)

            if root_id not in groups:
                groups[root_id] = {"score": score, "hits": [result]}
            else:
                groups[root_id]["score"] = max(groups[root_id]["score"], score)
                groups[root_id]["hits"].append(result)

        # Build grouped results with rating weighting
        grouped_results = []
        for root_id, group_data in groups.items():
            root_uuid = uuid.UUID(root_id)
            root_exp = await self.get_with_children(root_uuid)
            if root_exp is None:
                continue

            # Strategy B: filter by min_avg_rating
            if min_avg_rating > 0 and root_exp.avg_rating < min_avg_rating:
                continue

            # Strategy A: apply rating weight to score
            # When avg_rating=0 (no feedback), skip weighting to avoid penalizing
            # unrated experiences
            similarity = group_data["score"]
            avg_rating = root_exp.avg_rating or 0.0
            if avg_rating > 0:
                final_score = similarity * (
                    1.0 - rating_weight + rating_weight * avg_rating / 5.0
                )
            else:
                final_score = similarity

            # Get top-K children
            children = root_exp.children or []
            children_dicts = [c.to_dict() for c in children[:top_k_children]]

            parent_dict = root_exp.to_dict()
            result_entry = {
                "group_id": root_id,
                "id": root_id,
                "score": round(final_score, 4),
                "similarity": round(similarity, 4),
                "avg_rating": round(avg_rating, 2),
                "parent": parent_dict,
                "children": children_dicts,
                "total_children": len(children),
            }
            for key in ("title", "solution", "problem", "description",
                        "tags", "experience_type", "project", "scope",
                        "created_by", "created_at", "view_count", "use_count",
                        "root_cause", "category", "publish_status"):
                if key in parent_dict:
                    result_entry[key] = parent_dict[key]
            grouped_results.append(result_entry)

        # Sort by final score (descending) and limit
        grouped_results.sort(key=lambda x: x["score"], reverse=True)
        return grouped_results[:max_results]

    # ======================== DEDUP-ON-SAVE CHECK ========================

    async def check_similar(
        self,
        embedding: list[float],
        threshold: float = 0.90,
        limit: int = 5,
        project: str | None = None,
        current_user: str | None = None,
    ) -> list[dict]:
        """Quick similarity check for dedup-on-save.

        Searches published, non-deleted root experiences for similar embeddings.
        Returns a lightweight list of candidates: {id, title, tags, similarity}.
        """
        similarity_expr = (
            1 - Experience.embedding.cosine_distance(embedding)
        ).label("similarity")

        query = (
            select(Experience, similarity_expr)
            .where(Experience.embedding.is_not(None))
            .where(Experience.embedding_status == "ready")
            .where(Experience.parent_id.is_(None))  # Only root experiences
            .where(Experience.project == self._project_value(project))
            .where(similarity_expr >= threshold)
            .order_by(desc(similarity_expr))
            .limit(limit)
        )
        for f in self._active_filter(current_user):
            query = query.where(f)

        result = await self._session.execute(query)
        rows = result.all()

        return [
            {
                "id": str(exp.id),
                "title": exp.title,
                "tags": exp.tags or [],
                "similarity": round(float(sim), 4),
                "description": (
                    (exp.description[:200] + "...")
                    if len(exp.description) > 200
                    else exp.description
                ),
            }
            for exp, sim in rows
        ]

    # ======================== DRAFT LISTING ========================

    async def list_drafts(
        self,
        created_by: str | None = None,
        limit: int = 50,
        offset: int = 0,
        project: str | None = None,
    ) -> list[Experience]:
        """List draft experiences, optionally filtered by creator."""
        query = (
            select(Experience)
            .where(Experience.is_deleted == False)  # noqa: E712
            .where(Experience.exp_status == "draft")
            .where(Experience.parent_id.is_(None))
            .where(Experience.project == self._project_value(project))
            .options(selectinload(Experience.children))
            .order_by(desc(Experience.created_at))
        )
        if created_by:
            query = query.where(Experience.created_by == created_by)
        result = await self._session.execute(query.limit(limit).offset(offset))
        return list(result.scalars().all())

    async def count_drafts(
        self, created_by: str | None = None, project: str | None = None
    ) -> int:
        """Count draft experiences."""
        query = (
            select(func.count())
            .select_from(Experience)
            .where(Experience.is_deleted == False)  # noqa: E712
            .where(Experience.exp_status == "draft")
            .where(Experience.parent_id.is_(None))
            .where(Experience.project == self._project_value(project))
        )
        if created_by:
            query = query.where(Experience.created_by == created_by)
        result = await self._session.execute(query)
        return result.scalar_one()

    async def change_status(
        self,
        experience_id: uuid.UUID,
        new_status: str,
        visibility: str | None = None,
        changed_by: str = "system",
        is_admin: bool = False,
    ) -> Experience | None:
        """Change experience status with validation of allowed transitions.

        Admin users can skip review and go directly from draft to published.
        """
        experience = await self.get_by_id(experience_id, include_deleted=False)
        if experience is None:
            return None

        current_status = experience.exp_status
        allowed = self.VALID_STATUS_TRANSITIONS.get(current_status, [])

        if is_admin and current_status == "draft" and new_status == "published":
            pass  # admin can skip review
        elif new_status not in allowed:
            raise ValueError(
                f"Cannot transition from '{current_status}' to '{new_status}'. "
                f"Allowed: {allowed}"
            )

        # Sync old fields for backward compatibility
        status_to_publish = {
            "draft": "draft",
            "review": "pending_team",
            "published": "published",
            "rejected": "rejected",
        }
        status_to_review = {
            "draft": "approved",
            "review": "pending",
            "published": "approved",
            "rejected": "rejected",
        }
        experience.exp_status = new_status
        experience.publish_status = status_to_publish.get(new_status, new_status)
        experience.review_status = status_to_review.get(new_status, "approved")

        if new_status in ("published", "rejected"):
            experience.reviewed_by = changed_by
            experience.reviewed_at = _utcnow()

        if visibility is not None:
            experience.visibility = visibility
            vis_to_scope = {"private": "personal", "project": "team", "global": "global"}
            experience.scope = vis_to_scope.get(visibility, "team")

        experience.updated_at = _utcnow()
        await self._session.flush()
        return experience

    # ======================== SUMMARY (P0-4) ========================

    async def get_experiences_without_summary(
        self,
        limit: int = 10,
        min_content_length: int = 500,
        current_user: str | None = None,
    ) -> list[Experience]:
        """Get published experiences without a summary for batch generation."""
        query = (
            select(Experience)
            .where(Experience.summary.is_(None))
            .where(Experience.parent_id.is_(None))
            .where(
                func.length(Experience.description)
                + func.length(Experience.solution)
                > min_content_length
            )
            .order_by(desc(Experience.created_at))
            .limit(limit)
        )
        for f in self._active_filter(current_user):
            query = query.where(f)
        result = await self._session.execute(query)
        return list(result.scalars().all())

    # ======================== FTS SEARCH ========================

    async def search_by_fts(
        self,
        query_text: str,
        max_results: int = 5,
        tags: list[str] | None = None,
        project: str | None = None,
        current_user: str | None = None,
    ) -> list[dict]:
        """Full-text search using PostgreSQL tsvector/tsquery.

        Uses jieba tokenizer for Chinese text when available, falling back
        to the 'simple' PG configuration.
        """
        from team_memory.services.tokenizer import tokenize

        tokenized_query = tokenize(query_text)
        ts_query = func.plainto_tsquery("simple", tokenized_query)
        rank_expr = func.ts_rank(Experience.fts, ts_query).label("rank")

        query = (
            select(Experience, rank_expr)
            .where(Experience.fts.op("@@")(ts_query))
            .where(Experience.project == self._project_value(project))
            .order_by(desc(rank_expr))
            .limit(max_results)
        )

        # Only published, non-deleted
        for f in self._active_filter(current_user):
            query = query.where(f)

        if tags:
            query = query.where(Experience.tags.overlap(tags))

        result = await self._session.execute(query)
        rows = result.all()

        results = []
        for exp, rank in rows:
            data = exp.to_dict()
            data["fts_rank"] = round(float(rank), 4)
            results.append(data)

        return results

    # ======================== UPDATE ========================

    async def update(
        self,
        experience_id: uuid.UUID,
        **kwargs,
    ) -> Experience | None:
        """Update an experience record.

        All provided kwargs are applied, including None values
        (to allow clearing fields like severity, solution, etc.).
        """
        experience = await self.get_by_id(experience_id, include_deleted=True)
        if experience is None:
            return None

        for key, value in kwargs.items():
            if hasattr(experience, key):
                setattr(experience, key, value)

        experience.updated_at = datetime.now(timezone.utc)
        await self._session.flush()
        return experience

    async def review(
        self,
        experience_id: uuid.UUID,
        review_status: str,
        reviewed_by: str,
        review_note: str | None = None,
    ) -> Experience | None:
        """Review an experience: approve or reject."""
        experience = await self.get_by_id(experience_id, include_deleted=True)
        if experience is None:
            return None

        experience.review_status = review_status
        experience.reviewed_by = reviewed_by
        experience.review_note = review_note

        if review_status == "approved":
            experience.publish_status = "published"
        elif review_status == "rejected":
            experience.publish_status = "rejected"

        experience.updated_at = datetime.now(timezone.utc)
        await self._session.flush()
        return experience

    async def publish_to_team(
        self,
        experience_id: uuid.UUID,
        is_admin: bool = False,
    ) -> Experience | None:
        """Request publishing an experience to the team.

        Admin users go directly to 'published'; others go to 'pending_team'.
        """
        experience = await self.get_by_id(experience_id, include_deleted=False)
        if experience is None:
            return None
        if is_admin:
            experience.publish_status = "published"
            experience.review_status = "approved"
        else:
            experience.publish_status = "pending_team"
            experience.review_status = "pending"
        experience.updated_at = _utcnow()
        await self._session.flush()
        return experience

    async def publish_personal(
        self,
        experience_id: uuid.UUID,
    ) -> Experience | None:
        """Move a draft experience to personal (visible to creator)."""
        experience = await self.get_by_id(experience_id, include_deleted=False)
        if experience is None:
            return None
        if experience.publish_status == "draft":
            experience.publish_status = "personal"
            experience.updated_at = _utcnow()
            await self._session.flush()
        return experience

    async def increment_use_count(self, experience_id: uuid.UUID) -> None:
        """Increment the use_count of an experience and update last_used_at."""
        await self._session.execute(
            update(Experience)
            .where(Experience.id == experience_id)
            .values(
                use_count=Experience.use_count + 1,
                last_used_at=func.now(),
            )
        )

    async def increment_view_count(self, experience_id: uuid.UUID) -> None:
        """Increment the view_count of an experience."""
        await self._session.execute(
            update(Experience)
            .where(Experience.id == experience_id)
            .values(view_count=Experience.view_count + 1)
        )

    # ======================== DELETE ========================

    async def soft_delete(self, experience_id: uuid.UUID) -> bool:
        """Soft-delete an experience by ID."""
        experience = await self.get_by_id(experience_id)
        if experience is None:
            return False
        experience.is_deleted = True
        experience.deleted_at = datetime.now(timezone.utc)
        await self._session.flush()
        return True

    async def restore(self, experience_id: uuid.UUID) -> bool:
        """Restore a soft-deleted experience."""
        experience = await self.get_by_id(experience_id, include_deleted=True)
        if experience is None or not experience.is_deleted:
            return False
        experience.is_deleted = False
        experience.deleted_at = None
        await self._session.flush()
        return True

    async def delete(self, experience_id: uuid.UUID) -> bool:
        """Hard-delete an experience by ID (permanent).

        If the experience is a parent, CASCADE will delete children too.

        Returns:
            True if deleted, False if not found.
        """
        experience = await self.get_by_id(experience_id, include_deleted=True)
        if experience is None:
            return False
        await self._session.delete(experience)
        await self._session.flush()
        return True

    async def delete_group(self, root_id: uuid.UUID) -> bool:
        """Hard-delete a parent and all its children.

        Returns:
            True if the root was found and deleted, False otherwise.
        """
        root = await self.get_with_children(root_id, include_deleted=True)
        if root is None:
            return False
        # Children are CASCADE-deleted via FK
        await self._session.delete(root)
        await self._session.flush()
        return True

    # ======================== FEEDBACK ========================

    async def add_feedback(
        self,
        experience_id: uuid.UUID,
        rating: int,
        feedback_by: str,
        comment: str | None = None,
        fitness_score: int | None = None,
    ) -> ExperienceFeedback:
        """Add feedback to an experience. rating: 1-5, 5=best."""
        feedback = ExperienceFeedback(
            experience_id=experience_id,
            rating=rating,
            fitness_score=fitness_score,
            comment=comment,
            feedback_by=feedback_by,
        )
        self._session.add(feedback)
        await self._session.flush()
        await self._update_avg_rating(experience_id)
        return feedback

    async def _update_avg_rating(self, experience_id: uuid.UUID) -> None:
        """Recalculate and update the average rating for an experience."""
        result = await self._session.execute(
            select(func.avg(ExperienceFeedback.rating.cast(Float)))
            .where(ExperienceFeedback.experience_id == experience_id)
        )
        avg = result.scalar_one_or_none()
        if avg is not None:
            await self._session.execute(
                update(Experience)
                .where(Experience.id == experience_id)
                .values(avg_rating=float(avg))
            )

    # ======================== QUERY LOG ========================

    async def log_query(
        self,
        query: str,
        user_name: str,
        source: str = "mcp",
        result_count: int = 0,
        search_type: str = "vector",
        duration_ms: int | None = None,
    ) -> None:
        """Record a search query for analytics."""
        log = QueryLog(
            query=query,
            user_name=user_name,
            source=source,
            result_count=result_count,
            search_type=search_type,
            duration_ms=duration_ms,
        )
        self._session.add(log)
        await self._session.flush()

    async def get_query_logs(
        self, limit: int = 100, offset: int = 0
    ) -> list[dict]:
        """Get recent query logs."""
        result = await self._session.execute(
            select(QueryLog)
            .order_by(desc(QueryLog.created_at))
            .limit(limit)
            .offset(offset)
        )
        logs = result.scalars().all()
        return [
            {
                "id": log.id,
                "query": log.query,
                "user_name": log.user_name,
                "source": log.source,
                "result_count": log.result_count,
                "search_type": log.search_type,
                "duration_ms": log.duration_ms,
                "created_at": log.created_at.isoformat() if log.created_at else None,
            }
            for log in logs
        ]

    async def get_query_stats(self) -> dict:
        """Get query analytics summary."""
        total_queries = await self._session.execute(
            select(func.count()).select_from(QueryLog)
        )
        total = total_queries.scalar_one()

        recent_query = text("""
            SELECT COUNT(*)
            FROM query_logs
            WHERE created_at > NOW() - INTERVAL '7 days'
        """)
        recent_result = await self._session.execute(recent_query)
        recent = recent_result.scalar_one()

        type_dist = await self._session.execute(
            select(QueryLog.search_type, func.count())
            .group_by(QueryLog.search_type)
        )
        type_distribution = {row[0]: row[1] for row in type_dist.all()}

        return {
            "total_queries": total,
            "recent_7days": recent,
            "search_type_distribution": type_distribution,
        }

    # ======================== VERSION HISTORY (B3) ========================

    async def save_version(
        self,
        experience_id: uuid.UUID,
        changed_by: str,
        change_summary: str | None = None,
    ) -> ExperienceVersion:
        """Save a JSONB snapshot of the current experience state.

        Automatically increments version_number based on existing versions.
        """
        # Get the current experience
        exp = await self.get_with_children(experience_id, include_deleted=True)
        if exp is None:
            raise ValueError(f"Experience {experience_id} not found")

        # Get the next version number
        result = await self._session.execute(
            select(func.coalesce(func.max(ExperienceVersion.version_number), 0))
            .where(ExperienceVersion.experience_id == experience_id)
        )
        max_ver = result.scalar_one()

        # Build snapshot (all content fields, not embedding)
        snapshot = {
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
            "avg_rating": exp.avg_rating,
            "use_count": exp.use_count,
            "view_count": exp.view_count,
        }
        # Include children snapshots if any
        if exp.children:
            snapshot["children"] = [
                {
                    "id": str(c.id),
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

        version = ExperienceVersion(
            id=uuid.uuid4(),
            experience_id=experience_id,
            version_number=max_ver + 1,
            snapshot=snapshot,
            changed_by=changed_by,
            change_summary=change_summary,
        )
        self._session.add(version)
        await self._session.flush()
        return version

    async def get_versions(
        self, experience_id: uuid.UUID
    ) -> list[ExperienceVersion]:
        """Get all version snapshots for an experience, newest first."""
        result = await self._session.execute(
            select(ExperienceVersion)
            .where(ExperienceVersion.experience_id == experience_id)
            .order_by(desc(ExperienceVersion.version_number))
        )
        return list(result.scalars().all())

    async def get_version_detail(
        self, version_id: uuid.UUID
    ) -> ExperienceVersion | None:
        """Get a single version by its ID."""
        result = await self._session.execute(
            select(ExperienceVersion)
            .where(ExperienceVersion.id == version_id)
        )
        return result.scalar_one_or_none()

    # ======================== STALE DETECTION (B1) ========================

    async def scan_stale(
        self,
        months: int = 6,
        current_user: str | None = None,
    ) -> list[Experience]:
        """Find root/standalone experiences not used in the last N months.

        Returns experiences where last_used_at < now() - N months.
        Only considers non-deleted, published root experiences.
        """
        cutoff = text(f"NOW() - INTERVAL '{months} months'")
        query = (
            select(Experience)
            .where(Experience.parent_id.is_(None))
            .where(
                Experience.last_used_at < cutoff
            )
            .order_by(Experience.last_used_at.asc())
        )
        for f in self._active_filter(current_user):
            query = query.where(f)
        result = await self._session.execute(query)
        return list(result.scalars().all())

    async def count_stale(
        self,
        months: int = 6,
        current_user: str | None = None,
    ) -> int:
        """Count stale experiences."""
        cutoff = text(f"NOW() - INTERVAL '{months} months'")
        query = (
            select(func.count())
            .select_from(Experience)
            .where(Experience.parent_id.is_(None))
            .where(Experience.last_used_at < cutoff)
        )
        for f in self._active_filter(current_user):
            query = query.where(f)
        result = await self._session.execute(query)
        return result.scalar_one()

    # ======================== DEDUPLICATION (B2) ========================

    async def find_duplicates(
        self,
        threshold: float = 0.92,
        limit: int = 20,
    ) -> list[dict]:
        """Find near-duplicate experience pairs based on embedding similarity.

        Uses cosine similarity via pgvector. Only compares root/standalone
        experiences that are published and not deleted.

        Returns list of dicts: {exp_a, exp_b, similarity}
        """
        # Raw SQL for self-join with cosine similarity
        sql = text("""
            SELECT
                a.id AS a_id,
                b.id AS b_id,
                1 - (a.embedding <=> b.embedding) AS similarity
            FROM experiences a
            JOIN experiences b ON a.id < b.id
            WHERE a.embedding IS NOT NULL
              AND b.embedding IS NOT NULL
              AND a.is_deleted = false
              AND b.is_deleted = false
              AND a.exp_status = 'published'
              AND b.exp_status = 'published'
              AND a.parent_id IS NULL
              AND b.parent_id IS NULL
              AND 1 - (a.embedding <=> b.embedding) > :threshold
            ORDER BY similarity DESC
            LIMIT :lim
        """)
        result = await self._session.execute(
            sql, {"threshold": threshold, "lim": limit}
        )
        rows = result.all()

        pairs = []
        for a_id, b_id, sim in rows:
            exp_a = await self.get_by_id(a_id)
            exp_b = await self.get_by_id(b_id)
            if exp_a and exp_b:
                pairs.append({
                    "exp_a": exp_a.to_dict(),
                    "exp_b": exp_b.to_dict(),
                    "similarity": round(float(sim), 4),
                })
        return pairs

    async def merge_experiences(
        self,
        primary_id: uuid.UUID,
        secondary_id: uuid.UUID,
    ) -> Experience | None:
        """Merge secondary experience into primary.

        1. Union tags
        2. Migrate feedbacks from secondary to primary
        3. Accumulate use_count and view_count
        4. Recalculate avg_rating
        5. Hard-delete secondary (and its children via CASCADE)

        Returns the updated primary experience.
        """
        primary = await self.get_with_children(primary_id, include_deleted=True)
        secondary = await self.get_with_children(secondary_id, include_deleted=True)
        if primary is None or secondary is None:
            return None

        # 1. Union tags
        primary_tags = set(primary.tags or [])
        secondary_tags = set(secondary.tags or [])
        primary.tags = list(primary_tags | secondary_tags)

        # 2. Migrate feedbacks
        await self._session.execute(
            update(ExperienceFeedback)
            .where(ExperienceFeedback.experience_id == secondary_id)
            .values(experience_id=primary_id)
        )

        # 3. Accumulate counts
        primary.use_count += secondary.use_count
        primary.view_count += secondary.view_count

        # 4. Update last_used_at to the more recent one
        if secondary.last_used_at and (
            not primary.last_used_at or secondary.last_used_at > primary.last_used_at
        ):
            primary.last_used_at = secondary.last_used_at

        primary.updated_at = datetime.now(timezone.utc)
        await self._session.flush()

        # 5. Recalculate avg_rating for primary
        await self._update_avg_rating(primary_id)

        # 6. Hard-delete secondary
        await self._session.delete(secondary)
        await self._session.flush()

        # Re-fetch to get updated state
        return await self.get_with_children(primary_id)

    # ======================== BULK OPERATIONS (B4) ========================

    async def bulk_create(
        self,
        experiences_data: list[dict],
    ) -> list[Experience]:
        """Create multiple experiences in one transaction.

        Each dict should contain all fields needed for Experience creation.
        Returns the list of created Experience objects.
        """
        created = []
        for data in experiences_data:
            children_data = data.pop("children", None)
            exp = Experience(
                id=uuid.uuid4(),
                **data,
            )
            self._session.add(exp)
            await self._session.flush()

            if children_data:
                for child_data in children_data:
                    child_data.pop("children", None)
                    child = Experience(
                        id=uuid.uuid4(),
                        parent_id=exp.id,
                        **child_data,
                    )
                    self._session.add(child)

            created.append(exp)

        await self._session.flush()
        return created

    async def export_filtered(
        self,
        tag: str | None = None,
        start: datetime | None = None,
        end: datetime | None = None,
        current_user: str | None = None,
    ) -> list[Experience]:
        """Export experiences filtered by tag and/or time range.

        Only returns root/standalone, non-deleted, published experiences
        with children eagerly loaded.
        """
        query = (
            select(Experience)
            .where(Experience.parent_id.is_(None))
            .options(selectinload(Experience.children))
            .order_by(desc(Experience.created_at))
        )
        for f in self._active_filter(current_user):
            query = query.where(f)

        if tag:
            query = query.where(Experience.tags.overlap([tag]))
        if start:
            query = query.where(Experience.created_at >= start)
        if end:
            query = query.where(Experience.created_at <= end)

        result = await self._session.execute(query)
        return list(result.scalars().all())

    # ======================== STATS ========================

    async def get_stats(
        self,
        project: str | None = None,
        scope: str | None = None,
        current_user: str | None = None,
    ) -> dict:
        """Get aggregate statistics about the experience database."""
        proj = self._project_value(project)
        total = await self.count(
            project=project, scope=scope, current_user=current_user
        )

        all_total_result = await self._session.execute(
            text("""
                SELECT COUNT(*) FROM experiences
                WHERE is_deleted = false AND exp_status = 'published'
                  AND parent_id IS NULL
            """)
        )
        all_total = all_total_result.scalar_one()

        # Map old scope param to new visibility values for raw SQL
        vis_map = {"personal": "private", "team": "project", "global": "global"}
        vis_val = vis_map.get(scope, scope) if scope else None

        if vis_val and vis_val != "all":
            if vis_val == "global":
                scope_clause = "visibility = 'global'"
            elif vis_val == "private":
                scope_clause = (
                    "(visibility = 'private' AND created_by = :cuser)"
                    if current_user else "false"
                )
            else:
                scope_clause = "(project = :proj AND visibility = 'project')"
        else:
            scope_clause = "(project = :proj OR visibility = 'global')"
        params: dict = {"proj": proj}
        if current_user:
            params["cuser"] = current_user

        tag_query = text(f"""
            SELECT tag, COUNT(*) as cnt
            FROM experiences, unnest(tags) AS tag
            WHERE is_deleted = false AND exp_status = 'published'
              AND parent_id IS NULL
              AND {scope_clause}
            GROUP BY tag
            ORDER BY cnt DESC
            LIMIT 20
        """)
        tag_result = await self._session.execute(tag_query, params)
        tag_distribution = {row[0]: row[1] for row in tag_result.all()}

        recent_query = text(f"""
            SELECT COUNT(*)
            FROM experiences
            WHERE created_at > NOW() - INTERVAL '7 days'
              AND is_deleted = false AND exp_status = 'published'
              AND parent_id IS NULL
              AND {scope_clause}
        """)
        recent_result = await self._session.execute(recent_query, params)
        recent_count = recent_result.scalar_one()

        pending_query = text(f"""
            SELECT COUNT(*)
            FROM experiences
            WHERE exp_status = 'review' AND is_deleted = false
              AND parent_id IS NULL
              AND {scope_clause}
        """)
        pending_result = await self._session.execute(pending_query, params)
        pending_count = pending_result.scalar_one()

        stale_query = text(f"""
            SELECT COUNT(*)
            FROM experiences
            WHERE last_used_at < NOW() - INTERVAL '6 months'
              AND is_deleted = false AND exp_status = 'published'
              AND parent_id IS NULL
              AND {scope_clause}
        """)
        stale_result = await self._session.execute(stale_query, params)
        stale_count = stale_result.scalar_one()

        return {
            "total_experiences": total,
            "total_all_projects": all_total,
            "tag_distribution": tag_distribution,
            "recent_7days": recent_count,
            "pending_reviews": pending_count,
            "stale_count": stale_count,
        }

    async def list_projects(self) -> list[str]:
        """Return distinct project names."""
        result = await self._session.execute(
            text("""
                SELECT DISTINCT project FROM experiences
                WHERE is_deleted = false
                ORDER BY project
            """)
        )
        return [row[0] for row in result.all()]


class TaskRepository:
    """Repository for TaskGroup and PersonalTask CRUD."""

    WIP_LIMIT = 5

    def __init__(self, session: AsyncSession):
        self._session = session

    @staticmethod
    def _project_value(project: str | None) -> str:
        """Normalize project value for filtering/writes."""
        if project and project.strip():
            value = project.strip()
            alias_map = {
                "team-memory": "team_memory",
                "team_doc": "team_memory",
            }
            return alias_map.get(value, value)
        return "default"

    # ---------- TaskGroup ----------

    async def create_group(
        self,
        title: str,
        user_id: str,
        project: str = "default",
        description: str | None = None,
        source_doc: str | None = None,
        content_hash: str | None = None,
    ) -> TaskGroup:
        group = TaskGroup(
            title=title,
            user_id=user_id,
            project=self._project_value(project),
            description=description,
            source_doc=source_doc,
            content_hash=content_hash,
        )
        self._session.add(group)
        await self._session.flush()
        return group

    async def get_group(self, group_id: uuid.UUID) -> TaskGroup | None:
        result = await self._session.execute(
            select(TaskGroup)
            .options(selectinload(TaskGroup.tasks))
            .where(TaskGroup.id == group_id)
        )
        return result.scalar_one_or_none()

    async def get_group_by_source(self, source_doc: str) -> TaskGroup | None:
        result = await self._session.execute(
            select(TaskGroup).where(TaskGroup.source_doc == source_doc)
        )
        return result.scalar_one_or_none()

    async def list_groups(
        self,
        project: str = "default",
        user_id: str | None = None,
        include_archived: bool = False,
    ) -> list[TaskGroup]:
        q = (
            select(TaskGroup)
            .options(selectinload(TaskGroup.tasks))
            .where(TaskGroup.project == self._project_value(project))
        )
        if user_id:
            q = q.where(TaskGroup.user_id == user_id)
        if not include_archived:
            q = q.where(TaskGroup.archived == False)  # noqa: E712
        q = q.order_by(TaskGroup.sort_order, TaskGroup.created_at)
        result = await self._session.execute(q)
        return list(result.scalars().all())

    async def update_group(
        self, group_id: uuid.UUID, **kwargs
    ) -> TaskGroup | None:
        result = await self._session.execute(
            select(TaskGroup).where(TaskGroup.id == group_id)
        )
        group = result.scalar_one_or_none()
        if not group:
            return None
        for k, v in kwargs.items():
            if hasattr(group, k):
                setattr(group, k, v)
        await self._session.flush()
        return group

    async def delete_group(self, group_id: uuid.UUID) -> bool:
        result = await self._session.execute(
            select(TaskGroup).where(TaskGroup.id == group_id)
        )
        group = result.scalar_one_or_none()
        if not group:
            return False
        await self._session.execute(
            delete(PersonalTask).where(PersonalTask.group_id == group_id)
        )
        await self._session.delete(group)
        await self._session.flush()
        return True

    # ---------- PersonalTask ----------

    async def create_task(
        self,
        title: str,
        user_id: str,
        project: str = "default",
        group_id: uuid.UUID | None = None,
        description: str | None = None,
        status: str = "wait",
        priority: str = "medium",
        importance: int = 3,
        due_date: datetime | None = None,
        labels: list[str] | None = None,
        experience_id: uuid.UUID | None = None,
    ) -> PersonalTask:
        task = PersonalTask(
            title=title,
            user_id=user_id,
            project=self._project_value(project),
            group_id=group_id,
            description=description,
            status=status,
            priority=priority,
            importance=importance,
            due_date=due_date,
            labels=labels or [],
            experience_id=experience_id,
        )
        self._session.add(task)
        await self._session.flush()
        return task

    async def get_task(self, task_id: uuid.UUID) -> PersonalTask | None:
        result = await self._session.execute(
            select(PersonalTask).where(PersonalTask.id == task_id)
        )
        return result.scalar_one_or_none()

    async def delete_task(self, task_id: uuid.UUID) -> bool:
        task = await self.get_task(task_id)
        if not task:
            return False
        await self._session.delete(task)
        await self._session.flush()
        return True

    async def update_task(self, task_id: uuid.UUID, **kwargs) -> PersonalTask | None:
        task = await self.get_task(task_id)
        if not task:
            return None
        for k, v in kwargs.items():
            if hasattr(task, k) and v is not None:
                setattr(task, k, v)
        await self._session.flush()
        return task

    async def list_tasks(
        self,
        project: str = "default",
        user_id: str | None = None,
        status: str | None = None,
        group_id: uuid.UUID | None = None,
    ) -> list[PersonalTask]:
        q = select(PersonalTask).where(
            PersonalTask.project == self._project_value(project)
        )
        if user_id:
            q = q.where(PersonalTask.user_id == user_id)
        if status:
            q = q.where(PersonalTask.status == status)
        if group_id:
            q = q.where(PersonalTask.group_id == group_id)
        q = q.order_by(
            desc(PersonalTask.importance),
            PersonalTask.due_date.asc().nullslast(),
            PersonalTask.sort_order,
        )
        result = await self._session.execute(q)
        return list(result.scalars().all())

    async def count_in_progress(self, project: str, user_id: str) -> int:
        result = await self._session.execute(
            select(func.count())
            .select_from(PersonalTask)
            .where(PersonalTask.project == self._project_value(project))
            .where(PersonalTask.user_id == user_id)
            .where(PersonalTask.status == "in_progress")
        )
        return result.scalar_one()

    async def check_wip(self, project: str, user_id: str) -> tuple[bool, int]:
        """Return (within_limit, current_count)."""
        count = await self.count_in_progress(project, user_id)
        return count < self.WIP_LIMIT, count

    # ---------- Task Dependencies ----------

    async def add_dependency(
        self,
        source_id: uuid.UUID,
        target_id: uuid.UUID,
        dep_type: str = "blocks",
        created_by: str | None = None,
    ) -> TaskDependency:
        dep = TaskDependency(
            source_task_id=source_id,
            target_task_id=target_id,
            dep_type=dep_type,
            created_by=created_by,
        )
        self._session.add(dep)
        await self._session.flush()
        return dep

    async def remove_dependency(
        self, source_id: uuid.UUID, target_id: uuid.UUID
    ) -> bool:
        result = await self._session.execute(
            delete(TaskDependency).where(
                TaskDependency.source_task_id == source_id,
                TaskDependency.target_task_id == target_id,
            )
        )
        return result.rowcount > 0

    async def get_dependencies(self, task_id: uuid.UUID) -> list[TaskDependency]:
        result = await self._session.execute(
            select(TaskDependency).where(
                or_(
                    TaskDependency.source_task_id == task_id,
                    TaskDependency.target_task_id == task_id,
                )
            )
        )
        return list(result.scalars().all())

    async def get_ready_tasks(
        self, project: str, user_id: str | None = None,
    ) -> list[PersonalTask]:
        """Return tasks with no unresolved blocking dependencies."""
        blocked_ids = (
            select(TaskDependency.target_task_id)
            .join(
                PersonalTask,
                PersonalTask.id == TaskDependency.source_task_id,
            )
            .where(TaskDependency.dep_type == "blocks")
            .where(PersonalTask.status.notin_(["completed", "cancelled"]))
        ).scalar_subquery()

        q = (
            select(PersonalTask)
            .where(PersonalTask.project == self._project_value(project))
            .where(PersonalTask.status.in_(["wait", "plan"]))
            .where(PersonalTask.id.notin_(blocked_ids))
        )
        if user_id:
            q = q.where(PersonalTask.user_id == user_id)
        q = q.order_by(
            desc(PersonalTask.importance),
            PersonalTask.due_date.asc().nullslast(),
        )
        result = await self._session.execute(q)
        return list(result.scalars().all())

    # ---------- Atomic Claim ----------

    async def claim_task(
        self, task_id: uuid.UUID, assignee: str, project: str,
    ) -> PersonalTask | None:
        """Atomically claim a task using SELECT FOR UPDATE."""
        result = await self._session.execute(
            select(PersonalTask)
            .where(PersonalTask.id == task_id)
            .with_for_update()
        )
        task = result.scalar_one_or_none()
        if not task:
            return None
        if task.assignee and task.assignee != assignee:
            raise ValueError(f"Task already claimed by {task.assignee}")

        within_limit, count = await self.check_wip(project, assignee)
        if not within_limit:
            raise ValueError(f"WIP limit reached ({count}/{self.WIP_LIMIT})")

        task.assignee = assignee
        task.claimed_at = _utcnow()
        task.status = "in_progress"
        await self._session.flush()
        return task

    async def unclaim_task(self, task_id: uuid.UUID) -> PersonalTask | None:
        task = await self.get_task(task_id)
        if not task:
            return None
        task.assignee = None
        task.claimed_at = None
        if task.status == "in_progress":
            task.status = "plan"
        await self._session.flush()
        return task

    # ---------- Task Messages ----------

    async def add_message(
        self,
        task_id: uuid.UUID,
        author: str,
        content: str,
        thread_id: uuid.UUID | None = None,
    ) -> TaskMessage:
        msg = TaskMessage(
            task_id=task_id,
            author=author,
            content=content,
            thread_id=thread_id,
        )
        self._session.add(msg)
        await self._session.flush()
        return msg

    async def list_messages(self, task_id: uuid.UUID) -> list[TaskMessage]:
        result = await self._session.execute(
            select(TaskMessage)
            .where(TaskMessage.task_id == task_id)
            .order_by(TaskMessage.created_at)
        )
        return list(result.scalars().all())

    # ---------- Task Dedup ----------

    async def find_duplicate_tasks(
        self, project: str,
    ) -> list[tuple[str, list[PersonalTask]]]:
        """Return groups of tasks sharing the same content_hash."""
        q = (
            select(PersonalTask)
            .where(PersonalTask.project == self._project_value(project))
            .where(PersonalTask.content_hash.isnot(None))
            .order_by(PersonalTask.content_hash, PersonalTask.created_at)
        )
        result = await self._session.execute(q)
        tasks = list(result.scalars().all())

        from itertools import groupby
        groups = []
        for key, grp in groupby(tasks, key=lambda t: t.content_hash):
            items = list(grp)
            if len(items) > 1:
                groups.append((key, items))
        return groups

"""Experience data repository — CRUD operations + vector/FTS search (MVP)."""

from __future__ import annotations

import logging
import uuid
from datetime import datetime, timezone

from sqlalchemy import desc, func, select, text, update
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from team_memory.storage.models import (
    Experience,
    ExperienceFeedback,
)

logger = logging.getLogger("team_memory.storage.repository")

UNBOUNDED_QUERY_LIMIT = 1000


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


def _fts_tokenized_text(title: str, description: str, solution: str | None) -> str:
    """Return concatenated jieba-tokenized text for FTS trigger."""
    from team_memory.services.tokenizer import tokenize

    parts = [tokenize(title or ""), tokenize(description or ""), tokenize(solution or "")]
    return " ".join(p for p in parts if p)


# Dedup scan (Web UI): keep pair only if child-title Jaccard ≥ this (see test_find_duplicates_*).
CHILD_TITLE_JACCARD_MIN = 0.2


def _pair_passes_child_title_filter(exp_a: Experience, exp_b: Experience) -> bool:
    """Exclude pairs of groups whose child titles barely overlap (same parent embedding trap)."""
    ch_a = list(exp_a.children or [])
    ch_b = list(exp_b.children or [])
    if not ch_a or not ch_b:
        return True
    ta = {(c.title or "").strip().lower() for c in ch_a if (c.title or "").strip()}
    tb = {(c.title or "").strip().lower() for c in ch_b if (c.title or "").strip()}
    if not ta or not tb:
        return True
    union = ta | tb
    if not union:
        return True
    jaccard = len(ta & tb) / len(union)
    return jaccard >= CHILD_TITLE_JACCARD_MIN


def _summarize_exp_for_dedup(exp: Experience) -> dict:
    ch = list(exp.children or [])
    previews = [(c.title or "")[:80] for c in ch[:5]]
    return {
        "id": str(exp.id),
        "title": exp.title,
        "project": exp.project,
        "children_count": len(ch),
        "children_preview": previews,
    }


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
        """Visibility filter: published for all; drafts for creator only."""
        from team_memory.storage.query_builders import active_filter

        return active_filter(current_user)

    @staticmethod
    def _project_value(project: str | None) -> str:
        """Normalize project value for filtering/writes."""
        from team_memory.storage.query_builders import project_value

        return project_value(project)

    @staticmethod
    def _apply_scope_filter(query, project: str, scope: str | None, current_user: str | None):
        """Apply visibility-aware project filter."""
        from team_memory.storage.query_builders import apply_scope_filter

        return apply_scope_filter(query, project, scope, current_user)

    # Valid status transitions
    VALID_STATUS_TRANSITIONS = {
        "draft": ["published"],
        "published": ["draft"],
    }

    # ======================== CREATE ========================

    async def create(
        self,
        title: str,
        description: str,
        solution: str | None = None,
        created_by: str = "system",
        tags: list[str] | None = None,
        embedding: list[float] | None = None,
        source: str = "manual",
        parent_id: uuid.UUID | None = None,
        experience_type: str = "general",
        project: str = "default",
        group_key: str | None = None,
        visibility: str = "project",
        exp_status: str = "published",
    ) -> Experience:
        """Create a new experience record."""
        experience = Experience(
            id=uuid.uuid4(),
            title=title,
            description=description,
            solution=solution,
            created_by=created_by,
            tags=tags or [],
            embedding=embedding,
            source=source,
            parent_id=parent_id,
            experience_type=experience_type,
            project=self._project_value(project),
            group_key=group_key,
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
        """Create a parent experience with children in one transaction."""
        parent_fields = {**parent_data}
        parent_fields.pop("created_by", None)
        parent = await self.create(created_by=created_by, **parent_fields)
        parent_project = parent.project
        for child_data in children_data:
            child_fields = {**child_data}
            child_fields.pop("created_by", None)
            if "project" not in child_fields:
                child_fields["project"] = parent_project
            await self.create(
                created_by=created_by,
                parent_id=parent.id,
                **child_fields,
            )
        return await self.get_with_children(parent.id)

    async def find_or_create_group_parent(
        self,
        project: str,
        group_key: str,
        created_by: str,
    ) -> Experience:
        """Find existing parent by group_key, or create a new one.

        Used for automatic grouping: experiences with the same group_key
        are attached as children of a shared parent.
        """
        proj = self._project_value(project)
        query = (
            select(Experience)
            .where(Experience.parent_id.is_(None))
            .where(Experience.group_key == group_key)
            .where(Experience.project == proj)
            .where(Experience.is_deleted == False)  # noqa: E712
        )
        result = await self._session.execute(query)
        parent = result.scalar_one_or_none()
        if parent is not None:
            return parent

        # Create a new parent with group_key as title
        return await self.create(
            title=group_key,
            description=f"Auto-created group: {group_key}",
            created_by=created_by,
            project=project,
            group_key=group_key,
            exp_status="published",
        )

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

    # ======================== LIST / COUNT ========================

    async def list_root_ids_with_children(self, project: str | None = None) -> list[uuid.UUID]:
        """Return root experience IDs that have at least one non-deleted child."""
        subq = (
            select(Experience.parent_id)
            .where(Experience.parent_id.isnot(None))
            .where(Experience.is_deleted == False)  # noqa: E712
            .distinct()
        )
        query = (
            select(Experience.id)
            .where(Experience.parent_id.is_(None))
            .where(Experience.id.in_(subq))
            .where(Experience.is_deleted == False)  # noqa: E712
            .where(Experience.project == self._project_value(project))
            .limit(UNBOUNDED_QUERY_LIMIT)
        )
        result = await self._session.execute(query)
        rows = [row[0] for row in result.all()]
        if len(rows) >= UNBOUNDED_QUERY_LIMIT:
            logger.warning(
                "list_root_ids_with_children hit %d limit for project=%s",
                UNBOUNDED_QUERY_LIMIT,
                project,
            )
        return rows

    async def list_recent(
        self,
        limit: int = 10,
        offset: int = 0,
        include_all_statuses: bool = False,
        project: str | None = None,
        scope: str | None = None,
        current_user: str | None = None,
    ) -> list[Experience]:
        """List root/standalone experiences ordered by creation time (newest first)."""
        proj = self._project_value(project)
        query = (
            select(Experience)
            .where(Experience.parent_id.is_(None))
            .options(selectinload(Experience.children))
            .order_by(desc(Experience.created_at))
        )
        query = self._apply_scope_filter(query, proj, scope, current_user)
        if not include_all_statuses:
            for f in self._active_filter(current_user):
                query = query.where(f)
        else:
            query = query.where(Experience.is_deleted == False)  # noqa: E712
        result = await self._session.execute(query.limit(limit).offset(offset))
        return list(result.scalars().all())

    async def count(
        self,
        include_deleted: bool = False,
        project: str | None = None,
        scope: str | None = None,
        current_user: str | None = None,
    ) -> int:
        """Count root/standalone experiences (excludes children)."""
        query = select(func.count()).select_from(Experience).where(Experience.parent_id.is_(None))
        proj = self._project_value(project)
        query = self._apply_scope_filter(query, proj, scope, current_user)
        if not include_deleted:
            for f in self._active_filter(current_user):
                query = query.where(f)
        result = await self._session.execute(query)
        return result.scalar_one()

    # ======================== SEARCH ========================

    async def search_by_vector(
        self,
        query_embedding: list[float],
        max_results: int = 5,
        min_similarity: float = 0.6,
        tags: list[str] | None = None,
        project: str | None = None,
        current_user: str | None = None,
        offset: int = 0,
    ) -> list[dict]:
        """Search experiences by vector similarity using pgvector."""
        from team_memory.storage.query_builders import build_vector_search

        stmt = build_vector_search(
            query_embedding,
            project=project,
            current_user=current_user,
            tags=tags,
            min_similarity=min_similarity,
            limit=max_results,
            offset=offset,
        )
        result = await self._session.execute(stmt)
        rows = result.all()

        results = []
        for exp, similarity in rows:
            data = exp.to_dict()
            data["similarity"] = round(float(similarity), 4)
            results.append(data)
        return results

    async def search_by_fts(
        self,
        query_text: str,
        max_results: int = 5,
        tags: list[str] | None = None,
        project: str | None = None,
        current_user: str | None = None,
        offset: int = 0,
    ) -> list[dict]:
        """Full-text search using PostgreSQL tsvector/tsquery with jieba."""
        from team_memory.services.tokenizer import tokenize
        from team_memory.storage.query_builders import build_fts_search

        tokenized_query = tokenize(query_text)
        stmt = build_fts_search(
            tokenized_query,
            project=project,
            current_user=current_user,
            tags=tags,
            limit=max_results,
            offset=offset,
        )
        result = await self._session.execute(stmt)
        rows = result.all()

        results = []
        for exp, rank in rows:
            data = exp.to_dict()
            data["fts_rank"] = round(float(rank), 4)
            results.append(data)
        return results

    async def check_similar(
        self,
        embedding: list[float],
        threshold: float = 0.90,
        limit: int = 5,
        project: str | None = None,
        current_user: str | None = None,
    ) -> list[dict]:
        """Quick similarity check for dedup-on-save."""
        similarity_expr = (1 - Experience.embedding.cosine_distance(embedding)).label("similarity")

        query = (
            select(Experience, similarity_expr)
            .where(Experience.embedding.is_not(None))
            .where(Experience.parent_id.is_(None))
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

    async def find_duplicates(
        self,
        threshold: float = 0.9,
        limit: int = 50,
        project: str | None = None,
    ) -> list[dict]:
        """Find published root pairs with cosine similarity >= threshold (pgvector <=>).

        Applies a child-title Jaccard filter when both roots have children, so groups
        that share a parent embedding but differ in substance are not listed.
        """
        from team_memory.storage.query_builders import build_duplicate_detection, project_value

        proj = project_value(project)
        pair_cap = min(max(limit * 8, limit), 500)

        sql = build_duplicate_detection(project=project, threshold=threshold, pair_cap=pair_cap)
        result = await self._session.execute(
            sql,
            {
                "proj": proj,
                "threshold": float(threshold),
                "pair_cap": int(pair_cap),
            },
        )
        raw_pairs = result.all()
        if not raw_pairs:
            return []

        ids: set[uuid.UUID] = set()
        for row in raw_pairs:
            ids.add(uuid.UUID(str(row.id_a)))
            ids.add(uuid.UUID(str(row.id_b)))

        exps_result = await self._session.execute(
            select(Experience)
            .where(Experience.id.in_(ids))
            .options(selectinload(Experience.children))
        )
        by_id: dict[uuid.UUID, Experience] = {e.id: e for e in exps_result.scalars()}

        out: list[dict] = []
        for row in raw_pairs:
            if len(out) >= limit:
                break
            ua = uuid.UUID(str(row.id_a))
            ub = uuid.UUID(str(row.id_b))
            ea = by_id.get(ua)
            eb = by_id.get(ub)
            if ea is None or eb is None:
                continue
            if not _pair_passes_child_title_filter(ea, eb):
                continue
            sim_v = float(row.similarity)
            out.append(
                {
                    "similarity": round(sim_v, 4),
                    "exp_a": _summarize_exp_for_dedup(ea),
                    "exp_b": _summarize_exp_for_dedup(eb),
                }
            )
        return out

    # ======================== UPDATE ========================

    async def update(
        self,
        experience_id: uuid.UUID,
        **kwargs,
    ) -> Experience | None:
        """Update an experience record."""
        experience = await self.get_by_id(experience_id, include_deleted=True)
        if experience is None:
            return None

        for key, value in kwargs.items():
            if hasattr(experience, key):
                setattr(experience, key, value)

        experience.updated_at = datetime.now(timezone.utc)
        await self._session.flush()
        return experience

    async def change_status(
        self,
        experience_id: uuid.UUID,
        new_status: str,
        visibility: str | None = None,
        changed_by: str = "system",
        is_admin: bool = False,
    ) -> Experience | None:
        """Change experience status with validation."""
        experience = await self.get_by_id(experience_id, include_deleted=False)
        if experience is None:
            return None

        current_status = experience.exp_status
        if current_status not in ("draft", "published"):
            current_status = "draft"
        allowed = self.VALID_STATUS_TRANSITIONS.get(current_status, [])

        if is_admin and current_status == "draft" and new_status == "published":
            pass
        elif new_status not in allowed:
            raise ValueError(
                f"Cannot transition from '{current_status}' to '{new_status}'. Allowed: {allowed}"
            )

        experience.exp_status = new_status
        if visibility is not None:
            experience.visibility = visibility
        experience.updated_at = _utcnow()
        await self._session.flush()
        return experience

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
        """Hard-delete an experience (CASCADE deletes children)."""
        experience = await self.get_by_id(experience_id, include_deleted=True)
        if experience is None:
            return False
        await self._session.delete(experience)
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
        """Add feedback to an experience. rating: 1-5."""
        feedback = ExperienceFeedback(
            experience_id=experience_id,
            rating=rating,
            fitness_score=fitness_score,
            comment=comment,
            feedback_by=feedback_by,
        )
        self._session.add(feedback)
        await self._session.flush()
        return feedback

    # ======================== METRICS ========================

    async def increment_use_count(self, experience_id: uuid.UUID) -> None:
        """Increment use_count (implicit feedback on recall hit)."""
        await self._session.execute(
            update(Experience)
            .where(Experience.id == experience_id)
            .values(use_count=Experience.use_count + 1)
        )

    # ======================== PROJECT ========================

    async def list_projects(self) -> list[str]:
        """Return distinct project names."""
        result = await self._session.execute(
            text("""
                SELECT DISTINCT project FROM experiences
                WHERE is_deleted = false
                ORDER BY project
                LIMIT :lim
            """),
            {"lim": UNBOUNDED_QUERY_LIMIT},
        )
        rows = [row[0] for row in result.all()]
        if len(rows) >= UNBOUNDED_QUERY_LIMIT:
            logger.warning(
                "list_projects hit %d limit",
                UNBOUNDED_QUERY_LIMIT,
            )
        return rows

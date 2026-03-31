"""Experience data repository — CRUD operations + vector/FTS search (MVP)."""

from __future__ import annotations

import uuid
from datetime import datetime, timezone

from sqlalchemy import delete, desc, func, or_, select, text, update
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from team_memory.storage.models import (
    Experience,
    ExperienceFeedback,
    PersonalMemory,
)


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

    @staticmethod
    def _apply_scope_filter(
        query, project: str, scope: str | None, current_user: str | None
    ):
        """Apply visibility-aware project filter."""
        visibility = scope
        if visibility and visibility != "all":
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
        return query

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

    async def get_root_id(self, experience_id: uuid.UUID) -> uuid.UUID:
        """Find the root (topmost parent) of an experience."""
        current_id = experience_id
        for _ in range(10):
            exp = await self.get_by_id(current_id, include_deleted=True)
            if exp is None or exp.parent_id is None:
                return current_id
            current_id = exp.parent_id
        return current_id

    # ======================== LIST / COUNT ========================

    async def list_root_ids_with_children(
        self, project: str | None = None
    ) -> list[uuid.UUID]:
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
        )
        result = await self._session.execute(query)
        return [row[0] for row in result.all()]

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
            .where(Experience.parent_id.is_(None))
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

    # ======================== SEARCH ========================

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
            .where(Experience.project == self._project_value(project))
            .where(similarity_expr >= min_similarity)
            .order_by(desc(similarity_expr))
            .limit(max_results)
        )

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

    async def search_by_fts(
        self,
        query_text: str,
        max_results: int = 5,
        tags: list[str] | None = None,
        project: str | None = None,
        current_user: str | None = None,
    ) -> list[dict]:
        """Full-text search using PostgreSQL tsvector/tsquery with jieba."""
        from team_memory.services.tokenizer import tokenize

        tokenized_query = tokenize(query_text)
        ts_query = func.plainto_tsquery("simple", tokenized_query)
        rank_expr = func.ts_rank_cd(Experience.fts, ts_query).label("rank")

        query = (
            select(Experience, rank_expr)
            .where(Experience.fts.op("@@")(ts_query))
            .where(Experience.project == self._project_value(project))
            .order_by(desc(rank_expr))
            .limit(max_results)
        )

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

    async def check_similar(
        self,
        embedding: list[float],
        threshold: float = 0.90,
        limit: int = 5,
        project: str | None = None,
        current_user: str | None = None,
    ) -> list[dict]:
        """Quick similarity check for dedup-on-save."""
        similarity_expr = (
            1 - Experience.embedding.cosine_distance(embedding)
        ).label("similarity")

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
        """Find published root pairs with cosine similarity ≥ threshold (pgvector <=>).

        Applies a child-title Jaccard filter when both roots have children, so groups
        that share a parent embedding but differ in substance are not listed.
        """
        proj = self._project_value(project)
        pair_cap = min(max(limit * 8, limit), 500)

        sql = text(
            """
            SELECT a.id::text AS id_a, b.id::text AS id_b,
                   (1 - (a.embedding <=> b.embedding))::double precision AS similarity
            FROM experiences a
            INNER JOIN experiences b ON a.id < b.id
            WHERE a.parent_id IS NULL AND b.parent_id IS NULL
              AND a.embedding IS NOT NULL AND b.embedding IS NOT NULL
              AND a.is_deleted = false AND b.is_deleted = false
              AND a.project = :proj AND b.project = :proj
              AND a.exp_status = 'published' AND b.exp_status = 'published'
              AND (1 - (a.embedding <=> b.embedding)) >= :threshold
            ORDER BY similarity DESC
            LIMIT :pair_cap
            """
        )
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
                f"Cannot transition from '{current_status}' to '{new_status}'. "
                f"Allowed: {allowed}"
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

    async def delete_group(self, root_id: uuid.UUID) -> bool:
        """Hard-delete a parent and all its children."""
        root = await self.get_with_children(root_id, include_deleted=True)
        if root is None:
            return False
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
            """)
        )
        return [row[0] for row in result.all()]


# ============================================================
# PersonalMemoryRepository
# ============================================================

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
        similarity_expr = (
            1 - PersonalMemory.embedding.cosine_distance(context_embedding)
        ).label("similarity")
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

    async def get_by_id(
        self, memory_id: uuid.UUID, user_id: str
    ) -> PersonalMemory | None:
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
        similarity_expr = (
            1 - PersonalMemory.embedding.cosine_distance(embedding)
        ).label("similarity")
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

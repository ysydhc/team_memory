"""Query, listing, similarity search, and batch re-embedding for ExperienceService."""

from __future__ import annotations

import logging
import uuid

from sqlalchemy import select

from team_memory.services.infra.event_bus import Events
from team_memory.services.memory.experience.crud import _experience_repo_cls

logger = logging.getLogger("team_memory.service")


class ExperienceQueryMixin:
    """Mixin providing listing, filtering, similarity lookup, and batch re-embedding."""

    _REEMBED_BATCH_SIZE = 50

    async def reembed_group_parent_vectors(self, project: str | None) -> dict:
        """Recompute embeddings for parents that have children (group aggregate text).

        Collects all parent texts first, then batch-encodes in chunks of
        _REEMBED_BATCH_SIZE to reduce round-trips to the embedding provider.
        """
        repo_cls = _experience_repo_cls()
        proj_norm = repo_cls._project_value(project)
        updated = 0
        errors = 0
        total_groups = 0
        async with self._session() as session:
            repo = repo_cls(session)
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

        if self._event_bus:
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

    async def list_published(
        self,
        project: str | None = None,
        quality_tier: str | None = None,
        since: object | None = None,
        limit: int = 100,
        include_embedding: bool = False,
        active_only: bool = False,
    ) -> list[dict]:
        """List published/promoted experiences with optional filters.

        Args:
            project: Optional project name filter.
            quality_tier: Optional quality tier filter.
            since: Optional datetime — only return experiences created after this.
            limit: Maximum number of results (default 100).
            include_embedding: Whether to include raw vector embeddings in returned dicts.
            active_only: If True, filter out superseded experiences (valid_until IS NULL).

        Returns:
            List of experience dicts with exp_status in ('published', 'promoted').
        """
        from team_memory.storage.models import Experience

        async with self._session() as session:
            stmt = select(Experience).where(
                Experience.exp_status.in_(["published", "promoted"]),
                Experience.is_deleted == False,  # noqa: E712
            )
            if active_only:
                stmt = stmt.where(Experience.valid_until.is_(None))
            if project is not None:
                stmt = stmt.where(Experience.project == project)
            if quality_tier is not None:
                stmt = stmt.where(Experience.quality_tier == quality_tier)
            if since is not None:
                stmt = stmt.where(Experience.created_at >= since)
            stmt = stmt.order_by(Experience.created_at.desc()).limit(limit)
            result = await session.execute(stmt)
            rows = result.scalars().all()
            out: list[dict] = []
            for exp in rows:
                d = exp.to_dict()
                if include_embedding:
                    emb = getattr(exp, "embedding", None)
                    d["embedding"] = list(emb) if emb is not None else None
                out.append(d)
            return out

    async def list_low_adoption(
        self,
        *,
        min_recall_count: int = 5,
        max_used_count: int = 0,
        project: str | None = None,
        limit: int = 50,
        include_embedding: bool = True,
    ) -> list[dict]:
        """Sample active published experiences with high recall but zero/low adoption."""
        from team_memory.storage.models import Experience

        async with self._session() as session:
            stmt = select(Experience).where(
                Experience.exp_status.in_(["published", "promoted"]),
                Experience.is_deleted == False,  # noqa: E712
                Experience.valid_until.is_(None),
                Experience.recall_count >= min_recall_count,
                Experience.used_count <= max_used_count,
            )
            if project is not None:
                stmt = stmt.where(Experience.project == project)
            stmt = stmt.order_by(Experience.recall_count.desc()).limit(limit)
            result = await session.execute(stmt)
            rows = result.scalars().all()
            out: list[dict] = []
            for exp in rows:
                d = exp.to_dict()
                if include_embedding:
                    emb = getattr(exp, "embedding", None)
                    d["embedding"] = list(emb) if emb is not None else None
                out.append(d)
            return out

    async def list_stale_candidates(
        self,
        *,
        project: str | None = None,
        limit: int = 50,
        include_embedding: bool = True,
    ) -> list[dict]:
        """Sample active published/promoted experiences flagged as possibly_stale or stale."""
        from team_memory.storage.models import Experience

        async with self._session() as session:
            stmt = select(Experience).where(
                Experience.exp_status.in_(["published", "promoted"]),
                Experience.is_deleted == False,  # noqa: E712
                Experience.valid_until.is_(None),
                Experience.staleness_status.in_(["possibly_stale", "stale"]),
            )
            if project is not None:
                stmt = stmt.where(Experience.project == project)
            stmt = stmt.order_by(Experience.updated_at.desc()).limit(limit)
            result = await session.execute(stmt)
            rows = result.scalars().all()
            out: list[dict] = []
            for exp in rows:
                d = exp.to_dict()
                if include_embedding:
                    emb = getattr(exp, "embedding", None)
                    d["embedding"] = list(emb) if emb is not None else None
                out.append(d)
            return out

    async def find_similar_active(
        self,
        embedding: list[float],
        *,
        threshold: float = 0.78,
        limit: int = 10,
        project: str | None = None,
        include_cross_project: bool = False,
        exclude_ids: list[str] | None = None,
    ) -> list[dict]:
        """Find active published experiences by pgvector cosine similarity without search_logs pollution."""
        from sqlalchemy import desc

        from team_memory.storage.models import Experience

        similarity_expr = (1 - Experience.embedding.cosine_distance(embedding)).label("similarity")

        async with self._session() as session:
            stmt = (
                select(Experience, similarity_expr)
                .where(
                    Experience.exp_status.in_(["published", "promoted"]),
                    Experience.is_deleted == False,  # noqa: E712
                    Experience.valid_until.is_(None),
                    Experience.embedding.is_not(None),
                    similarity_expr >= threshold,
                )
                .order_by(desc(similarity_expr))
                .limit(limit)
            )
            if project is not None and not include_cross_project:
                stmt = stmt.where(Experience.project == project)
            if exclude_ids:
                valid_uuids = []
                for eid in exclude_ids:
                    try:
                        valid_uuids.append(uuid.UUID(str(eid)))
                    except (ValueError, TypeError):
                        continue
                if valid_uuids:
                    stmt = stmt.where(Experience.id.notin_(valid_uuids))

            result = await session.execute(stmt)
            out: list[dict] = []
            for exp, sim in result.all():
                d = exp.to_dict()
                d["similarity"] = round(float(sim), 4)
                out.append(d)
            return out

    async def count_published(
        self,
        project: str | None = None,
        since: object | None = None,
    ) -> int:
        """Count published/promoted experiences.

        Args:
            project: Optional project name filter.
            since: Optional datetime — only count experiences created after this.

        Returns:
            Integer count of matching published/promoted experiences.
        """
        from sqlalchemy import func

        from team_memory.storage.models import Experience

        async with self._session() as session:
            stmt = (
                select(func.count())
                .select_from(Experience)
                .where(
                    Experience.exp_status.in_(["published", "promoted"]),
                    Experience.is_deleted == False,  # noqa: E712
                )
            )
            if project is not None:
                stmt = stmt.where(Experience.project == project)
            if since is not None:
                stmt = stmt.where(Experience.created_at >= since)
            result = await session.execute(stmt)
            return result.scalar() or 0

    async def list_projects(self) -> list[str]:
        """Return distinct project names."""
        async with self._session() as session:
            repo = _experience_repo_cls()(session)
            return await repo.list_projects()

    async def list_experiences_filtered(
        self,
        *,
        limit: int,
        offset: int,
        tag: str | None = None,
        project: str | None = None,
        status: str | None = None,
        visibility: str | None = None,
        experience_type: str | None = None,
        current_user: str,
    ) -> dict:
        """Full filtered paginated list matching routes/experiences.py logic."""
        from sqlalchemy import func as sa_func
        from sqlalchemy import or_
        from sqlalchemy.orm import selectinload

        from team_memory.storage.models import Experience

        project_list = [p.strip() for p in project.split(",") if p.strip()] if project and "," in project else None
        resolved_project = self._resolve_project(project.split(",")[0] if project else project)

        def _project_filter(q):
            if project_list:
                return q.where(
                    or_(
                        Experience.project.in_(project_list),
                        Experience.visibility == "global",
                    )
                )
            return q.where(
                or_(
                    Experience.project == resolved_project,
                    Experience.visibility == "global",
                )
            )

        def _vis_where(q):
            if visibility and visibility != "all":
                vis_map = {"personal": "private", "team": "project", "global": "global"}
                vis_val = vis_map.get(visibility, visibility)
                q = q.where(Experience.visibility == vis_val)
                if vis_val == "private" and current_user:
                    q = q.where(Experience.created_by == current_user)
                if vis_val != "global":
                    q = _project_filter(q)
            else:
                q = _project_filter(q)
                if current_user:
                    q = q.where(
                        or_(
                            Experience.visibility != "private",
                            Experience.created_by == current_user,
                        )
                    )
                else:
                    q = q.where(Experience.visibility != "private")
            return q

        include_all = status in ("all", "draft")

        async with self._session() as session:
            repo = _experience_repo_cls()(session)

            if tag or experience_type or status:
                base_q = (
                    select(Experience)
                    .where(Experience.parent_id.is_(None))
                    .where(Experience.is_deleted == False)  # noqa: E712
                    .options(selectinload(Experience.children))
                )
                base_q = _vis_where(base_q)
                count_q = (
                    select(sa_func.count())
                    .select_from(Experience)
                    .where(Experience.parent_id.is_(None))
                    .where(Experience.is_deleted == False)  # noqa: E712
                )
                count_q = _vis_where(count_q)
                if status and status not in ("all",):
                    base_q = base_q.where(Experience.exp_status == status)
                    count_q = count_q.where(Experience.exp_status == status)
                elif not include_all:
                    base_q = base_q.where(Experience.exp_status == "published")
                    count_q = count_q.where(Experience.exp_status == "published")
                if tag:
                    base_q = base_q.where(Experience.tags.overlap([tag]))
                    count_q = count_q.where(Experience.tags.overlap([tag]))
                if experience_type:
                    base_q = base_q.where(Experience.experience_type == experience_type)
                    count_q = count_q.where(Experience.experience_type == experience_type)

                base_q = base_q.order_by(Experience.created_at.desc()).limit(limit).offset(offset)
                result = await session.execute(base_q)
                experiences = list(result.scalars().all())
                total_result = await session.execute(count_q)
                total = total_result.scalar() or 0
            else:
                total = await repo.count(
                    include_deleted=False if include_all else True,
                    project=resolved_project,
                    scope=visibility,
                    current_user=current_user,
                )
                experiences = await repo.list_recent(
                    limit=limit,
                    offset=offset,
                    include_all_statuses=include_all,
                    project=resolved_project,
                    scope=visibility,
                    current_user=current_user,
                )

            exp_list = []
            for exp in experiences:
                d = exp.to_dict()
                d["children_count"] = len(exp.children) if exp.children else 0
                exp_list.append(d)

            return {
                "items": exp_list,
                "total": total,
                "limit": limit,
                "offset": offset,
            }

    async def list_outdated(
        self,
        *,
        limit: int,
        offset: int,
        project: str | None = None,
    ) -> dict:
        """List Outdated-tier experiences for janitor route."""
        from team_memory.storage.models import Experience

        resolved_project = self._resolve_project(project) if project else None

        async with self._session() as session:
            query = (
                select(Experience)
                .where(Experience.quality_tier == "Outdated")
                .where(Experience.is_deleted == False)  # noqa: E712
                .order_by(Experience.last_scored_at.desc())
            )

            if resolved_project:
                query = query.where(Experience.project == resolved_project)

            count_query = (
                select(Experience.id).where(Experience.quality_tier == "Outdated").where(Experience.is_deleted == False)  # noqa: E712
            )
            if resolved_project:
                count_query = count_query.where(Experience.project == resolved_project)

            total_result = await session.execute(count_query)
            total = len(total_result.all())

            paginated_query = query.limit(limit).offset(offset)
            result = await session.execute(paginated_query)
            experiences = result.scalars().all()

            exp_list = []
            for exp in experiences:
                exp_dict = exp.to_dict()
                exp_dict["cleanup_info"] = {
                    "quality_score": exp.quality_score,
                    "quality_tier": exp.quality_tier,
                    "last_scored_at": exp.last_scored_at.isoformat() if exp.last_scored_at else None,
                    "is_pinned": exp.is_pinned,
                }
                exp_list.append(exp_dict)

            return {
                "items": exp_list,
                "total": total,
                "limit": limit,
                "offset": offset,
                "project": resolved_project,
                "message": f"Found {total} outdated experiences"
                + (f" in project '{resolved_project}'" if resolved_project else ""),
            }

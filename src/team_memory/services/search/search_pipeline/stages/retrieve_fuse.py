"""Vector, FTS retrieval and RRF fusion stage for SearchPipeline."""

from __future__ import annotations

import asyncio
import logging
from typing import Any

from team_memory.services.search.search_pipeline.models import (
    SearchRequest,
    SearchResultItem,
    _pkg_attr,
)
from team_memory.storage.database import get_session as _default_get_session
from team_memory.storage.repository import ExperienceRepository as _DefaultExperienceRepository

logger = logging.getLogger("team_memory.search_pipeline")


class SearchRetrieveFuseMixin:
    """Mixin implementing vector/FTS retrieval and 2/3/4-way RRF fusion."""

    async def _retrieve_and_fuse(
        self,
        repo: Any,
        request: SearchRequest,
        query_embedding: list[float] | None,
        mode: str,
        retrieval_query: str | None = None,
        effective_min_similarity: float | None = None,
        graph_weighted_hits: list[tuple[str, float]] | None = None,
        stage_counts: dict[str, int] | None = None,
        prefetched_fts: list[SearchResultItem] | None = None,
    ) -> list[SearchResultItem]:
        """Execute retrieval and RRF fusion."""
        q = (retrieval_query or request.query).strip()
        over_fetch = request.max_results * 5
        min_sim = effective_min_similarity if effective_min_similarity is not None else request.min_similarity
        enable_graph = getattr(self._search_config, "enable_graph", True) is not False

        if mode == "vector" and query_embedding:
            vec_res = await self._vector_search(repo, query_embedding, over_fetch, request, min_similarity=min_sim)
            if stage_counts is not None:
                stage_counts["vec_count"] = len(vec_res)
                stage_counts["fts_count"] = 0
                stage_counts["topic_count"] = 0
                stage_counts["graph_count"] = 0
                stage_counts["graph_unique_added"] = 0
                stage_counts["fused_count"] = len(vec_res)
            return vec_res

        if mode == "fts" or query_embedding is None:
            fts_res = (
                prefetched_fts
                if prefetched_fts is not None
                else await self._fts_search(repo, q, over_fetch, request)
            )
            graph_results: list[SearchResultItem] = []
            if enable_graph and graph_weighted_hits:
                existing_by_id = {
                    str(item.data.get("id")): item.data
                    for item in fts_res
                    if isinstance(item.data, dict) and item.data.get("id")
                }
                try:
                    graph_results = await self._graph_search(
                        repo,
                        graph_weighted_hits,
                        over_fetch,
                        request,
                        existing_by_id=existing_by_id,
                    )
                except Exception as e:
                    logger.warning("Graph search failed in FTS/degraded mode: %s", e, exc_info=True)
                    graph_results = []

            if graph_results:
                rank_profile = None
                if getattr(self._search_config, "adaptive_ranking", True):
                    from team_memory.services.search.scoring import resolve_rank_profile

                    rank_profile = resolve_rank_profile(request.query, request.intent_type)
                    if stage_counts is not None:
                        stage_counts["rank_intent"] = rank_profile.name
                fused_degraded = self._rrf_fuse4(
                    [],
                    fts_res,
                    [],
                    graph_results,
                    over_fetch,
                    rank_profile=rank_profile,
                )
                fts_ids = {
                    item.data.get("id") or item.data.get("group_id")
                    for item in fts_res
                    if item.data.get("id") or item.data.get("group_id")
                }
                g_ids = {
                    item.data.get("id") or item.data.get("group_id")
                    for item in graph_results
                    if item.data.get("id") or item.data.get("group_id")
                }
                if stage_counts is not None:
                    stage_counts["vec_count"] = 0
                    stage_counts["fts_count"] = len(fts_res)
                    stage_counts["topic_count"] = 0
                    stage_counts["graph_count"] = len(graph_results)
                    stage_counts["graph_unique_added"] = len(g_ids - fts_ids)
                    stage_counts["fused_count"] = len(fused_degraded)
                return fused_degraded

            if stage_counts is not None:
                stage_counts["vec_count"] = 0
                stage_counts["fts_count"] = len(fts_res)
                stage_counts["topic_count"] = 0
                stage_counts["graph_count"] = 0
                stage_counts["graph_unique_added"] = 0
                stage_counts["fused_count"] = len(fts_res)
            return fts_res

        # Hybrid mode: parallel vector + FTS (or pre-fetched FTS) + topic + graph
        enable_topic = getattr(self._search_config, "enable_topic", True) is not False
        vector_task = self._vector_search(repo, query_embedding, over_fetch, request, min_similarity=min_sim)
        fts_task = (
            asyncio.sleep(0, result=prefetched_fts)
            if prefetched_fts is not None
            else self._fts_search(repo, q, over_fetch, request)
        )
        topic_task = (
            self._topic_search(query_embedding, over_fetch, request)
            if enable_topic
            else asyncio.sleep(0, result=[])
        )

        vector_results, fts_results, topic_results = await asyncio.gather(
            vector_task, fts_task, topic_task, return_exceptions=True
        )

        if isinstance(vector_results, Exception):
            logger.warning("Vector search failed in hybrid mode: %s", vector_results, exc_info=True)
            vector_results = []
        if isinstance(fts_results, Exception):
            logger.warning("FTS failed in hybrid mode: %s", fts_results, exc_info=True)
            fts_results = []
        if isinstance(topic_results, Exception):
            logger.warning("Topic search failed in hybrid mode: %s", topic_results, exc_info=True)
            topic_results = []

        existing_by_id = {
            str(item.data.get("id")): item.data
            for item in (vector_results + fts_results + topic_results)
            if isinstance(item.data, dict) and item.data.get("id")
        }

        if enable_graph:
            try:
                graph_results = await self._graph_search(
                    repo,
                    graph_weighted_hits or [],
                    over_fetch,
                    request,
                    existing_by_id=existing_by_id,
                )
            except Exception as e:
                logger.warning("Graph search failed in hybrid mode: %s", e, exc_info=True)
                graph_results = []
        else:
            graph_results = []

        non_graph_ids = {
            item.data.get("id") or item.data.get("group_id")
            for item in (vector_results + fts_results + topic_results)
            if item.data.get("id") or item.data.get("group_id")
        }
        graph_ids = {
            item.data.get("id") or item.data.get("group_id")
            for item in graph_results
            if item.data.get("id") or item.data.get("group_id")
        }

        if stage_counts is not None:
            stage_counts["vec_count"] = len(vector_results)
            stage_counts["fts_count"] = len(fts_results)
            stage_counts["topic_count"] = len(topic_results)
            stage_counts["graph_count"] = len(graph_results)
            stage_counts["graph_unique_added"] = len(graph_ids - non_graph_ids)

        rank_profile = None
        if getattr(self._search_config, "adaptive_ranking", True):
            from team_memory.services.search.scoring import resolve_rank_profile

            rank_profile = resolve_rank_profile(request.query, request.intent_type)
            if stage_counts is not None:
                stage_counts["rank_intent"] = rank_profile.name

        if graph_results:
            fused = self._rrf_fuse4(
                vector_results,
                fts_results,
                topic_results,
                graph_results,
                over_fetch,
                rank_profile=rank_profile,
            )
        else:
            fused = self._rrf_fuse3(
                vector_results,
                fts_results,
                topic_results,
                over_fetch,
                rank_profile=rank_profile,
            )

        if stage_counts is not None:
            stage_counts["fused_count"] = len(fused)

        return fused

    async def _vector_search(
        self,
        repo: Any,
        query_embedding: list[float],
        limit: int,
        request: SearchRequest,
        min_similarity: float | None = None,
    ) -> list[SearchResultItem]:
        """Execute vector similarity search with optional cross-project fallback."""
        min_sim = min_similarity if min_similarity is not None else request.min_similarity
        prom_kw: dict[str, Any] = {}
        if getattr(request, "include_promoted", False):
            prom_kw["include_promoted"] = True
        if getattr(request, "include_superseded", False):
            prom_kw["include_superseded"] = True
        raw = await repo.search_by_vector(
            query_embedding=query_embedding,
            max_results=limit,
            min_similarity=min_sim,
            tags=request.tags,
            project=request.project,
            current_user=request.current_user,
            **prom_kw,
        )
        best_sim = max((float(r.get("similarity", 0.0)) for r in raw), default=0.0) if raw else 0.0
        is_default_proj = (request.project or "default") in ("default", "team_memory")
        should_fallback = bool(request.fallback_projects) and (
            not raw or is_default_proj or best_sim < 0.80
        )
        if should_fallback:
            seen_ids = {str(r.get("id")) for r in raw if r.get("id")}
            for fb_proj in request.fallback_projects:
                if fb_proj == request.project:
                    continue
                fb_raw = await repo.search_by_vector(
                    query_embedding=query_embedding,
                    max_results=limit,
                    min_similarity=min_sim,
                    tags=request.tags,
                    project=fb_proj,
                    current_user=request.current_user,
                    **prom_kw,
                )
                for r in fb_raw:
                    rid = str(r.get("id", ""))
                    if rid and rid not in seen_ids:
                        seen_ids.add(rid)
                        raw.append(r)
            raw.sort(key=lambda x: float(x.get("similarity", 0.0)), reverse=True)
            raw = raw[:limit]
        items = []
        for r in raw:
            items.append(
                SearchResultItem(
                    data=r,
                    score=r.get("similarity", 0),
                    similarity=r.get("similarity", 0),
                    source_type="vector",
                )
            )
        return items

    async def _fts_search(
        self,
        repo: Any,
        query_text: str,
        limit: int,
        request: SearchRequest,
    ) -> list[SearchResultItem]:
        """Execute full-text search.

        When self._db_url is set (hybrid mode), uses a separate session to avoid
        SQLAlchemy "concurrent operations not permitted" on the same session.
        """
        prom_kw: dict[str, Any] = {}
        if getattr(request, "include_promoted", False):
            prom_kw["include_promoted"] = True
        if getattr(request, "include_superseded", False):
            prom_kw["include_superseded"] = True
        is_default_proj = (request.project or "default") in ("default", "team_memory")
        get_session_fn = _pkg_attr("get_session", _default_get_session)
        exp_repo_cls = _pkg_attr("ExperienceRepository", _DefaultExperienceRepository)
        if self._db_url:
            async with get_session_fn(self._db_url) as session2:
                repo_fts = exp_repo_cls(session2)
                raw = await repo_fts.search_by_fts(
                    query_text=query_text,
                    max_results=limit,
                    tags=request.tags,
                    project=request.project,
                    current_user=request.current_user,
                    **prom_kw,
                )
                if request.fallback_projects and (not raw or is_default_proj):
                    seen_ids = {str(r.get("id")) for r in raw if r.get("id")}
                    for fb_proj in request.fallback_projects:
                        if fb_proj == request.project:
                            continue
                        fb_raw = await repo_fts.search_by_fts(
                            query_text=query_text,
                            max_results=limit,
                            tags=request.tags,
                            project=fb_proj,
                            current_user=request.current_user,
                            **prom_kw,
                        )
                        for r in fb_raw:
                            rid = str(r.get("id", ""))
                            if rid and rid not in seen_ids:
                                seen_ids.add(rid)
                                raw.append(r)
                    raw.sort(key=lambda x: float(x.get("fts_rank", 0.0)), reverse=True)
                    raw = raw[:limit]
        else:
            raw = await repo.search_by_fts(
                query_text=query_text,
                max_results=limit,
                tags=request.tags,
                project=request.project,
                current_user=request.current_user,
                **prom_kw,
            )
            if request.fallback_projects and (not raw or is_default_proj):
                seen_ids = {str(r.get("id")) for r in raw if r.get("id")}
                for fb_proj in request.fallback_projects:
                    if fb_proj == request.project:
                        continue
                    fb_raw = await repo.search_by_fts(
                        query_text=query_text,
                        max_results=limit,
                        tags=request.tags,
                        project=fb_proj,
                        current_user=request.current_user,
                        **prom_kw,
                    )
                    for r in fb_raw:
                        rid = str(r.get("id", ""))
                        if rid and rid not in seen_ids:
                            seen_ids.add(rid)
                            raw.append(r)
                raw.sort(key=lambda x: float(x.get("fts_rank", 0.0)), reverse=True)
                raw = raw[:limit]
        items = []
        for r in raw:
            items.append(
                SearchResultItem(
                    data=r,
                    score=r.get("fts_rank", 0),
                    fts_rank=r.get("fts_rank", 0),
                    source_type="fts",
                )
            )
        return items

    def _rrf_fuse(
        self,
        vector_results: list[SearchResultItem],
        fts_results: list[SearchResultItem],
        limit: int,
    ) -> list[SearchResultItem]:
        """Merge vector and FTS results using Reciprocal Rank Fusion."""
        from team_memory.services.search.scoring import rrf_fuse

        return rrf_fuse(
            vector_results,
            fts_results,
            limit,
            rrf_k=self._search_config.rrf_k,
            vector_weight=self._search_config.vector_weight,
            fts_weight=self._search_config.fts_weight,
        )

    def _rrf_fuse3(
        self,
        vector_results: list[SearchResultItem],
        fts_results: list[SearchResultItem],
        topic_results: list[SearchResultItem],
        limit: int,
        *,
        rank_profile: Any = None,
    ) -> list[SearchResultItem]:
        """Merge vector + FTS + topic results using 3-way RRF."""
        from team_memory.services.search.scoring import rrf_fuse3

        vec_w = rank_profile.vector_weight if rank_profile is not None else self._search_config.vector_weight
        fts_w = rank_profile.fts_weight if rank_profile is not None else self._search_config.fts_weight
        return rrf_fuse3(
            vector_results,
            fts_results,
            topic_results,
            limit,
            rrf_k=self._search_config.rrf_k,
            vector_weight=vec_w,
            fts_weight=fts_w,
        )

    def _rrf_fuse4(
        self,
        vector_results: list[SearchResultItem],
        fts_results: list[SearchResultItem],
        topic_results: list[SearchResultItem],
        graph_results: list[SearchResultItem],
        limit: int,
        *,
        rank_profile: Any = None,
    ) -> list[SearchResultItem]:
        """Merge vector + FTS + topic + graph results using 4-way RRF."""
        from team_memory.services.search.scoring import rrf_fuse4

        vec_w = rank_profile.vector_weight if rank_profile is not None else self._search_config.vector_weight
        fts_w = rank_profile.fts_weight if rank_profile is not None else self._search_config.fts_weight
        g_w = rank_profile.graph_weight if rank_profile is not None else None
        g_only = rank_profile.graph_only_factor if rank_profile is not None else 0.35
        g_confirm = rank_profile.graph_confirm_factor if rank_profile is not None else 0.05
        return rrf_fuse4(
            vector_results,
            fts_results,
            topic_results,
            graph_results,
            limit,
            rrf_k=self._search_config.rrf_k,
            vector_weight=vec_w,
            fts_weight=fts_w,
            graph_weight=g_w,
            graph_only_factor=g_only,
            graph_confirm_factor=g_confirm,
        )

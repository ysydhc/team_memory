"""SearchPipeline orchestrator composing all retrieval, ranking, and postprocessing stages."""

from __future__ import annotations

import logging
import time
from typing import TYPE_CHECKING, Any

from sqlalchemy.ext.asyncio import AsyncSession

from team_memory import io_logger
from team_memory.config import CacheConfig, RetrievalConfig, SearchConfig
from team_memory.embedding.base import EmbeddingProvider
from team_memory.services.search.cache import SearchCache
from team_memory.services.search.search_pipeline.models import (
    SearchPipelineResult,
    SearchRequest,
    _pkg_attr,
    coerce_cached_search_pipeline_result,
)
from team_memory.services.search.search_pipeline.stages.graph_topic import SearchGraphTopicMixin
from team_memory.services.search.search_pipeline.stages.postprocess import SearchPostprocessMixin
from team_memory.services.search.search_pipeline.stages.query_prep import (
    SearchQueryPrepMixin,
)
from team_memory.services.search.search_pipeline.stages.query_prep import (
    _llm_expand_query as _default_llm_expand_query,
)
from team_memory.services.search.search_pipeline.stages.ranking import SearchRankingMixin
from team_memory.services.search.search_pipeline.stages.retrieve_fuse import SearchRetrieveFuseMixin
from team_memory.storage.repository import ExperienceRepository as _DefaultExperienceRepository

if TYPE_CHECKING:
    from team_memory.reranker.base import RerankerProvider

logger = logging.getLogger("team_memory.search_pipeline")


class SearchPipeline(
    SearchQueryPrepMixin,
    SearchRetrieveFuseMixin,
    SearchGraphTopicMixin,
    SearchRankingMixin,
    SearchPostprocessMixin,
):
    """Orchestrates the complete search pipeline.

    Coordinates EmbeddingProvider, ExperienceRepository, SearchCache
    for hybrid retrieval (vector + FTS + RRF fusion).
    """

    def __init__(
        self,
        embedding_provider: EmbeddingProvider,
        search_config: SearchConfig,
        retrieval_config: RetrievalConfig,
        cache_config: CacheConfig,
        llm_config=None,
        tag_synonyms: dict[str, str] | None = None,
        db_url: str | None = None,
        reranker_provider: RerankerProvider | None = None,
        rerank_enabled: bool = False,
        reranker_signature: str = "none",
        rerank_max_document_chars: int = 8000,
    ):
        self._embedding = embedding_provider
        self._db_url = db_url
        self._search_config = search_config
        self._retrieval_config = retrieval_config
        self._llm_config = llm_config
        self._tag_synonyms = tag_synonyms or {}
        self._reranker = reranker_provider
        self._rerank_enabled = bool(rerank_enabled and reranker_provider is not None)
        self._reranker_signature = reranker_signature or "none"
        self._rerank_max_document_chars = max(256, int(rerank_max_document_chars))
        self._cache = SearchCache(
            max_size=cache_config.max_size,
            ttl_seconds=cache_config.ttl_seconds,
            embedding_cache_size=cache_config.embedding_cache_size,
            enabled=cache_config.enabled,
            backend=getattr(cache_config, "backend", "memory"),
            redis_url=getattr(cache_config, "redis_url", "redis://localhost:6379/0"),
        )

    async def search(
        self,
        session: AsyncSession,
        request: SearchRequest,
    ) -> SearchPipelineResult:
        """Execute the full search pipeline."""
        start = time.monotonic()
        stage_metrics: dict[str, int] = {}
        stage_counts: dict[str, int] = {}
        exp_repo_cls = _pkg_attr("ExperienceRepository", _DefaultExperienceRepository)
        llm_expand_fn = _pkg_attr("_llm_expand_query", _default_llm_expand_query)
        repo = exp_repo_cls(session)

        # Stage 1: Cache check
        stage_begin = time.monotonic()
        if self._cache.enabled:
            cached = await self._cache.get(
                request.query,
                request.tags,
                project=request.project,
                current_user=request.current_user,
                include_archives=request.include_archives,
                reranker_signature=self._reranker_signature,
            )
            if cached is not None:
                hit = coerce_cached_search_pipeline_result(cached)
                hit.cached = True
                hit.duration_ms = int((time.monotonic() - start) * 1000)
                return hit
        stage_metrics["cache_check_ms"] = int((time.monotonic() - stage_begin) * 1000)

        # Query expansion: tag synonyms (simple key<->value); LLM expansion defaults to fallback-only
        stage_begin = time.monotonic()
        retrieval_query = self._expand_query_synonyms(request.query)
        qe_enabled = bool(getattr(self._search_config, "query_expansion_enabled", False) and self._llm_config)
        qe_mode = getattr(self._search_config, "query_expansion_mode", "fallback")
        qe_timeout_s = float(getattr(self._search_config, "query_expansion_timeout_seconds", 3.0))
        qe_triggered = 0

        if qe_enabled and qe_mode == "always":
            expanded = await llm_expand_fn(request.query, self._llm_config, qe_timeout_s)
            if expanded and expanded.strip():
                retrieval_query = (request.query.strip() + " " + expanded.strip()).strip()[:2000]
                qe_triggered = 1

        query_expansion_ms = int((time.monotonic() - stage_begin) * 1000)
        io_logger.log_internal(
            "query_expansion",
            {
                "query": (request.query or "")[:50],
                "retrieval_query": (retrieval_query or "")[:50],
                "mode": qe_mode,
                "triggered": qe_triggered,
            },
            duration_ms=query_expansion_ms,
        )

        # Lower min_similarity for short queries
        short_threshold = getattr(self._search_config, "short_query_max_chars", 20)
        min_sim_short = getattr(self._search_config, "min_similarity_short", 0.45)
        effective_min_similarity = (
            min_sim_short if len(request.query.strip()) <= short_threshold else request.min_similarity
        )
        if request.tags:
            tags_lower = [t.lower() for t in request.tags if isinstance(t, str)]
            if "workflow" in tags_lower:
                effective_min_similarity = min(effective_min_similarity, 0.5)

        # Stage 1.5/1.6 + Stage 2 + FTS pre-fetch: run independent I/O concurrently via asyncio.gather
        enable_graph = getattr(self._search_config, "enable_graph", True) is not False
        emb_timeout = float(getattr(self._search_config, "embedding_timeout_seconds", 2.5))
        over_fetch = request.max_results * 5
        should_prefetch_fts = (
            bool(self._db_url)
            and self._search_config.mode in ("hybrid", "fts")
            and not hasattr(self._retrieve_and_fuse, "_mock_name")
        )

        (
            (_entity_names, graph_weighted_hits, graph_expand_ms),
            (query_embedding, embedding_ms, degraded, degraded_reason),
            (prefetched_fts, fts_prefetch_ms),
            parallel_fetch_wall_ms,
        ) = await self._run_parallel_prefetch(
            session,
            repo,
            request,
            retrieval_query,
            enable_graph=enable_graph,
            emb_timeout=emb_timeout,
            over_fetch=over_fetch,
            should_prefetch_fts=should_prefetch_fts,
        )

        graph_experience_ids = [eid for eid, _ in graph_weighted_hits]
        stage_metrics["graph_expand_ms"] = graph_expand_ms
        stage_metrics["embedding_ms"] = embedding_ms
        stage_metrics["parallel_fetch_wall_ms"] = parallel_fetch_wall_ms
        if should_prefetch_fts:
            stage_metrics["fts_ms"] = fts_prefetch_ms

        # Stage 3 & 4: Retrieval + RRF Fusion
        stage_begin = time.monotonic()
        mode = self._search_config.mode
        if query_embedding is None:
            mode = "fts"

        retrieve_kwargs: dict[str, Any] = {
            "retrieval_query": retrieval_query,
            "effective_min_similarity": effective_min_similarity,
            "graph_weighted_hits": graph_weighted_hits,
            "stage_counts": stage_counts,
        }
        if prefetched_fts is not None:
            retrieve_kwargs["prefetched_fts"] = prefetched_fts

        candidates = await self._retrieve_and_fuse(
            repo,
            request,
            query_embedding,
            mode,
            **retrieve_kwargs,
        )
        stage_metrics["retrieve_fuse_ms"] = int((time.monotonic() - stage_begin) * 1000)

        # Stage 4.5: Fallback-only LLM query expansion (triggered only on low count / low confidence)
        if qe_enabled and qe_mode == "fallback" and self._needs_fallback_expansion(candidates, mode):
            qe_fb_begin = time.monotonic()
            expanded = await llm_expand_fn(request.query, self._llm_config, qe_timeout_s)
            query_expansion_ms += int((time.monotonic() - qe_fb_begin) * 1000)
            if expanded and expanded.strip():
                qe_triggered = 1
                expanded_query = (request.query.strip() + " " + expanded.strip()).strip()[:2000]
                fb_embedding = None
                if self._search_config.mode != "fts":
                    try:
                        fb_embedding = await self._cache.get_or_compute_embedding(
                            expanded_query,
                            self._embedding,
                            timeout_seconds=emb_timeout,
                        )
                    except Exception as e:
                        logger.debug("Fallback expansion embedding failed, using FTS: %s", e)
                fb_mode = self._search_config.mode if fb_embedding is not None else "fts"
                fb_rf_begin = time.monotonic()
                fb_candidates = await self._retrieve_and_fuse(
                    repo,
                    request,
                    fb_embedding,
                    fb_mode,
                    retrieval_query=expanded_query,
                    effective_min_similarity=effective_min_similarity,
                    graph_weighted_hits=graph_weighted_hits,
                    stage_counts=None,
                )
                stage_metrics["retrieve_fuse_ms"] += int((time.monotonic() - fb_rf_begin) * 1000)
                candidates = self._merge_candidates(candidates, fb_candidates)

        stage_metrics["query_expansion_ms"] = query_expansion_ms
        stage_counts["query_expansion_triggered"] = qe_triggered
        stage_counts["embedding_degraded"] = 1 if degraded else 0

        io_logger.log_internal(
            "retrieve_fuse",
            {
                "query": (retrieval_query or request.query or "")[:50],
                "mode": mode,
                "candidates": len(candidates),
                "degraded": degraded,
            },
            duration_ms=stage_metrics["retrieve_fuse_ms"],
        )

        # Exclude promoted experiences before boost/adaptive-filter/truncation unless requested
        if not getattr(request, "include_promoted", False):
            candidates = [
                c
                for c in candidates
                if (c.data.get("status") or c.data.get("exp_status", "published")) != "promoted"
            ]

        # Stage 5: Exact match boost
        if getattr(self._search_config, "enable_exact_boost", True) is not False:
            candidates = self._apply_exact_match_boost(candidates, request.query)

        # Stage 5.5: Entity-graph boost — promote experiences found via graph
        # traversal when not already fused via 4-way RRF in hybrid mode.
        if graph_experience_ids and mode != "hybrid":
            graph_id_set = set(graph_experience_ids)
            graph_score_map = {eid: float(g_score) for eid, g_score in graph_weighted_hits}
            boosted_count = 0
            for c in candidates:
                cid = c.data.get("id") or c.data.get("group_id")
                if cid in graph_id_set:
                    old_score = float(c.score or 0.0)
                    g_factor = min(1.0, max(0.0, graph_score_map.get(str(cid), 1.0)))
                    if 0.0 < old_score < 0.1:
                        c.score = old_score * (1.0 + 0.08 * g_factor)
                    else:
                        c.score = old_score + 0.15 * g_factor
                    boosted_count += 1
                    logger.debug(
                        "[GRAPH]  boosted candidate %s from %.4f to %.4f",
                        str(cid)[:8],
                        old_score,
                        c.score,
                    )
            candidates.sort(key=lambda x: x.score, reverse=True)
            logger.info("[GRAPH]  boosted %d candidate(s) from graph traversal", boosted_count)

        # Stage 5b: Popularity boost (反哺排序)
        stage_begin = time.monotonic()
        if self._search_config.popularity_enabled and candidates:
            candidates = self._apply_popularity_boost(candidates)
        stage_metrics["popularity_boost_ms"] = int((time.monotonic() - stage_begin) * 1000)

        # Stage 5c: Intent-adaptive recency boost (for release/version/latest queries)
        if getattr(self._search_config, "adaptive_ranking", True) and mode == "hybrid" and candidates:
            from team_memory.services.search.scoring import apply_recency_boost, resolve_rank_profile

            rank_profile = resolve_rank_profile(request.query, request.intent_type)
            if rank_profile.recency_boost > 0.0:
                candidates = apply_recency_boost(candidates, weight=rank_profile.recency_boost)

        total_candidates = len(candidates)

        reranked = False
        if self._rerank_enabled and candidates:
            stage_begin = time.monotonic()
            candidates, reranked = await self._apply_rerank(request, candidates)
            stage_metrics["rerank_ms"] = int((time.monotonic() - stage_begin) * 1000)
            io_logger.log_internal(
                "rerank",
                {
                    "query": (request.query or "")[:50],
                    "candidates": len(candidates),
                    "reranked": reranked,
                },
                duration_ms=stage_metrics.get("rerank_ms", 0),
            )

        # Stage 6: Adaptive Filtering (moved AFTER reranking)
        stage_begin = time.monotonic()
        if self._search_config.adaptive_filter and candidates:
            candidates = self._apply_adaptive_filter(candidates)
        stage_metrics["adaptive_filter_ms"] = int((time.monotonic() - stage_begin) * 1000)
        io_logger.log_internal(
            "adaptive_filter",
            {
                "query": (request.query or "")[:50],
                "before": total_candidates,
                "after": len(candidates),
            },
            duration_ms=stage_metrics["adaptive_filter_ms"],
        )

        # Confidence labels BEFORE epistemic/staleness adjustment so possibly_stale
        # down-weighting never pushes a medium-relevance item into 'low' and gets dropped by _guard_output.
        candidates = self._label_confidence(candidates)

        # Stage 6b: Defense-4 epistemic & staleness calibration (post-filter/confidence re-rank)
        if getattr(self._search_config, "epistemic_staleness_ranking", True) and candidates:
            from team_memory.services.search.scoring import apply_epistemic_and_staleness_adjustment

            candidates = apply_epistemic_and_staleness_adjustment(candidates)

        # Stage 6c: Defense-4 CRAG multi-signal low-confidence detection (0ms default + optional 1.5s Verifier)
        low_confidence = False
        if candidates:
            from team_memory.services.search.scoring import detect_low_confidence

            crag_sim_thr = float(getattr(self._search_config, "crag_low_confidence_threshold", 0.58))
            low_confidence = detect_low_confidence(candidates, sim_threshold=crag_sim_thr)
            if low_confidence:
                stage_counts["crag_low_confidence"] = 1
                if (
                    getattr(self._search_config, "crag_verifier_enabled", False)
                    and self._llm_config is not None
                ):
                    crag_timeout = float(
                        getattr(self._search_config, "crag_verifier_timeout_seconds", 1.5)
                    )
                    ver_t0 = time.monotonic()
                    candidates = await self._crag_verify_candidates(
                        request.query,
                        candidates,
                        timeout_seconds=crag_timeout,
                    )
                    stage_metrics["crag_verifier_ms"] = int((time.monotonic() - ver_t0) * 1000)

        return await self._postprocess_and_assemble(
            session,
            request,
            candidates,
            query_embedding=query_embedding,
            effective_min_similarity=effective_min_similarity,
            total_candidates=total_candidates,
            mode=mode,
            reranked=reranked,
            degraded=degraded,
            degraded_reason=degraded_reason,
            low_confidence=low_confidence,
            start=start,
            stage_metrics=stage_metrics,
            stage_counts=stage_counts,
        )

    async def invalidate_cache(self):
        """Invalidate all cached results. Call after data mutations."""
        await self._cache.clear()

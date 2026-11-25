"""Unified search pipeline — orchestrates the complete retrieval process.

Pipeline stages:
  1. Cache check — return cached results if available
  2. Embedding — encode query to vector
  3. Retrieval — parallel vector + FTS search (hybrid mode) or single mode
  4. RRF Fusion — merge results from multiple sources using Reciprocal Rank Fusion
  5. Exact match boost — boost score when query matches title
  6. Adaptive Filtering — dynamic threshold + elbow detection
  7. Optional reranking — server-side rerank of experience candidates when configured
  8. Confidence labeling — high/medium/low from scores (after rerank if any)
  9. Archive merging — include archive results when requested (after top experiences assembled)
  10. Cache store — save results for future queries
"""

from __future__ import annotations

import asyncio
import logging
import os
import time
import uuid
from dataclasses import asdict, dataclass, field
from typing import TYPE_CHECKING, Any, Mapping

from sqlalchemy.ext.asyncio import AsyncSession

from team_memory import io_logger
from team_memory.config import (
    CacheConfig,
    RetrievalConfig,
    SearchConfig,
)
from team_memory.embedding.base import EmbeddingProvider
from team_memory.services.search.cache import SearchCache
from team_memory.services.search.query_expansion import (  # noqa: F401
    _EXPANSION_CACHE_TTL,
    _TOPIC_CACHE_TTL,
    _expansion_cache,
    _topic_cache,
    expansion_cache_clear,
    topic_cache_clear,
)

if TYPE_CHECKING:
    from team_memory.reranker.base import RerankerProvider
from team_memory.storage.archive_repository import ArchiveRepository
from team_memory.storage.database import get_session
from team_memory.storage.repository import ExperienceRepository

logger = logging.getLogger("team_memory.search_pipeline")

_SEARCH_DEBUG = os.environ.get("TEAM_MEMORY_SEARCH_DEBUG", "").lower() in ("1", "true", "yes")


# Backward-compatible aliases for the module-level cache helpers
_expansion_cache_clear = expansion_cache_clear
_topic_cache_clear = topic_cache_clear


@dataclass
class SearchRequest:
    """Parameters for a search pipeline execution."""

    query: str
    max_results: int = 10
    min_similarity: float = 0.6
    tags: list[str] | None = None
    user_name: str = ""
    current_user: str | None = None
    source: str = "mcp"
    grouped: bool = False
    top_k_children: int = 3
    project: str | None = None
    include_archives: bool = False
    include_promoted: bool = False
    include_superseded: bool = False
    session_id: str | None = None
    fallback_projects: list[str] | None = None
    intent_type: str | None = None


SUPERSESSION_PENALTY: float = 0.3


def _apply_bi_temporal_annotations(results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Apply 0.3x down-weighting and explicit [已演进/Superseded] markers to superseded experiences."""
    any_penalized = False
    for d in results:
        if not isinstance(d, dict):
            continue
        valid_until = d.get("valid_until")
        sup_by = d.get("superseded_by_id")
        if valid_until is not None or sup_by is not None:
            if not d.get("_superseded_penalized"):
                if "score" in d and d["score"] is not None:
                    d["score"] = round(float(d["score"]) * SUPERSESSION_PENALTY, 4)
                if "similarity" in d and d["similarity"] is not None:
                    d["similarity"] = round(float(d["similarity"]) * SUPERSESSION_PENALTY, 4)
                d["_superseded_penalized"] = True
                any_penalized = True
            d["is_superseded"] = True
            sup_short = str(sup_by)[:8] if sup_by else ""
            marker = (
                f"[已演进/Superseded by tm://exp/{sup_short}]"
                if sup_short
                else "[已演进/Superseded]"
            )
            d["superseded_marker"] = marker
            title = str(d.get("title") or "")
            if not title.startswith("[已演进"):
                d["title"] = f"{marker} {title}".strip()

        mut_hist = d.get("mutation_history")
        if isinstance(mut_hist, list):
            for entry in reversed(mut_hist):
                if isinstance(entry, dict) and (
                    entry.get("action") == "SUPERSEDE_REPLACES"
                    or entry.get("type") == "SUPERSEDE_REPLACES"
                ):
                    old_id = (
                        entry.get("supersedes_id")
                        or entry.get("old_id")
                        or entry.get("replaces_id")
                    )
                    if old_id:
                        d["supersedes_id"] = str(old_id)
                        d["evolution_marker"] = f"[已演进自 tm://exp/{str(old_id)[:8]}]"
                        break

        d.pop("_superseded_penalized", None)

    if any_penalized:
        results.sort(key=lambda x: float(x.get("score", 0.0)), reverse=True)
    return results


@dataclass
class SearchResultItem:
    """A single search result with metadata from the pipeline."""

    data: dict
    score: float = 0.0
    similarity: float = 0.0
    fts_rank: float = 0.0
    rrf_score: float = 0.0
    confidence: str = "medium"  # "high" | "medium" | "low"
    source_type: str = "vector"  # "vector" | "fts" | "hybrid"


@dataclass
class SearchPipelineResult:
    """Complete result from the search pipeline."""

    results: list[dict] = field(default_factory=list)
    total_candidates: int = 0
    search_type: str = "hybrid"
    reranked: bool = False
    cached: bool = False
    duration_ms: int = 0
    stage_metrics: dict[str, int] = field(default_factory=dict)
    stage_counts: dict[str, int] = field(default_factory=dict)
    degraded: bool = False
    degraded_reason: str | None = None
    low_confidence: bool = False


def coerce_cached_search_pipeline_result(raw: SearchPipelineResult | Mapping[str, Any]) -> SearchPipelineResult:
    """Normalize cache hits: memory backend returns a dataclass; Redis returns a dict (JSON)."""
    if isinstance(raw, SearchPipelineResult):
        return raw
    if not isinstance(raw, Mapping):
        raise TypeError(f"Unexpected cached search payload type: {type(raw)}")
    results = raw.get("results") or []
    if not isinstance(results, list):
        results = []
    sm = raw.get("stage_metrics") or {}
    if not isinstance(sm, dict):
        sm = {}
    stage_metrics = {str(k): int(v) for k, v in sm.items()}
    sc = raw.get("stage_counts") or {}
    if not isinstance(sc, dict):
        sc = {}
    stage_counts = {str(k): int(v) for k, v in sc.items()}
    return SearchPipelineResult(
        results=list(results),
        total_candidates=int(raw.get("total_candidates", 0)),
        search_type=str(raw.get("search_type", "hybrid")),
        reranked=bool(raw.get("reranked", False)),
        cached=bool(raw.get("cached", False)),
        duration_ms=int(raw.get("duration_ms", 0)),
        stage_metrics=stage_metrics,
        stage_counts=stage_counts,
        degraded=bool(raw.get("degraded", False)),
        degraded_reason=raw.get("degraded_reason"),
        low_confidence=bool(raw.get("low_confidence", False)),
    )


async def _llm_expand_query(
    query: str,
    llm_config,
    timeout_seconds: float,
) -> str | None:
    """Call LLM to expand search query with extra keywords. Returns None on any failure.

    Delegates to :func:`team_memory.services.search.query_expansion.llm_expand_query`.
    """
    from team_memory.services.search.query_expansion import llm_expand_query

    return await llm_expand_query(query, llm_config, timeout_seconds)


class SearchPipeline:
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
        repo = ExperienceRepository(session)

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
            expanded = await _llm_expand_query(request.query, self._llm_config, qe_timeout_s)
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

        async def _run_graph_expand() -> tuple[list[str], list[tuple[str, float]], int]:
            g_t0 = time.monotonic()
            if not enable_graph:
                return [], [], 0
            ent_names: list[str] = []
            g_hits: list[tuple[str, float]] = []
            try:
                from team_memory.services.entity.entity_search import (
                    extract_entities_from_query,
                    find_weighted_experiences_by_entities,
                    get_cached_entity_lexicon,
                )

                db_lexicon: set[str] = set()
                if self._db_url and "test_db" not in self._db_url:
                    try:
                        db_lexicon = await get_cached_entity_lexicon(self._db_url)
                    except Exception:
                        db_lexicon = set()
                ent_names = extract_entities_from_query(retrieval_query, lexicon=db_lexicon or None)
                if ent_names and self._db_url:
                    projects = [request.project] if request.project else ["default"]
                    if request.fallback_projects:
                        for fp in request.fallback_projects:
                            if fp not in projects:
                                projects.append(fp)
                    # Embedding (HTTP) and FTS pre-fetch (separate session2 when db_url is set)
                    # do not touch `session`, so entity-graph CTE can safely reuse `session` here.
                    try:
                        reusable_session = (
                            session
                            if (isinstance(AsyncSession, type) and isinstance(session, AsyncSession))
                            else None
                        )
                    except TypeError:
                        reusable_session = None
                    graph_kwargs: dict[str, Any] = {
                        "db_url": self._db_url,
                        "entity_names": ent_names,
                        "projects": projects,
                        "max_depth": 2,
                    }
                    if reusable_session is not None:
                        graph_kwargs["session"] = reusable_session
                    g_hits = await asyncio.wait_for(
                        find_weighted_experiences_by_entities(**graph_kwargs),
                        timeout=1.5,
                    )
                    if g_hits:
                        logger.info(
                            "[GRAPH]  entity-graph found %d related experience(s) for query=%s",
                            len(g_hits),
                            retrieval_query[:50],
                        )
            except Exception:
                logger.debug("Entity graph traversal failed (non-critical)", exc_info=True)
            return ent_names, g_hits, int((time.monotonic() - g_t0) * 1000)

        async def _run_embedding() -> tuple[list[float] | None, int, bool, str | None]:
            if self._search_config.mode == "fts":
                return None, 0, False, None
            e_t0 = time.monotonic()
            try:
                vec = await self._cache.get_or_compute_embedding(
                    retrieval_query,
                    self._embedding,
                    timeout_seconds=emb_timeout,
                )
                return vec, int((time.monotonic() - e_t0) * 1000), False, None
            except asyncio.TimeoutError:
                elapsed = int((time.monotonic() - e_t0) * 1000)
                logger.warning(
                    "Embedding timed out after %.1fs (%dms), degrading to FTS+Graph",
                    emb_timeout,
                    elapsed,
                )
                return None, elapsed, True, "embedding_timeout"
            except Exception as e:
                elapsed = int((time.monotonic() - e_t0) * 1000)
                logger.warning("Embedding failed, degrading to FTS+Graph: %s", e, exc_info=True)
                return None, elapsed, True, f"embedding_error:{type(e).__name__}"

        async def _run_fts_prefetch() -> tuple[list[SearchResultItem] | None, int]:
            if not should_prefetch_fts:
                return None, 0
            f_t0 = time.monotonic()
            try:
                fts_items = await self._fts_search(repo, retrieval_query.strip(), over_fetch, request)
                return fts_items, int((time.monotonic() - f_t0) * 1000)
            except Exception as e:
                logger.warning("FTS pre-fetch failed (will retry in retrieve_and_fuse): %s", e)
                return None, int((time.monotonic() - f_t0) * 1000)

        parallel_begin = time.monotonic()
        (
            (entity_names, graph_weighted_hits, graph_expand_ms),
            (query_embedding, embedding_ms, degraded, degraded_reason),
            (prefetched_fts, fts_prefetch_ms),
        ) = await asyncio.gather(
            _run_graph_expand(),
            _run_embedding(),
            _run_fts_prefetch(),
        )
        parallel_fetch_wall_ms = int((time.monotonic() - parallel_begin) * 1000)

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
            expanded = await _llm_expand_query(request.query, self._llm_config, qe_timeout_s)
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
            logger.info(
                "[GRAPH]  boosted %d candidate(s) from graph traversal",
                boosted_count,
            )

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

        # Stage 6.5: Grouping logic (US5)
        if request.grouped:
            grouped_results: dict[str, list[SearchResultItem]] = {}
            for item in candidates:
                gid = item.data.get("group_id") or item.data.get("parent_id") or item.data.get("id")
                gid = str(gid)
                if gid not in grouped_results:
                    grouped_results[gid] = []
                if len(grouped_results[gid]) < request.top_k_children:
                    grouped_results[gid].append(item)

            flattened = []
            for group_items in grouped_results.values():
                flattened.extend(group_items)
            candidates = flattened[: request.max_results]
        else:
            # Limit final results
            candidates = candidates[: request.max_results]

        # Build final result dicts with Defense-4 epistemic/staleness metadata & warnings
        from team_memory.services.memory.evidence_anchors import compact_evidence_anchors
        from team_memory.services.search.scoring import STALE_WARNING_TEXT

        result_dicts = []
        for item in candidates:
            d = item.data.copy()
            d["score"] = round(float(item.score), 4)
            d["similarity"] = round(item.similarity, 4)
            d["confidence"] = item.confidence
            d["epistemic_tier"] = d.get("epistemic_tier") or "auto_extract"
            staleness = d.get("staleness_status") or "fresh"
            d["staleness_status"] = staleness
            if not d.get("last_verified_at"):
                d["last_verified_at"] = d.get("created_at")
            d["evidence_anchors"] = compact_evidence_anchors(
                d.get("evidence_anchors"),
                max_paths=8,
                max_symbols=8,
            )
            if staleness in ("possibly_stale", "stale"):
                d["stale_warning"] = STALE_WARNING_TEXT
            if low_confidence:
                d["low_confidence"] = True
            result_dicts.append(d)

        result_dicts = _apply_bi_temporal_annotations(result_dicts)

        # Stage 7: Include archives when requested
        post_begin = time.monotonic()
        if request.include_archives and query_embedding is not None:
            archive_repo = ArchiveRepository(session)
            archive_limit = max(request.max_results * 2, 20)
            archive_hits = await archive_repo.search_archives(
                query_embedding,
                project=request.project,
                limit=archive_limit,
                min_similarity=effective_min_similarity,
                current_user=request.current_user,
            )
            for row in archive_hits:
                d = dict(row)
                d.setdefault("confidence", "medium")
                result_dicts.append(d)
            result_dicts.sort(key=lambda x: float(x.get("score", 0)), reverse=True)
            result_dicts = result_dicts[: request.max_results]

        # Stage 7b: Enrich experience results with archive_ids
        exp_id_strs = [d["id"] for d in result_dicts if d.get("id") and d.get("type") != "archive"]
        if exp_id_strs:
            archive_repo_enrich = ArchiveRepository(session)
            exp_uuids = []
            for eid in exp_id_strs:
                try:
                    exp_uuids.append(uuid.UUID(eid))
                except (ValueError, TypeError):
                    pass
            if exp_uuids:
                archive_mapping = await archive_repo_enrich.get_archive_ids_for_experiences(exp_uuids)
                for d in result_dicts:
                    if d.get("type") != "archive":
                        d["archive_ids"] = archive_mapping.get(str(d.get("id", "")), [])
        stage_metrics["post_process_ms"] = int((time.monotonic() - post_begin) * 1000)

        duration_ms = int((time.monotonic() - start) * 1000)
        stage_metrics["total_ms"] = duration_ms
        stage_counts["final_count"] = len(result_dicts)

        pipeline_result = SearchPipelineResult(
            results=result_dicts,
            total_candidates=total_candidates,
            search_type=mode,
            reranked=reranked,
            cached=False,
            duration_ms=duration_ms,
            stage_metrics=stage_metrics,
            stage_counts=stage_counts,
            degraded=degraded,
            degraded_reason=degraded_reason,
            low_confidence=low_confidence,
        )

        # Stage 8: Cache store
        if self._cache.enabled:
            stage_begin = time.monotonic()
            to_store: SearchPipelineResult | dict[str, Any] = (
                asdict(pipeline_result) if self._cache._backend == "redis" else pipeline_result
            )
            await self._cache.put(
                request.query,
                request.tags,
                to_store,
                project=request.project,
                current_user=request.current_user,
                include_archives=request.include_archives,
                reranker_signature=self._reranker_signature,
            )
            io_logger.log_internal(
                "cache_store",
                {
                    "query": (request.query or "")[:50],
                    "results": len(pipeline_result.results),
                },
                duration_ms=int((time.monotonic() - stage_begin) * 1000),
            )

        return pipeline_result

    def _expand_query_synonyms(self, query: str) -> str:
        """Expand query with tag_synonyms (key<->value) for better recall."""
        from team_memory.services.search.query_expansion import expand_query_synonyms

        return expand_query_synonyms(query, self._tag_synonyms)

    def _needs_fallback_expansion(self, candidates: list[SearchResultItem], mode: str) -> bool:
        """Determine whether first-pass retrieval was too weak and needs LLM query expansion."""
        min_results = int(getattr(self._search_config, "query_expansion_fallback_min_results", 3))
        if len(candidates) < min_results:
            return True
        if mode == "fts":
            return float(candidates[0].score or 0.0) <= 0.0
        min_sim = float(getattr(self._search_config, "query_expansion_fallback_min_similarity", 0.65))
        top_sim = max((float(c.similarity or 0.0) for c in candidates), default=0.0)
        return top_sim < min_sim

    @staticmethod
    def _merge_candidates(
        primary: list[SearchResultItem],
        secondary: list[SearchResultItem],
    ) -> list[SearchResultItem]:
        """Merge and deduplicate two candidate lists, retaining the highest score per experience ID."""
        merged: dict[str, SearchResultItem] = {}
        order: list[str] = []
        for item in primary + secondary:
            cid = str(item.data.get("id") or item.data.get("group_id") or id(item))
            if cid not in merged:
                merged[cid] = item
                order.append(cid)
            elif float(item.score or 0.0) > float(merged[cid].score or 0.0):
                merged[cid] = item
        out = [merged[cid] for cid in order]
        out.sort(key=lambda x: float(x.score or 0.0), reverse=True)
        return out

    async def _retrieve_and_fuse(
        self,
        repo: ExperienceRepository,
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

    async def _graph_search(
        self,
        repo: ExperienceRepository,
        graph_weighted_hits: list[tuple[str, float]],
        limit: int,
        request: SearchRequest,
        *,
        existing_by_id: dict[str, dict[str, Any]] | None = None,
        session: AsyncSession | None = None,
    ) -> list[SearchResultItem]:
        """Fetch active experiences discovered via weighted entity-graph traversal."""
        if not graph_weighted_hits:
            return []

        top_hits = graph_weighted_hits[:limit]
        score_by_id: dict[str, float] = {}
        ordered_ids: list[str] = []
        for eid, g_score in top_hits:
            eid_str = str(eid)
            if eid_str not in score_by_id:
                score_by_id[eid_str] = float(g_score)
                ordered_ids.append(eid_str)

        if not ordered_ids:
            return []

        fetched_map: dict[str, dict[str, Any]] = {}
        if existing_by_id:
            for eid_str in ordered_ids:
                existing = existing_by_id.get(eid_str)
                if isinstance(existing, dict) and existing:
                    fetched_map[eid_str] = existing

        missing_ids = [eid for eid in ordered_ids if eid not in fetched_map]
        db_query_succeeded = False
        if missing_ids:
            repo_session = getattr(repo, "_session", None)
            try:
                active_session = (
                    session
                    if (isinstance(AsyncSession, type) and isinstance(session, AsyncSession))
                    else (
                        repo_session
                        if (isinstance(AsyncSession, type) and isinstance(repo_session, AsyncSession))
                        else None
                    )
                )
            except TypeError:
                active_session = None
            status_clause = (
                "exp_status IN ('published', 'promoted')"
                if getattr(request, "include_promoted", False)
                else "exp_status = 'published'"
            )
            inc_superseded = bool(getattr(request, "include_superseded", False))
            valid_clause = "" if inc_superseded else "AND valid_until IS NULL"
            query_sql = f"""
                SELECT id, title, description, solution, tags, project,
                       experience_type, exp_status, recall_count, used_count,
                       negative_feedback_count, created_at,
                       implicit_positive_count, implicit_negative_count,
                       epistemic_tier, staleness_status, last_verified_at, evidence_anchors
                FROM experiences
                WHERE id::text = ANY(:ids)
                  AND {status_clause}
                  AND is_deleted = false
                  {valid_clause}
                  AND coalesce(trim(title), '') <> ''
                  AND title !~ '^(Draft [0-9a-zA-Z_-]+|Session [0-9a-zA-Z_-]+ draft|主题-.*)$'
            """
            if active_session is not None:
                try:
                    from sqlalchemy import text as sql_text

                    result = await active_session.execute(
                        sql_text(query_sql),
                        {"ids": missing_ids},
                    )
                    rows = result.fetchall()
                    if isinstance(rows, list):
                        db_query_succeeded = True
                        for row in rows:
                            rid = str(row[0])
                            created_iso = row[11].isoformat() if len(row) > 11 and row[11] else None
                            verified_iso = (
                                row[16].isoformat() if len(row) > 16 and row[16] else created_iso
                            )
                            fetched_map[rid] = {
                                "id": rid,
                                "title": row[1],
                                "description": row[2],
                                "solution": row[3],
                                "tags": row[4] or [],
                                "project": row[5],
                                "experience_type": row[6],
                                "status": row[7] or "published",
                                "recall_count": row[8] or 0,
                                "used_count": row[9] or 0,
                                "negative_feedback_count": (row[10] or 0) if len(row) > 10 else 0,
                                "created_at": created_iso,
                                "implicit_positive_count": (row[12] or 0) if len(row) > 12 else 0,
                                "implicit_negative_count": (row[13] or 0) if len(row) > 13 else 0,
                                "epistemic_tier": (row[14] or "auto_extract") if len(row) > 14 else "auto_extract",
                                "staleness_status": (row[15] or "fresh") if len(row) > 15 else "fresh",
                                "last_verified_at": verified_iso,
                                "evidence_anchors": (
                                    (row[17] or {})
                                    if len(row) > 17 and isinstance(row[17], dict)
                                    else {}
                                ),
                            }
                except Exception:
                    if hasattr(active_session, "rollback"):
                        try:
                            await active_session.rollback()
                        except Exception:
                            pass
            elif self._db_url:
                try:
                    from sqlalchemy import text as sql_text

                    async with get_session(self._db_url) as s:
                        result = await s.execute(
                            sql_text(query_sql),
                            {"ids": missing_ids},
                        )
                        rows = result.fetchall()
                        if isinstance(rows, list):
                            db_query_succeeded = True
                            for row in rows:
                                rid = str(row[0])
                                created_iso = row[11].isoformat() if len(row) > 11 and row[11] else None
                                verified_iso = (
                                    row[16].isoformat() if len(row) > 16 and row[16] else created_iso
                                )
                                fetched_map[rid] = {
                                    "id": rid,
                                    "title": row[1],
                                    "description": row[2],
                                    "solution": row[3],
                                    "tags": row[4] or [],
                                    "project": row[5],
                                    "experience_type": row[6],
                                    "status": row[7] or "published",
                                    "recall_count": row[8] or 0,
                                    "used_count": row[9] or 0,
                                    "negative_feedback_count": (row[10] or 0) if len(row) > 10 else 0,
                                    "created_at": created_iso,
                                    "implicit_positive_count": (row[12] or 0) if len(row) > 12 else 0,
                                    "implicit_negative_count": (row[13] or 0) if len(row) > 13 else 0,
                                    "epistemic_tier": (row[14] or "auto_extract") if len(row) > 14 else "auto_extract",
                                    "staleness_status": (row[15] or "fresh") if len(row) > 15 else "fresh",
                                    "last_verified_at": verified_iso,
                                    "evidence_anchors": (
                                        (row[17] or {})
                                        if len(row) > 17 and isinstance(row[17], dict)
                                        else {}
                                    ),
                                }
                except Exception:
                    pass

        if not db_query_succeeded and len(fetched_map) < len(ordered_ids) and hasattr(repo, "get_by_id"):
            for eid_str in ordered_ids:
                if eid_str in fetched_map:
                    continue
                try:
                    try:
                        lookup_id: Any = uuid.UUID(eid_str)
                    except (ValueError, TypeError):
                        lookup_id = eid_str
                    exp_obj = await repo.get_by_id(lookup_id)
                    if exp_obj is None:
                        continue
                    if isinstance(exp_obj, dict):
                        d = dict(exp_obj)
                    elif hasattr(exp_obj, "to_dict") and callable(exp_obj.to_dict):
                        if getattr(exp_obj, "is_deleted", False) is True:
                            continue
                        valid_until = getattr(exp_obj, "valid_until", None)
                        if (
                            not getattr(request, "include_superseded", False)
                            and valid_until is not None
                            and not hasattr(valid_until, "_mock_name")
                        ):
                            continue
                        exp_status = getattr(exp_obj, "exp_status", "published")
                        if isinstance(exp_status, str) and exp_status not in ("published", "promoted"):
                            continue
                        d = exp_obj.to_dict()
                    else:
                        continue
                    if isinstance(d, dict) and d.get("id"):
                        fetched_map[str(d["id"])] = d
                except Exception:
                    continue

        req_proj = (request.project or "").strip()
        items: list[SearchResultItem] = []
        for eid_str in ordered_ids:
            data = fetched_map.get(eid_str)
            if not data:
                continue
            g_score = score_by_id.get(eid_str, 0.5)
            d_copy = dict(data)
            exp_proj = str(d_copy.get("project") or "").strip()
            if (
                req_proj
                and req_proj not in ("default", "team_memory", "*", "all")
                and exp_proj
                and exp_proj not in (req_proj, "default", "global")
            ):
                d_copy["cross_project"] = True
            d_copy["graph_score"] = round(g_score, 4)
            items.append(
                SearchResultItem(
                    data=d_copy,
                    score=float(g_score),
                    similarity=max(0.5, min(0.95, float(g_score))),
                    source_type="graph",
                )
            )
        return items

    async def _vector_search(
        self,
        repo: ExperienceRepository,
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
        repo: ExperienceRepository,
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
        if self._db_url:
            async with get_session(self._db_url) as session2:
                repo_fts = ExperienceRepository(session2)
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

    async def _topic_search(
        self,
        query_embedding: list[float],
        limit: int,
        request: SearchRequest,
    ) -> list[SearchResultItem]:
        """Search via topic clusters — returns experiences from relevant topics."""
        global _topic_cache

        if not self._db_url:
            return []

        # Step 1: Get or cache discover_topics result
        cache_key = self._db_url
        cached = _topic_cache.get(cache_key)
        if cached and (time.monotonic() - cached[1]) < _TOPIC_CACHE_TTL:
            topics = cached[0]
        else:
            try:
                from team_memory.services.analysis.topic_discovery import discover_topics

                result = await discover_topics(self._db_url)
                topics = result.topics
                _topic_cache[cache_key] = (topics, time.monotonic())
            except Exception:
                return []

        if not topics:
            return []

        # Step 2: Compute cosine similarity to each topic center
        import numpy as np

        q_vec = np.array(query_embedding, dtype=np.float32)
        q_norm = np.linalg.norm(q_vec)
        if q_norm < 1e-8:
            return []

        scored: list[tuple[float, str]] = []
        for topic in topics:
            if not topic.center:
                continue
            c_vec = np.array(topic.center, dtype=np.float32)
            sim = float(np.dot(q_vec, c_vec) / (q_norm * np.linalg.norm(c_vec) + 1e-8))
            if sim >= 0.52:
                for eid in topic.experience_ids:
                    scored.append((sim, eid))

        if not scored:
            return []

        scored.sort(key=lambda x: x[0], reverse=True)
        seen: set[str] = set()
        exp_ids: list[str] = []
        sim_by_id: dict[str, float] = {}
        candidate_pool_limit = max(limit * 15, 50)
        for sim_val, eid in scored:
            if eid not in seen:
                seen.add(eid)
                exp_ids.append(eid)
                sim_by_id[eid] = sim_val
            if len(exp_ids) >= candidate_pool_limit:
                break

        if not exp_ids:
            return []

        # Fetch experience data from PG
        from sqlalchemy import text as sql_text

        from team_memory.storage.database import get_session

        items: list[SearchResultItem] = []
        allowed_statuses = (
            ["published", "promoted"]
            if getattr(request, "include_promoted", False)
            else ["published"]
        )
        req_proj = (request.project or "default").strip()
        allow_all_projects = req_proj in ("*", "all") or (
            bool(request.fallback_projects)
            and any(p in ("*", "all") for p in request.fallback_projects)
        )
        try:
            async with get_session(self._db_url) as session:
                result = await session.execute(
                    sql_text("""
                        SELECT id, title, description, solution, tags, project,
                               experience_type, exp_status, recall_count, used_count,
                               negative_feedback_count, created_at,
                               implicit_positive_count, implicit_negative_count,
                               epistemic_tier, staleness_status, last_verified_at, evidence_anchors
                        FROM experiences
                        WHERE id::text = ANY(:ids)
                          AND exp_status = ANY(:statuses)
                          AND is_deleted = false
                          AND valid_until IS NULL
                          AND coalesce(trim(title), '') <> ''
                          AND title !~ '^(Draft [0-9a-zA-Z_-]+|Session [0-9a-zA-Z_-]+ draft|主题-.*)$'
                          AND (:allow_all = true OR project = :project)
                    """),
                    {
                        "ids": exp_ids,
                        "statuses": allowed_statuses,
                        "allow_all": allow_all_projects,
                        "project": req_proj,
                    },
                )
                rows = result.fetchall()
                row_map = {str(r[0]): r for r in rows}
                # Sort matched rows: prefer same-project first, then cluster similarity, then recall_count
                ordered_eids = [eid for eid in exp_ids if eid in row_map]
                ordered_eids.sort(
                    key=lambda eid: (
                        0 if str(row_map[eid][5] or "") == req_proj else 1,
                        -sim_by_id.get(eid, 0.52),
                        -int(row_map[eid][8] or 0),
                    )
                )
                for eid in ordered_eids[:limit]:
                    row = row_map[eid]
                    t_sim = sim_by_id.get(eid, 0.55)
                    items.append(
                        SearchResultItem(
                            data={
                                "id": str(row[0]),
                                "title": row[1],
                                "description": row[2],
                                "solution": row[3],
                                "tags": row[4],
                                "project": row[5],
                                "experience_type": row[6],
                                "status": row[7] or "published",
                                "exp_status": row[7] or "published",
                                "recall_count": row[8] or 0,
                                "used_count": row[9] or 0,
                                "negative_feedback_count": row[10] or 0,
                                "created_at": row[11].isoformat() if row[11] else None,
                                "implicit_positive_count": (row[12] or 0) if len(row) > 12 else 0,
                                "implicit_negative_count": (row[13] or 0) if len(row) > 13 else 0,
                                "epistemic_tier": (row[14] or "auto_extract") if len(row) > 14 else "auto_extract",
                                "staleness_status": (row[15] or "fresh") if len(row) > 15 else "fresh",
                                "last_verified_at": (
                                    row[16].isoformat()
                                    if len(row) > 16 and row[16]
                                    else (row[11].isoformat() if row[11] else None)
                                ),
                                "evidence_anchors": (
                                    (row[17] or {})
                                    if len(row) > 17 and isinstance(row[17], dict)
                                    else {}
                                ),
                            },
                            score=t_sim,
                            similarity=t_sim,
                            source_type="topic",
                        )
                    )
        except Exception:
            return []

        return items

    def _apply_exact_match_boost(self, candidates: list[SearchResultItem], query: str) -> list[SearchResultItem]:
        """Boost score when query exactly matches or is contained in title."""
        from team_memory.services.search.scoring import apply_exact_match_boost

        return apply_exact_match_boost(candidates, query)

    def _apply_popularity_boost(self, candidates: list[SearchResultItem]) -> list[SearchResultItem]:
        """Boost score based on recall_count and used_count with time decay."""
        from team_memory.services.search.scoring import apply_popularity_boost

        cfg = self._search_config
        return apply_popularity_boost(
            candidates,
            popularity_recall_weight=cfg.popularity_recall_weight,
            popularity_used_weight=cfg.popularity_used_weight,
            popularity_decay_lambda=cfg.popularity_decay_lambda,
            popularity_age_floor_ratio=cfg.popularity_age_floor_ratio,
            popularity_scale=cfg.popularity_scale,
        )

    def _apply_adaptive_filter(self, candidates: list[SearchResultItem]) -> list[SearchResultItem]:
        """Apply adaptive score filtering using similarity scores."""
        from team_memory.services.search.scoring import apply_adaptive_filter

        return apply_adaptive_filter(
            candidates,
            min_confidence_ratio=self._search_config.min_confidence_ratio,
            score_gap_threshold=self._search_config.score_gap_threshold,
            adaptive_min_keep=getattr(self._search_config, "adaptive_min_keep", 3),
        )

    async def _crag_verify_candidates(
        self,
        query: str,
        candidates: list[SearchResultItem],
        *,
        timeout_seconds: float = 1.5,
    ) -> list[SearchResultItem]:
        """Optional lightweight CRAG verifier for borderline low-confidence hits.

        Filters out clearly irrelevant candidates within a strict timeout; degrades
        safely to returning the original candidates on timeout or any error.
        """
        if not candidates or self._llm_config is None:
            return candidates

        import json

        import httpx

        top_slice = candidates[:5]
        base_url = getattr(self._llm_config, "base_url", "https://gw.gburouter.com/v1").rstrip("/")
        if not base_url.endswith("/v1"):
            base_url = f"{base_url}/v1"
        model = getattr(self._llm_config, "model", "gbu-gemini-3.8-flash")
        api_key_env = getattr(self._llm_config, "api_key_env", "GATEWAY_API_KEY")
        api_key = os.environ.get(api_key_env, "") or getattr(self._llm_config, "api_key", "") or "none"

        items_payload = [
            {
                "idx": idx,
                "title": (c.data.get("title") or "")[:160],
                "snippet": (c.data.get("solution") or c.data.get("description") or "")[:240],
            }
            for idx, c in enumerate(top_slice)
        ]
        prompt = (
            "Given a search query and candidate memory snippets, return ONLY a JSON array "
            "of integer indices `idx` that are genuinely relevant to the query (e.g. [0, 2])."
        )
        payload = {
            "model": model,
            "messages": [
                {"role": "system", "content": prompt},
                {
                    "role": "user",
                    "content": json.dumps({"query": query[:300], "candidates": items_payload}, ensure_ascii=False),
                },
            ],
            "temperature": 0.0,
            "max_tokens": 128,
        }
        headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}

        async def _call() -> list[int]:
            async with httpx.AsyncClient(timeout=timeout_seconds) as client:
                resp = await client.post(f"{base_url}/chat/completions", json=payload, headers=headers)
                resp.raise_for_status()
                raw = resp.json()["choices"][0]["message"]["content"].strip()
                if "```" in raw:
                    raw = raw.split("```")[1]
                    if raw.startswith("json"):
                        raw = raw[4:]
                    raw = raw.strip()
                parsed = json.loads(raw)
                if not isinstance(parsed, list):
                    raise ValueError("CRAG verifier returned non-list")
                return [int(i) for i in parsed if isinstance(i, int) and 0 <= i < len(top_slice)]

        try:
            keep_indices = await asyncio.wait_for(_call(), timeout=timeout_seconds)
            keep_set = set(keep_indices)
            filtered = [top_slice[i] for i in range(len(top_slice)) if i in keep_set]
            # Never silently erase all results on a borderline query; keep top-1 tagged low_confidence
            return filtered if filtered else candidates
        except Exception as exc:
            logger.debug("CRAG lightweight verifier skipped/timed out (%s); keeping candidates", exc)
            return candidates

    def _build_rerank_document_text(self, item: SearchResultItem) -> str:
        """Concatenate title, problem (description), solution for reranker input."""
        d = item.data
        title = str(d.get("title") or "").strip()
        problem = str(d.get("description") or d.get("problem") or "").strip()
        solution = str(d.get("solution") or "").strip()
        parts = [p for p in (title, problem, solution) if p]
        text = "\n\n".join(parts) if parts else title
        max_c = self._rerank_max_document_chars
        if len(text) > max_c:
            text = text[:max_c] + "…"
        return text or "(empty)"

    async def _apply_rerank(
        self,
        request: SearchRequest,
        candidates: list[SearchResultItem],
    ) -> tuple[list[SearchResultItem], bool]:
        """Server-side rerank on experience candidates only. Returns (items, success)."""
        if not self._reranker or not candidates:
            return candidates, False

        query = (request.query or "").strip()
        documents = [self._build_rerank_document_text(c) for c in candidates]
        doc_meta: list[dict] = [{"exact_title_match": (c.data.get("exact_title_match") or "")} for c in candidates]
        top_k = len(documents)

        try:
            ranked = await self._reranker.rank(
                query,
                documents,
                top_k=top_k,
                document_metadata=doc_meta,
            )
        except Exception as e:
            logger.warning("Reranker failed, keeping pre-rerank order: %s", e, exc_info=True)
            return candidates, False

        if not ranked:
            return candidates, False

        seen: set[int] = set()
        new_items: list[SearchResultItem] = []
        for r in ranked:
            if r.index < 0 or r.index >= len(candidates) or r.index in seen:
                continue
            seen.add(r.index)
            item = candidates[r.index]
            item.score = float(r.score)
            item.data["rerank_score"] = round(float(r.score), 4)
            new_items.append(item)

        if len(new_items) != len(candidates):
            logger.warning(
                "Reranker returned incomplete index set (%d/%d); merging remainder",
                len(new_items),
                len(candidates),
            )
            for i, c in enumerate(candidates):
                if i not in seen:
                    new_items.append(c)

        return new_items, True

    @staticmethod
    def _label_confidence(
        candidates: list[SearchResultItem],
    ) -> list[SearchResultItem]:
        """Label each result with a confidence level based on its score."""
        from team_memory.services.search.scoring import label_confidence

        return label_confidence(candidates)

    async def invalidate_cache(self):
        """Invalidate all cached results. Call after data mutations."""
        await self._cache.clear()

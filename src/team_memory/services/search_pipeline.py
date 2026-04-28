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
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from sqlalchemy.ext.asyncio import AsyncSession

from team_memory import io_logger
from team_memory.config import (
    CacheConfig,
    RetrievalConfig,
    SearchConfig,
)
from team_memory.embedding.base import EmbeddingProvider
from team_memory.services.cache import SearchCache

if TYPE_CHECKING:
    from team_memory.reranker.base import RerankerProvider
from team_memory.storage.archive_repository import ArchiveRepository
from team_memory.storage.database import get_session
from team_memory.storage.repository import ExperienceRepository

logger = logging.getLogger("team_memory.search_pipeline")

_SEARCH_DEBUG = os.environ.get("TEAM_MEMORY_SEARCH_DEBUG", "").lower() in ("1", "true", "yes")

# LLM query expansion cache: normalized query -> (expanded_text, monotonic_timestamp)
_expansion_cache: dict[str, tuple[str, float]] = {}
_EXPANSION_CACHE_TTL = 600  # 10 minutes


def _expansion_cache_clear() -> None:
    """Clear the LLM expansion cache. Useful for testing and cache invalidation."""
    _expansion_cache.clear()


@dataclass
class SearchRequest:
    """Parameters for a search pipeline execution."""

    query: str
    max_results: int = 5
    min_similarity: float = 0.6
    tags: list[str] | None = None
    user_name: str = ""
    current_user: str | None = None
    source: str = "mcp"
    grouped: bool = False
    top_k_children: int = 3
    project: str | None = None
    include_archives: bool = False


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


async def _llm_expand_query(
    query: str,
    llm_config,
    timeout_seconds: float,
) -> str | None:
    """Call LLM to expand search query with extra keywords. Returns None on any failure.

    Results are cached for ``_EXPANSION_CACHE_TTL`` seconds keyed by the
    normalised (stripped + lowercased) query text so repeated identical
    searches skip the LLM round-trip.
    """
    if not query or not query.strip() or not llm_config:
        return None

    cache_key = query.strip().lower()

    # Check cache
    if cache_key in _expansion_cache:
        expanded, ts = _expansion_cache[cache_key]
        if time.monotonic() - ts < _EXPANSION_CACHE_TTL:
            logger.debug("LLM expansion cache hit for: %s", cache_key[:50])
            return expanded
        # Expired -- remove stale entry
        _expansion_cache.pop(cache_key, None)

    try:
        from team_memory.services.llm_client import LLMClient

        client = LLMClient.from_config(llm_config)
        system = (
            "You are a search keyword expansion assistant. Given the user's search intent, "
            "output a single line of space-separated expanded search keywords only, "
            "no explanation, no newlines. Example: 'database connection timeout' -> "
            "'database connection timeout pool connect'. Output only the keyword line."
        )
        text = await asyncio.wait_for(
            client.chat(system, query.strip(), temperature=0.2, timeout=timeout_seconds),
            timeout=timeout_seconds + 1.0,
        )
        if not text:
            return None
        line = text.strip().split("\n")[0].strip()
        result = line if line else None

        # Cache successful result
        if result:
            _expansion_cache[cache_key] = (result, time.monotonic())

        return result
    except (asyncio.TimeoutError, Exception) as e:
        logger.debug("LLM query expansion failed (fallback to original): %s", e)
        return None


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
        repo = ExperienceRepository(session)

        # Stage 1: Cache check
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
                cached.cached = True
                cached.duration_ms = int((time.monotonic() - start) * 1000)
                return cached

        # Query expansion: tag synonyms (simple key<->value)
        stage_begin = time.monotonic()
        retrieval_query = self._expand_query_synonyms(request.query)

        # Optional LLM query expansion
        if getattr(self._search_config, "query_expansion_enabled", False) and self._llm_config:
            timeout_s = getattr(self._search_config, "query_expansion_timeout_seconds", 3.0)
            expanded = await _llm_expand_query(request.query, self._llm_config, timeout_s)
            if expanded and expanded.strip():
                retrieval_query = (request.query.strip() + " " + expanded.strip()).strip()[:2000]

        query_expansion_ms = int((time.monotonic() - stage_begin) * 1000)
        io_logger.log_internal(
            "query_expansion",
            {"query": (request.query or "")[:50], "retrieval_query": (retrieval_query or "")[:50]},
            duration_ms=query_expansion_ms,
        )

        # Lower min_similarity for short queries
        short_threshold = getattr(self._search_config, "short_query_max_chars", 20)
        min_sim_short = getattr(self._search_config, "min_similarity_short", 0.45)
        effective_min_similarity = (
            min_sim_short
            if len(request.query.strip()) <= short_threshold
            else request.min_similarity
        )
        if request.tags:
            tags_lower = [t.lower() for t in request.tags if isinstance(t, str)]
            if "workflow" in tags_lower:
                effective_min_similarity = min(effective_min_similarity, 0.5)

        # Stage 1.5: Entity-graph query enrichment (rule-based, best-effort)
        # Extracts entity names from query and appends them for better recall.
        try:
            from team_memory.services.entity_search import extract_entities_from_query
            entity_names = extract_entities_from_query(retrieval_query)
            if entity_names and self._db_url:
                retrieval_query = retrieval_query.rstrip()
                # Append entity names as additional search terms (deduplicated)
                extra = " ".join(
                    n for n in entity_names
                    if n.lower() not in retrieval_query.lower()
                )
                if extra:
                    retrieval_query = f"{retrieval_query} {extra}"
                retrieval_query = retrieval_query[:2000]
        except Exception:
            pass  # entity enrichment must never break search

        # Stage 2: Embedding
        stage_begin = time.monotonic()
        query_embedding = None
        try:
            query_embedding = await self._cache.get_or_compute_embedding(
                retrieval_query, self._embedding
            )
        except Exception as e:
            logger.warning("Embedding failed, will use FTS only: %s", e, exc_info=True)
        stage_metrics["embedding_ms"] = int((time.monotonic() - stage_begin) * 1000)

        # Stage 3 & 4: Retrieval + RRF Fusion
        stage_begin = time.monotonic()
        mode = self._search_config.mode
        if query_embedding is None:
            mode = "fts"

        candidates = await self._retrieve_and_fuse(
            repo,
            request,
            query_embedding,
            mode,
            retrieval_query=retrieval_query,
            effective_min_similarity=effective_min_similarity,
        )
        stage_metrics["retrieve_fuse_ms"] = int((time.monotonic() - stage_begin) * 1000)
        io_logger.log_internal(
            "retrieve_fuse",
            {
                "query": (retrieval_query or request.query or "")[:50],
                "mode": mode,
                "candidates": len(candidates),
            },
            duration_ms=stage_metrics["retrieve_fuse_ms"],
        )

        # Stage 5: Exact match boost
        candidates = self._apply_exact_match_boost(candidates, request.query)

        total_candidates = len(candidates)

        # Stage 6: Adaptive Filtering
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

        # Confidence labels (after fusion / filter / optional rerank scores)
        candidates = self._label_confidence(candidates)

        # Limit final results
        candidates = candidates[: request.max_results]

        # Build final result dicts
        result_dicts = []
        for item in candidates:
            d = item.data.copy()
            d["score"] = round(float(item.score), 4)
            d["similarity"] = round(item.similarity, 4)
            d["confidence"] = item.confidence
            result_dicts.append(d)

        # Stage 7: Include archives when requested
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
            exp_uuids = [uuid.UUID(eid) for eid in exp_id_strs]
            archive_mapping = await archive_repo_enrich.get_archive_ids_for_experiences(exp_uuids)
            for d in result_dicts:
                if d.get("type") != "archive":
                    d["archive_ids"] = archive_mapping.get(str(d.get("id", "")), [])

        duration_ms = int((time.monotonic() - start) * 1000)

        pipeline_result = SearchPipelineResult(
            results=result_dicts,
            total_candidates=total_candidates,
            search_type=mode,
            reranked=reranked,
            cached=False,
            duration_ms=duration_ms,
            stage_metrics=stage_metrics,
        )

        # Stage 8: Cache store
        if self._cache.enabled:
            stage_begin = time.monotonic()
            await self._cache.put(
                request.query,
                request.tags,
                pipeline_result,
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
        if not query or not self._tag_synonyms:
            return query.strip()
        q = query.strip()
        added: list[str] = []
        for key, value in self._tag_synonyms.items():
            if key and value and key != value:
                if key in q and value not in q:
                    added.append(value)
                if value in q and key not in q:
                    added.append(key)
        if not added:
            return q
        return q + " " + " ".join(dict.fromkeys(added))

    async def _retrieve_and_fuse(
        self,
        repo: ExperienceRepository,
        request: SearchRequest,
        query_embedding: list[float] | None,
        mode: str,
        retrieval_query: str | None = None,
        effective_min_similarity: float | None = None,
    ) -> list[SearchResultItem]:
        """Execute retrieval and RRF fusion."""
        q = (retrieval_query or request.query).strip()
        over_fetch = request.max_results * 3
        min_sim = (
            effective_min_similarity
            if effective_min_similarity is not None
            else request.min_similarity
        )

        if mode == "vector" and query_embedding:
            return await self._vector_search(
                repo, query_embedding, over_fetch, request, min_similarity=min_sim
            )

        if mode == "fts":
            return await self._fts_search(repo, q, over_fetch, request)

        # Hybrid mode: parallel vector + FTS
        if query_embedding is None:
            return await self._fts_search(repo, q, over_fetch, request)

        vector_task = self._vector_search(
            repo, query_embedding, over_fetch, request, min_similarity=min_sim
        )
        fts_task = self._fts_search(repo, q, over_fetch, request)

        vector_results, fts_results = await asyncio.gather(
            vector_task, fts_task, return_exceptions=True
        )

        if isinstance(vector_results, Exception):
            logger.warning("Vector search failed in hybrid mode: %s", vector_results, exc_info=True)
            vector_results = []
        if isinstance(fts_results, Exception):
            logger.warning("FTS failed in hybrid mode: %s", fts_results, exc_info=True)
            fts_results = []

        return self._rrf_fuse(vector_results, fts_results, over_fetch)

    async def _vector_search(
        self,
        repo: ExperienceRepository,
        query_embedding: list[float],
        limit: int,
        request: SearchRequest,
        min_similarity: float | None = None,
    ) -> list[SearchResultItem]:
        """Execute vector similarity search."""
        min_sim = min_similarity if min_similarity is not None else request.min_similarity
        raw = await repo.search_by_vector(
            query_embedding=query_embedding,
            max_results=limit,
            min_similarity=min_sim,
            tags=request.tags,
            project=request.project,
            current_user=request.current_user,
        )
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
        if self._db_url:
            async with get_session(self._db_url) as session2:
                repo_fts = ExperienceRepository(session2)
                raw = await repo_fts.search_by_fts(
                    query_text=query_text,
                    max_results=limit,
                    tags=request.tags,
                    project=request.project,
                    current_user=request.current_user,
                )
        else:
            raw = await repo.search_by_fts(
                query_text=query_text,
                max_results=limit,
                tags=request.tags,
                project=request.project,
                current_user=request.current_user,
            )
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
        """Merge vector and FTS results using Reciprocal Rank Fusion.

        RRF formula: score = sum(weight / (rank + k)) for each source.
        The constant k=60 is empirically optimal (Cormack et al., 2009).
        """
        k = self._search_config.rrf_k
        v_weight = self._search_config.vector_weight
        f_weight = self._search_config.fts_weight

        score_map: dict[str, dict] = {}

        for rank, item in enumerate(vector_results):
            exp_id = item.data.get("id") or item.data.get("group_id", "")
            rrf_contribution = v_weight / (rank + k)
            if exp_id not in score_map:
                score_map[exp_id] = {
                    "rrf_score": 0.0,
                    "item": item,
                    "similarity": item.similarity,
                }
            score_map[exp_id]["rrf_score"] += rrf_contribution
            if item.similarity > score_map[exp_id]["similarity"]:
                score_map[exp_id]["item"] = item
                score_map[exp_id]["similarity"] = item.similarity

        for rank, item in enumerate(fts_results):
            exp_id = item.data.get("id") or item.data.get("group_id", "")
            rrf_contribution = f_weight / (rank + k)
            if exp_id not in score_map:
                score_map[exp_id] = {
                    "rrf_score": 0.0,
                    "item": item,
                    "similarity": item.similarity,
                }
            score_map[exp_id]["rrf_score"] += rrf_contribution

        fused = []
        for _exp_id, data in score_map.items():
            item = data["item"]
            item.rrf_score = data["rrf_score"]
            item.score = data["rrf_score"]
            item.source_type = "hybrid"
            fused.append(item)

        fused.sort(key=lambda x: x.score, reverse=True)
        return fused[:limit]

    def _apply_exact_match_boost(
        self, candidates: list[SearchResultItem], query: str
    ) -> list[SearchResultItem]:
        """Boost score when query exactly matches or is contained in title."""
        if not query or not query.strip():
            return candidates
        q = query.strip()
        q_lower = q.lower()
        exact_boost = 0.4
        contains_boost = 0.15

        for item in candidates:
            data = item.data
            titles_to_check: list[str] = []
            if data.get("title"):
                titles_to_check.append(data["title"])
            for child in data.get("children", []):
                if child.get("title"):
                    titles_to_check.append(child["title"])

            match_type = ""
            for t in titles_to_check:
                t_stripped = (t or "").strip()
                if not t_stripped:
                    continue
                if q == t_stripped:
                    match_type = "exact"
                    break
                if q_lower in t_stripped.lower():
                    match_type = "contains"
                    break

            if match_type:
                data["exact_title_match"] = match_type
                boost = exact_boost if match_type == "exact" else contains_boost
                item.score = float(item.score) + boost
            else:
                data["exact_title_match"] = ""

        candidates.sort(key=lambda x: x.score, reverse=True)
        return candidates

    def _apply_adaptive_filter(self, candidates: list[SearchResultItem]) -> list[SearchResultItem]:
        """Apply adaptive score filtering using similarity scores.

        Two strategies:
        1. Dynamic threshold: filter results below min_confidence_ratio * top-1.
        2. Elbow detection: cut when score gap exceeds threshold.
        """
        if not candidates:
            return candidates

        def _score(item: SearchResultItem) -> float:
            return item.similarity if item.similarity > 0 else item.score

        top_sim = _score(candidates[0]) or 1.0
        min_ratio = self._search_config.min_confidence_ratio
        gap_threshold = self._search_config.score_gap_threshold
        threshold = top_sim * min_ratio
        min_keep = getattr(self._search_config, "adaptive_min_keep", 3)

        filtered = [candidates[0]]

        for i in range(1, len(candidates)):
            curr = candidates[i]
            curr_s = _score(curr)

            if curr.data.get("exact_title_match"):
                filtered.append(curr)
                continue

            if curr_s < threshold:
                break

            prev_s = _score(candidates[i - 1])
            if (
                len(filtered) >= min_keep
                and prev_s > 0
                and (prev_s - curr_s) / prev_s > gap_threshold
            ):
                break

            filtered.append(curr)

        return filtered

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
        doc_meta: list[dict] = [
            {"exact_title_match": (c.data.get("exact_title_match") or "")} for c in candidates
        ]
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
        """Label each result with a confidence level based on its score.

        - high: >= 80% of top score
        - medium: >= 50% of top score
        - low: < 50% of top score
        """
        if not candidates:
            return candidates

        top_score = candidates[0].score if candidates[0].score > 0 else 1.0

        for item in candidates:
            ratio = item.score / top_score if top_score > 0 else 0
            if ratio >= 0.8:
                item.confidence = "high"
            elif ratio >= 0.5:
                item.confidence = "medium"
            else:
                item.confidence = "low"

        return candidates

    async def invalidate_cache(self):
        """Invalidate all cached results. Call after data mutations."""
        await self._cache.clear()

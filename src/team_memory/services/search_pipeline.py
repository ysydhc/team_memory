"""Unified search pipeline — orchestrates the complete retrieval process.

Pipeline stages:
  1. Cache check — return cached results if available
  2. Embedding — encode query to vector
  3. Retrieval — parallel vector + FTS search (hybrid mode) or single mode
  4. RRF Fusion — merge results from multiple sources using Reciprocal Rank Fusion
  5. Adaptive Filtering — dynamic threshold + elbow detection + confidence labeling
  6. Reranking — optional server-side reranking (LLM/cross-encoder/Jina)
  7. Context Compression — trim results to fit token budget
  8. Cache store — save results for future queries
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field

from sqlalchemy.ext.asyncio import AsyncSession

from team_memory.config import CacheConfig, PageIndexLiteConfig, RetrievalConfig, SearchConfig
from team_memory.embedding.base import EmbeddingProvider
from team_memory.reranker.base import RerankerProvider
from team_memory.services.cache import SearchCache
from team_memory.services.context_trimmer import ContextTrimmer
from team_memory.storage.repository import ExperienceRepository

logger = logging.getLogger("team_memory.search_pipeline")


@dataclass
class SearchRequest:
    """Parameters for a search pipeline execution."""

    query: str
    max_results: int = 5
    min_similarity: float = 0.6
    tags: list[str] | None = None
    user_name: str = "anonymous"
    current_user: str | None = None
    source: str = "mcp"
    # Grouped search params
    grouped: bool = False
    top_k_children: int = 3
    min_avg_rating: float = 0.0
    rating_weight: float = 0.3
    use_pageindex_lite: bool | None = None
    project: str | None = None


@dataclass
class SearchResultItem:
    """A single search result with metadata from the pipeline."""

    data: dict  # The experience dict
    score: float = 0.0
    similarity: float = 0.0
    fts_rank: float = 0.0
    rrf_score: float = 0.0
    rerank_score: float | None = None
    confidence: str = "medium"  # "high" | "medium" | "low"
    reranked: bool = False
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
    tree_hits: int = 0
    stage_metrics: dict[str, int] = field(default_factory=dict)


def _expand_query_synonyms(query: str, tag_synonyms: dict[str, str]) -> str:
    """Expand query with synonym terms from tag_synonyms (key<->value) for better recall.

    Cache key must remain the original query; only retrieval uses expanded query.
    """
    if not query or not tag_synonyms:
        return query.strip()
    q = query.strip()
    added: list[str] = []
    for key, value in tag_synonyms.items():
        if key and value and key != value:
            if key in q and value not in q:
                added.append(value)
            if value in q and key not in q:
                added.append(key)
    if not added:
        return q
    return q + " " + " ".join(dict.fromkeys(added))


async def _llm_expand_query(
    query: str,
    llm_config,
    timeout_seconds: float,
) -> str | None:
    """Call LLM to expand search query with extra keywords. Returns None on any failure."""
    if not query or not query.strip() or not llm_config:
        return None
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
        return line if line else None
    except (asyncio.TimeoutError, Exception) as e:
        logger.debug("LLM query expansion failed (fallback to original): %s", e)
        return None


class SearchPipeline:
    """Orchestrates the complete search pipeline.

    This is the main entry point for all search operations. It coordinates:
    - EmbeddingProvider for query vectorization
    - ExperienceRepository for database queries
    - SearchCache for result caching
    - RerankerProvider for optional reranking
    - ContextTrimmer for token budget management
    """

    def __init__(
        self,
        embedding_provider: EmbeddingProvider,
        reranker_provider: RerankerProvider,
        search_config: SearchConfig,
        retrieval_config: RetrievalConfig,
        cache_config: CacheConfig,
        pageindex_lite_config: PageIndexLiteConfig | None = None,
        llm_config=None,
        tag_synonyms: dict[str, str] | None = None,
    ):
        self._embedding = embedding_provider
        self._reranker = reranker_provider
        self._search_config = search_config
        self._retrieval_config = retrieval_config
        self._pageindex_lite_config = pageindex_lite_config
        self._llm_config = llm_config
        self._tag_synonyms = tag_synonyms or {}
        self._cache = SearchCache(
            max_size=cache_config.max_size,
            ttl_seconds=cache_config.ttl_seconds,
            embedding_cache_size=cache_config.embedding_cache_size,
            enabled=cache_config.enabled,
            backend=getattr(cache_config, "backend", "memory"),
            redis_url=getattr(cache_config, "redis_url", "redis://localhost:6379/0"),
        )
        self._trimmer = ContextTrimmer(
            max_tokens=retrieval_config.max_tokens,
            trim_strategy=retrieval_config.trim_strategy,
            summary_model=retrieval_config.summary_model,
            llm_config=llm_config,
        )

    async def search(
        self,
        session: AsyncSession,
        request: SearchRequest,
    ) -> SearchPipelineResult:
        """Execute the full search pipeline."""
        import time

        start = time.monotonic()
        stage_metrics: dict[str, int] = {}
        repo = ExperienceRepository(session)

        # Stage 1: Cache check
        if self._cache.enabled:
            cached = await self._cache.get(
                request.query, request.tags, project=request.project
            )
            if cached is not None:
                duration_ms = int((time.monotonic() - start) * 1000)
                cached.cached = True
                cached.duration_ms = duration_ms
                return cached

        # Query expansion (synonyms) for retrieval only; cache key stays request.query
        retrieval_query = _expand_query_synonyms(
            request.query, self._tag_synonyms
        )
        # Optional LLM query expansion (P1-5): merge LLM keywords; fallback on timeout/failure
        if getattr(self._search_config, "query_expansion_enabled", False) and self._llm_config:
            timeout_s = getattr(
                self._search_config, "query_expansion_timeout_seconds", 3.0
            )
            expanded = await _llm_expand_query(
                request.query, self._llm_config, timeout_s
            )
            if expanded and expanded.strip():
                retrieval_query = (
                    request.query.strip() + " " + expanded.strip()
                ).strip()[:2000]

        # Lower min_similarity for short queries to improve recall
        short_threshold = getattr(
            self._search_config, "short_query_max_chars", 20
        )
        min_sim_short = getattr(
            self._search_config, "min_similarity_short", 0.45
        )
        effective_min_similarity = (
            min_sim_short
            if len(request.query.strip()) <= short_threshold
            else request.min_similarity
        )

        # Stage 2: Embedding
        stage_begin = time.monotonic()
        query_embedding = None
        try:
            query_embedding = await self._cache.get_or_compute_embedding(
                retrieval_query, self._embedding
            )
        except Exception as e:
            logger.warning("Embedding failed, will use FTS only: %s", e)
        stage_metrics["embedding_ms"] = int((time.monotonic() - stage_begin) * 1000)

        # Stage 3 & 4: Retrieval + RRF Fusion
        stage_begin = time.monotonic()
        mode = self._search_config.mode
        if query_embedding is None:
            mode = "fts"  # Force FTS if embedding unavailable

        candidates = await self._retrieve_and_fuse(
            repo,
            request,
            query_embedding,
            mode,
            retrieval_query=retrieval_query,
            effective_min_similarity=effective_min_similarity,
        )
        stage_metrics["retrieve_fuse_ms"] = int((time.monotonic() - stage_begin) * 1000)

        total_candidates = len(candidates)

        # Stage 5: Adaptive Filtering
        stage_begin = time.monotonic()
        if self._search_config.adaptive_filter and candidates:
            candidates = self._apply_adaptive_filter(candidates)
        stage_metrics["adaptive_filter_ms"] = int((time.monotonic() - stage_begin) * 1000)

        # Stage 5.5: PageIndex-Lite candidate enhancement
        stage_begin = time.monotonic()
        tree_hits = 0
        if candidates and self._should_use_pageindex(request):
            candidates, tree_hits = await self._apply_pageindex_lite(
                repo=repo,
                query=request.query,
                candidates=candidates,
            )
        stage_metrics["pageindex_lite_ms"] = int((time.monotonic() - stage_begin) * 1000)

        # Stage 6: Reranking
        stage_begin = time.monotonic()
        reranked = False
        if self._reranker.provider_name != "noop" and candidates:
            candidates = await self._apply_reranker(
                request.query, candidates
            )
            reranked = True
        stage_metrics["rerank_ms"] = int((time.monotonic() - stage_begin) * 1000)

        # Apply confidence labels (always, regardless of reranker)
        candidates = self._label_confidence(candidates)

        # Stage 7: Context Compression
        stage_begin = time.monotonic()
        candidates = await self._trimmer.trim(candidates)
        stage_metrics["trim_ms"] = int((time.monotonic() - stage_begin) * 1000)

        # Limit final results
        candidates = candidates[: request.max_results]

        # Build final result dicts
        result_dicts = []
        for item in candidates:
            d = item.data.copy()
            d["score"] = round(item.score, 4)
            d["similarity"] = round(item.similarity, 4)
            d["confidence"] = item.confidence
            d["reranked"] = item.reranked
            if item.rerank_score is not None:
                d["rerank_score"] = round(item.rerank_score, 4)
            result_dicts.append(d)

        duration_ms = int((time.monotonic() - start) * 1000)

        pipeline_result = SearchPipelineResult(
            results=result_dicts,
            total_candidates=total_candidates,
            search_type=mode,
            reranked=reranked,
            cached=False,
            duration_ms=duration_ms,
            tree_hits=tree_hits,
            stage_metrics=stage_metrics,
        )

        # Stage 8: Cache store
        if self._cache.enabled:
            await self._cache.put(
                request.query,
                request.tags,
                pipeline_result,
                project=request.project,
            )

        return pipeline_result

    def _should_use_pageindex(self, request: SearchRequest) -> bool:
        """Decide whether to apply PageIndex-Lite for this search call."""
        cfg = self._pageindex_lite_config
        if cfg is None:
            return False
        if request.use_pageindex_lite is not None:
            return request.use_pageindex_lite
        return bool(getattr(cfg, "enabled", False))

    async def _apply_pageindex_lite(
        self,
        repo: ExperienceRepository,
        query: str,
        candidates: list[SearchResultItem],
    ) -> tuple[list[SearchResultItem], int]:
        """Apply PageIndex-Lite node matching and score boost to candidates."""
        cfg = self._pageindex_lite_config
        if cfg is None:
            return candidates, 0

        candidate_uuids = []
        child_to_root: dict[str, str] = {}
        for item in candidates:
            data = item.data
            candidate_id = data.get("group_id") or data.get("id")
            if not candidate_id:
                continue
            try:
                import uuid

                candidate_uuids.append(uuid.UUID(candidate_id))
            except Exception:
                continue
            for child in data.get("children", []):
                child_id = child.get("id")
                if not child_id:
                    continue
                child_to_root[child_id] = candidate_id
                try:
                    candidate_uuids.append(uuid.UUID(child_id))
                except Exception:
                    continue

        if not candidate_uuids:
            return candidates, 0
        # De-duplicate IDs before querying tree nodes.
        candidate_uuids = list(dict.fromkeys(candidate_uuids))

        node_map = await repo.search_tree_nodes(
            query_text=query,
            experience_ids=candidate_uuids,
            max_results=max(len(candidate_uuids) * 3, 15),
            min_score=getattr(cfg, "min_node_score", 0.01),
            max_nodes_per_experience=3,
        )

        tree_hits = 0
        tree_weight = getattr(cfg, "tree_weight", 0.15)
        include_nodes = getattr(cfg, "include_matched_nodes", True)

        # Merge child node hits into their root candidate.
        for child_id, root_id in child_to_root.items():
            child_nodes = node_map.get(child_id, [])
            if not child_nodes:
                continue
            prefixed = []
            for n in child_nodes:
                n2 = dict(n)
                n2["path"] = f"child:{n2.get('path', '')}"
                prefixed.append(n2)
            node_map.setdefault(root_id, [])
            node_map[root_id].extend(prefixed)

        for item in candidates:
            data = item.data
            candidate_id = data.get("group_id") or data.get("id")
            if not candidate_id:
                continue
            matched = node_map.get(candidate_id, [])
            if not matched:
                continue
            tree_hits += len(matched)
            best_score = max(m.get("score", 0.0) for m in matched)
            item.score = float(item.score) + float(tree_weight) * float(best_score)
            data["tree_score"] = round(float(best_score), 4)
            if include_nodes:
                data["matched_nodes"] = matched

        candidates.sort(key=lambda x: x.score, reverse=True)
        return candidates, tree_hits

    async def _retrieve_and_fuse(
        self,
        repo: ExperienceRepository,
        request: SearchRequest,
        query_embedding: list[float] | None,
        mode: str,
        retrieval_query: str | None = None,
        effective_min_similarity: float | None = None,
    ) -> list[SearchResultItem]:
        """Execute retrieval and RRF fusion.

        retrieval_query: query used for retrieval (may be synonym-expanded);
        defaults to request.query if not provided.
        effective_min_similarity: override for short-query lower threshold.
        """
        q = (retrieval_query or request.query).strip()
        over_fetch = request.max_results * 3  # Over-fetch for filtering
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

        # Handle exceptions
        if isinstance(vector_results, Exception):
            logger.warning("Vector search failed in hybrid mode: %s", vector_results)
            vector_results = []
        if isinstance(fts_results, Exception):
            logger.warning("FTS failed in hybrid mode: %s", fts_results)
            fts_results = []

        # RRF Fusion
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
        if request.grouped:
            raw = await repo.search_by_vector_grouped(
                query_embedding=query_embedding,
                max_results=limit,
                min_similarity=min_sim,
                tags=request.tags,
                top_k_children=request.top_k_children,
                min_avg_rating=request.min_avg_rating,
                rating_weight=request.rating_weight,
                project=request.project,
                current_user=request.current_user,
            )
            # Grouped results have a different structure
            items = []
            for r in raw:
                items.append(
                    SearchResultItem(
                        data=r,
                        score=r.get("score", 0),
                        similarity=r.get("similarity", 0),
                        source_type="vector",
                    )
                )
            return items

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
        """Execute full-text search."""
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

        # Build score map: experience_id -> {rrf_score, best_item}
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
            # Keep the item with higher similarity
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

        # Build fused results
        fused = []
        for exp_id, data in score_map.items():
            item = data["item"]
            item.rrf_score = data["rrf_score"]
            item.score = data["rrf_score"]
            item.source_type = "hybrid"
            fused.append(item)

        # Sort by RRF score descending
        fused.sort(key=lambda x: x.score, reverse=True)
        return fused[:limit]

    def _apply_adaptive_filter(
        self, candidates: list[SearchResultItem]
    ) -> list[SearchResultItem]:
        """Apply adaptive score filtering using similarity scores.

        Uses the primary score for filtering: similarity when set (e.g. from vector
        search), otherwise score (e.g. from RRF or tests). RRF scores are low
        (~0.01-0.02 with k=60) so pipeline callers pass similarity; tests may pass
        only score.

        Two strategies:
        1. Dynamic threshold: filter results below min_confidence_ratio * top-1.
        2. Elbow detection: cut when score gap (relative to previous) exceeds threshold.
        """
        if not candidates:
            return candidates

        def _score(item: SearchResultItem) -> float:
            return item.similarity if item.similarity > 0 else item.score

        top_sim = _score(candidates[0]) or 1.0
        min_ratio = self._search_config.min_confidence_ratio
        gap_threshold = self._search_config.score_gap_threshold
        threshold = top_sim * min_ratio

        filtered = [candidates[0]]

        for i in range(1, len(candidates)):
            curr = candidates[i]
            curr_s = _score(curr)

            if curr_s < threshold:
                break

            prev_s = _score(candidates[i - 1])
            if prev_s > 0 and (prev_s - curr_s) / prev_s > gap_threshold:
                break

            filtered.append(curr)

        return filtered

    async def _apply_reranker(
        self, query: str, candidates: list[SearchResultItem]
    ) -> list[SearchResultItem]:
        """Apply server-side reranking."""
        # Extract document texts for reranking
        doc_texts = []
        for item in candidates:
            data = item.data
            # Build a text representation for the reranker to score
            parts = []
            if data.get("title"):
                parts.append(data["title"])
            if data.get("description"):
                parts.append(data["description"])
            if data.get("solution"):
                parts.append(data["solution"][:500])  # Truncate long solutions
            doc_texts.append("\n".join(parts) if parts else str(data))

        try:
            rerank_results = await self._reranker.rank(
                query=query,
                documents=doc_texts,
                top_k=len(candidates),  # Rerank all, filter later
            )

            # Map rerank scores back to candidates
            reranked = []
            for rr in rerank_results:
                if rr.index < len(candidates):
                    item = candidates[rr.index]
                    item.rerank_score = rr.score
                    item.reranked = True
                    # Use rerank score as primary score
                    item.score = rr.score
                    reranked.append(item)

            return reranked

        except Exception as e:
            logger.warning("Reranking failed, using original order: %s", e)
            return candidates

    @staticmethod
    def _label_confidence(
        candidates: list[SearchResultItem],
    ) -> list[SearchResultItem]:
        """Label each result with a confidence level based on its score.

        Score boundaries (relative to top-1):
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

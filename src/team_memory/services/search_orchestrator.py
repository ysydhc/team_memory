"""Search orchestrator — owns the read path (search + implicit feedback).

Extracted from ExperienceService to separate read (search) from write
(save/update/delete/feedback) concerns.
"""

from __future__ import annotations

import logging
import time
import uuid
from dataclasses import dataclass

from team_memory import io_logger
from team_memory.embedding.base import EmbeddingProvider
from team_memory.services.evaluation import EvaluationService
from team_memory.services.intent_router import DefaultIntentRouter, IntentRouter
from team_memory.storage.repository import ExperienceRepository

logger = logging.getLogger("team_memory.search_orchestrator")


@dataclass
class OrchestratedSearchResult:
    """Search hits plus pipeline metadata for MCP/API."""

    results: list[dict]
    reranked: bool = False
    cached: bool = False
    intent_type: str = "general"


class SearchOrchestrator:
    """Orchestrates search queries through the pipeline with implicit feedback.

    Accepts only search-related dependencies: SearchPipeline, EmbeddingProvider,
    and db_url.  ExperienceService no longer owns search.
    """

    def __init__(
        self,
        search_pipeline: object | None,
        embedding_provider: EmbeddingProvider,
        db_url: str,
        intent_router: IntentRouter | None = None,
        evaluation_service: EvaluationService | None = None,
    ) -> None:
        self._search_pipeline = search_pipeline
        self._embedding = embedding_provider
        self._db_url = db_url
        self._intent_router = intent_router or DefaultIntentRouter()
        self._evaluation_service = evaluation_service

    async def search(
        self,
        query: str,
        tags: list[str] | None = None,
        max_results: int = 5,
        min_similarity: float = 0.6,
        user_name: str = "",
        source: str = "mcp",
        grouped: bool = False,
        top_k_children: int = 3,
        project: str | None = None,
        include_archives: bool = False,
        include_promoted: bool = False,
    ) -> OrchestratedSearchResult:
        """Search experiences using the enhanced search pipeline.

        If a SearchPipeline is configured, uses the full pipeline
        (hybrid search +
        RRF fusion + adaptive filter + optional rerank + compression).
        Otherwise, falls back to legacy vector/FTS search.
        """
        from team_memory.storage.database import get_session

        # Classify query intent
        intent = await self._intent_router.classify(query)

        async with get_session(self._db_url) as session:
            search_start = time.monotonic()
            repo = ExperienceRepository(session)

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
                    project=project,
                    include_archives=include_archives,
                )
                pipeline_result = await self._search_pipeline.search(session, request)
                duration_ms = int((time.monotonic() - search_start) * 1000)
                io_logger.log_internal(
                    "search",
                    {
                        "query": (query or "")[:50],
                        "result_count": len(pipeline_result.results),
                        "search_type": pipeline_result.search_type,
                    },
                    duration_ms=duration_ms,
                )

                # Implicit feedback: increment use_count for top results
                result_ids: list[uuid.UUID] = []
                for r in pipeline_result.results:
                    eid = r.get("group_id") or r.get("id")
                    if eid:
                        try:
                            result_ids.append(uuid.UUID(str(eid)))
                        except (ValueError, TypeError):
                            pass
                if result_ids:
                    for rid in result_ids:
                        try:
                            await repo.increment_use_count(rid)
                            # Add quality score bonus for recalled experiences
                            await repo.increment_quality_score(rid, 2.0)
                        except Exception:
                            pass

                # Apply status-based post-processing: exclude promoted, penalize draft
                processed_results = self._apply_status_filter(
                    pipeline_result.results, include_promoted=include_promoted
                )

                # Inject [mem:xxx] markers if evaluation_service is configured
                processed_results = self._maybe_inject_markers(processed_results)

                return OrchestratedSearchResult(
                    results=processed_results,
                    reranked=pipeline_result.reranked,
                    cached=pipeline_result.cached,
                    intent_type=intent.intent_type,
                )

            # Legacy fallback: direct vector/FTS search
            results = await self._legacy_search(
                session,
                query=query,
                tags=tags,
                max_results=max_results,
                min_similarity=min_similarity,
                user_name=user_name,
                source=source,
                grouped=grouped,
                top_k_children=top_k_children,
                project=project,
            )
            duration_ms = int((time.monotonic() - search_start) * 1000)
            io_logger.log_internal(
                "search",
                {
                    "query": (query or "")[:50],
                    "result_count": len(results),
                    "search_type": "legacy",
                },
                duration_ms=duration_ms,
            )
            # Apply status-based post-processing to legacy results
            processed_legacy = self._apply_status_filter(
                results, include_promoted=include_promoted
            )

            # Inject [mem:xxx] markers if evaluation_service is configured
            processed_legacy = self._maybe_inject_markers(processed_legacy)

            return OrchestratedSearchResult(
                results=processed_legacy,
                reranked=False,
                cached=False,
                intent_type=intent.intent_type,
            )

    _DRAFT_SCORE_MULTIPLIER = 0.7

    def _maybe_inject_markers(self, results: list[dict]) -> list[dict]:
        """Inject [mem:xxx] markers if evaluation_service is configured."""
        if self._evaluation_service is not None:
            return self._evaluation_service.inject_markers(results)
        return results

    def _apply_status_filter(
        self,
        results: list[dict],
        include_promoted: bool = False,
    ) -> list[dict]:
        """Post-process search results based on exp_status.

        - Promoted experiences are excluded unless include_promoted=True.
        - Draft experiences have their score multiplied by _DRAFT_SCORE_MULTIPLIER.
        """
        filtered: list[dict] = []
        for r in results:
            status = r.get("status") or r.get("exp_status", "published")
            # Exclude promoted unless explicitly requested
            if status == "promoted" and not include_promoted:
                continue
            # Penalize draft scores
            if status == "draft" and "score" in r:
                r = {**r, "score": round(r["score"] * self._DRAFT_SCORE_MULTIPLIER, 4)}
            filtered.append(r)
        return filtered

    async def _legacy_search(
        self,
        session: object,
        query: str,
        tags: list[str] | None = None,
        max_results: int = 5,
        min_similarity: float = 0.6,
        user_name: str = "",
        source: str = "mcp",
        grouped: bool = False,
        top_k_children: int = 3,
        project: str | None = None,
    ) -> list[dict]:
        """Legacy search (vector with FTS fallback, no pipeline)."""
        repo = ExperienceRepository(session)

        try:
            query_embedding = await self._embedding.encode_single(query)
            results = await repo.search_by_vector(
                query_embedding=query_embedding,
                max_results=max_results,
                min_similarity=min_similarity,
                tags=tags,
                project=project,
                current_user=user_name,
            )
        except Exception as e:
            logger.warning("Vector search failed, falling back to FTS: %s", str(e))
            results = await repo.search_by_fts(
                query_text=query,
                max_results=max_results,
                tags=tags,
                project=project,
                current_user=user_name,
            )

        return results

    async def invalidate_cache(self) -> None:
        """Invalidate search cache after data mutations."""
        if self._search_pipeline is not None:
            await self._search_pipeline.invalidate_cache()

"""Post-processing stage: grouping, annotations, archive merge/enrichment, and cache store."""

from __future__ import annotations

import time
import uuid
from dataclasses import asdict
from typing import Any

from sqlalchemy.ext.asyncio import AsyncSession

from team_memory import io_logger
from team_memory.services.search.search_pipeline.models import (
    SearchPipelineResult,
    SearchRequest,
    SearchResultItem,
    _apply_bi_temporal_annotations,
    _pkg_attr,
)
from team_memory.storage.archive_repository import ArchiveRepository as _DefaultArchiveRepository


class SearchPostprocessMixin:
    """Mixin implementing grouping, result dict assembly, archive merge/enrichment, and caching."""

    async def _postprocess_and_assemble(
        self,
        session: AsyncSession,
        request: SearchRequest,
        candidates: list[SearchResultItem],
        *,
        query_embedding: list[float] | None,
        effective_min_similarity: float,
        total_candidates: int,
        mode: str,
        reranked: bool,
        degraded: bool,
        degraded_reason: str | None,
        low_confidence: bool,
        start: float,
        stage_metrics: dict[str, int],
        stage_counts: dict[str, int],
    ) -> SearchPipelineResult:
        """Assemble final SearchPipelineResult, enrich with archives, and store in cache."""
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
        archive_repo_cls = _pkg_attr("ArchiveRepository", _DefaultArchiveRepository)
        post_begin = time.monotonic()
        if request.include_archives and query_embedding is not None:
            archive_repo = archive_repo_cls(session)
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
            archive_repo_enrich = archive_repo_cls(session)
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

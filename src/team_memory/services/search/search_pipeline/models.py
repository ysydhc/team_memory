"""Data models, bi-temporal annotation helpers, and patch-aware lookups for SearchPipeline."""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass, field
from typing import Any, Mapping

_SEARCH_DEBUG = os.environ.get("TEAM_MEMORY_SEARCH_DEBUG", "").lower() in ("1", "true", "yes")

SUPERSESSION_PENALTY: float = 0.3


def _pkg_attr(name: str, default: Any) -> Any:
    """Resolve an attribute from the parent search_pipeline package for mock.patch compatibility."""
    pkg = sys.modules.get("team_memory.services.search.search_pipeline")
    return getattr(pkg, name, default)


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

"""Search routes."""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException

from team_memory.auth.provider import User
from team_memory.storage.database import get_session
from team_memory.web import app as app_module
from team_memory.web.app import (
    SearchRequest,
    _get_db_url,
    _resolve_project,
    get_current_user,
    get_optional_user,
)

router = APIRouter(tags=["search"])


@router.post("/search")
async def search_experiences_api(
    req: SearchRequest,
    user: User | None = Depends(get_optional_user),
):
    """Semantic search for experiences. Supports anonymous access."""
    _service = app_module._service
    _settings = app_module._settings
    db_url = _get_db_url()
    user_name = user.name if user else "anonymous"

    retrieval_cfg = _settings.retrieval if _settings else None
    resolved_project = _resolve_project(req.project)
    max_results = req.max_results or (retrieval_cfg.max_count if retrieval_cfg else 10)
    top_k_children = req.top_k_children or (
        retrieval_cfg.top_k_children if retrieval_cfg else 3
    )
    if req.min_avg_rating is not None:
        min_avg_rating = req.min_avg_rating
    elif retrieval_cfg:
        min_avg_rating = retrieval_cfg.min_avg_rating
    else:
        min_avg_rating = 0.0
    rating_weight = retrieval_cfg.rating_weight if retrieval_cfg else 0.3

    async with get_session(db_url) as session:
        results = await _service.search(
            session=session,
            query=req.query,
            tags=req.tags,
            max_results=max_results,
            min_similarity=req.min_similarity,
            user_name=user_name,
            source="web",
            grouped=req.grouped,
            top_k_children=top_k_children,
            min_avg_rating=min_avg_rating,
            rating_weight=rating_weight,
            use_pageindex_lite=req.use_pageindex_lite,
            project=resolved_project,
        )
        return {
            "results": results,
            "total": len(results),
            "project": resolved_project,
        }


@router.post("/search/debug")
async def search_experiences_debug(
    req: SearchRequest,
    user: User = Depends(get_current_user),
):
    """Debug search pipeline internals for tuning and troubleshooting."""
    _service = app_module._service
    _settings = app_module._settings
    if _service is None or _service._search_pipeline is None:
        raise HTTPException(
            status_code=400, detail="Search pipeline is not enabled"
        )

    db_url = _get_db_url()
    retrieval_cfg = _settings.retrieval if _settings else None
    resolved_project = _resolve_project(req.project)
    max_results = req.max_results or (retrieval_cfg.max_count if retrieval_cfg else 10)
    top_k_children = req.top_k_children or (
        retrieval_cfg.top_k_children if retrieval_cfg else 3
    )
    if req.min_avg_rating is not None:
        min_avg_rating = req.min_avg_rating
    elif retrieval_cfg:
        min_avg_rating = retrieval_cfg.min_avg_rating
    else:
        min_avg_rating = 0.0
    rating_weight = retrieval_cfg.rating_weight if retrieval_cfg else 0.3

    from team_memory.services.search_pipeline import SearchRequest as PipelineSearchRequest

    async with get_session(db_url) as session:
        result = await _service._search_pipeline.search(  # noqa: SLF001
            session,
            PipelineSearchRequest(
                query=req.query,
                tags=req.tags,
                max_results=max_results,
                min_similarity=req.min_similarity,
                user_name=user.name,
                source="web-debug",
                grouped=req.grouped,
                top_k_children=top_k_children,
                min_avg_rating=min_avg_rating,
                rating_weight=rating_weight,
                use_pageindex_lite=req.use_pageindex_lite,
                project=resolved_project,
            ),
        )
    return {
        "results": result.results,
        "total": len(result.results),
        "total_candidates": result.total_candidates,
        "search_type": result.search_type,
        "reranked": result.reranked,
        "cached": result.cached,
        "duration_ms": result.duration_ms,
        "tree_hits": result.tree_hits,
        "stage_metrics": result.stage_metrics,
        "project": resolved_project,
    }

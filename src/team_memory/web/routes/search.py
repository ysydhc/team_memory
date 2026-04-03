"""Search routes."""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException

from team_memory.auth.provider import User
from team_memory.bootstrap import get_context
from team_memory.storage.database import get_session
from team_memory.web import app as app_module
from team_memory.web.app import _get_db_url, _resolve_project
from team_memory.web.auth_session import get_current_user, get_optional_user
from team_memory.web.schemas import SearchRequest

router = APIRouter(tags=["search"])


def _get_search_orchestrator():
    """Get SearchOrchestrator from bootstrap context."""
    return get_context().search_orchestrator


@router.post("/search")
async def search_experiences_api(
    req: SearchRequest,
    user: User | None = Depends(get_optional_user),
):
    """Semantic search for experiences. Supports anonymous access."""
    _settings = app_module._settings
    user_name = user.name if user else "anonymous"

    retrieval_cfg = _settings.retrieval if _settings else None
    resolved_project = _resolve_project(req.project)
    max_results = req.max_results or (retrieval_cfg.max_count if retrieval_cfg else 10)
    top_k_children = req.top_k_children or (retrieval_cfg.top_k_children if retrieval_cfg else 3)

    search_orchestrator = _get_search_orchestrator()
    results = await search_orchestrator.search(
        query=req.query,
        tags=req.tags,
        max_results=max_results,
        min_similarity=req.min_similarity,
        user_name=user_name,
        source="web",
        grouped=req.grouped,
        top_k_children=top_k_children,
        project=resolved_project,
        include_archives=req.include_archives,
    )
    return {
        "results": results,
        "total": len(results),
        "limit": max_results,
    }


@router.post("/search/debug")
async def search_experiences_debug(
    req: SearchRequest,
    user: User = Depends(get_current_user),
):
    """Debug search pipeline internals for tuning and troubleshooting."""
    _settings = app_module._settings
    search_orchestrator = _get_search_orchestrator()
    if search_orchestrator is None or search_orchestrator._search_pipeline is None:
        raise HTTPException(status_code=400, detail="Search pipeline is not enabled")

    db_url = _get_db_url()
    retrieval_cfg = _settings.retrieval if _settings else None
    resolved_project = _resolve_project(req.project)
    max_results = req.max_results or (retrieval_cfg.max_count if retrieval_cfg else 10)
    top_k_children = req.top_k_children or (retrieval_cfg.top_k_children if retrieval_cfg else 3)

    from team_memory.services.search_pipeline import SearchRequest as PipelineSearchRequest

    async with get_session(db_url) as session:
        result = await search_orchestrator._search_pipeline.search(  # noqa: SLF001
            session,
            PipelineSearchRequest(
                query=req.query,
                tags=req.tags,
                max_results=max_results,
                min_similarity=req.min_similarity,
                user_name=user.name,
                current_user=user.name,
                source="web-debug",
                grouped=req.grouped,
                top_k_children=top_k_children,
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

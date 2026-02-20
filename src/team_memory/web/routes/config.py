"""Config: retrieval, search, cache, reranker, project, lifecycle, review, memory, webhooks."""

from __future__ import annotations

import json as json_mod
import os
import time

import httpx
from fastapi import APIRouter, Depends, HTTPException

from team_memory.auth.permissions import require_role
from team_memory.auth.provider import User
from team_memory.web import app as app_module
from team_memory.web.app import (
    CacheConfigUpdate,
    DefaultProjectConfigUpdate,
    LifecycleConfigUpdate,
    MemoryConfigUpdate,
    PageIndexLiteConfigUpdate,
    RerankerConfigUpdate,
    RetrievalConfigUpdate,
    ReviewConfigUpdate,
    SearchConfigUpdate,
    _all_config_dict,
    _pageindex_lite_config_dict,
    _retrieval_config_dict,
    get_current_user,
)

router = APIRouter(tags=["config"])


def _cfg():
    return app_module._settings


def _svc():
    return app_module._service


@router.get("/config/retrieval")
async def get_retrieval_config(user: User = Depends(get_current_user)):
    """Get current retrieval configuration (requires auth)."""
    from team_memory.config import RetrievalConfig

    cfg = _cfg().retrieval if _cfg() else RetrievalConfig()
    return _retrieval_config_dict(cfg)


@router.put("/config/retrieval")
async def update_retrieval_config(
    req: RetrievalConfigUpdate,
    user: User = Depends(require_role("admin")),
):
    """Update retrieval configuration in-memory (requires auth)."""
    if not _cfg():
        raise HTTPException(status_code=500, detail="Settings not initialized")

    if req.trim_strategy not in ("top_k", "summary"):
        raise HTTPException(
            status_code=422, detail="trim_strategy must be 'top_k' or 'summary'"
        )

    cfg = _cfg().retrieval
    cfg.max_tokens = req.max_tokens
    cfg.max_count = req.max_count
    cfg.trim_strategy = req.trim_strategy
    cfg.top_k_children = req.top_k_children
    cfg.min_avg_rating = req.min_avg_rating
    cfg.rating_weight = req.rating_weight
    cfg.summary_model = req.summary_model if req.summary_model else None

    return {
        "message": "Retrieval config updated",
        "config": _retrieval_config_dict(cfg),
    }


@router.get("/config/pageindex-lite")
async def get_pageindex_lite_config(user: User = Depends(get_current_user)):
    """Get current PageIndex-Lite configuration."""
    from team_memory.config import PageIndexLiteConfig

    cfg = _cfg().pageindex_lite if _cfg() else PageIndexLiteConfig()
    return _pageindex_lite_config_dict(cfg)


@router.put("/config/pageindex-lite")
async def update_pageindex_lite_config(
    req: PageIndexLiteConfigUpdate,
    user: User = Depends(require_role("admin")),
):
    """Update PageIndex-Lite config in-memory."""
    if not _cfg():
        raise HTTPException(status_code=500, detail="Settings not initialized")

    cfg = _cfg().pageindex_lite
    cfg.enabled = req.enabled
    cfg.only_long_docs = req.only_long_docs
    cfg.min_doc_chars = req.min_doc_chars
    cfg.max_tree_depth = req.max_tree_depth
    cfg.max_nodes_per_doc = req.max_nodes_per_doc
    cfg.max_node_chars = req.max_node_chars
    cfg.tree_weight = req.tree_weight
    cfg.min_node_score = req.min_node_score
    cfg.include_matched_nodes = req.include_matched_nodes
    return {
        "message": "PageIndex-Lite config updated",
        "config": _pageindex_lite_config_dict(cfg),
    }


@router.get("/config/project")
async def get_default_project_config(user: User = Depends(get_current_user)):
    """Get current default project config."""
    if not _cfg():
        return {"default_project": "default"}
    return {
        "default_project": _cfg().default_project,
        "env_project": os.environ.get("TEAM_MEMORY_PROJECT", ""),
    }


@router.put("/config/project")
async def update_default_project_config(
    req: DefaultProjectConfigUpdate,
    user: User = Depends(require_role("admin")),
):
    """Update default project in memory."""
    if not _cfg():
        raise HTTPException(status_code=500, detail="Settings not initialized")
    project = (req.default_project or "").strip()
    if not project:
        raise HTTPException(
            status_code=422, detail="default_project cannot be empty"
        )
    app_module._settings.default_project = project
    return {"message": "Default project updated", "default_project": project}


@router.get("/config/all")
async def get_all_config(user: User = Depends(get_current_user)):
    """Get all pipeline-related configuration sections."""
    return _all_config_dict()


@router.put("/config/search")
async def update_search_config(
    req: SearchConfigUpdate,
    user: User = Depends(require_role("admin")),
):
    """Update search pipeline configuration in-memory."""
    if not _cfg():
        raise HTTPException(status_code=500, detail="Settings not initialized")
    if req.mode not in ("hybrid", "vector", "fts"):
        raise HTTPException(
            status_code=422, detail="mode must be 'hybrid', 'vector', or 'fts'"
        )
    cfg = _cfg().search
    cfg.mode = req.mode
    cfg.rrf_k = req.rrf_k
    cfg.vector_weight = req.vector_weight
    cfg.fts_weight = req.fts_weight
    cfg.adaptive_filter = req.adaptive_filter
    cfg.score_gap_threshold = req.score_gap_threshold
    cfg.min_confidence_ratio = req.min_confidence_ratio
    return {"message": "Search config updated"}


@router.put("/config/cache")
async def update_cache_config(
    req: CacheConfigUpdate,
    user: User = Depends(require_role("admin")),
):
    """Update cache configuration in-memory."""
    if not _cfg():
        raise HTTPException(status_code=500, detail="Settings not initialized")
    cfg = _cfg().cache
    cfg.enabled = req.enabled
    cfg.ttl_seconds = req.ttl_seconds
    cfg.max_size = req.max_size
    cfg.embedding_cache_size = req.embedding_cache_size
    return {"message": "Cache config updated"}


@router.put("/config/reranker")
async def update_reranker_config(
    req: RerankerConfigUpdate,
    user: User = Depends(require_role("admin")),
):
    """Update reranker provider in-memory."""
    if not _cfg():
        raise HTTPException(status_code=500, detail="Settings not initialized")
    if req.provider not in ("none", "ollama_llm", "cross_encoder", "jina"):
        raise HTTPException(status_code=422, detail="Invalid provider")
    app_module._settings.reranker.provider = req.provider
    return {
        "message": f"Reranker provider set to '{req.provider}'. Restart server to apply."
    }


@router.get("/config/webhooks")
async def get_webhook_config(user: User = Depends(get_current_user)):
    """Get current webhook configuration."""
    if not _cfg():
        return []
    return [w.model_dump() for w in _cfg().webhooks]


@router.put("/config/webhooks")
async def update_webhook_config(
    req: list[dict],
    user: User = Depends(get_current_user),
):
    """Update webhook configuration at runtime."""
    from team_memory.config import WebhookItemConfig

    try:
        new_webhooks = [WebhookItemConfig.model_validate(w) for w in req]
    except Exception as e:
        raise HTTPException(status_code=422, detail=str(e))

    if _cfg():
        app_module._settings.webhooks = new_webhooks
    return {
        "message": f"Webhook 配置已更新 ({len(new_webhooks)} 条目)",
        "count": len(new_webhooks),
    }


@router.post("/config/webhooks/test")
async def test_webhook(
    req: dict,
    user: User = Depends(get_current_user),
):
    """Send a test webhook payload to a specified URL."""
    url = req.get("url", "")
    if not url:
        raise HTTPException(status_code=422, detail="url is required")

    test_payload = {
        "event": "test.ping",
        "payload": {
            "message": "team_memory webhook test",
            "timestamp": time.time(),
        },
        "timestamp": time.time(),
    }
    body = json_mod.dumps(test_payload)

    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            resp = await client.post(
                url,
                content=body.encode(),
                headers={"Content-Type": "application/json"},
            )
            return {
                "status": resp.status_code,
                "success": resp.status_code < 400,
                "response_text": resp.text[:500],
            }
    except Exception as e:
        return {"status": 0, "success": False, "error": str(e)}


@router.get("/embedding-queue/status")
async def embedding_queue_status(user: User = Depends(get_current_user)):
    """Get embedding queue status (D2)."""
    if _svc() and _svc()._embedding_queue:
        return _svc()._embedding_queue.status
    return {"running": False, "message": "Embedding queue not configured"}


@router.post("/cache/clear")
async def clear_cache(user: User = Depends(get_current_user)):
    """Clear the search result cache."""
    if _svc():
        await _svc().invalidate_search_cache()
    return {"message": "Cache cleared"}


@router.get("/config/lifecycle")
async def get_lifecycle_config(user: User = Depends(get_current_user)):
    """Get lifecycle configuration."""
    if not _cfg():
        return {}
    c = _cfg()
    return {
        "stale_months": c.lifecycle.stale_months,
        "scan_interval_hours": c.lifecycle.scan_interval_hours,
        "duplicate_threshold": c.lifecycle.duplicate_threshold,
        "dedup_on_save": c.lifecycle.dedup_on_save,
        "dedup_on_save_threshold": c.lifecycle.dedup_on_save_threshold,
    }


@router.put("/config/lifecycle")
async def update_lifecycle_config(
    req: LifecycleConfigUpdate,
    user: User = Depends(require_role("admin")),
):
    """Update lifecycle configuration in-memory."""
    if not _cfg():
        raise HTTPException(status_code=500, detail="Settings not initialized")
    s = app_module._settings
    s.lifecycle.stale_months = req.stale_months
    s.lifecycle.scan_interval_hours = req.scan_interval_hours
    s.lifecycle.duplicate_threshold = req.duplicate_threshold
    s.lifecycle.dedup_on_save = req.dedup_on_save
    s.lifecycle.dedup_on_save_threshold = req.dedup_on_save_threshold
    return {"message": "Lifecycle config updated"}


@router.get("/config/review")
async def get_review_config(user: User = Depends(get_current_user)):
    """Get review workflow configuration."""
    if not _cfg():
        return {}
    c = _cfg()
    return {
        "enabled": c.review.enabled,
        "auto_publish_threshold": c.review.auto_publish_threshold,
        "require_review_for_ai": c.review.require_review_for_ai,
    }


@router.put("/config/review")
async def update_review_config(
    req: ReviewConfigUpdate,
    user: User = Depends(require_role("admin")),
):
    """Update review configuration in-memory."""
    if not _cfg():
        raise HTTPException(status_code=500, detail="Settings not initialized")
    s = app_module._settings
    s.review.enabled = req.enabled
    s.review.auto_publish_threshold = req.auto_publish_threshold
    s.review.require_review_for_ai = req.require_review_for_ai
    if _svc():
        _svc()._review_config = s.review
    return {"message": "Review config updated"}


@router.get("/config/memory")
async def get_memory_config(user: User = Depends(get_current_user)):
    """Get memory/summary configuration."""
    if not _cfg():
        return {}
    c = _cfg()
    return {
        "auto_summarize": c.memory.auto_summarize,
        "summary_threshold_tokens": c.memory.summary_threshold_tokens,
        "summary_model": c.memory.summary_model,
        "batch_size": c.memory.batch_size,
    }


@router.put("/config/memory")
async def update_memory_config(
    req: MemoryConfigUpdate,
    user: User = Depends(require_role("admin")),
):
    """Update memory/summary configuration in-memory."""
    if not _cfg():
        raise HTTPException(status_code=500, detail="Settings not initialized")
    s = app_module._settings
    s.memory.auto_summarize = req.auto_summarize
    s.memory.summary_threshold_tokens = req.summary_threshold_tokens
    s.memory.summary_model = req.summary_model
    s.memory.batch_size = req.batch_size
    if _svc():
        _svc()._memory_config = s.memory
    return {"message": "Memory config updated"}

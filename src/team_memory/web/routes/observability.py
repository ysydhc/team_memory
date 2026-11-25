"""Observability dashboard & Golden Eval trigger routes."""

from __future__ import annotations

import asyncio
import logging

from fastapi import APIRouter, Depends, HTTPException, Query

from team_memory.auth.provider import User
from team_memory.bootstrap import get_context
from team_memory.services.analysis.golden_eval import run_ablation_eval
from team_memory.services.analysis.observability import get_observability_overview
from team_memory.web.auth_session import get_current_user

logger = logging.getLogger("team_memory.web.observability")

router = APIRouter(tags=["observability"])

_eval_lock = asyncio.Lock()


@router.get("/observability/overview")
async def observability_overview(
    days: int = Query(30, ge=7, le=365),
    user: User = Depends(get_current_user),
):
    """Return 5-subsystem health status lamps, daily trends, and Golden Eval ablation summary."""
    ctx = get_context()
    return await get_observability_overview(ctx.db_url, days=days)


@router.post("/observability/eval/run")
async def trigger_golden_eval(
    limit: int | None = Query(None, ge=1, le=100),
    user: User = Depends(get_current_user),
):
    """Run 3-way Golden Eval ablation (Vector vs Hybrid vs Full) and persist to history.jsonl."""
    if _eval_lock.locked():
        raise HTTPException(status_code=409, detail="Golden Eval is already running")

    ctx = get_context()
    async with _eval_lock:
        try:
            reranker = getattr(ctx.search_pipeline, "_reranker", None)
            record = await run_ablation_eval(
                db_url=ctx.db_url,
                embedding_provider=ctx.embedding,
                settings=ctx.settings,
                reranker_provider=reranker,
                limit=limit,
            )
            return {"success": True, "eval": record}
        except FileNotFoundError as exc:
            raise HTTPException(status_code=400, detail=str(exc))
        except Exception as exc:
            logger.exception("Golden Eval run failed")
            raise HTTPException(status_code=500, detail=f"Golden Eval failed: {exc}")

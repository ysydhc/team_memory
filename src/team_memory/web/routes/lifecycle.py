"""Lifecycle routes: stale scan, duplicates, merge, quality scoring."""

from __future__ import annotations

import logging

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel

from team_memory.auth.provider import User  # noqa: F401 (used by Depends type hint)
from team_memory.web import app as app_module
from team_memory.web.app import (
    MergeRequest,
    _get_db_url,
    get_current_user,
    get_optional_user,
)

logger = logging.getLogger(__name__)

router = APIRouter(tags=["lifecycle"])


@router.post("/lifecycle/scan-stale")
async def scan_stale_experiences(
    user: User | None = Depends(get_optional_user),
):
    """Manually trigger a scan for stale experiences."""
    _service = app_module._service
    _settings = app_module._settings
    months = _settings.lifecycle.stale_months if _settings else 6
    results = await _service.scan_stale(months=months)
    return {
        "stale_experiences": results,
        "total": len(results),
        "threshold_months": months,
    }


@router.get("/lifecycle/stale")
async def get_stale_experiences(
    user: User | None = Depends(get_optional_user),
):
    """Get currently stale experiences (unused > N months)."""
    _service = app_module._service
    _settings = app_module._settings
    months = _settings.lifecycle.stale_months if _settings else 6
    results = await _service.scan_stale(months=months)
    return {
        "stale_experiences": results,
        "total": len(results),
        "threshold_months": months,
    }


@router.get("/lifecycle/duplicates")
async def find_duplicates(
    threshold: float = 0.92,
    limit: int = 20,
    user: User | None = Depends(get_optional_user),
):
    """Find near-duplicate experience pairs based on embedding similarity."""
    _service = app_module._service
    _settings = app_module._settings
    if _settings:
        threshold = threshold or _settings.lifecycle.duplicate_threshold
    pairs = await _service.find_duplicates(
        threshold=threshold, limit=limit
    )
    return {"duplicates": pairs, "total": len(pairs), "threshold": threshold}


@router.post("/lifecycle/merge")
async def merge_experiences(
    req: MergeRequest,
    user: User = Depends(get_current_user),
):
    """Merge secondary experience into primary."""
    _service = app_module._service
    result = await _service.merge_experiences(
        primary_id=req.primary_id,
        secondary_id=req.secondary_id,
        user=user.name,
    )
    if result is None:
        raise HTTPException(
            status_code=404, detail="One or both experiences not found"
        )
    return {"message": "Merge successful", "experience": result}


# ── Quality scoring endpoints ──────────────────────────────────────


@router.post("/lifecycle/refresh-scores")
async def refresh_quality_scores(
    user: User = Depends(get_current_user),
):
    """Run quality-score decay on all experiences."""
    from team_memory.services.scoring import get_scoring_config, run_decay_batch
    from team_memory.storage.database import get_session

    db_url = _get_db_url()
    cfg = get_scoring_config()
    async with get_session(db_url) as session:
        updated = await run_decay_batch(session, cfg)
        await session.commit()
    return {"message": f"已更新 {updated} 条经验的质量分值", "updated": updated}


@router.get("/lifecycle/outdated")
async def list_outdated_experiences(
    user: User | None = Depends(get_optional_user),
):
    """List experiences with quality_score <= outdated threshold."""
    from sqlalchemy import select as sa_select

    from team_memory.services.scoring import get_scoring_config
    from team_memory.storage.database import get_session
    from team_memory.storage.models import Experience

    cfg = get_scoring_config()
    threshold = cfg.get("outdated_threshold", 0)
    db_url = _get_db_url()
    async with get_session(db_url) as session:
        q = (
            sa_select(Experience)
            .where(Experience.quality_score <= threshold)
            .where(Experience.is_deleted == False)  # noqa: E712
            .order_by(Experience.quality_score.asc())
            .limit(100)
        )
        result = await session.execute(q)
        exps = [e.to_dict() for e in result.scalars().all()]
    return {"experiences": exps, "total": len(exps)}


class ScoreActionRequest(BaseModel):
    action: str  # "restore" | "pin" | "delete"


@router.post("/lifecycle/experiences/{exp_id}/score-action")
async def score_action(
    exp_id: str,
    req: ScoreActionRequest,
    user: User = Depends(get_current_user),
):
    """Manage an experience's quality score: restore, pin, or delete."""
    from team_memory.services.scoring import get_scoring_config
    from team_memory.storage.database import get_session
    from team_memory.storage.models import Experience

    db_url = _get_db_url()
    cfg = get_scoring_config()
    async with get_session(db_url) as session:
        from sqlalchemy import select as sa_select

        q = sa_select(Experience).where(Experience.id == exp_id)
        result = await session.execute(q)
        exp = result.scalar_one_or_none()
        if not exp:
            raise HTTPException(status_code=404, detail="Experience not found")

        if req.action == "restore":
            exp.quality_score = cfg["initial_score"]
            exp.last_decay_date = None
            msg = f"经验已恢复至初始分值 {cfg['initial_score']}"
        elif req.action == "pin":
            exp.pinned = True
            msg = "经验已置顶，不再衰减"
        elif req.action == "unpin":
            exp.pinned = False
            msg = "经验已取消置顶"
        elif req.action == "delete":
            exp.is_deleted = True
            msg = "经验已删除"
        else:
            raise HTTPException(status_code=400, detail=f"Unknown action: {req.action}")

        await session.commit()
        return {"message": msg, "experience": exp.to_dict()}


# ── Merge suggestions ──────────────────────────────────────────────


@router.get("/lifecycle/merge-suggestions")
async def get_merge_suggestions(
    threshold: float = 0.85,
    limit: int = 20,
    user: User | None = Depends(get_optional_user),
):
    """Return experience pairs that are candidates for merging.

    Unlike plain duplicates, merge suggestions only include pairs where both
    experiences have been referenced (use_count > 0).
    """
    _service = app_module._service
    _settings = app_module._settings
    if _settings:
        threshold = threshold or _settings.lifecycle.duplicate_threshold
    pairs = await _service.find_duplicates(threshold=threshold, limit=limit)
    suggestions = [
        p for p in pairs
        if (p.get("a", {}).get("use_count", 0) > 0
            or p.get("b", {}).get("use_count", 0) > 0)
    ]
    return {
        "suggestions": suggestions,
        "total": len(suggestions),
        "threshold": threshold,
    }


class MergePreviewRequest(BaseModel):
    primary_id: str
    secondary_id: str


@router.post("/lifecycle/merge-preview")
async def merge_preview(
    req: MergePreviewRequest,
    user: User = Depends(get_current_user),
):
    """Generate a merged preview of two experiences using LLM."""
    from team_memory.storage.database import get_session
    from team_memory.storage.models import Experience

    db_url = _get_db_url()
    async with get_session(db_url) as session:
        from sqlalchemy import select as sa_select

        qa = sa_select(Experience).where(Experience.id == req.primary_id)
        qb = sa_select(Experience).where(Experience.id == req.secondary_id)
        ra = await session.execute(qa)
        rb = await session.execute(qb)
        exp_a = ra.scalar_one_or_none()
        exp_b = rb.scalar_one_or_none()
        if not exp_a or not exp_b:
            raise HTTPException(404, "One or both experiences not found")

        merged_title = exp_a.title
        merged_desc = _simple_merge_text(
            exp_a.description or "", exp_b.description or ""
        )
        merged_solution = _simple_merge_text(
            exp_a.solution or "", exp_b.solution or ""
        )
        merged_tags = list(set((exp_a.tags or []) + (exp_b.tags or [])))

        return {
            "primary": exp_a.to_dict(),
            "secondary": exp_b.to_dict(),
            "merged": {
                "title": merged_title,
                "description": merged_desc,
                "solution": merged_solution,
                "tags": merged_tags,
            },
        }


def _simple_merge_text(a: str, b: str) -> str:
    """Combine two text blocks, deduplicating identical lines."""
    lines_a = a.strip().splitlines()
    lines_b = b.strip().splitlines()
    seen = set()
    merged = []
    for line in lines_a + lines_b:
        key = line.strip()
        if key and key not in seen:
            seen.add(key)
            merged.append(line)
    return "\n".join(merged)

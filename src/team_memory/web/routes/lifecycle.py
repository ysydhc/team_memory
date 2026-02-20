"""Lifecycle routes: stale scan, duplicates, merge."""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException

from team_memory.auth.provider import User
from team_memory.storage.database import get_session
from team_memory.web import app as app_module
from team_memory.web.app import (
    MergeRequest,
    _get_db_url,
    get_current_user,
)

router = APIRouter(tags=["lifecycle"])


@router.post("/lifecycle/scan-stale")
async def scan_stale_experiences(
    user: User = Depends(get_current_user),
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
    user: User = Depends(get_current_user),
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
    user: User = Depends(get_current_user),
):
    """Find near-duplicate experience pairs based on embedding similarity."""
    _service = app_module._service
    _settings = app_module._settings
    if _settings:
        threshold = threshold or _settings.lifecycle.duplicate_threshold
    db_url = _get_db_url()
    async with get_session(db_url) as session:
        pairs = await _service.find_duplicates(
            session, threshold=threshold, limit=limit
        )
        return {"duplicates": pairs, "total": len(pairs), "threshold": threshold}


@router.post("/lifecycle/merge")
async def merge_experiences(
    req: MergeRequest,
    user: User = Depends(get_current_user),
):
    """Merge secondary experience into primary."""
    _service = app_module._service
    db_url = _get_db_url()
    async with get_session(db_url) as session:
        result = await _service.merge_experiences(
            session=session,
            primary_id=req.primary_id,
            secondary_id=req.secondary_id,
            user=user.name,
        )
        if result is None:
            raise HTTPException(
                status_code=404, detail="One or both experiences not found"
            )
        return {"message": "Merge successful", "experience": result}

"""Semantic duplicate scan and group re-embedding (Web UI).

Paths live under ``/dedup/*`` so they never collide with ``/experiences/{experience_id}``.
"""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException

from team_memory.auth.provider import User
from team_memory.web import app as app_module
from team_memory.web.app import _resolve_project
from team_memory.web.auth_session import get_current_user
from team_memory.web.dependencies import require_role

router = APIRouter(tags=["dedup"])


def _svc():
    if app_module._service is None:
        raise HTTPException(
            status_code=503,
            detail="Service not ready (bootstrap may not have run)",
        )
    return app_module._service


@router.get("/dedup/pairs")
async def list_duplicate_pairs(
    threshold: float = 0.85,
    limit: int = 50,
    project: str | None = None,
    _user: User = Depends(get_current_user),
):
    """Scan published root experiences for high vector similarity."""
    svc = _svc()
    resolved = _resolve_project(project)
    return await svc.find_duplicate_pairs(threshold=threshold, limit=limit, project=resolved)


@router.post("/dedup/reembed-group-vectors")
async def reembed_group_vectors(
    project: str | None = None,
    _user: User = Depends(require_role("update")),
):
    """Recompute parent embeddings from parent+children text for all groups."""
    svc = _svc()
    resolved = _resolve_project(project)
    return await svc.reembed_group_parent_vectors(project=resolved)

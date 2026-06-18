"""Experience CRUD, feedback, status routes."""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, Query, Request
from fastapi.responses import JSONResponse

from team_memory.auth.provider import User
from team_memory.web import app as app_module
from team_memory.web.app import _resolve_project
from team_memory.web.auth_session import get_current_user
from team_memory.web.dependencies import require_role
from team_memory.web.schemas import (
    ExperienceCreate,
    ExperienceUpdate,
    FeedbackCreate,
)

router = APIRouter(tags=["experiences"])


def _svc():
    if app_module._service is None:
        raise HTTPException(
            status_code=503,
            detail="Service not ready (bootstrap may not have run)",
        )
    return app_module._service


@router.get("/projects")
async def list_projects(
    user: User = Depends(get_current_user),
):
    """List distinct project names from experiences."""
    projects = await _svc().list_projects()
    return {"projects": projects}


@router.get("/experiences")
async def list_experiences(
    limit: int = Query(20, ge=1, le=100),
    offset: int = Query(0, ge=0),
    tag: str | None = None,
    project: str | None = None,
    status: str | None = None,
    visibility: str | None = None,
    experience_type: str | None = None,
    user: User = Depends(get_current_user),
):
    """List experiences with pagination and filters."""
    return await _svc().list_experiences_filtered(
        limit=limit,
        offset=offset,
        tag=tag,
        project=project,
        status=status,
        visibility=visibility,
        experience_type=experience_type,
        current_user=user.name,
    )


@router.get("/experiences/{experience_id}")
async def get_experience(
    experience_id: str,
    user: User = Depends(get_current_user),
):
    """Get a single experience by ID."""
    data = await _svc().get_experience_detail(experience_id)
    if data is None:
        raise HTTPException(status_code=404, detail="Experience not found")
    return data


@router.post("/experiences")
async def create_experience(
    req: ExperienceCreate,
    user: User = Depends(require_role("create")),
):
    """Create a new experience."""
    resolved_project = _resolve_project(req.project)
    result = await _svc().save(
        title=req.title,
        problem=req.problem,
        solution=req.solution,
        created_by=user.name,
        tags=req.tags,
        source="web",
        exp_status=req.status,
        visibility=req.visibility,
        skip_dedup=req.skip_dedup_check,
        experience_type=req.experience_type,
        project=resolved_project,
        group_key=req.group_key,
    )
    if result.get("error"):
        return JSONResponse(
            status_code=500,
            content={
                "error": {
                    "code": "embedding_failure",
                    "message": result.get("message", "Internal error"),
                }
            },
        )
    if result.get("status") == "duplicate_detected":
        return JSONResponse(
            status_code=409,
            content={
                "error": {
                    "code": "duplicate_detected",
                    "message": result.get("message", "Duplicate detected"),
                },
                "candidates": result.get("candidates", []),
            },
        )
    return {"item": result, "message": "Created successfully"}


@router.put("/experiences/{experience_id}")
async def update_experience_route(
    experience_id: str,
    req: ExperienceUpdate,
    user: User = Depends(require_role("update")),
):
    """In-place update: only provided fields are modified."""
    raw = req.model_dump(exclude_unset=True)

    kwargs: dict = {}
    if "title" in raw:
        kwargs["title"] = raw["title"]
    if "problem" in raw:
        kwargs["description"] = raw["problem"]
    if "solution" in raw:
        kwargs["solution"] = raw["solution"]
    if "tags" in raw:
        kwargs["tags"] = raw["tags"]
    if "experience_type" in raw:
        kwargs["experience_type"] = raw["experience_type"]
    if "exp_status" in raw:
        kwargs["exp_status"] = raw["exp_status"]
    if "visibility" in raw:
        kwargs["visibility"] = raw["visibility"]
    if "solution_addendum" in raw:
        kwargs["solution_addendum"] = raw["solution_addendum"]

    result = await _svc().update(
        experience_id=experience_id,
        user=user.name,
        **kwargs,
    )
    if result is None:
        raise HTTPException(status_code=404, detail="Experience not found")
    return result


@router.delete("/experiences/{experience_id}")
async def delete_experience(
    experience_id: str,
    hard: bool = False,
    user: User = Depends(require_role("delete")),
):
    """Soft-delete an experience (or hard-delete with ?hard=true)."""
    if hard:
        deleted = await _svc().hard_delete(experience_id)
    else:
        deleted = await _svc().soft_delete(experience_id)
    if not deleted:
        raise HTTPException(status_code=404, detail="Experience not found")
    return {"message": "Experience deleted"}


@router.post("/experiences/{experience_id}/restore")
async def restore_experience(
    experience_id: str,
    user: User = Depends(get_current_user),
):
    """Restore a soft-deleted experience."""
    restored = await _svc().restore(experience_id)
    if not restored:
        raise HTTPException(status_code=404, detail="Experience not found or not deleted")
    return {"message": "Experience restored"}


@router.post("/experiences/{experience_id}/feedback")
async def add_feedback(
    experience_id: str,
    req: FeedbackCreate,
    user: User = Depends(get_current_user),
):
    """Add feedback to an experience."""
    success = await _svc().feedback(
        experience_id=experience_id,
        rating=req.rating,
        feedback_by=user.name,
        comment=req.comment,
    )
    if not success:
        raise HTTPException(status_code=404, detail="Experience not found")
    return {"message": "Feedback recorded"}


@router.patch("/experiences/{experience_id}/pin")
async def toggle_pin_experience(
    experience_id: str,
    user: User = Depends(require_role("update")),
):
    """Toggle pin status for an experience.

    Pinned experiences are exempt from quality score decay.
    Body: { "pinned": true | false }
    """
    try:
        result = await _svc().toggle_pin(experience_id)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    return result


@router.post("/experiences/{experience_id}/revive")
async def revive_outdated_experience(
    experience_id: str,
    user: User = Depends(require_role("update")),
):
    """Revive an Outdated experience by resetting quality_score to 100.

    Sets quality_score=100, quality_tier='Silver', last_scored_at=now.
    """
    try:
        result = await _svc().revive(experience_id)
    except ValueError as e:
        msg = str(e)
        status_code = 404 if "not found" in msg.lower() else 400
        raise HTTPException(status_code=status_code, detail=msg)
    return result


@router.patch("/experiences/{experience_id}/status")
async def change_experience_status(
    experience_id: str,
    request: Request,
    user: User = Depends(get_current_user),
):
    """Change experience status.

    Body: { "status": "published"|"draft", "visibility": "private"|"project"|"global" }
    """
    body = {}
    try:
        body = await request.json()
    except Exception:
        pass

    new_status = body.get("status")
    new_visibility = body.get("visibility")

    if new_status and new_status not in ("draft", "published"):
        raise HTTPException(
            status_code=400,
            detail="Only status=draft or status=published is allowed",
        )
    if not new_status and not new_visibility:
        raise HTTPException(status_code=400, detail="Must provide status or visibility")

    is_admin = user.role == "admin"
    try:
        result = await _svc().change_status(
            experience_id,
            new_status=new_status,
            visibility=new_visibility,
            changed_by=user.name,
            is_admin=is_admin,
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    if result is None:
        raise HTTPException(status_code=404, detail="Experience not found")

    status_messages = {
        "draft": "已退回草稿",
        "published": "已发布",
    }
    msg = status_messages.get(new_status, "状态已更新")
    if new_visibility:
        vis_labels = {"private": "仅自己", "project": "项目内", "global": "全局"}
        msg += f"，可见范围：{vis_labels.get(new_visibility, new_visibility)}"

    return {"message": msg, "experience": result}

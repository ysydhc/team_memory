"""Experience CRUD, feedback, status routes."""

from __future__ import annotations

import uuid

from fastapi import APIRouter, Depends, HTTPException, Query, Request
from fastapi.responses import JSONResponse
from sqlalchemy import distinct, or_, select
from sqlalchemy import func as sa_func
from sqlalchemy.orm import selectinload

from team_memory.auth.provider import User
from team_memory.storage.models import Experience
from team_memory.web import app as app_module
from team_memory.web.app import _get_db_url, _resolve_project
from team_memory.web.auth_session import get_current_user
from team_memory.web.dependencies import require_role
from team_memory.web.schemas import ExperienceCreate, ExperienceUpdate, FeedbackCreate

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
    db_url = _get_db_url()
    async with app_module.get_session(db_url) as session:
        result = await session.execute(
            select(distinct(Experience.project))
            .where(Experience.is_deleted == False)  # noqa: E712
            .where(Experience.project.is_not(None))
            .order_by(Experience.project)
        )
        projects = [row[0] for row in result.all() if row[0]]
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
    db_url = _get_db_url()
    project_list = (
        [p.strip() for p in project.split(",") if p.strip()] if project and "," in project else None
    )
    resolved_project = _resolve_project(project.split(",")[0] if project else project)
    current_user = user.name

    def _project_filter(q):
        if project_list:
            return q.where(
                or_(
                    Experience.project.in_(project_list),
                    Experience.visibility == "global",
                )
            )
        return q.where(
            or_(
                Experience.project == resolved_project,
                Experience.visibility == "global",
            )
        )

    def _vis_where(q):
        if visibility and visibility != "all":
            vis_map = {"personal": "private", "team": "project", "global": "global"}
            vis_val = vis_map.get(visibility, visibility)
            q = q.where(Experience.visibility == vis_val)
            if vis_val == "private" and current_user:
                q = q.where(Experience.created_by == current_user)
            if vis_val != "global":
                q = _project_filter(q)
        else:
            q = _project_filter(q)
            if current_user:
                q = q.where(
                    or_(
                        Experience.visibility != "private",
                        Experience.created_by == current_user,
                    )
                )
            else:
                q = q.where(Experience.visibility != "private")
        return q

    include_all = status in ("all", "draft")

    async with app_module.get_session(db_url) as session:
        from team_memory.storage.repository import ExperienceRepository

        repo = ExperienceRepository(session)

        if tag or experience_type or status:
            base_q = (
                select(Experience)
                .where(Experience.parent_id.is_(None))
                .where(Experience.is_deleted == False)  # noqa: E712
                .options(selectinload(Experience.children))
            )
            base_q = _vis_where(base_q)
            count_q = (
                select(sa_func.count())
                .select_from(Experience)
                .where(Experience.parent_id.is_(None))
                .where(Experience.is_deleted == False)  # noqa: E712
            )
            count_q = _vis_where(count_q)
            if status and status not in ("all",):
                base_q = base_q.where(Experience.exp_status == status)
                count_q = count_q.where(Experience.exp_status == status)
            elif not include_all:
                base_q = base_q.where(Experience.exp_status == "published")
                count_q = count_q.where(Experience.exp_status == "published")
            if tag:
                base_q = base_q.where(Experience.tags.overlap([tag]))
                count_q = count_q.where(Experience.tags.overlap([tag]))
            if experience_type:
                base_q = base_q.where(Experience.experience_type == experience_type)
                count_q = count_q.where(Experience.experience_type == experience_type)

            base_q = base_q.order_by(Experience.created_at.desc()).limit(limit).offset(offset)
            result = await session.execute(base_q)
            experiences = list(result.scalars().all())
            total_result = await session.execute(count_q)
            total = total_result.scalar() or 0
        else:
            total = await repo.count(
                include_deleted=False if include_all else True,
                project=resolved_project,
                scope=visibility,
                current_user=current_user,
            )
            experiences = await repo.list_recent(
                limit=limit,
                offset=offset,
                include_all_statuses=include_all,
                project=resolved_project,
                scope=visibility,
                current_user=current_user,
            )

        exp_list = []
        for exp in experiences:
            d = exp.to_dict()
            d["children_count"] = len(exp.children) if exp.children else 0
            exp_list.append(d)

        return {
            "items": exp_list,
            "total": total,
            "limit": limit,
            "offset": offset,
        }


@router.get("/experiences/{experience_id}")
async def get_experience(
    experience_id: str,
    user: User = Depends(get_current_user),
):
    """Get a single experience by ID."""
    db_url = _get_db_url()
    async with app_module.get_session(db_url) as session:
        result = await session.execute(
            select(Experience)
            .options(
                selectinload(Experience.feedbacks),
                selectinload(Experience.children),
            )
            .where(Experience.id == uuid.UUID(experience_id))
            .where(Experience.is_deleted == False)  # noqa: E712
        )
        exp = result.scalar_one_or_none()
        if exp is None:
            raise HTTPException(status_code=404, detail="Experience not found")

        data = exp.to_dict()
        data["feedbacks"] = [fb.to_dict() for fb in exp.feedbacks]
        data["children"] = [c.to_dict() for c in (exp.children or [])]
        return data


@router.post("/experiences")
async def create_experience(
    req: ExperienceCreate,
    user: User = Depends(require_role("create")),
):
    """Create a new experience."""
    db_url = _get_db_url()
    resolved_project = _resolve_project(req.project)
    async with app_module.get_session(db_url) as session:
        result = await _svc().save(
            session=session,
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
    db_url = _get_db_url()
    if hard:
        async with app_module.get_session(db_url) as session:
            from team_memory.storage.repository import ExperienceRepository

            repo = ExperienceRepository(session)
            deleted = await repo.delete(uuid.UUID(experience_id))
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
    db_url = _get_db_url()
    async with app_module.get_session(db_url) as session:
        from team_memory.storage.repository import ExperienceRepository

        repo = ExperienceRepository(session)
        exp_uuid = uuid.UUID(experience_id)
        exp = await repo.get_by_id(exp_uuid)
        if exp is None:
            raise HTTPException(status_code=404, detail="Experience not found")

        new_pinned = not exp.is_pinned
        await repo.update(exp_uuid, is_pinned=new_pinned)

        label = "已置顶" if new_pinned else "已取消置顶"
        return {
            "message": label,
            "is_pinned": new_pinned,
        }


@router.post("/experiences/{experience_id}/revive")
async def revive_outdated_experience(
    experience_id: str,
    user: User = Depends(require_role("update")),
):
    """Revive an Outdated experience by resetting quality_score to 100.

    Sets quality_score=100, quality_tier='Silver', last_scored_at=now.
    """
    db_url = _get_db_url()
    async with app_module.get_session(db_url) as session:
        from datetime import datetime, timezone

        from team_memory.storage.repository import ExperienceRepository

        repo = ExperienceRepository(session)
        exp_uuid = uuid.UUID(experience_id)
        exp = await repo.get_by_id(exp_uuid)
        if exp is None:
            raise HTTPException(status_code=404, detail="Experience not found")

        if exp.quality_tier != "Outdated":
            raise HTTPException(
                status_code=400,
                detail="Only Outdated experiences can be revived",
            )

        now = datetime.now(timezone.utc)
        await repo.update(
            exp_uuid,
            quality_score=100.0,
            quality_tier="Silver",
            last_scored_at=now,
        )

        return {
            "message": "经验已恢复，质量评分重置为 100",
            "quality_score": 100.0,
            "quality_tier": "Silver",
        }


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

    db_url = _get_db_url()
    async with app_module.get_session(db_url) as session:
        from team_memory.storage.repository import ExperienceRepository

        repo = ExperienceRepository(session)
        exp_uuid = uuid.UUID(experience_id)

        if new_status:
            is_admin = user.role == "admin"
            try:
                result = await repo.change_status(
                    experience_id=exp_uuid,
                    new_status=new_status,
                    visibility=new_visibility,
                    changed_by=user.name,
                    is_admin=is_admin,
                )
            except ValueError as e:
                raise HTTPException(status_code=400, detail=str(e))
        else:
            result = await repo.update(exp_uuid, visibility=new_visibility)

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

    if new_status and result is not None:
        try:
            from team_memory.bootstrap import get_context

            await get_context().archive_service.update_archive_status_for_experience(exp_uuid)
        except Exception:
            pass
    return {"message": msg, "experience": result.to_dict()}

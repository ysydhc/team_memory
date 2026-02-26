"""Experience CRUD, links, versions, groups, drafts, publish routes."""

from __future__ import annotations

import uuid

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import JSONResponse
from sqlalchemy import func as sa_func
from sqlalchemy import or_, select
from sqlalchemy.orm import selectinload

from team_memory.auth.permissions import require_role
from team_memory.auth.provider import User
from team_memory.storage.models import Experience, ExperienceLink, ReviewHistory
from team_memory.web import app as app_module
from team_memory.web.app import (
    ExperienceCreate,
    ExperienceGroupCreate,
    ExperienceLinkCreate,
    ExperienceUpdate,
    FeedbackCreate,
    ReviewRequest,
    _get_db_url,
    _resolve_project,
    get_current_user,
    get_optional_user,
)

router = APIRouter(tags=["experiences"])


def _svc():
    if app_module._service is None:
        raise HTTPException(
            status_code=503,
            detail="Service not ready (bootstrap may not have run)",
        )
    return app_module._service


def _cfg():
    return app_module._settings


@router.get("/experiences")
async def list_experiences(
    page: int = 1,
    page_size: int = 20,
    tag: str | None = None,
    project: str | None = None,
    status: str | None = None,
    scope: str | None = None,
    experience_type: str | None = None,
    severity: str | None = None,
    category: str | None = None,
    progress_status: str | None = None,
    user: User | None = Depends(get_optional_user),
):
    """List experiences with pagination and multi-dimensional filters."""
    db_url = _get_db_url()
    resolved_project = _resolve_project(project)
    include_all = status in ("all", "draft")
    async with app_module.get_session(db_url) as session:
        repo = app_module.ExperienceRepository(session)

        if status == "draft":
            experiences = await repo.list_drafts(
                limit=page_size,
                offset=(page - 1) * page_size,
                project=resolved_project,
            )
            total = await repo.count_drafts(project=resolved_project)
        elif tag or experience_type or severity or category or progress_status:
            offset = (page - 1) * page_size
            base_q = (
                select(Experience)
                .where(Experience.parent_id.is_(None))
                .where(Experience.is_deleted == False)  # noqa: E712
                .where(Experience.project == resolved_project)
                .options(selectinload(Experience.children))
            )
            count_q = (
                select(sa_func.count())
                .select_from(Experience)
                .where(Experience.parent_id.is_(None))
                .where(Experience.is_deleted == False)  # noqa: E712
                .where(Experience.project == resolved_project)
            )
            if not include_all:
                base_q = base_q.where(Experience.publish_status == "published")
                count_q = count_q.where(Experience.publish_status == "published")
            if tag:
                base_q = base_q.where(Experience.tags.overlap([tag]))
                count_q = count_q.where(Experience.tags.overlap([tag]))
            if experience_type:
                base_q = base_q.where(Experience.experience_type == experience_type)
                count_q = count_q.where(Experience.experience_type == experience_type)
            if severity:
                base_q = base_q.where(Experience.severity == severity)
                count_q = count_q.where(Experience.severity == severity)
            if category:
                base_q = base_q.where(Experience.category == category)
                count_q = count_q.where(Experience.category == category)
            if progress_status:
                base_q = base_q.where(Experience.progress_status == progress_status)
                count_q = count_q.where(Experience.progress_status == progress_status)

            base_q = (
                base_q.order_by(Experience.created_at.desc())
                .limit(page_size)
                .offset(offset)
            )
            result = await session.execute(base_q)
            experiences = list(result.scalars().all())
            total_result = await session.execute(count_q)
            total = total_result.scalar() or 0
        else:
            if not include_all:
                total = await repo.count(project=resolved_project)
            else:
                total = await repo.count(include_deleted=False, project=resolved_project)
            offset = (page - 1) * page_size
            experiences = await repo.list_recent(
                limit=page_size,
                offset=offset,
                include_all_statuses=include_all,
                project=resolved_project,
            )

        exp_list = []
        for exp in experiences:
            d = exp.to_dict()
            d["children_count"] = len(exp.children) if exp.children else 0
            exp_list.append(d)

        return {
            "experiences": exp_list,
            "total": total,
            "page": page,
            "page_size": page_size,
            "total_pages": (total + page_size - 1) // page_size,
            "project": resolved_project,
        }


@router.get("/experiences/drafts")
async def list_drafts(
    page: int = 1,
    page_size: int = 20,
    project: str | None = None,
    user: User = Depends(get_current_user),
):
    """List draft experiences for the current user."""
    resolved_project = _resolve_project(project)
    drafts = await _svc().get_drafts(
        created_by=user.name,
        limit=page_size,
        offset=(page - 1) * page_size,
        project=resolved_project,
    )
    return {
        "experiences": drafts,
        "total": len(drafts),
        "page": page,
        "project": resolved_project,
    }


@router.post("/experiences/batch-summarize")
async def batch_summarize(
    limit: int = 10,
    user: User = Depends(get_current_user),
):
    """Batch generate summaries for experiences without one."""
    result = await _svc().batch_generate_summaries(limit=limit)
    return result


@router.get("/experiences/{experience_id}")
async def get_experience(
    experience_id: str,
    user: User | None = Depends(get_optional_user),
):
    """Get a single experience by ID with feedbacks eagerly loaded."""
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

        repo = app_module.ExperienceRepository(session)
        await repo.increment_view_count(exp.id)

        data = exp.to_dict()
        data["feedbacks"] = [fb.to_dict() for fb in exp.feedbacks]
        data["children"] = [c.to_dict() for c in (exp.children or [])]
        return data


@router.get("/experiences/{experience_id}/tree-nodes")
async def get_experience_tree_nodes(
    experience_id: str,
    user: User | None = Depends(get_optional_user),
):
    """Get PageIndex-Lite tree nodes for one experience."""
    db_url = _get_db_url()
    async with app_module.get_session(db_url) as session:
        repo = app_module.ExperienceRepository(session)
        nodes = await repo.get_tree_nodes(uuid.UUID(experience_id))
        return {"experience_id": experience_id, "nodes": [n.to_dict() for n in nodes]}


@router.post("/experiences")
async def create_experience(
    req: ExperienceCreate,
    user: User = Depends(require_role("create")),
):
    """Create a new experience (requires create permission)."""
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
            code_snippets=req.code_snippets,
            language=req.language,
            framework=req.framework,
            source="web",
            root_cause=req.root_cause,
            publish_status=req.publish_status,
            skip_dedup=req.skip_dedup_check,
            experience_type=req.experience_type,
            severity=req.severity,
            category=req.category,
            progress_status=req.progress_status,
            structured_data=req.structured_data,
            git_refs=req.git_refs,
            related_links=req.related_links,
            project=resolved_project,
        )
        if result.get("status") == "duplicate_detected":
            return JSONResponse(status_code=409, content=result)
        return result


@router.post("/experiences/groups")
async def create_experience_group(
    req: ExperienceGroupCreate,
    user: User = Depends(require_role("create")),
):
    """Create a parent experience with children (requires create permission)."""
    db_url = _get_db_url()
    resolved_project = _resolve_project(req.parent.project)

    parent_data = {
        "title": req.parent.title,
        "problem": req.parent.problem,
        "solution": req.parent.solution,
        "tags": req.parent.tags,
        "code_snippets": req.parent.code_snippets,
        "root_cause": req.parent.root_cause,
        "language": req.parent.language,
        "framework": req.parent.framework,
        "source": "web",
        "project": resolved_project,
    }

    children_data = [
        {
            "title": c.title,
            "problem": c.problem,
            "solution": c.solution,
            "tags": c.tags,
            "code_snippets": c.code_snippets,
            "root_cause": c.root_cause,
            "language": c.language,
            "framework": c.framework,
            "source": "web",
            "project": resolved_project,
        }
        for c in req.children
    ]

    async with app_module.get_session(db_url) as session:
        result = await _svc().save_group(
            session=session,
            parent=parent_data,
            children=children_data,
            created_by=user.name,
            project=resolved_project,
        )
        return result


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
    if "root_cause" in raw:
        kwargs["root_cause"] = raw["root_cause"]
    if "tags" in raw:
        kwargs["tags"] = raw["tags"]
    if "code_snippets" in raw:
        kwargs["code_snippets"] = raw["code_snippets"]
    if "language" in raw:
        kwargs["language"] = raw["language"]
    if "framework" in raw:
        kwargs["framework"] = raw["framework"]
    if "experience_type" in raw:
        kwargs["experience_type"] = raw["experience_type"]
    if "severity" in raw:
        kwargs["severity"] = raw["severity"]
    if "category" in raw:
        kwargs["category"] = raw["category"]
    if "progress_status" in raw:
        kwargs["progress_status"] = raw["progress_status"]
    if "structured_data" in raw:
        kwargs["structured_data"] = raw["structured_data"]
    if "git_refs" in raw:
        kwargs["git_refs"] = raw["git_refs"]
    if "related_links" in raw:
        kwargs["related_links"] = raw["related_links"]
    if "publish_status" in raw:
        kwargs["publish_status"] = raw["publish_status"]
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
    async with app_module.get_session(db_url) as session:
        if hard:
            repo = app_module.ExperienceRepository(session)
            deleted = await repo.delete(uuid.UUID(experience_id))
        else:
            deleted = await _svc().soft_delete(experience_id)
        if not deleted:
            raise HTTPException(status_code=404, detail="Experience not found")
        return {"message": "Experience deleted"}


@router.post("/experiences/{experience_id}/promote")
async def promote_experience(
    experience_id: str,
    user: User = Depends(require_role("create")),
):
    """Promote a personal experience to team scope. May trigger review."""
    db_url = _get_db_url()
    async with app_module.get_session(db_url) as session:
        result = await session.execute(
            select(Experience).where(
                Experience.id == uuid.UUID(experience_id),
                Experience.is_deleted == False,  # noqa: E712
            )
        )
        exp = result.scalar_one_or_none()
        if exp is None:
            raise HTTPException(status_code=404, detail="Experience not found")
        if exp.scope == "team":
            return {"message": "Already team scope"}
        exp.scope = "team"
        if _cfg() and _cfg().review.enabled:
            exp.review_status = "pending"
            exp.publish_status = "draft"
        await session.commit()
        return {"message": "Experience promoted to team scope", "experience": exp.to_dict()}


@router.post("/experiences/{experience_id}/links")
async def create_experience_link(
    experience_id: str,
    req: ExperienceLinkCreate,
    user: User = Depends(require_role("create")),
):
    """Create a link between two experiences."""
    db_url = _get_db_url()
    async with app_module.get_session(db_url) as session:
        link = ExperienceLink(
            source_id=uuid.UUID(experience_id),
            target_id=uuid.UUID(req.target_id),
            link_type=req.link_type,
            created_by=user.name,
        )
        session.add(link)
        try:
            await session.commit()
        except Exception:
            await session.rollback()
            raise HTTPException(
                status_code=409, detail="Link already exists or invalid IDs"
            )
        return {"message": "Link created", "link": link.to_dict()}


@router.get("/experiences/{experience_id}/links")
async def get_experience_links(
    experience_id: str,
    user: User | None = Depends(get_optional_user),
):
    """Get all links for an experience (both directions)."""
    db_url = _get_db_url()
    async with app_module.get_session(db_url) as session:
        result = await session.execute(
            select(ExperienceLink).where(
                or_(
                    ExperienceLink.source_id == uuid.UUID(experience_id),
                    ExperienceLink.target_id == uuid.UUID(experience_id),
                )
            )
        )
        links = result.scalars().all()
        return {"links": [link.to_dict() for link in links]}


@router.delete("/experiences/{experience_id}/links/{link_id}")
async def delete_experience_link(
    experience_id: str,
    link_id: str,
    user: User = Depends(require_role("delete")),
):
    """Delete a link between experiences."""
    from sqlalchemy import delete

    db_url = _get_db_url()
    async with app_module.get_session(db_url) as session:
        result = await session.execute(
            delete(ExperienceLink).where(ExperienceLink.id == uuid.UUID(link_id))
        )
        await session.commit()
        if result.rowcount == 0:
            raise HTTPException(status_code=404, detail="Link not found")
        return {"message": "Link deleted"}


@router.get("/experiences/{experience_id}/review-history")
async def get_review_history(
    experience_id: str,
    user: User = Depends(get_current_user),
):
    """Get review history for an experience."""
    db_url = _get_db_url()
    async with app_module.get_session(db_url) as session:
        result = await session.execute(
            select(ReviewHistory)
            .where(ReviewHistory.experience_id == uuid.UUID(experience_id))
            .order_by(ReviewHistory.created_at.desc())
        )
        history = result.scalars().all()
        return {"history": [h.to_dict() for h in history]}


@router.post("/experiences/{experience_id}/restore")
async def restore_experience(
    experience_id: str,
    user: User = Depends(get_current_user),
):
    """Restore a soft-deleted experience (requires auth)."""
    restored = await _svc().restore(experience_id)
    if not restored:
        raise HTTPException(
            status_code=404, detail="Experience not found or not deleted"
        )
    return {"message": "Experience restored"}


@router.post("/experiences/{experience_id}/feedback")
async def add_feedback(
    experience_id: str,
    req: FeedbackCreate,
    user: User = Depends(get_current_user),
):
    """Add feedback to an experience (requires auth)."""
    success = await _svc().feedback(
        experience_id=experience_id,
        rating=req.rating,
        feedback_by=user.name,
        comment=req.comment,
    )
    if not success:
        raise HTTPException(status_code=404, detail="Experience not found")
    return {"message": "Feedback recorded"}


@router.get("/experiences/{experience_id}/versions")
async def get_versions(
    experience_id: str,
    user: User = Depends(get_current_user),
):
    """Get version history for an experience."""
    versions = await _svc().get_versions(experience_id)
    return {"versions": versions, "total": len(versions)}


@router.get("/experiences/{experience_id}/versions/{version_id}")
async def get_version_detail(
    experience_id: str,
    version_id: str,
    user: User = Depends(get_current_user),
):
    """Get a specific version snapshot."""
    version = await _svc().get_version_detail(version_id)
    if version is None:
        raise HTTPException(status_code=404, detail="Version not found")
    return version


@router.post("/experiences/{experience_id}/rollback/{version_id}")
async def rollback_to_version(
    experience_id: str,
    version_id: str,
    user: User = Depends(get_current_user),
):
    """Rollback an experience to a specific version."""
    result = await _svc().rollback_to_version(
        experience_id=experience_id,
        version_id=version_id,
        user=user.name,
    )
    if result is None:
        raise HTTPException(
            status_code=404, detail="Experience or version not found"
        )
    return {"message": "Rollback successful", "experience": result}


@router.get("/reviews/pending")
async def list_pending_reviews(
    user: User = Depends(get_current_user),
):
    """List experiences pending review (requires auth, admin recommended)."""
    results = await _svc().get_pending_reviews()
    return {"experiences": results, "total": len(results)}


@router.post("/experiences/{experience_id}/review")
async def review_experience(
    experience_id: str,
    req: ReviewRequest,
    user: User = Depends(require_role("review")),
):
    """Review an experience: approve or reject (requires review permission)."""
    if req.review_status not in ("approved", "rejected"):
        raise HTTPException(
            status_code=400, detail="review_status must be 'approved' or 'rejected'"
        )
    result = await _svc().review(
        experience_id=experience_id,
        review_status=req.review_status,
        reviewed_by=user.name,
        review_note=req.review_note,
    )
    if result is None:
        raise HTTPException(status_code=404, detail="Experience not found")
    return result


@router.post("/experiences/{experience_id}/publish")
async def publish_experience(
    experience_id: str,
    user: User = Depends(get_current_user),
):
    """Publish a draft experience (set publish_status='published')."""
    result = await _svc().publish_experience(
        experience_id=experience_id,
        user=user.name,
    )
    if result is None:
        raise HTTPException(status_code=404, detail="Experience not found")
    return {"message": "Experience published", "experience": result}


@router.post("/experiences/{experience_id}/summarize")
async def summarize_experience(
    experience_id: str,
    user: User = Depends(get_current_user),
):
    """Generate an LLM summary for a single experience."""
    result = await _svc().generate_summary(experience_id)
    if result is None:
        raise HTTPException(
            status_code=404,
            detail="Experience not found or summary generation failed",
        )
    return {"message": "Summary generated", "experience": result}

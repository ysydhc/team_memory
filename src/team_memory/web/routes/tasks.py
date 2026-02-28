"""TaskGroup and PersonalTask CRUD API routes."""

from __future__ import annotations

import uuid
from datetime import datetime, timezone

from fastapi import APIRouter, Body, Depends, HTTPException
from pydantic import BaseModel, Field

from team_memory.auth.provider import User
from team_memory.storage.database import get_session
from team_memory.storage.models import PersonalTask
from team_memory.storage.repository import ExperienceRepository, TaskRepository
from team_memory.web.app import (
    _get_db_url,
    _resolve_project,
    get_current_user,
    get_optional_user,
)

router = APIRouter(tags=["tasks"])

# Priority to numeric urgency for sorting (importance * urgency)
_PRIORITY_WEIGHT = {"urgent": 4, "high": 3, "medium": 2, "low": 1}


# ---------- Request/Response Schemas ----------


class TaskCreate(BaseModel):
    """Request body for creating a task."""

    title: str
    description: str | None = None
    status: str = "wait"  # wait, plan, in_progress, completed, cancelled
    priority: str = "medium"  # low, medium, high, urgent
    importance: int = Field(default=3, ge=1, le=5)
    project: str | None = None
    group_id: str | None = None
    due_date: datetime | None = None
    labels: list[str] = Field(default_factory=list)
    experience_id: str | None = None


class TaskUpdate(BaseModel):
    """Request body for updating a task (all optional)."""

    title: str | None = None
    description: str | None = None
    status: str | None = None
    priority: str | None = None
    importance: int | None = None
    project: str | None = None
    group_id: str | None = None
    due_date: datetime | None = None
    labels: list[str] | None = None
    experience_id: str | None = None


class TaskGroupCreate(BaseModel):
    """Request body for creating a task group."""

    title: str
    description: str | None = None
    project: str | None = None
    source_doc: str | None = None
    content_hash: str | None = None


def _task_sort_key(t: PersonalTask) -> tuple:
    """Sort by importance * urgency descending, then due_date ascending."""
    weight = _PRIORITY_WEIGHT.get(t.priority or "medium", 2)
    score = (t.importance or 3) * weight
    # Tasks without due_date sort last; use far-future sentinel for comparability
    due = t.due_date or datetime(9999, 12, 31, tzinfo=timezone.utc)
    return (-score, due, t.sort_order or 0)


# ---------- Task Endpoints ----------


@router.get("/tasks")
async def list_tasks(
    project: str | None = None,
    status: str | None = None,
    group_id: str | None = None,
    user_id: str | None = None,
    user: User | None = Depends(get_optional_user),
):
    """List tasks with filters. Returns tasks sorted by importance * urgency."""
    db_url = _get_db_url()
    resolved_project = _resolve_project(project)
    filter_user_id = user_id if user_id else (user.name if user else None)

    async with get_session(db_url) as session:
        repo = TaskRepository(session)
        tasks = await repo.list_tasks(
            project=resolved_project,
            user_id=filter_user_id,
            status=status,
            group_id=uuid.UUID(group_id) if group_id else None,
        )
        await session.commit()

    tasks_sorted = sorted(tasks, key=_task_sort_key)
    return {"tasks": [t.to_dict() for t in tasks_sorted]}


@router.post("/tasks")
async def create_task(
    req: TaskCreate,
    user: User = Depends(get_current_user),
):
    """Create a task. Checks WIP limit when status=in_progress."""
    db_url = _get_db_url()
    resolved_project = _resolve_project(req.project)

    async with get_session(db_url) as session:
        repo = TaskRepository(session)

        if req.status == "in_progress":
            within_limit, count = await repo.check_wip(resolved_project, user.name)
            if not within_limit:
                raise HTTPException(
                    status_code=400,
                    detail=f"WIP limit ({TaskRepository.WIP_LIMIT}) exceeded. "
                    f"You have {count} tasks in progress.",
                )

        task = await repo.create_task(
            title=req.title,
            user_id=user.name,
            project=resolved_project,
            group_id=uuid.UUID(req.group_id) if req.group_id else None,
            description=req.description,
            status=req.status,
            priority=req.priority,
            importance=req.importance,
            due_date=req.due_date,
            labels=req.labels,
            experience_id=uuid.UUID(req.experience_id) if req.experience_id else None,
        )
        await session.commit()

    return task.to_dict()


@router.put("/tasks/{task_id}")
async def update_task(
    task_id: str,
    req: TaskUpdate,
    user: User = Depends(get_current_user),
):
    """Update a task. When status changes to in_progress, checks WIP limit (5).
    Returns warning if over limit but still allows the update."""
    db_url = _get_db_url()

    async with get_session(db_url) as session:
        repo = TaskRepository(session)
        task = await repo.get_task(uuid.UUID(task_id))
        if not task:
            raise HTTPException(status_code=404, detail="Task not found")

        raw = req.model_dump(exclude_unset=True)
        new_status = raw.get("status")

        warning = None
        if new_status == "in_progress" and task.status != "in_progress":
            within_limit, count = await repo.check_wip(task.project, task.user_id)
            if not within_limit:
                warning = (
                    f"WIP limit ({TaskRepository.WIP_LIMIT}) exceeded. "
                    f"You have {count} tasks in progress."
                )

        kwargs = {}
        if "title" in raw:
            kwargs["title"] = raw["title"]
        if "description" in raw:
            kwargs["description"] = raw["description"]
        if "status" in raw:
            kwargs["status"] = raw["status"]
        if "priority" in raw:
            kwargs["priority"] = raw["priority"]
        if "importance" in raw:
            kwargs["importance"] = raw["importance"]
        if "project" in raw:
            kwargs["project"] = _resolve_project(raw["project"])
        if "group_id" in raw:
            kwargs["group_id"] = (
                uuid.UUID(raw["group_id"]) if raw["group_id"] else None
            )
        if "due_date" in raw:
            kwargs["due_date"] = raw["due_date"]
        if "labels" in raw:
            kwargs["labels"] = raw["labels"]
        if "experience_id" in raw:
            kwargs["experience_id"] = (
                uuid.UUID(raw["experience_id"]) if raw["experience_id"] else None
            )

        updated = await repo.update_task(task.id, **kwargs)
        await session.commit()

    result = updated.to_dict() if updated else {}
    if warning:
        result["warning"] = warning
    return result


@router.delete("/tasks/{task_id}")
async def delete_task(
    task_id: str,
    user: User | None = Depends(get_optional_user),
):
    """Delete a task by ID."""
    db_url = _get_db_url()
    async with get_session(db_url) as session:
        repo = TaskRepository(session)
        ok = await repo.delete_task(uuid.UUID(task_id))
        if not ok:
            raise HTTPException(status_code=404, detail="Task not found")
        await session.commit()
    return {"message": "Task deleted"}


@router.get("/tasks/{task_id}")
async def get_task(
    task_id: str,
    with_context: bool = False,
    user: User | None = Depends(get_optional_user),
):
    """Get task detail. If with_context=true, also returns linked experience content."""
    db_url = _get_db_url()

    async with get_session(db_url) as session:
        repo = TaskRepository(session)
        task = await repo.get_task(uuid.UUID(task_id))
        if not task:
            raise HTTPException(status_code=404, detail="Task not found")

        data = task.to_dict()

        if task.group_id:
            group = await repo.get_group(task.group_id)
            if group:
                data["group_title"] = group.title

        if with_context and task.experience_id:
            exp_repo = ExperienceRepository(session)
            exp = await exp_repo.get_by_id(task.experience_id)
            if exp:
                data["experience_context"] = exp.to_dict()
            else:
                data["experience_context"] = None

        await session.commit()

    return data


# ---------- TaskGroup Endpoints ----------


@router.get("/task-groups")
async def list_task_groups(
    project: str | None = None,
    user_id: str | None = None,
    include_archived: bool = False,
    user: User | None = Depends(get_optional_user),
):
    """List task groups with their tasks. By default excludes archived groups."""
    db_url = _get_db_url()
    resolved_project = _resolve_project(project)
    filter_user_id = user_id if user_id else (user.name if user else None)

    async with get_session(db_url) as session:
        repo = TaskRepository(session)
        groups = await repo.list_groups(
            project=resolved_project,
            user_id=filter_user_id,
            include_archived=include_archived,
        )
        await session.commit()

    return {"groups": [g.to_dict(include_tasks=True) for g in groups]}


@router.post("/task-groups")
async def create_task_group(
    req: TaskGroupCreate,
    user: User = Depends(get_current_user),
):
    """Create a task group."""
    db_url = _get_db_url()
    resolved_project = _resolve_project(req.project)

    async with get_session(db_url) as session:
        repo = TaskRepository(session)
        group = await repo.create_group(
            title=req.title,
            user_id=user.name,
            project=resolved_project,
            description=req.description,
            source_doc=req.source_doc,
            content_hash=req.content_hash,
        )
        await session.commit()

    return group.to_dict()


@router.put("/task-groups/{group_id}")
async def update_task_group(
    group_id: str,
    body: dict = Body(...),
    user: User | None = Depends(get_optional_user),
):
    """Update a task group by ID."""
    db_url = _get_db_url()
    async with get_session(db_url) as session:
        repo = TaskRepository(session)
        group = await repo.update_group(uuid.UUID(group_id), **body)
        if not group:
            raise HTTPException(status_code=404, detail="Task group not found")
        await session.commit()
    return {"group": group.to_dict(), "message": "Group updated"}


@router.delete("/task-groups/{group_id}")
async def delete_task_group(
    group_id: str,
    user: User | None = Depends(get_optional_user),
):
    """Delete a task group and its tasks."""
    db_url = _get_db_url()
    async with get_session(db_url) as session:
        repo = TaskRepository(session)
        ok = await repo.delete_group(uuid.UUID(group_id))
        if not ok:
            raise HTTPException(status_code=404, detail="Task group not found")
        await session.commit()
    return {"message": "Task group deleted"}


@router.get("/task-groups/{group_id}")
async def get_task_group(
    group_id: str,
    user: User = Depends(get_current_user),
):
    """Get a task group with its tasks."""
    db_url = _get_db_url()

    async with get_session(db_url) as session:
        repo = TaskRepository(session)
        group = await repo.get_group(uuid.UUID(group_id))
        if not group:
            raise HTTPException(status_code=404, detail="Task group not found")
        await session.commit()

    return group.to_dict(include_tasks=True)


# ---------- Task Dependencies ----------


class DependencyCreate(BaseModel):
    target_task_id: str
    dep_type: str = "blocks"  # blocks, related, discovered_from


@router.post("/tasks/{task_id}/dependencies")
async def add_dependency(
    task_id: str,
    req: DependencyCreate,
    user: User = Depends(get_current_user),
):
    """Add a dependency from task_id (source) to target_task_id."""
    db_url = _get_db_url()
    async with get_session(db_url) as session:
        repo = TaskRepository(session)
        dep = await repo.add_dependency(
            source_id=uuid.UUID(task_id),
            target_id=uuid.UUID(req.target_task_id),
            dep_type=req.dep_type,
            created_by=user.name,
        )
        await session.commit()
    return dep.to_dict()


@router.get("/tasks/{task_id}/dependencies")
async def get_dependencies(
    task_id: str,
    user: User | None = Depends(get_optional_user),
):
    """Get all dependencies for a task."""
    db_url = _get_db_url()
    async with get_session(db_url) as session:
        repo = TaskRepository(session)
        deps = await repo.get_dependencies(uuid.UUID(task_id))
        await session.commit()
    return {"dependencies": [d.to_dict() for d in deps]}


@router.delete("/tasks/{task_id}/dependencies/{target_id}")
async def remove_dependency(
    task_id: str,
    target_id: str,
    user: User = Depends(get_current_user),
):
    """Remove a dependency between two tasks."""
    db_url = _get_db_url()
    async with get_session(db_url) as session:
        repo = TaskRepository(session)
        removed = await repo.remove_dependency(
            uuid.UUID(task_id), uuid.UUID(target_id),
        )
        await session.commit()
    if not removed:
        raise HTTPException(status_code=404, detail="Dependency not found")
    return {"message": "Dependency removed"}


@router.get("/tasks-ready")
async def get_ready_tasks(
    project: str | None = None,
    user_id: str | None = None,
    user: User | None = Depends(get_optional_user),
):
    """Get tasks ready to start (no unresolved blockers)."""
    db_url = _get_db_url()
    resolved_project = _resolve_project(project)
    filter_user_id = user_id if user_id else (user.name if user else None)

    async with get_session(db_url) as session:
        repo = TaskRepository(session)
        tasks = await repo.get_ready_tasks(resolved_project, filter_user_id)
        await session.commit()
    return {"tasks": [t.to_dict() for t in tasks]}


# ---------- Atomic Claim ----------


@router.post("/tasks/{task_id}/claim")
async def claim_task(
    task_id: str,
    user: User = Depends(get_current_user),
):
    """Atomically claim a task (assigns + sets in_progress)."""
    db_url = _get_db_url()
    async with get_session(db_url) as session:
        repo = TaskRepository(session)
        try:
            task = await repo.claim_task(
                uuid.UUID(task_id), user.name,
                project=_resolve_project(None),
            )
        except ValueError as e:
            raise HTTPException(status_code=409, detail=str(e))
        if not task:
            raise HTTPException(status_code=404, detail="Task not found")
        await session.commit()
    return task.to_dict()


@router.post("/tasks/{task_id}/unclaim")
async def unclaim_task(
    task_id: str,
    user: User = Depends(get_current_user),
):
    """Release a task claim."""
    db_url = _get_db_url()
    async with get_session(db_url) as session:
        repo = TaskRepository(session)
        task = await repo.unclaim_task(uuid.UUID(task_id))
        if not task:
            raise HTTPException(status_code=404, detail="Task not found")
        await session.commit()
    return task.to_dict()


# ---------- Task Messages ----------


class MessageCreate(BaseModel):
    content: str
    thread_id: str | None = None


@router.get("/tasks/{task_id}/messages")
async def list_messages(
    task_id: str,
    user: User | None = Depends(get_optional_user),
):
    """Get all messages for a task."""
    db_url = _get_db_url()
    async with get_session(db_url) as session:
        repo = TaskRepository(session)
        msgs = await repo.list_messages(uuid.UUID(task_id))
        await session.commit()
    return {"messages": [m.to_dict() for m in msgs]}


@router.post("/tasks/{task_id}/messages")
async def add_message(
    task_id: str,
    req: MessageCreate,
    user: User = Depends(get_current_user),
):
    """Add a message to a task."""
    db_url = _get_db_url()
    async with get_session(db_url) as session:
        repo = TaskRepository(session)
        msg = await repo.add_message(
            task_id=uuid.UUID(task_id),
            author=user.name,
            content=req.content,
            thread_id=uuid.UUID(req.thread_id) if req.thread_id else None,
        )
        await session.commit()
    return msg.to_dict()


# ---------- Task Duplicates ----------


@router.get("/tasks-duplicates")
async def find_duplicate_tasks(
    project: str | None = None,
    user: User | None = Depends(get_optional_user),
):
    """Find duplicate tasks by content hash."""
    db_url = _get_db_url()
    resolved_project = _resolve_project(project)
    async with get_session(db_url) as session:
        repo = TaskRepository(session)
        groups = await repo.find_duplicate_tasks(resolved_project)
        await session.commit()
    return {
        "duplicate_groups": [
            {"content_hash": ch, "tasks": [t.to_dict() for t in tasks]}
            for ch, tasks in groups
        ],
    }

"""Personal memory: pull API for Agent + CRUD for Web (list/get/put/delete)."""

from __future__ import annotations

import logging

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.exc import ProgrammingError

from team_memory.auth.provider import User
from team_memory.bootstrap import get_context
from team_memory.services.personal_memory import PersonalMemoryService
from team_memory.web.app import get_current_user, get_optional_user

logger = logging.getLogger("team_memory.web.personal_memory")

router = APIRouter(tags=["personal-memory"])


def _personal_memory_service() -> PersonalMemoryService:
    ctx = get_context()
    return PersonalMemoryService(
        embedding_provider=ctx.embedding,
        db_url=ctx.db_url,
    )


@router.post("/personal-memory")
async def create_personal_memory(
    body: dict,
    user: User = Depends(get_current_user),
):
    """Create one personal memory. Requires auth."""
    if str(user.name).strip().lower() == "anonymous":
        raise HTTPException(status_code=401, detail="Anonymous cannot create")
    content = body.get("content")
    if not content or not str(content).strip():
        raise HTTPException(status_code=400, detail="content is required")
    scope = body.get("scope") or "generic"
    context_hint = body.get("context_hint")
    if scope not in ("generic", "context"):
        scope = "generic"
    pk = body.get("profile_kind")
    if pk not in ("static", "dynamic"):
        pk = None
    svc = _personal_memory_service()
    mem = await svc.write(
        user.name,
        content=str(content).strip(),
        scope=scope,
        context_hint=str(context_hint).strip() if context_hint else None,
        profile_kind=pk,
    )
    return mem


@router.get("/personal-memory/list")
async def list_personal_memory(
    scope: str | None = None,
    profile_kind: str | None = None,
    user: User = Depends(get_current_user),
):
    """List current user's personal memories. Requires auth.
    Supports scope and profile_kind filter."""
    if str(user.name).strip().lower() == "anonymous":
        raise HTTPException(status_code=401, detail="Anonymous cannot list")
    if profile_kind not in (None, "static", "dynamic"):
        profile_kind = None
    svc = _personal_memory_service()
    try:
        items = await svc.list_by_user(
            user.name, scope=scope, profile_kind=profile_kind
        )
    except ProgrammingError as e:
        orig = str(getattr(e, "orig", e) or e)
        logger.exception("personal_memory list query failed: %s", orig)
        if "personal_memories" in orig:
            raise HTTPException(
                status_code=503,
                detail="数据库缺少 personal_memories 表，请运行: alembic upgrade head",
            ) from e
        raise HTTPException(
            status_code=503,
            detail="用户画像列表查询失败，请查看服务端日志",
        ) from e
    return {"items": items}


@router.get("/personal-memory/{memory_id}")
async def get_personal_memory(
    memory_id: str,
    user: User = Depends(get_current_user),
):
    """Get one personal memory by id. Requires auth, must own."""
    if str(user.name).strip().lower() == "anonymous":
        raise HTTPException(status_code=401, detail="Anonymous cannot get")
    svc = _personal_memory_service()
    mem = await svc.get_by_id(memory_id, user.name)
    if not mem:
        raise HTTPException(status_code=404, detail="Not found")
    return mem


@router.put("/personal-memory/{memory_id}")
async def put_personal_memory(
    memory_id: str,
    body: dict,
    user: User = Depends(get_current_user),
):
    """Update one personal memory. Requires auth, must own."""
    if str(user.name).strip().lower() == "anonymous":
        raise HTTPException(status_code=401, detail="Anonymous cannot update")
    content = body.get("content")
    scope = body.get("scope")
    context_hint = body.get("context_hint")
    profile_kind = body.get("profile_kind")
    if profile_kind not in (None, "static", "dynamic"):
        profile_kind = None
    svc = _personal_memory_service()
    mem = await svc.update(
        memory_id,
        user.name,
        content=content,
        scope=scope,
        context_hint=context_hint,
        profile_kind=profile_kind,
    )
    if not mem:
        raise HTTPException(status_code=404, detail="Not found")
    return mem


@router.delete("/personal-memory/{memory_id}")
async def delete_personal_memory(
    memory_id: str,
    user: User = Depends(get_current_user),
):
    """Delete one personal memory. Requires auth, must own."""
    if str(user.name).strip().lower() == "anonymous":
        raise HTTPException(status_code=401, detail="Anonymous cannot delete")
    svc = _personal_memory_service()
    ok = await svc.delete(memory_id, user.name)
    if not ok:
        raise HTTPException(status_code=404, detail="Not found")
    return {"deleted": True}


@router.get("/personal-memory")
async def pull_personal_memory(
    current_context: str | None = None,
    user: User | None = Depends(get_optional_user),
):
    """Pull personal memories for Agent: generic + context-matched.

    Anonymous (no auth or user name 'anonymous'): returns [].
    Else returns scope=generic all; if current_context given,
    also scope=context items semantically matching current_context.
    """
    user_id = user.name if user else None
    svc = _personal_memory_service()
    items = await svc.pull(
        user_id=user_id,
        current_context=current_context,
    )
    return {"items": items}

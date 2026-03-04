"""Personal memory: pull API for Agent context (generic + context-matched)."""

from __future__ import annotations

from fastapi import APIRouter, Depends

from team_memory.auth.provider import User
from team_memory.bootstrap import get_context
from team_memory.services.personal_memory import PersonalMemoryService
from team_memory.web.app import get_optional_user

router = APIRouter(tags=["personal-memory"])


def _personal_memory_service() -> PersonalMemoryService:
    ctx = get_context()
    return PersonalMemoryService(
        embedding_provider=ctx.embedding,
        db_url=ctx.db_url,
    )


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

"""User expansion config: per-user tag_synonyms for query expansion (GET/PUT)."""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException

from team_memory.auth.provider import User
from team_memory.bootstrap import get_context
from team_memory.storage.database import get_session
from team_memory.storage.repository import UserExpansionRepository
from team_memory.web.app import get_current_user, get_optional_user

router = APIRouter(tags=["user-expansion"])


@router.get("/user-expansion-config")
async def get_user_expansion_config(
    user: User | None = Depends(get_optional_user),
):
    """Get current user's tag_synonyms for query expansion.

    Anonymous: returns { tag_synonyms: {} }.
    Logged-in: returns stored config or empty.
    """
    if not user or str(user.name).strip().lower() == "anonymous":
        return {"tag_synonyms": {}}
    ctx = get_context()
    async with get_session(ctx.db_url) as session:
        repo = UserExpansionRepository(session)
        tag_synonyms = await repo.get_by_user(user.name)
    return {"tag_synonyms": tag_synonyms}


@router.put("/user-expansion-config")
async def put_user_expansion_config(
    body: dict,
    user: User = Depends(get_current_user),
):
    """Update current user's tag_synonyms. Requires auth."""
    if str(user.name).strip().lower() == "anonymous":
        raise HTTPException(status_code=401, detail="Anonymous cannot update")
    tag_synonyms = body.get("tag_synonyms")
    if tag_synonyms is not None and not isinstance(tag_synonyms, dict):
        raise HTTPException(status_code=400, detail="tag_synonyms must be object")
    tag_synonyms = dict(tag_synonyms or {})
    # Ensure values are str (key->value mapping)
    for k, v in list(tag_synonyms.items()):
        if not isinstance(k, str) or not isinstance(v, str):
            raise HTTPException(
                status_code=400,
                detail="tag_synonyms keys and values must be strings",
            )
    ctx = get_context()
    async with get_session(ctx.db_url) as session:
        repo = UserExpansionRepository(session)
        await repo.upsert(user.name, tag_synonyms)
    return {"tag_synonyms": tag_synonyms}

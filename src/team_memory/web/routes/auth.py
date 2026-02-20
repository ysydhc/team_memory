"""Auth and API key management routes."""

from __future__ import annotations

import logging

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import JSONResponse
from sqlalchemy import update

from team_memory.auth.permissions import require_role
from team_memory.auth.provider import ApiKeyAuth, DbApiKeyAuth, User
from team_memory.storage.database import get_session
from team_memory.storage.models import ApiKey
from team_memory.web import app as app_module
from team_memory.web.app import (
    ApiKeyCreateRequest,
    ApiKeyUpdateRequest,
    LoginRequest,
    LoginResponse,
    _encode_api_key_cookie,
    _get_db_url,
    get_current_user,
)

logger = logging.getLogger("team_memory.web")

router = APIRouter(tags=["auth"])


@router.post("/auth/login", response_model=LoginResponse)
async def login(req: LoginRequest):
    """Validate API key and return user info."""
    _auth = app_module._auth
    if not _auth:
        raise HTTPException(status_code=500, detail="Auth not initialized")

    user = await _auth.authenticate({"api_key": req.api_key})
    if user is None:
        return LoginResponse(success=False, message="Invalid API key")

    from team_memory.auth.permissions import ROLE_PERMISSIONS

    perms = ROLE_PERMISSIONS.get(user.role, set())
    resp_data = LoginResponse(
        success=True,
        user=user.name,
        role=user.role,
        message="Login successful",
    ).model_dump()
    resp_data["permissions"] = sorted(perms) if "*" not in perms else ["*"]
    response = JSONResponse(content=resp_data)
    response.set_cookie(
        key="api_key",
        value=_encode_api_key_cookie(req.api_key),
        httponly=True,
        samesite="strict",
        max_age=86400 * 7,
    )
    return response


@router.post("/auth/logout")
async def logout():
    """Clear the auth cookie."""
    response = JSONResponse(content={"message": "Logged out"})
    response.delete_cookie("api_key")
    return response


@router.get("/auth/me")
async def auth_me(user: User = Depends(get_current_user)):
    """Check current auth status and permissions."""
    from team_memory.auth.permissions import ROLE_PERMISSIONS

    perms = ROLE_PERMISSIONS.get(user.role, set())
    return {
        "user": user.name,
        "role": user.role,
        "permissions": sorted(perms) if "*" not in perms else ["*"],
    }


@router.get("/keys")
async def list_api_keys(user: User = Depends(require_role("admin"))):
    """List all API keys (admin only)."""
    _auth = app_module._auth
    if not isinstance(_auth, DbApiKeyAuth):
        return {
            "keys": [],
            "message": "DB key management not enabled. Using in-memory keys.",
        }

    db_url = _get_db_url()
    async with get_session(db_url) as session:
        keys = await _auth.list_keys_db(session)
        return {"keys": keys}


@router.post("/keys")
async def create_api_key(
    req: ApiKeyCreateRequest,
    user: User = Depends(require_role("admin")),
):
    """Create a new API key (admin only)."""
    _auth = app_module._auth
    if not isinstance(_auth, DbApiKeyAuth):
        if isinstance(_auth, ApiKeyAuth):
            _auth.register_key(req.api_key, req.user_name, req.role)
            return {"message": "Key registered in memory (not persisted to DB)"}
        raise HTTPException(
            status_code=400,
            detail="Auth provider does not support key management",
        )

    db_url = _get_db_url()
    async with get_session(db_url) as session:
        result = await _auth.register_key_db(
            session=session,
            api_key=req.api_key,
            user_name=req.user_name,
            role=req.role,
        )
        return result


@router.delete("/keys/{key_id}")
async def deactivate_api_key(
    key_id: int,
    user: User = Depends(require_role("admin")),
):
    """Deactivate an API key (admin only)."""
    _auth = app_module._auth
    if not isinstance(_auth, DbApiKeyAuth):
        raise HTTPException(
            status_code=400, detail="DB key management not enabled"
        )

    db_url = _get_db_url()
    async with get_session(db_url) as session:
        success = await _auth.deactivate_key_db(session, key_id)
        if not success:
            raise HTTPException(status_code=404, detail="Key not found")
        return {"message": "Key deactivated"}


@router.put("/keys/{key_id}")
async def update_api_key(
    key_id: int,
    req: ApiKeyUpdateRequest,
    user: User = Depends(require_role("admin")),
):
    """Update an API key's role or active status (admin only)."""
    db_url = _get_db_url()
    async with get_session(db_url) as session:
        values = {}
        if req.role is not None:
            if req.role not in ("admin", "editor", "viewer"):
                raise HTTPException(
                    status_code=422, detail="role must be admin, editor, or viewer"
                )
            values["role"] = req.role
        if req.is_active is not None:
            values["is_active"] = req.is_active
        if not values:
            raise HTTPException(status_code=400, detail="No fields to update")

        result = await session.execute(
            update(ApiKey)
            .where(ApiKey.id == key_id)
            .values(**values)
            .returning(ApiKey.id)
        )
        if not result.first():
            raise HTTPException(status_code=404, detail="Key not found")
        return {"message": "Key updated"}

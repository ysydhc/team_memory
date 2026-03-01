"""Auth, registration, and API key management routes."""

from __future__ import annotations

import logging

from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import JSONResponse
from sqlalchemy import select, update

from team_memory.auth.permissions import require_role
from team_memory.auth.provider import DbApiKeyAuth, User
from team_memory.storage.audit import write_audit_log
from team_memory.storage.database import get_session
from team_memory.storage.models import ApiKey
from team_memory.web import app as app_module
from team_memory.web.app import (
    ApiKeyCreateRequest,
    ApiKeyUpdateRequest,
    ChangePasswordRequest,
    LoginRequest,
    LoginResponse,
    RegisterRequest,
    _encode_api_key_cookie,
    _get_db_url,
    get_current_user,
)

logger = logging.getLogger("team_memory.web")

router = APIRouter(tags=["auth"])


# ------------------------------------------------------------------
# Registration (no auth required)
# ------------------------------------------------------------------
@router.post("/auth/register")
async def register(req: RegisterRequest):
    """Self-register a new account (pending admin approval)."""
    _auth = app_module._auth
    if not isinstance(_auth, DbApiKeyAuth):
        raise HTTPException(
            status_code=400,
            detail="注册功能需要启用 db_api_key 认证模式",
        )

    if len(req.username.strip()) < 2:
        raise HTTPException(status_code=422, detail="用户名至少 2 个字符")
    if len(req.password) < 6:
        raise HTTPException(status_code=422, detail="密码至少 6 个字符")

    try:
        result = await _auth.register_user_db(req.username.strip(), req.password)
        return {"success": True, "message": "注册成功，请等待管理员审批", **result}
    except ValueError as e:
        raise HTTPException(status_code=409, detail=str(e))


# ------------------------------------------------------------------
# Login (supports api_key OR username+password)
# ------------------------------------------------------------------
@router.post("/auth/login", response_model=LoginResponse)
async def login(req: LoginRequest):
    """Validate credentials and return user info."""
    _auth = app_module._auth
    if not _auth:
        raise HTTPException(status_code=500, detail="Auth not initialized")

    credentials: dict = {}
    cookie_value = ""

    if req.api_key:
        credentials = {"api_key": req.api_key}
        cookie_value = req.api_key
    elif req.username and req.password:
        credentials = {"username": req.username.strip(), "password": req.password}
    else:
        return LoginResponse(success=False, message="请提供 API Key 或用户名密码")

    user = await _auth.authenticate(credentials)

    if user is None:
        if req.username and isinstance(_auth, DbApiKeyAuth):
            status = await _auth.check_user_status(req.username.strip())
            if status == "pending":
                return LoginResponse(success=False, message="账号待审批，请联系管理员")
            elif status == "not_found":
                return LoginResponse(success=False, message="用户不存在")
        return LoginResponse(
            success=False,
            message="API Key 无效" if req.api_key else "用户名或密码错误",
        )

    from team_memory.auth.permissions import ROLE_PERMISSIONS

    perms = ROLE_PERMISSIONS.get(user.role, set())
    resp_data = LoginResponse(
        success=True, user=user.name, role=user.role, message="Login successful"
    ).model_dump()
    resp_data["permissions"] = sorted(perms) if "*" not in perms else ["*"]

    response = JSONResponse(content=resp_data)
    if cookie_value:
        response.set_cookie(
            key="api_key",
            value=_encode_api_key_cookie(cookie_value),
            httponly=True,
            samesite="strict",
            max_age=86400 * 7,
        )
    else:
        response.set_cookie(
            key="api_key",
            value=_encode_api_key_cookie(f"pwd:{req.username}:{req.password}"),
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


# ------------------------------------------------------------------
# Change password (any authenticated user)
# ------------------------------------------------------------------
@router.put("/auth/password")
async def change_password(
    req: ChangePasswordRequest,
    user: User = Depends(get_current_user),
):
    """Change the current user's password."""
    _auth = app_module._auth
    if not isinstance(_auth, DbApiKeyAuth):
        raise HTTPException(status_code=400, detail="密码功能需要 db_api_key 模式")

    if len(req.new_password) < 6:
        raise HTTPException(status_code=422, detail="新密码至少 6 个字符")

    ok = await _auth.update_password_db(user.name, req.old_password, req.new_password)
    if not ok:
        raise HTTPException(status_code=400, detail="旧密码不正确")
    return {"message": "密码修改成功"}


# ------------------------------------------------------------------
# API Key management (admin only)
# ------------------------------------------------------------------
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
    request: Request,
    req: ApiKeyCreateRequest,
    user: User = Depends(require_role("admin")),
):
    """Admin create a new user (immediately active, auto-generated API key)."""
    _auth = app_module._auth
    if not isinstance(_auth, DbApiKeyAuth):
        raise HTTPException(status_code=400, detail="需要 db_api_key 认证模式")

    db_url = _get_db_url()
    try:
        async with get_session(db_url) as session:
            result = await _auth.register_key_db(
                session=session,
                user_name=req.user_name,
                role=req.role,
                password=req.password,
            )
            ip = request.client.host if request.client else None
            await write_audit_log(
                session,
                user_name=user.name,
                action="create",
                target_type="api_key",
                target_id=req.user_name,
                detail={"role": req.role},
                ip_address=ip,
            )
            return result
    except ValueError as e:
        raise HTTPException(status_code=409, detail=str(e))


@router.delete("/keys/{key_id}")
async def delete_or_deactivate_key(
    request: Request,
    key_id: int,
    user: User = Depends(require_role("admin")),
):
    """Delete a pending user or deactivate an active user."""
    _auth = app_module._auth
    if not isinstance(_auth, DbApiKeyAuth):
        raise HTTPException(status_code=400, detail="DB key management not enabled")

    db_url = _get_db_url()
    async with get_session(db_url) as session:
        result = await session.execute(select(ApiKey).where(ApiKey.id == key_id))
        db_key = result.scalar_one_or_none()
        if not db_key:
            raise HTTPException(status_code=404, detail="用户不存在")

        if not db_key.is_active and db_key.key_hash is None:
            ok = await _auth.delete_key_db(session, key_id)
            if ok:
                ip = request.client.host if request.client else None
                await write_audit_log(
                    session,
                    user_name=user.name,
                    action="delete",
                    target_type="api_key",
                    target_id=str(key_id),
                    detail={"user_name": getattr(db_key, "user_name", None)},
                    ip_address=ip,
                )
            return {"message": "待审批用户已拒绝并删除" if ok else "删除失败"}
        else:
            ok = await _auth.deactivate_key_db(session, key_id)
            if ok:
                ip = request.client.host if request.client else None
                await write_audit_log(
                    session,
                    user_name=user.name,
                    action="deactivate",
                    target_type="api_key",
                    target_id=str(key_id),
                    detail={"user_name": getattr(db_key, "user_name", None)},
                    ip_address=ip,
                )
            return {"message": "用户已停用" if ok else "停用失败"}


@router.put("/keys/{key_id}")
async def update_api_key(
    request: Request,
    key_id: int,
    req: ApiKeyUpdateRequest,
    user: User = Depends(require_role("admin")),
):
    """Update role/active status. Auto-generates API key on first activation."""
    _auth = app_module._auth
    if not isinstance(_auth, DbApiKeyAuth):
        raise HTTPException(status_code=400, detail="需要 db_api_key 认证模式")

    db_url = _get_db_url()
    async with get_session(db_url) as session:
        db_record = await session.execute(select(ApiKey).where(ApiKey.id == key_id))
        db_key = db_record.scalar_one_or_none()
        if not db_key:
            raise HTTPException(status_code=404, detail="用户不存在")

        is_approval = (
            req.is_active is True
            and not db_key.is_active
            and db_key.key_hash is None
        )

        if is_approval:
            result = await _auth.approve_user_db(session, key_id)
            if req.role and req.role != db_key.role:
                await session.execute(
                    update(ApiKey).where(ApiKey.id == key_id).values(role=req.role)
                )
                result["role"] = req.role
            ip = request.client.host if request.client else None
            await write_audit_log(
                session,
                user_name=user.name,
                action="approve",
                target_type="api_key",
                target_id=str(key_id),
                detail={"role": req.role or db_key.role},
                ip_address=ip,
            )
            return result

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

        await session.execute(
            update(ApiKey).where(ApiKey.id == key_id).values(**values)
        )

        if req.is_active is False and db_key.key_hash:
            self_keys = getattr(_auth, "_keys", {})
            self_keys.pop(db_key.key_hash, None)

        ip = request.client.host if request.client else None
        await write_audit_log(
            session,
            user_name=user.name,
            action="update",
            target_type="api_key",
            target_id=str(key_id),
            detail=values,
            ip_address=ip,
        )
        return {"message": "用户信息已更新"}

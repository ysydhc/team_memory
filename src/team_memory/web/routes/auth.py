"""Auth, registration, and API key management routes."""

from __future__ import annotations

import hashlib
import logging
import os
import time
from collections import defaultdict

from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import JSONResponse
from sqlalchemy import select, update

from team_memory.auth.provider import DbApiKeyAuth, User, _hash_password
from team_memory.config import get_settings
from team_memory.storage.database import get_session
from team_memory.storage.models import ApiKey
from team_memory.web import app as app_module
from team_memory.web.app import (
    AdminResetPasswordRequest,
    ApiKeyCreateRequest,
    ApiKeyUpdateRequest,
    ChangePasswordRequest,
    ForgotPasswordResetRequest,
    LoginRequest,
    LoginResponse,
    RegisterRequest,
    _encode_api_key_cookie,
    _encode_session_token,
    _get_db_url,
    get_current_user,
)
from team_memory.web.dependencies import require_role

_forgot_password_attempts: dict[str, list[float]] = defaultdict(list)
_FORGOT_PASSWORD_WINDOW = 60.0
_FORGOT_PASSWORD_MAX_PER_WINDOW = 5


def _check_forgot_password_rate_limit(client_ip: str) -> bool:
    now = time.monotonic()
    attempts = _forgot_password_attempts[client_ip]
    attempts[:] = [t for t in attempts if now - t < _FORGOT_PASSWORD_WINDOW]
    if len(attempts) >= _FORGOT_PASSWORD_MAX_PER_WINDOW:
        return False
    attempts.append(now)
    return True


def _get_session_secret() -> str:
    secret = os.environ.get("TEAM_MEMORY_SESSION_SECRET")
    if secret:
        return secret
    settings = app_module._settings or get_settings()
    if settings.auth.session_secret:
        return settings.auth.session_secret
    return hashlib.sha256(settings.database.url.encode()).hexdigest()


logger = logging.getLogger("team_memory.web")

router = APIRouter(tags=["auth"])


@router.post("/auth/register")
async def register(req: RegisterRequest):
    """Self-register a new account (pending admin approval)."""
    _auth = app_module._auth
    if not isinstance(_auth, DbApiKeyAuth):
        raise HTTPException(status_code=400, detail="注册功能需要启用 db_api_key 认证模式")

    if len(req.username.strip()) < 2:
        raise HTTPException(status_code=422, detail="用户名至少 2 个字符")
    if len(req.password) < 6:
        raise HTTPException(status_code=422, detail="密码至少 6 个字符")

    try:
        result = await _auth.register_user_db(req.username.strip(), req.password)
        return {"success": True, "message": "注册成功，请等待管理员审批", **result}
    except ValueError as e:
        raise HTTPException(status_code=409, detail=str(e))


@router.post("/auth/login", response_model=LoginResponse)
async def login(req: LoginRequest):
    """Validate credentials and return user info."""
    _auth = app_module._auth
    if not _auth:
        raise HTTPException(status_code=500, detail="Auth not initialized")

    credentials: dict = {}
    cookie_value = ""

    if req.api_key:
        api_key_clean = (req.api_key or "").strip()
        credentials = {"api_key": api_key_clean}
        cookie_value = api_key_clean
    elif req.username and req.password:
        credentials = {"username": req.username.strip(), "password": req.password}
        cookie_value = ""
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

    if req.username and req.password:
        secret = _get_session_secret()
        cookie_value = _encode_session_token(user.name, secret)

    from team_memory.auth.permissions import ROLE_PERMISSIONS

    perms = ROLE_PERMISSIONS.get(user.role, set())
    resp_data = LoginResponse(
        success=True, user=user.name, role=user.role, message="Login successful"
    ).model_dump()
    resp_data["permissions"] = sorted(perms) if "*" not in perms else ["*"]

    response = JSONResponse(content=resp_data)
    response.set_cookie(
        key="api_key",
        value=_encode_api_key_cookie(cookie_value),
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
    """Current auth status and permissions."""
    from team_memory.auth.permissions import ROLE_PERMISSIONS

    perms = ROLE_PERMISSIONS.get(user.role, set())
    out = {
        "user": user.name,
        "role": user.role,
        "permissions": sorted(perms) if "*" not in perms else ["*"],
    }
    if isinstance(_auth := app_module._auth, DbApiKeyAuth):
        masked = await _auth.get_masked_key_for_user(user.name)
        out["api_key_masked"] = masked if masked else "••••****••••"
    else:
        out["api_key_masked"] = None
    return out


@router.post("/auth/forgot-password/reset")
async def forgot_password_reset(req: ForgotPasswordResetRequest, request: Request):
    """Reset password by username + API Key."""
    _auth = app_module._auth
    if not isinstance(_auth, DbApiKeyAuth):
        raise HTTPException(status_code=400, detail="密码重置需要 db_api_key 模式")

    if len(req.new_password) < 6:
        raise HTTPException(status_code=422, detail="新密码至少 6 个字符")

    client_ip = request.client.host if request.client else "unknown"
    if not _check_forgot_password_rate_limit(client_ip):
        raise HTTPException(status_code=429, detail="请求过于频繁，请稍后再试")

    username = (req.username or "").strip()
    if not username:
        raise HTTPException(status_code=422, detail="用户名或 API Key 不正确")

    user = await _auth.authenticate({"api_key": req.api_key})
    if user is None or user.name != username:
        raise HTTPException(status_code=400, detail="用户名或 API Key 不正确")

    ok = await _auth.update_password_by_api_key_db(username, req.api_key, req.new_password)
    if not ok:
        raise HTTPException(status_code=400, detail="用户名或 API Key 不正确")

    return {"message": "密码已重置"}


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


@router.post("/auth/admin/reset-password")
async def admin_reset_password(
    req: AdminResetPasswordRequest,
    user: User = Depends(require_role("admin")),
):
    """Admin reset a user's password."""
    _auth = app_module._auth
    if not isinstance(_auth, DbApiKeyAuth):
        raise HTTPException(status_code=400, detail="密码重置需要 db_api_key 模式")

    if len(req.new_password) < 6:
        raise HTTPException(status_code=422, detail="新密码至少 6 个字符")

    username = (req.username or "").strip()
    if not username:
        raise HTTPException(status_code=422, detail="用户名不能为空")

    db_url = _get_db_url()
    async with get_session(db_url) as session:
        result = await session.execute(
            select(ApiKey).where(ApiKey.user_name == username)
        )
        db_key = result.scalar_one_or_none()
        if not db_key:
            raise HTTPException(status_code=404, detail="用户不存在")

        db_key.password_hash = _hash_password(req.new_password)

    return {"message": "密码已重置"}


# ------------------------------------------------------------------
# API Key management (admin only)
# ------------------------------------------------------------------
@router.get("/keys")
async def list_api_keys(user: User = Depends(require_role("admin"))):
    """List all API keys (admin only)."""
    _auth = app_module._auth
    if not isinstance(_auth, DbApiKeyAuth):
        return {"keys": [], "message": "DB key management not enabled."}

    db_url = _get_db_url()
    async with get_session(db_url) as session:
        keys = await _auth.list_keys_db(session)
        return {"keys": keys}


@router.post("/keys")
async def create_api_key(
    req: ApiKeyCreateRequest,
    user: User = Depends(require_role("admin")),
):
    """Admin create a new user."""
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
                generate_api_key=req.generate_api_key,
            )
            return result
    except ValueError as e:
        raise HTTPException(status_code=409, detail=str(e))


@router.delete("/keys/{key_id}")
async def delete_or_deactivate_key(
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
            return {"message": "待审批用户已拒绝并删除" if ok else "删除失败"}
        else:
            ok = await _auth.deactivate_key_db(session, key_id)
            return {"message": "用户已停用" if ok else "停用失败"}


@router.post("/keys/{key_id}/generate")
async def generate_api_key_for_user(
    key_id: int,
    user: User = Depends(require_role("admin")),
):
    """Generate API key for a user without key (admin only)."""
    _auth = app_module._auth
    if not isinstance(_auth, DbApiKeyAuth):
        raise HTTPException(status_code=400, detail="需要 db_api_key 认证模式")

    db_url = _get_db_url()
    try:
        async with get_session(db_url) as session:
            return await _auth.generate_key_for_user_db(session, key_id)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.put("/keys/{key_id}")
async def update_api_key(
    key_id: int,
    req: ApiKeyUpdateRequest,
    user: User = Depends(require_role("admin")),
):
    """Update role/active status."""
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
            result = await _auth.approve_user_db(
                session,
                key_id,
                generate_key=req.generate_api_key or False,
            )
            if req.role and req.role != db_key.role:
                await session.execute(
                    update(ApiKey).where(ApiKey.id == key_id).values(role=req.role)
                )
                result["role"] = req.role
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

        return {"message": "用户信息已更新"}

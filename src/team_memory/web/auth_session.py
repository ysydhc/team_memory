"""Web-layer authentication session helpers.

Functions for session token encoding/decoding, cookie handling,
and FastAPI dependency functions for extracting the current user.
"""

from __future__ import annotations

import hashlib
import hmac
import logging
import os
import time
from urllib.parse import quote

from fastapi import HTTPException, Request
from sqlalchemy import select

from team_memory.auth.provider import (
    DbApiKeyAuth,
    User,
)
from team_memory.storage.database import get_session
from team_memory.storage.models import ApiKey, TenantMember
from team_memory.storage.tenant_context import (
    DEFAULT_TENANT_ID,
    parse_tenant_uuid,
    set_current_tenant_id,
)

logger = logging.getLogger("team_memory.web")


def _get_context():
    """Late-bind get_context so that patches on team_memory.bootstrap are respected."""
    from team_memory.bootstrap import get_context

    return get_context()


def _get_auth():
    """Get the auth provider from AppContext."""
    return _get_context().auth


def _get_settings_lazy():
    """Get settings from AppContext."""
    return _get_context().settings


def _get_db_url_lazy() -> str:
    """Get DB URL from env or AppContext."""
    return os.environ.get("TEAM_MEMORY_DB_URL") or _get_context().db_url


def _encode_api_key_cookie(api_key: str) -> str:
    return quote(api_key, safe="")


def _encode_session_token(user: str, secret: str, expiry_days: int = 7) -> str:
    expiry_ts = int(time.time()) + expiry_days * 86400
    msg = f"{user}:{expiry_ts}".encode()
    sig = hmac.new(secret.encode(), msg, hashlib.sha256).hexdigest()
    return f"sess:{user}:{expiry_ts}:{sig}"


def _decode_session_token(token: str, secret: str) -> str | None:
    if not token or not token.startswith("sess:"):
        return None
    parts = token.split(":", 3)
    if len(parts) != 4:
        return None
    _, user, expiry_str, sig = parts
    try:
        expiry_ts = int(expiry_str)
    except ValueError:
        return None
    if time.time() > expiry_ts:
        return None
    msg = f"{user}:{expiry_ts}".encode()
    expected = hmac.new(secret.encode(), msg, hashlib.sha256).hexdigest()
    if not hmac.compare_digest(sig, expected):
        return None
    return user


def _get_session_secret() -> str:
    secret = os.environ.get("TEAM_MEMORY_SESSION_SECRET")
    if secret:
        return secret
    settings = _get_settings_lazy()
    if settings.auth.session_secret:
        return settings.auth.session_secret
    raise RuntimeError(
        "Session secret not configured. Set TEAM_MEMORY_SESSION_SECRET env var or auth.session_secret in config."
    )


async def _get_user_role_from_db(user_name: str, tenant_id: str | None = None) -> str:
    _auth = _get_auth()
    if not isinstance(_auth, DbApiKeyAuth):
        return "viewer"
    db_url = _get_db_url_lazy()
    try:
        async with get_session(db_url) as session:
            tid = parse_tenant_uuid(tenant_id)
            if tid is not None:
                tm = await session.execute(
                    select(TenantMember.role).where(
                        TenantMember.tenant_id == tid,
                        TenantMember.user_id == user_name,
                    )
                )
                m_role = tm.scalar_one_or_none()
                if m_role:
                    return m_role
            result = await session.execute(
                select(ApiKey).where(ApiKey.user_name == user_name).order_by(ApiKey.id.asc()).limit(1)
            )
            row = result.scalar_one_or_none()
            if row:
                return row.role
    except Exception:
        pass
    return "viewer"


async def _resolve_session_tenant(user_name: str, requested_tenant_header: str | None) -> tuple[str, str | None]:
    """Resolve (tenant_id, role_override) for a session-cookie authenticated user.

    If ``requested_tenant_header`` is provided, verifies membership in ``tenant_members``
    and raises 403 if the user is not a member.
    """
    _auth = _get_auth()
    if not isinstance(_auth, DbApiKeyAuth):
        if requested_tenant_header:
            try:
                parsed = parse_tenant_uuid(requested_tenant_header)
                if parsed is not None:
                    return str(parsed), None
            except ValueError as exc:
                raise HTTPException(status_code=400, detail="Invalid X-Tenant-Id header") from exc
        return str(DEFAULT_TENANT_ID), None

    db_url = _get_db_url_lazy()
    if requested_tenant_header:
        try:
            req_tid = parse_tenant_uuid(requested_tenant_header)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail="Invalid X-Tenant-Id header") from exc
        if req_tid is not None:
            async with get_session(db_url) as session:
                res = await session.execute(
                    select(TenantMember).where(
                        TenantMember.tenant_id == req_tid,
                        TenantMember.user_id == user_name,
                    )
                )
                member = res.scalar_one_or_none()
                if member is None:
                    raise HTTPException(
                        status_code=403,
                        detail="Not a member of the requested tenant",
                    )
                return str(member.tenant_id), member.role

    # No explicit header: prefer DEFAULT_TENANT_ID if user is a member, else first membership
    try:
        async with get_session(db_url) as session:
            res = await session.execute(
                select(TenantMember)
                .where(TenantMember.user_id == user_name)
                .order_by(TenantMember.created_at.asc())
            )
            memberships = res.scalars().all()
            for m in memberships:
                if m.tenant_id == DEFAULT_TENANT_ID:
                    return str(m.tenant_id), m.role
            if memberships:
                return str(memberships[0].tenant_id), memberships[0].role
    except Exception:
        pass
    return str(DEFAULT_TENANT_ID), None


async def get_current_user(request: Request) -> User:
    """Extract and validate credentials from header or cookie, and bind tenant context."""
    cached = getattr(request.state, "user", None)
    if cached is not None:
        set_current_tenant_id(getattr(cached, "tenant_id", None) or DEFAULT_TENANT_ID)
        return cached

    _auth = _get_auth()
    if not _auth:
        raise HTTPException(status_code=401, detail="Not authenticated")

    auth_header = request.headers.get("Authorization", "")
    api_key = ""
    from_bearer = False
    if auth_header.startswith("Bearer "):
        api_key = auth_header[7:]
        from_bearer = True

    if not api_key:
        from team_memory.web.middleware import _decode_api_key_cookie

        api_key = _decode_api_key_cookie(request.cookies.get("api_key", ""))

    if not api_key:
        raise HTTPException(status_code=401, detail="Not authenticated")

    requested_tenant = (request.headers.get("X-Tenant-Id") or "").strip() or None

    if api_key.startswith("sess:"):
        secret = _get_session_secret()
        user_str = _decode_session_token(api_key, secret)
        if user_str:
            tid, member_role = await _resolve_session_tenant(user_str, requested_tenant)
            role = member_role or await _get_user_role_from_db(user_str, tid)
            user = User(name=user_str, role=role, tenant_id=tid)
            request.state.user = user
            request.state.tenant_id = tid
            set_current_tenant_id(tid)
            return user
        raise HTTPException(status_code=401, detail="Invalid or expired session")

    if api_key.startswith("pwd:"):
        parts = api_key.split(":", 2)
        if len(parts) == 3:
            user = await _auth.authenticate({"username": parts[1], "password": parts[2]})
        else:
            user = None
    else:
        user = await _auth.authenticate({"api_key": api_key})

    if user is None:
        raise HTTPException(status_code=401, detail="Invalid credentials")

    # Bearer API keys are strictly bound to their tenant_id; browser cookies may
    # switch to another tenant only after verifying membership in tenant_members.
    if requested_tenant:
        try:
            req_tid = parse_tenant_uuid(requested_tenant)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail="Invalid X-Tenant-Id header") from exc
        if req_tid is not None and str(req_tid) != str(user.tenant_id):
            if from_bearer:
                raise HTTPException(
                    status_code=403,
                    detail="API key is not authorized for the requested tenant",
                )
            tid, member_role = await _resolve_session_tenant(user.name, str(req_tid))
            role = member_role or user.role
            user = User(name=user.name, role=role, tenant_id=tid)

    request.state.user = user
    request.state.tenant_id = user.tenant_id
    set_current_tenant_id(user.tenant_id)
    return user

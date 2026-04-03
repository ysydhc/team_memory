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
from team_memory.bootstrap import get_context
from team_memory.storage.database import get_session
from team_memory.storage.models import ApiKey

logger = logging.getLogger("team_memory.web")


def _get_auth():
    """Get the module-level _auth from web.app (lazy import to avoid circular)."""
    from team_memory.web import app as app_module

    return app_module._auth


def _get_settings_lazy():
    """Get the module-level _settings from web.app (lazy import to avoid circular)."""
    from team_memory.web import app as app_module

    return app_module._settings


def _get_db_url_lazy() -> str:
    """Get DB URL via the web app module."""
    from team_memory.web.app import _get_db_url

    return _get_db_url()


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
    settings = _get_settings_lazy() or get_context().settings
    if settings.auth.session_secret:
        return settings.auth.session_secret
    raise RuntimeError(
        "Session secret not configured. Set TEAM_MEMORY_SESSION_SECRET env var "
        "or auth.session_secret in config."
    )


async def _get_user_role_from_db(user_name: str) -> str:
    _auth = _get_auth()
    if not isinstance(_auth, DbApiKeyAuth):
        return "viewer"
    db_url = _get_db_url_lazy()
    try:
        async with get_session(db_url) as session:
            result = await session.execute(select(ApiKey).where(ApiKey.user_name == user_name))
            row = result.scalar_one_or_none()
            if row:
                return row.role
    except Exception:
        pass
    return "viewer"


async def get_current_user(request: Request) -> User:
    """Extract and validate credentials from header or cookie."""
    cached = getattr(request.state, "user", None)
    if cached is not None:
        return cached

    _auth = _get_auth()
    if not _auth:
        raise HTTPException(status_code=401, detail="Not authenticated")

    auth_header = request.headers.get("Authorization", "")
    api_key = ""
    if auth_header.startswith("Bearer "):
        api_key = auth_header[7:]

    if not api_key:
        from team_memory.web.middleware import _decode_api_key_cookie

        api_key = _decode_api_key_cookie(request.cookies.get("api_key", ""))

    if not api_key:
        raise HTTPException(status_code=401, detail="Not authenticated")

    if api_key.startswith("sess:"):
        secret = _get_session_secret()
        user_str = _decode_session_token(api_key, secret)
        if user_str:
            role = await _get_user_role_from_db(user_str)
            user = User(name=user_str, role=role)
            request.state.user = user
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

    request.state.user = user
    return user


async def get_optional_user(request: Request) -> User | None:
    """Try to authenticate, return None instead of raising 401."""
    auth_header = request.headers.get("Authorization", "")
    api_key = ""
    if auth_header.startswith("Bearer "):
        api_key = auth_header[7:]

    if not api_key:
        from team_memory.web.middleware import _decode_api_key_cookie

        api_key = _decode_api_key_cookie(request.cookies.get("api_key", ""))

    _auth = _get_auth()
    _settings = _get_settings_lazy()
    if not api_key or not _auth:
        if _settings and _settings.auth.allow_anonymous_search:
            return User(name="anonymous", role="viewer")
        return None

    if api_key.startswith("sess:"):
        secret = _get_session_secret()
        user_str = _decode_session_token(api_key, secret)
        if user_str:
            role = await _get_user_role_from_db(user_str)
            return User(name=user_str, role=role)
        return None

    if api_key.startswith("pwd:"):
        parts = api_key.split(":", 2)
        if len(parts) == 3:
            return await _auth.authenticate({"username": parts[1], "password": parts[2]})
        return None

    return await _auth.authenticate({"api_key": api_key})

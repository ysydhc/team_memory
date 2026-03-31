"""FastAPI web application for TeamMemory management UI.

Provides a REST API and single-page web interface for browsing,
searching, creating, and managing team experiences.

Authentication is via API Key (same keys used for MCP access).
"""

from __future__ import annotations

import asyncio
import hashlib
import hmac
import logging
import os
import time
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import AsyncGenerator
from urllib.parse import quote

import httpx
import uvicorn
from fastapi import (
    APIRouter,
    FastAPI,
    HTTPException,
    Request,
)
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field, field_validator
from sqlalchemy import select

from team_memory.auth.provider import (
    AuthProvider,
    DbApiKeyAuth,
    User,
)
from team_memory.bootstrap import (
    AppContext,
    bootstrap,
    get_context,
    start_background_tasks,
    stop_background_tasks,
)
from team_memory.config import Settings, load_settings
from team_memory.services.experience import ExperienceService
from team_memory.storage.database import get_session
from team_memory.storage.models import ApiKey

logger = logging.getLogger("team_memory.web")
request_logger = logging.getLogger("team_memory.web.request")

# ============================================================
# Global state — backed by bootstrap.AppContext singleton
# ============================================================
_settings: Settings | None = None
_service: ExperienceService | None = None
_auth: AuthProvider | None = None


def _init_from_context(ctx: AppContext) -> None:
    """Populate module-level aliases from AppContext."""
    global _settings, _service, _auth
    _settings = ctx.settings
    _service = ctx.service
    _auth = ctx.auth


def _get_db_url() -> str:
    if _settings is not None:
        env_url = os.environ.get("TEAM_MEMORY_DB_URL")
        return env_url or _settings.database.url
    return get_context().db_url


def _resolve_project(project: str | None) -> str:
    """Resolve project from request > env > config default."""
    if project and project.strip():
        return project.strip()
    env_project = (os.environ.get("TEAM_MEMORY_PROJECT") or "").strip()
    if env_project:
        return env_project
    settings = _settings or get_context().settings
    default_project = (settings.default_project or "").strip()
    return default_project or "default"


# ============================================================
# Lifespan
# ============================================================
@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    ctx = bootstrap()
    _init_from_context(ctx)

    # Ensure default admin when db_api_key auth and no users exist
    if ctx.settings.auth.type == "db_api_key":
        password = (
            os.environ.get("TEAM_MEMORY_ADMIN_PASSWORD")
            or ctx.settings.auth.default_admin_password
            or ""
        )
        db_url = ctx.db_url
        if not password:
            from team_memory.auth.init_admin import is_api_keys_empty

            if await is_api_keys_empty(db_url):
                raise RuntimeError(
                    "首次启动需设置 TEAM_MEMORY_ADMIN_PASSWORD 以创建默认 admin，"
                    "或手动在 DB 中创建用户"
                )
        else:
            from team_memory.auth.init_admin import ensure_default_admin

            await ensure_default_admin(db_url, password)

    # Defer route registration to avoid circular import
    from team_memory.web.routes import mount_all

    v1_router = APIRouter(prefix="/api/v1")
    mount_all(v1_router)
    app.include_router(v1_router)
    await start_background_tasks(ctx)
    logger.info("TeamMemory web server started")
    yield
    await stop_background_tasks(ctx)
    logger.info("TeamMemory web server stopped")


# ============================================================
# FastAPI App + MCP mount (single-process)
# ============================================================
from team_memory.server import mcp  # noqa: E402

mcp_app = mcp.http_app(path="/mcp")

app = FastAPI(
    title="TeamMemory",
    description="Team Experience Database Management",
    version="0.1.2",
    lifespan=lifespan,
)
app.mount("/mcp", mcp_app)


# ============================================================
# Health Check & Readiness
# ============================================================


async def _check_database() -> dict:
    try:
        from sqlalchemy import text as sa_text

        db_url = _get_db_url()
        start = time.monotonic()
        async with get_session(db_url) as session:
            await session.execute(sa_text("SELECT 1"))
        latency_ms = round((time.monotonic() - start) * 1000, 1)
        return {"status": "up", "latency_ms": latency_ms}
    except Exception as e:
        return {"status": "down", "error": str(e)}


async def _check_ollama() -> dict:
    try:
        base_url = "http://localhost:11434"
        if _settings and _settings.embedding.provider == "ollama":
            base_url = _settings.embedding.ollama.base_url
        start = time.monotonic()
        async with httpx.AsyncClient(timeout=5.0) as client:
            resp = await client.get(f"{base_url}/api/tags")
            resp.raise_for_status()
        latency_ms = round((time.monotonic() - start) * 1000, 1)
        return {"status": "up", "latency_ms": latency_ms}
    except Exception as e:
        return {"status": "down", "error": str(e)}


async def _check_embedding_provider() -> dict:
    try:
        ctx = get_context()
        provider_name = ctx.settings.embedding.provider
        vec = await ctx.embedding.encode_single("health check")
        return {"status": "up", "provider": provider_name, "dimension": len(vec)}
    except RuntimeError:
        return {"status": "down", "reason": "not bootstrapped"}
    except Exception as e:
        return {"status": "down", "error": str(e)[:100]}


@app.get("/health")
async def health_check():
    """Health check endpoint — no authentication required."""
    db_check, ollama_check, embed_check = await asyncio.gather(
        _check_database(),
        _check_ollama(),
        _check_embedding_provider(),
    )

    checks = {
        "database": db_check,
        "ollama": ollama_check,
        "embedding_provider": embed_check,
    }

    if db_check["status"] == "down":
        status = "unhealthy"
    elif any(checks[k].get("status") == "down" for k in ["ollama", "embedding_provider"]):
        status = "degraded"
    else:
        status = "healthy"

    status_code = 200 if status != "unhealthy" else 503
    return JSONResponse(
        status_code=status_code,
        content={
            "status": status,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "version": app.version,
            "checks": checks,
        },
    )


@app.get("/ready")
async def readiness_check():
    """Readiness probe — returns 200 only when database is healthy."""
    db_check = await _check_database()
    if db_check["status"] == "down":
        return JSONResponse(
            status_code=503,
            content={"ready": False, "reason": "database unavailable"},
        )
    return {"ready": True}


# ============================================================
# Middleware
# ============================================================
from team_memory.web.middleware import api_version_compat  # noqa: E402

app.middleware("http")(api_version_compat)


@app.middleware("http")
async def request_log_middleware(request: Request, call_next):
    """Structured request log: path, method, status, duration_ms."""
    path = request.scope.get("path", "")
    if path in ("/health", "/ready"):
        return await call_next(request)
    start = time.monotonic()
    response = await call_next(request)
    duration_ms = int((time.monotonic() - start) * 1000)
    ip = request.client.host if request.client else None
    if not ip and request.headers.get("x-forwarded-for"):
        ip = request.headers.get("x-forwarded-for", "").split(",")[0].strip()
    user = getattr(request.state, "user", None)
    user_name = user.name if user and hasattr(user, "name") else None
    payload = {
        "event": "request",
        "path": path,
        "method": request.scope.get("method", "GET"),
        "status": response.status_code,
        "duration_ms": duration_ms,
        "ip": ip or "",
        "user": user_name or "",
    }
    request_logger.info("request", extra=payload)
    return response


@app.exception_handler(Exception)
async def _global_exception_handler(request: Request, exc: Exception):
    """Ensure unhandled exceptions return JSON for API clients."""
    err_id = f"web-{id(exc) & 0xFFFF:04x}"
    logger.exception(
        "Unhandled exception [%s] path=%s: %s",
        err_id,
        request.scope.get("path", "?") if hasattr(request, "scope") else "?",
        exc,
    )
    path = request.scope.get("path", "") if hasattr(request, "scope") else ""
    if path.startswith("/api/"):
        return JSONResponse(
            status_code=500,
            content={
                "detail": "Internal server error. See server logs for diagnosis.",
                "ops_error_id": err_id,
                "message": str(exc),
            },
        )
    raise exc


# ============================================================
# Auth Dependencies
# ============================================================
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
    settings = _settings or get_context().settings
    if settings.auth.session_secret:
        return settings.auth.session_secret
    return hashlib.sha256(settings.database.url.encode()).hexdigest()


async def _get_user_role_from_db(user_name: str) -> str:
    if not isinstance(_auth, DbApiKeyAuth):
        return "viewer"
    db_url = _get_db_url()
    try:
        async with get_session(db_url) as session:
            result = await session.execute(
                select(ApiKey).where(ApiKey.user_name == user_name)
            )
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


# ============================================================
# Request/Response Models
# ============================================================
class LoginRequest(BaseModel):
    api_key: str | None = None
    username: str | None = None
    password: str | None = None


class RegisterRequest(BaseModel):
    username: str
    password: str


class LoginResponse(BaseModel):
    success: bool
    user: str = ""
    role: str = ""
    message: str = ""


class ChangePasswordRequest(BaseModel):
    old_password: str
    new_password: str


class ForgotPasswordResetRequest(BaseModel):
    username: str
    api_key: str
    new_password: str


class AdminResetPasswordRequest(BaseModel):
    username: str
    new_password: str


class ExperienceCreate(BaseModel):
    title: str
    problem: str
    solution: str | None = None
    tags: list[str] = Field(default_factory=list)
    status: str = "published"
    visibility: str = "project"
    skip_dedup_check: bool = False
    experience_type: str = "general"
    project: str | None = None
    group_key: str | None = None


class ExperienceUpdate(BaseModel):
    title: str | None = None
    problem: str | None = None
    solution: str | None = None
    tags: list[str] | None = None
    experience_type: str | None = None
    exp_status: str | None = None
    visibility: str | None = None
    solution_addendum: str | None = None


class FeedbackCreate(BaseModel):
    rating: int
    comment: str | None = None

    @field_validator("rating")
    @classmethod
    def rating_range(cls, v: int) -> int:
        if not (1 <= v <= 5):
            raise ValueError("rating must be between 1 and 5")
        return v


class SearchRequest(BaseModel):
    query: str
    tags: list[str] | None = None
    max_results: int | None = None
    min_similarity: float = 0.5
    grouped: bool = True
    top_k_children: int | None = None
    project: str | None = None
    include_archives: bool = False


class ApiKeyCreateRequest(BaseModel):
    user_name: str
    role: str = "editor"
    password: str | None = None
    generate_api_key: bool = False

    @field_validator("role")
    @classmethod
    def validate_role(cls, v: str) -> str:
        if v not in ("admin", "editor", "viewer"):
            raise ValueError("role must be admin, editor, or viewer")
        return v


class ApiKeyUpdateRequest(BaseModel):
    role: str | None = None
    is_active: bool | None = None
    generate_api_key: bool | None = None


# ============================================================
# Static / SPA
# ============================================================
STATIC_DIR = Path(__file__).parent / "static"

app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


@app.middleware("http")
async def no_cache_static(request: Request, call_next):
    """Prevent browser caching of static JS/CSS during development."""
    response = await call_next(request)
    if request.url.path.startswith("/static/"):
        response.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
        response.headers["Pragma"] = "no-cache"
        response.headers["Expires"] = "0"
    return response


@app.get("/", response_class=HTMLResponse)
async def serve_spa():
    """Serve the single-page application."""
    index_file = STATIC_DIR / "index.html"
    return HTMLResponse(content=index_file.read_text(encoding="utf-8"))


# ============================================================
# Entry Point
# ============================================================
def main():
    """Run the web server."""
    settings = load_settings()
    host = os.environ.get("TEAM_MEMORY_WEB_HOST", settings.web.host)
    port = int(os.environ.get("TEAM_MEMORY_WEB_PORT", str(settings.web.port)))
    ssl_keyfile = os.environ.get("TEAM_MEMORY_WEB_SSL_KEYFILE") or settings.web.ssl_keyfile
    ssl_certfile = os.environ.get("TEAM_MEMORY_WEB_SSL_CERTFILE") or settings.web.ssl_certfile
    use_ssl = bool(ssl_keyfile and ssl_certfile)
    scheme = "https" if use_ssl else "http"
    logger.info("Starting team_memory web server at %s://%s:%d", scheme, host, port)
    run_kwargs = {
        "host": host,
        "port": port,
        "reload": False,
        "log_level": "info",
    }
    if use_ssl:
        run_kwargs["ssl_keyfile"] = ssl_keyfile
        run_kwargs["ssl_certfile"] = ssl_certfile
    uvicorn.run("team_memory.web.app:app", **run_kwargs)


if __name__ == "__main__":
    main()

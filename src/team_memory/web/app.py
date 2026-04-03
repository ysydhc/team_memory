"""FastAPI web application for TeamMemory management UI.

Provides a REST API and single-page web interface for browsing,
searching, creating, and managing team experiences.

Authentication is via API Key (same keys used for MCP access).
"""

from __future__ import annotations

import asyncio
import logging
import os
import time
from collections import defaultdict
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import AsyncGenerator

import httpx
import uvicorn
from fastapi import (
    FastAPI,
    Request,
)
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from team_memory.bootstrap import (
    AppContext,
    bootstrap,
    get_context,
    start_background_tasks,
    stop_background_tasks,
)
from team_memory.config import Settings, load_settings
from team_memory.services.experience import ExperienceService
from team_memory.storage.database import get_session  # noqa: F401 -- used by route modules
from team_memory.web.auth_session import (  # noqa: F401 -- re-exported for backward compat
    _decode_session_token,
    _encode_api_key_cookie,
    _encode_session_token,
    _get_session_secret,
    _get_user_role_from_db,
    get_current_user,
    get_optional_user,
)
from team_memory.web.schemas import (  # noqa: F401 — re-exported for backward compat
    AdminResetPasswordRequest,
    ApiKeyCreateRequest,
    ApiKeyUpdateRequest,
    ChangePasswordRequest,
    ExperienceCreate,
    ExperienceUpdate,
    FeedbackCreate,
    ForgotPasswordResetRequest,
    LoginRequest,
    LoginResponse,
    RegisterRequest,
    SearchRequest,
)

logger = logging.getLogger("team_memory.web")
request_logger = logging.getLogger("team_memory.web.request")

# ============================================================
# Global state -- backed by bootstrap.AppContext singleton
# ============================================================
_settings: Settings | None = None
_service: ExperienceService | None = None
_auth = None  # AuthProvider | None


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
    """Resolve project from request > env > config default.

    Delegates to utils.project.resolve_project but falls back to the
    module-level _settings when AppContext is not yet bootstrapped.
    """
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
    from fastapi import APIRouter

    from team_memory.web.routes import mount_all

    v1_router = APIRouter(prefix="/api/v1")
    mount_all(v1_router)
    app.include_router(v1_router)
    await start_background_tasks(ctx)
    logger.info("TeamMemory web server started")
    yield
    await stop_background_tasks(ctx)
    # Ensure DB connections are released even if stop_background_tasks changes
    from team_memory.storage.database import close_db

    await close_db()
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


async def _check_llm_provider() -> dict:
    """Check LLM provider connectivity."""
    try:
        ctx = get_context()
        settings = ctx.settings
        if settings.llm.provider == "ollama":
            async with httpx.AsyncClient(timeout=5) as client:
                resp = await client.get(f"{settings.llm.base_url}/api/tags")
                resp.raise_for_status()
                return {
                    "status": "up",
                    "provider": settings.llm.provider,
                    "model": settings.llm.model,
                }
        return {"status": "up", "provider": settings.llm.provider}
    except Exception as e:
        return {"status": "down", "error": str(e)[:100]}


async def _check_cache() -> dict:
    """Check cache backend availability."""
    try:
        ctx = get_context()
        pipeline = (
            ctx.search_orchestrator._search_pipeline
            if hasattr(ctx, "search_orchestrator")
            else None
        )
        if pipeline and pipeline._cache:
            return {"status": "up", "backend": ctx.settings.cache.backend}
        return {"status": "up", "backend": "none"}
    except Exception as e:
        return {"status": "down", "error": str(e)[:100]}


@app.get("/health")
async def health_check():
    """Health check endpoint -- no authentication required."""
    db_check, ollama_check, embed_check, llm_check, cache_check = await asyncio.gather(
        _check_database(),
        _check_ollama(),
        _check_embedding_provider(),
        _check_llm_provider(),
        _check_cache(),
    )

    checks = {
        "database": db_check,
        "ollama": ollama_check,
        "embedding_provider": embed_check,
        "llm_provider": llm_check,
        "cache": cache_check,
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
    """Readiness probe -- returns 200 only when database is healthy."""
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

# ---- Rate limiting (in-memory, per IP) ----
_rate_limit_store: dict[str, list[float]] = defaultdict(list)


@app.middleware("http")
async def rate_limit(request: Request, call_next):
    """Reject requests that exceed the per-IP rate limit."""
    if request.url.path in ("/health", "/ready"):
        return await call_next(request)

    client_ip = request.client.host if request.client else "unknown"
    limit = getattr(getattr(_settings, "web", None), "rate_limit_per_minute", None) or 200
    now = time.monotonic()
    window = 60.0

    timestamps = _rate_limit_store[client_ip]
    _rate_limit_store[client_ip] = [t for t in timestamps if now - t < window]

    if len(_rate_limit_store[client_ip]) >= limit:
        return JSONResponse(
            status_code=429,
            content={"detail": "Rate limit exceeded"},
        )

    _rate_limit_store[client_ip].append(now)
    return await call_next(request)


# ---- Request body size limit ----
@app.middleware("http")
async def limit_request_body(request: Request, call_next):
    """Return 413 when Content-Length exceeds the configured maximum."""
    content_length = request.headers.get("content-length")
    max_bytes = (
        getattr(getattr(_settings, "web", None), "max_request_body_bytes", None) or 20_971_520
    )
    if content_length and int(content_length) > int(max_bytes):
        return JSONResponse(
            status_code=413,
            content={"detail": f"Request body too large. Max {max_bytes} bytes."},
        )
    return await call_next(request)


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

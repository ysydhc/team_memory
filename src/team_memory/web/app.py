"""FastAPI web application for TeamMemory management UI.

Provides a REST API and single-page web interface for browsing,
searching, creating, and managing team experiences.

Authentication is via API Key (same keys used for MCP access).
"""

from __future__ import annotations

import asyncio
import logging
import os
import re
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

from team_memory.auth.provider import (
    AuthProvider,
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
from team_memory.services.installable_catalog import (
    InstallableCatalogService,
)
from team_memory.storage.database import get_session
from team_memory.storage.repository import ExperienceRepository  # noqa: F401

logger = logging.getLogger("team_memory.web")

# ============================================================
# Global state — backed by bootstrap.AppContext singleton
# ============================================================
_settings: Settings | None = None
_service: ExperienceService | None = None
_auth: AuthProvider | None = None


def _init_from_context(ctx: AppContext) -> None:
    """Populate module-level aliases from AppContext for backward compat."""
    global _settings, _service, _auth
    _settings = ctx.settings
    _service = ctx.service
    _auth = ctx.auth


def _get_db_url() -> str:
    if _settings is not None:
        env_url = os.environ.get("TEAM_MEMORY_DB_URL")
        return env_url or _settings.database.url
    return get_context().db_url


def _normalize_project_name(project: str | None) -> str:
    """Normalize legacy project aliases to a canonical project name."""
    if not project:
        return ""
    value = project.strip()
    alias_map = {
        "team-memory": "team_memory",
        "team_doc": "team_memory",
    }
    return alias_map.get(value, value)


def _resolve_project(project: str | None) -> str:
    """Resolve project from request > env > config default."""
    normalized = _normalize_project_name(project)
    if normalized:
        return normalized
    env_project = _normalize_project_name(os.environ.get("TEAM_MEMORY_PROJECT", ""))
    if env_project:
        return env_project
    settings = _settings or get_context().settings
    default_project = _normalize_project_name(settings.default_project)
    if default_project:
        return default_project
    return "default"


def _get_catalog_service() -> InstallableCatalogService:
    """Create installable catalog service from current settings."""
    settings = _settings or get_context().settings
    return InstallableCatalogService(
        config=settings.installable_catalog,
        workspace_root=Path.cwd(),
    )


# ============================================================
# Lifespan
# ============================================================
@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    ctx = bootstrap()
    _init_from_context(ctx)
    # Defer route registration to avoid circular import (routes -> app)
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
# Health Check & Readiness (D4) — No auth required
# ============================================================


async def _check_database() -> dict:
    """Check database connectivity."""
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
    """Check Ollama service availability."""
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


async def _check_cache() -> dict:
    """Check cache status."""
    if _service and _service._search_pipeline:
        cache = _service._search_pipeline._cache
        stats = await cache.stats
        return {"status": "up", **stats}
    return {"status": "unknown"}


async def _check_dashboard_stats() -> dict:
    """Check that dashboard /stats can run (DB schema and service ready).

    If this fails, the web UI dashboard will show '加载仪表盘失败'.
    Error message is included for ops: run make health or GET /health to see it.
    """
    if _service is None:
        return {
            "status": "down",
            "error": "Service not initialized (bootstrap not run or failed)",
            "ops_hint": "Ensure app started with valid config and DB; check server startup logs.",
        }
    try:
        await _service.get_stats()
        return {"status": "up"}
    except Exception as e:
        return {
            "status": "down",
            "error": str(e),
            "ops_hint": (
                "Dashboard /api/v1/stats failed. Common causes: "
                "DB migration not run (alembic upgrade head), "
                "wrong database url in config, or missing table 'experiences'."
            ),
        }


@app.get("/health")
async def health_check():
    """Health check endpoint — no authentication required.

    Returns overall status: healthy / degraded / unhealthy.
    Checks: database, ollama, cache, embedding, migration.
    """
    db_check, ollama_check, cache_check, dashboard_check = await asyncio.gather(
        _check_database(),
        _check_ollama(),
        _check_cache(),
        _check_dashboard_stats(),
    )

    event_bus_info = {}
    if _service and _service._event_bus:
        event_bus_info = _service._event_bus.stats()

    checks = {
        "database": db_check,
        "ollama": ollama_check,
        "cache": cache_check,
        "dashboard_stats": dashboard_check,
        "event_bus": {"status": "up", **event_bus_info},
    }

    if _service and _service._embedding_queue:
        checks["embedding_queue"] = {
            "status": "up",
            **_service._embedding_queue.status,
        }

    # Enhanced: Embedding provider test
    embed_check = await _check_embedding_provider()
    checks["embedding_provider"] = embed_check

    # Enhanced: Migration status
    migration_check = await _check_migration_status()
    checks["migration"] = migration_check

    if db_check["status"] == "down":
        status = "unhealthy"
    elif any(
        checks[k].get("status") == "down"
        for k in ["ollama", "dashboard_stats", "embedding_provider"]
    ):
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


async def _check_embedding_provider() -> dict:
    """Test embedding by encoding a short text."""
    try:
        from team_memory.bootstrap import get_context
        ctx = get_context()
        provider_name = ctx.settings.embedding.provider
        vec = await ctx.embedding.encode_single("health check")
        return {
            "status": "up",
            "provider": provider_name,
            "dimension": len(vec),
        }
    except RuntimeError:
        return {"status": "down", "reason": "not bootstrapped"}
    except Exception as e:
        return {"status": "down", "error": str(e)[:100]}


async def _check_migration_status() -> dict:
    """Check if alembic migrations are up to date."""
    try:
        import subprocess
        result = subprocess.run(
            ["alembic", "current", "--verbose"],
            capture_output=True, text=True, timeout=10,
            cwd=os.path.dirname(os.path.dirname(os.path.dirname(
                os.path.dirname(__file__)
            ))),
        )
        current = result.stdout.strip() if result.returncode == 0 else "unknown"
        result2 = subprocess.run(
            ["alembic", "heads", "--verbose"],
            capture_output=True, text=True, timeout=10,
            cwd=os.path.dirname(os.path.dirname(os.path.dirname(
                os.path.dirname(__file__)
            ))),
        )
        heads = result2.stdout.strip() if result2.returncode == 0 else "unknown"
        is_current = "head" in current.lower() or current == heads
        return {
            "status": "up" if is_current else "warning",
            "current": current[:80],
            "heads": heads[:80],
        }
    except Exception as e:
        return {"status": "unknown", "error": str(e)[:100]}


@app.get("/ready")
async def readiness_check():
    """Readiness probe — returns 200 only when critical dependencies are healthy.

    Used by Docker / Kubernetes to determine if the service can accept traffic.
    """
    db_check = await _check_database()
    if db_check["status"] == "down":
        return JSONResponse(
            status_code=503,
            content={"ready": False, "reason": "database unavailable"},
        )
    return {"ready": True}


# ============================================================
# Backward-compatible middleware: /api/xxx → /api/v1/xxx (D3)
# ============================================================
from team_memory.web.middleware import api_version_compat  # noqa: E402

app.middleware("http")(api_version_compat)


# ============================================================
# Global exception handler — always return JSON for API clients
# ============================================================
@app.exception_handler(Exception)
async def _global_exception_handler(request: Request, exc: Exception):
    """Ensure unhandled exceptions return JSON with ops hint (no HTML/plain text)."""
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
                "ops_hint": (
                    "Grep server log for '[%s]' or run: make health to check DB and services."
                    % err_id
                ),
                "message": str(exc),
            },
        )
    raise exc


# ============================================================
# Auth Dependencies
# ============================================================
def _encode_api_key_cookie(api_key: str) -> str:
    """Encode API key into ASCII-safe cookie value."""
    return quote(api_key, safe="")


async def get_current_user(request: Request) -> User:
    """Extract and validate credentials from header or cookie.

    Supports both API Key and password-based cookie (pwd:user:pass).
    Raises 401 if not authenticated.
    """
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


class ExperienceCreate(BaseModel):
    title: str
    problem: str
    solution: str | None = None       # nullable: allow incomplete experiences
    tags: list[str] = Field(default_factory=list)
    code_snippets: str | None = None
    language: str | None = None
    framework: str | None = None
    root_cause: str | None = None
    publish_status: str = "published"  # "published" or "draft"
    skip_dedup_check: bool = False
    # Type system (v3)
    experience_type: str = "general"
    severity: str | None = None
    category: str | None = None
    progress_status: str | None = None
    structured_data: dict | None = None
    git_refs: list | None = None
    related_links: list | None = None
    project: str | None = None


class ExperienceUpdate(BaseModel):
    """Full in-place update request for experience editing (v3).

    All fields are optional — only provided fields are updated.
    Set a field to null/None to clear it.
    Use model_dump(exclude_unset=True) to distinguish "not sent" from "sent as null".
    """
    title: str | None = None
    problem: str | None = None          # maps to description
    solution: str | None = None
    root_cause: str | None = None
    tags: list[str] | None = None
    code_snippets: str | None = None
    language: str | None = None
    framework: str | None = None
    # Type system
    experience_type: str | None = None
    severity: str | None = None
    category: str | None = None
    progress_status: str | None = None
    structured_data: dict | None = None
    git_refs: list | None = None
    related_links: list | None = None
    publish_status: str | None = None
    # New status model
    visibility: str | None = None
    project: str | None = None
    # Legacy append mode
    solution_addendum: str | None = None


class FeedbackCreate(BaseModel):
    rating: int  # 1-5, 5=best
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
    max_results: int | None = None  # None = use config default
    min_similarity: float = 0.5
    grouped: bool = True
    top_k_children: int | None = None  # None = use config default
    min_avg_rating: float | None = None  # None = use config default
    max_tokens: int | None = None  # Reserved for future token trimming
    use_pageindex_lite: bool | None = None
    project: str | None = None


class InstallableInstallRequest(BaseModel):
    id: str
    source: str | None = None


class ReviewRequest(BaseModel):
    review_status: str  # "approved" or "rejected"
    review_note: str | None = None


class ParseDocumentRequest(BaseModel):
    content: str


class ParseURLRequest(BaseModel):
    url: str


class SuggestTypeRequest(BaseModel):
    title: str
    problem: str = ""


# ============================================================
# Auth Routes (moved to routes/)
# Models and helpers below are used by route modules.
# ============================================================

# ============================================================
# Experience Routes (Anonymous read-only access supported)
# ============================================================
# ============================================================
# Draft Routes (P0-1) — MUST be before {experience_id} routes
# ============================================================
# ============================================================
# Summary Routes (P0-4) — batch route MUST be before {experience_id}
# ============================================================
class ExperienceGroupChild(BaseModel):
    title: str
    problem: str = ""
    solution: str = ""
    tags: list[str] | None = None
    code_snippets: str | None = None
    root_cause: str | None = None
    language: str | None = None
    framework: str | None = None


class ExperienceGroupCreate(BaseModel):
    parent: ExperienceCreate
    children: list[ExperienceGroupChild]


# ============================================================
# Scope / Promote Routes (P1-2)
# ============================================================
# ============================================================
# Experience Link Routes (P1-3)
# ============================================================
class ExperienceLinkCreate(BaseModel):
    target_id: str
    link_type: str = "related"  # related, supersedes, derived_from

    @field_validator("link_type")
    @classmethod
    def validate_link_type(cls, v: str) -> str:
        if v not in ("related", "supersedes", "derived_from"):
            raise ValueError("link_type must be related, supersedes, or derived_from")
        return v


# ============================================================
# Review History Routes (P1-6)
# ============================================================
# ============================================================
# Version History Routes (B3)
# ============================================================
# ============================================================
# Review Routes (admin only)
# ============================================================
# ============================================================
# Publish Route (P0-1) — publish a draft experience
# ============================================================
# ============================================================
# Summarize Route (P0-4) — generate summary for a single experience
# ============================================================
# ============================================================
# Search Route (Anonymous access supported)
# ============================================================
# ============================================================
# Stats & Tags (Anonymous access supported)
# ============================================================
async def _llm_parse_content(content: str) -> dict:
    """Call LLM to parse text content into structured experience fields.

    Delegates to the shared llm_parser module. Translates LLMParseError
    into HTTPException for the web API layer.
    """
    from team_memory.services.llm_parser import LLMParseError, parse_content

    llm_config = _settings.llm if _settings else None
    try:
        return await parse_content(content, llm_config=llm_config)
    except LLMParseError as e:
        detail = str(e)
        if "Cannot connect" in detail:
            raise HTTPException(status_code=503, detail=detail)
        raise HTTPException(status_code=502, detail=detail)


def _extract_text_from_html(html: str) -> str:
    """Extract readable text from HTML, stripping tags/scripts/styles.

    Uses a lightweight regex-based approach to avoid extra dependencies.
    """
    # Remove script and style blocks
    text = re.sub(r"<script[^>]*>[\s\S]*?</script>", "", html, flags=re.IGNORECASE)
    text = re.sub(r"<style[^>]*>[\s\S]*?</style>", "", text, flags=re.IGNORECASE)
    # Remove HTML comments
    text = re.sub(r"<!--[\s\S]*?-->", "", text)
    # Replace <br>, <p>, <div>, <li>, headings with newlines for readability
    text = re.sub(r"<(?:br|p|div|li|h[1-6])[^>]*>", "\n", text, flags=re.IGNORECASE)
    # Strip remaining tags
    text = re.sub(r"<[^>]+>", "", text)
    # Decode common HTML entities
    text = text.replace("&amp;", "&").replace("&lt;", "<").replace("&gt;", ">")
    text = text.replace("&nbsp;", " ").replace("&quot;", '"').replace("&#39;", "'")
    # Collapse whitespace
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


# ============================================================
# API Key Management Routes (admin only)
# ============================================================
class ApiKeyCreateRequest(BaseModel):
    user_name: str
    role: str = "editor"
    password: str | None = None

    @field_validator("role")
    @classmethod
    def validate_role(cls, v: str) -> str:
        if v not in ("admin", "editor", "viewer"):
            raise ValueError("role must be admin, editor, or viewer")
        return v


class ApiKeyUpdateRequest(BaseModel):
    role: str | None = None
    is_active: bool | None = None


# ============================================================
# Retrieval Config Routes (admin)
# ============================================================
class RetrievalConfigUpdate(BaseModel):
    """Full replacement model for retrieval configuration.

    All fields are provided by the frontend form.
    max_tokens and summary_model can be None (meaning "no limit" / "use default").
    """
    max_tokens: int | None = None
    max_count: int = 20
    trim_strategy: str = "top_k"
    top_k_children: int = 3
    min_avg_rating: float = 0.0
    rating_weight: float = 0.3
    summary_model: str | None = None


class PageIndexLiteConfigUpdate(BaseModel):
    enabled: bool = True
    only_long_docs: bool = True
    min_doc_chars: int = 800
    max_tree_depth: int = 4
    max_nodes_per_doc: int = 40
    max_node_chars: int = 1200
    tree_weight: float = 0.15
    min_node_score: float = 0.01
    include_matched_nodes: bool = True


class DefaultProjectConfigUpdate(BaseModel):
    default_project: str


def _retrieval_config_dict(cfg) -> dict:
    """Serialize retrieval config to a JSON-safe dict."""
    return {
        "max_tokens": cfg.max_tokens,
        "max_count": cfg.max_count,
        "trim_strategy": cfg.trim_strategy,
        "top_k_children": cfg.top_k_children,
        "min_avg_rating": cfg.min_avg_rating,
        "rating_weight": cfg.rating_weight,
        "summary_model": cfg.summary_model,
    }


def _pageindex_lite_config_dict(cfg) -> dict:
    """Serialize PageIndex-Lite config to JSON-safe dict."""
    return {
        "enabled": cfg.enabled,
        "only_long_docs": cfg.only_long_docs,
        "min_doc_chars": cfg.min_doc_chars,
        "max_tree_depth": cfg.max_tree_depth,
        "max_nodes_per_doc": cfg.max_nodes_per_doc,
        "max_node_chars": cfg.max_node_chars,
        "tree_weight": cfg.tree_weight,
        "min_node_score": cfg.min_node_score,
        "include_matched_nodes": cfg.include_matched_nodes,
    }


def _installable_catalog_config_dict(cfg) -> dict:
    """Serialize installable catalog config to JSON-safe dict."""
    return {
        "sources": list(cfg.sources or []),
        "local_base_dir": cfg.local_base_dir,
        "registry_manifest_url": cfg.registry_manifest_url,
        "target_rules_dir": cfg.target_rules_dir,
        "target_prompts_dir": cfg.target_prompts_dir,
        "request_timeout_seconds": cfg.request_timeout_seconds,
    }


# ============================================================
# Extended Config Routes (reranker, search, cache)
# ============================================================

def _all_config_dict() -> dict:
    """Serialize all pipeline-related config sections."""
    if not _settings:
        return {}
    return {
        "retrieval": _retrieval_config_dict(_settings.retrieval),
        "pageindex_lite": _pageindex_lite_config_dict(_settings.pageindex_lite),
        "default_project": _settings.default_project,
        "installable_catalog": _installable_catalog_config_dict(
            _settings.installable_catalog
        ),
        "reranker": {
            "provider": _settings.reranker.provider,
            "ollama_llm": {
                "model": _settings.reranker.ollama_llm.model,
                "base_url": _settings.reranker.ollama_llm.base_url,
                "top_k": _settings.reranker.ollama_llm.top_k,
                "batch_size": _settings.reranker.ollama_llm.batch_size,
            },
            "cross_encoder": {
                "model_name": _settings.reranker.cross_encoder.model_name,
                "device": _settings.reranker.cross_encoder.device,
                "top_k": _settings.reranker.cross_encoder.top_k,
            },
            "jina": {
                "model": _settings.reranker.jina.model,
                "top_k": _settings.reranker.jina.top_k,
                "has_api_key": bool(_settings.reranker.jina.api_key),
            },
        },
        "search": {
            "mode": _settings.search.mode,
            "rrf_k": _settings.search.rrf_k,
            "vector_weight": _settings.search.vector_weight,
            "fts_weight": _settings.search.fts_weight,
            "adaptive_filter": _settings.search.adaptive_filter,
            "score_gap_threshold": _settings.search.score_gap_threshold,
            "min_confidence_ratio": _settings.search.min_confidence_ratio,
        },
        "cache": {
            "enabled": _settings.cache.enabled,
            "ttl_seconds": _settings.cache.ttl_seconds,
            "max_size": _settings.cache.max_size,
            "embedding_cache_size": _settings.cache.embedding_cache_size,
        },
    }


class SearchConfigUpdate(BaseModel):
    mode: str = "hybrid"
    rrf_k: int = 60
    vector_weight: float = 0.7
    fts_weight: float = 0.3
    adaptive_filter: bool = True
    score_gap_threshold: float = 0.15
    min_confidence_ratio: float = 0.6


class CacheConfigUpdate(BaseModel):
    enabled: bool = True
    ttl_seconds: int = 300
    max_size: int = 100
    embedding_cache_size: int = 200


class RerankerConfigUpdate(BaseModel):
    provider: str = "none"


# ============================================================
# Schema Registry API
# ============================================================


class SchemaGenerateRequest(BaseModel):
    content: str = Field(..., min_length=10, max_length=20000)


class MergeRequest(BaseModel):
    primary_id: str
    secondary_id: str


# ============================================================
# Lifecycle Config Routes
# ============================================================
class LifecycleConfigUpdate(BaseModel):
    stale_months: int = 6
    scan_interval_hours: int = 24
    duplicate_threshold: float = 0.92
    dedup_on_save: bool = True
    dedup_on_save_threshold: float = 0.90


# ============================================================
# Review Config Routes (P0-2)
# ============================================================
class ReviewConfigUpdate(BaseModel):
    enabled: bool = True
    auto_publish_threshold: float = 0.0
    require_review_for_ai: bool = True


# ============================================================
# Memory Config Routes (P0-4)
# ============================================================
class MemoryConfigUpdate(BaseModel):
    auto_summarize: bool = True
    summary_threshold_tokens: int = 500
    summary_model: str = ""
    batch_size: int = 10


# ============================================================
# Import / Export Routes (B4)
# ============================================================
# ============================================================
# Batch Operations (P3-1 E3)
# ============================================================
class BatchActionRequest(BaseModel):
    ids: list[str]
    action: str  # delete, tag, publish, set_scope
    tags: list[str] | None = None
    scope: str | None = None


# ============================================================
# Workflow Templates (P2-5)
# ============================================================
# ============================================================
# Tag Suggest (P2-7)
# ============================================================
# ============================================================
# Metrics & Analytics (P2-2)
# ============================================================
@app.get("/metrics")
async def prometheus_metrics():
    """Prometheus /metrics endpoint (optional, requires prometheus-client)."""
    from team_memory.web.metrics import get_prometheus_metrics, is_prometheus_available
    if not is_prometheus_available():
        return JSONResponse(
            status_code=501,
            content={"detail": "prometheus-client not installed. pip install prometheus-client"},
        )
    data, content_type = get_prometheus_metrics()
    from fastapi.responses import Response
    return Response(content=data, media_type=content_type)


# ============================================================
# API v1 Router — registered in lifespan to avoid circular import
# ============================================================

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
    # Config.yaml defaults, env vars override
    host = os.environ.get("TEAM_MEMORY_WEB_HOST", settings.web.host)
    port = int(os.environ.get("TEAM_MEMORY_WEB_PORT", str(settings.web.port)))
    print(f"Starting team_memory web server at http://{host}:{port}")
    print(f"Open http://localhost:{port} in your browser")
    uvicorn.run(
        "team_memory.web.app:app",
        host=host,
        port=port,
        reload=False,
        log_level="info",
    )


if __name__ == "__main__":
    main()

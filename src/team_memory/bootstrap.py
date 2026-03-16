"""Unified application bootstrap — single source of truth for initialization.

Provides AppContext (a dataclass holding all runtime components) and a
module-level singleton factory so that both the FastAPI web process and
the MCP stdio process share the same objects when running in-process.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import queue
from dataclasses import dataclass
from datetime import datetime, timezone
from logging.handlers import QueueHandler, QueueListener, RotatingFileHandler
from pathlib import Path

# Load .env into os.environ so TEAM_MEMORY_API_KEY etc. are available for auth and config
try:
    from dotenv import load_dotenv

    _env_file = Path(".env")
    if _env_file.exists():
        load_dotenv(_env_file)
except ImportError:
    pass
from typing import TYPE_CHECKING

from team_memory.auth.provider import AuthProvider, create_auth_provider
from team_memory.config import Settings, load_settings
from team_memory.embedding.base import EmbeddingProvider
from team_memory.schemas import init_schema_registry
from team_memory.services.archive import ArchiveService
from team_memory.services.event_bus import EventBus, Events
from team_memory.services.experience import ExperienceService

if TYPE_CHECKING:
    from team_memory.schemas import SchemaRegistry
    from team_memory.services.embedding_queue import EmbeddingQueue
    from team_memory.services.search_pipeline import SearchPipeline
    from team_memory.services.webhook import WebhookService

logger = logging.getLogger("team_memory.bootstrap")


_LOG_RECORD_STD_ATTRS = frozenset({
    "name", "msg", "args", "created", "filename", "funcName",
    "levelname", "levelno", "lineno", "module", "msecs",
    "pathname", "process", "processName", "relativeCreated",
    "stack_info", "exc_info", "exc_text", "thread", "threadName",
    "message", "taskName",
})

# Sensitive keys to redact in extra (per docs/design-docs/logging-format.md)
_SENSITIVE_KEYS = frozenset({
    "api_key", "apikey", "api-key", "password", "secret", "token",
    "authorization", "auth_header",
})


def _redact_sensitive(extra: dict) -> dict:
    """Redact sensitive keys in extra dict. Returns new dict."""
    result = {}
    for k, v in extra.items():
        key_lower = k.lower().replace("-", "_")
        is_sensitive = (
            key_lower in _SENSITIVE_KEYS
            or any(s in key_lower for s in ("secret", "token", "password"))
        )
        result[k] = "***" if is_sensitive else v
    return result


class _JsonFormatter(logging.Formatter):
    """JSON Lines formatter per docs/design-docs/logging-format.md.

    Outputs single-line JSON with required fields: timestamp, level, logger, message.
    Redacts sensitive keys in extra per logging-format.md.
    """

    def format(self, record: logging.LogRecord) -> str:
        ts = datetime.fromtimestamp(record.created, tz=timezone.utc).strftime(
            "%Y-%m-%dT%H:%M:%S.%f"
        )[:-3] + "Z"
        log_record: dict = {
            "timestamp": ts,
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        if hasattr(record, "request_id") and record.request_id:
            log_record["request_id"] = record.request_id
        extra = {k: v for k, v in record.__dict__.items() if k not in _LOG_RECORD_STD_ATTRS}
        if extra:
            log_record["extra"] = _redact_sensitive(extra)
        return json.dumps(log_record, ensure_ascii=False)


def _is_debug_mode() -> bool:
    """True if TEAM_MEMORY_DEBUG=1 or team_memory logger effective level is DEBUG.

    Per docs/plans/2025-03-10-logging-system-design.md: when debug, LOG_FILE_MAX_BYTES
    does not apply (no size limit for local debugging).
    """
    if os.environ.get("TEAM_MEMORY_DEBUG", "0") == "1":
        return True
    tm_logger = logging.getLogger("team_memory")
    return tm_logger.getEffectiveLevel() == logging.DEBUG


class NonBlockingQueueHandler(QueueHandler):
    """QueueHandler that uses put(block=False); on queue.Full logs a warning and drops."""

    def enqueue(self, record: logging.LogRecord) -> None:
        try:
            self.queue.put(record, block=False)
        except queue.Full:
            logger = logging.getLogger("team_memory.bootstrap")
            logger.warning("io_log queue full, dropping log record")


def _configure_logging(settings: Settings) -> QueueListener | None:
    """Configure team_memory loggers per config.LOG_FORMAT (L3 only).

    When log_file_enabled, adds QueueHandler + QueueListener + FileHandler/RotatingFileHandler
    for async file logging. Uses RotatingFileHandler (size-based) unless DEBUG mode.
    Returns the QueueListener for lifecycle management.
    """
    is_debug = _is_debug_mode()  # Check before modifying logger level
    use_json = getattr(settings, "log_format", "human") == "json"
    root = logging.getLogger("team_memory")
    root.setLevel(logging.INFO)
    root.propagate = False  # Capture all team_memory.* here, avoid duplicate to root
    for h in root.handlers[:]:
        root.removeHandler(h)
    handler = logging.StreamHandler()
    handler.setLevel(logging.INFO)
    if use_json:
        handler.setFormatter(_JsonFormatter())
    else:
        handler.setFormatter(
            logging.Formatter("%(asctime)s %(levelname)s [%(name)s] %(message)s")
        )
    root.addHandler(handler)

    log_listener: QueueListener | None = None
    if getattr(settings, "logging", None) and getattr(
        settings.logging, "log_file_enabled", False
    ):
        log_cfg = settings.logging
        log_path = getattr(log_cfg, "log_file_path", "logs/team_memory.log")
        backup_count = getattr(log_cfg, "log_file_backup_count", 5)
        max_bytes = getattr(log_cfg, "log_file_max_bytes", 10 * 1024 * 1024)
        Path(log_path).parent.mkdir(parents=True, exist_ok=True)
        log_queue: queue.Queue[logging.LogRecord] = queue.Queue(maxsize=10000)
        if is_debug:
            file_handler = logging.FileHandler(log_path, encoding="utf-8")
        else:
            file_handler = RotatingFileHandler(
                log_path,
                maxBytes=max_bytes,
                backupCount=backup_count,
                encoding="utf-8",
            )
        file_handler.setLevel(logging.INFO)
        if use_json:
            file_handler.setFormatter(_JsonFormatter())
        else:
            file_handler.setFormatter(
                logging.Formatter("%(asctime)s %(levelname)s [%(name)s] %(message)s")
            )
        log_listener = QueueListener(log_queue, file_handler, respect_handler_level=True)
        log_listener.start()
        root.addHandler(NonBlockingQueueHandler(log_queue))

    # request_logger: remove app.py hardcoded handler, inherit from team_memory
    req_log = logging.getLogger("team_memory.web.request")
    for h in req_log.handlers[:]:
        req_log.removeHandler(h)
    req_log.propagate = True

    return log_listener


@dataclass
class AppContext:
    """Runtime context holding every application component."""

    settings: Settings
    db_url: str
    embedding: EmbeddingProvider
    auth: AuthProvider
    event_bus: EventBus
    schema_registry: SchemaRegistry
    search_pipeline: SearchPipeline
    service: ExperienceService
    archive_service: ArchiveService
    embedding_queue: EmbeddingQueue | None = None
    webhook_service: WebhookService | None = None
    _log_listener: QueueListener | None = None
    _stale_scanner_task: asyncio.Task | None = None
    _file_location_cleanup_task: asyncio.Task | None = None


_instance: AppContext | None = None


def _create_embedding_provider(settings: Settings) -> EmbeddingProvider:
    """Create the configured embedding provider with automatic degradation.

    Priority: configured provider → Ollama → OpenAI → generic → local.
    """
    cfg = settings.embedding
    provider = cfg.provider

    def _try_ollama() -> EmbeddingProvider | None:
        try:
            import httpx

            resp = httpx.get(f"{cfg.ollama.base_url}/api/tags", timeout=3)
            if resp.status_code == 200:
                from team_memory.embedding.ollama_provider import OllamaEmbedding

                return OllamaEmbedding(
                    model=cfg.ollama.model,
                    dim=cfg.ollama.dimension,
                    base_url=cfg.ollama.base_url,
                )
        except Exception:
            pass
        return None

    def _try_openai() -> EmbeddingProvider | None:
        api_key = cfg.openai.api_key or os.environ.get("OPENAI_API_KEY", "")
        if api_key:
            from team_memory.embedding.openai_provider import OpenAIEmbedding

            return OpenAIEmbedding(
                api_key=api_key, model=cfg.openai.model, dim=cfg.openai.dimension,
            )
        return None

    def _try_generic() -> EmbeddingProvider | None:
        api_key = cfg.generic.api_key or os.environ.get("EMBEDDING_API_KEY", "")
        if api_key or cfg.generic.base_url != "http://localhost:8080/v1":
            from team_memory.embedding.generic_provider import GenericEmbedding

            return GenericEmbedding(
                base_url=cfg.generic.base_url,
                api_key=api_key,
                model=cfg.generic.model,
                dim=cfg.generic.dimension,
            )
        return None

    def _try_local() -> EmbeddingProvider | None:
        try:
            from team_memory.embedding.local_provider import LocalEmbedding

            return LocalEmbedding(
                model_name=cfg.local.model_name,
                device=cfg.local.device,
                dim=cfg.local.dimension,
            )
        except Exception:
            return None

    factories = {
        "ollama": _try_ollama,
        "openai": _try_openai,
        "generic": _try_generic,
        "local": _try_local,
    }

    result = factories.get(provider, _try_ollama)()
    if result is not None:
        logger.info("Embedding provider: %s (configured)", provider)
        return result

    logger.warning("Configured embedding provider '%s' unavailable, trying fallbacks…", provider)
    fallback_order = ["ollama", "openai", "generic", "local"]
    for fb in fallback_order:
        if fb == provider:
            continue
        result = factories[fb]()
        if result is not None:
            logger.warning("Embedding provider degraded to: %s", fb)
            return result

    raise RuntimeError(
        "No embedding provider available. Install sentence-transformers, "
        "start Ollama, or set OPENAI_API_KEY."
    )


def _configure_auth(settings: Settings) -> AuthProvider:
    from team_memory.auth.provider import ApiKeyAuth

    db_url = _resolve_db_url(settings) if settings.auth.type == "db_api_key" else None
    auth = create_auth_provider(settings.auth.type, db_url=db_url)
    if isinstance(auth, ApiKeyAuth):
        if settings.auth.api_key:
            auth.register_key((settings.auth.api_key or "").strip(), settings.auth.user, "admin")
        env_key = (os.environ.get("TEAM_MEMORY_API_KEY") or "").strip()
        if env_key:
            user_name = (os.environ.get("TEAM_MEMORY_USER") or "admin").strip()
            auth.register_key(env_key, user_name, "admin")
            logger.debug("Registered API key from TEAM_MEMORY_API_KEY for user %s", user_name)
    return auth


def _resolve_db_url(settings: Settings) -> str:
    return os.environ.get("TEAM_MEMORY_DB_URL") or settings.database.url


def bootstrap(
    config_path: str | None = None,
    enable_background: bool = True,
) -> AppContext:
    """Module-level singleton factory.

    First call creates and caches, subsequent calls return the same instance.
    ``enable_background=True`` creates EmbeddingQueue and WebhookService;
    ``False`` skips them (for lightweight MCP stdio usage).
    """
    global _instance
    if _instance is not None:
        return _instance

    settings = load_settings(config_path)
    log_listener = _configure_logging(settings)
    db_url = _resolve_db_url(settings)
    embedding = _create_embedding_provider(settings)
    auth = _configure_auth(settings)

    from team_memory.reranker.factory import create_reranker
    from team_memory.services.search_pipeline import SearchPipeline

    reranker = create_reranker(settings.reranker, settings.llm)
    search_pipeline = SearchPipeline(
        embedding_provider=embedding,
        reranker_provider=reranker,
        search_config=settings.search,
        retrieval_config=settings.retrieval,
        cache_config=settings.cache,
        pageindex_lite_config=settings.pageindex_lite,
        llm_config=settings.llm,
        tag_synonyms=getattr(settings, "tag_synonyms", None) or {},
        db_url=db_url,
        file_location_config=settings.file_location_binding,
    )

    event_bus = EventBus()

    embedding_queue: EmbeddingQueue | None = None
    if enable_background:
        from team_memory.services.embedding_queue import EmbeddingQueue as EQClass

        embedding_queue = EQClass(
            embedding_provider=embedding,
            db_url=db_url,
            max_workers=3,
            max_retries=3,
            event_bus=event_bus,
        )

    archive_service = ArchiveService(
        embedding_provider=embedding,
        db_url=db_url,
    )

    service = ExperienceService(
        embedding_provider=embedding,
        auth_provider=auth,
        search_pipeline=search_pipeline,
        event_bus=event_bus,
        embedding_queue=embedding_queue,
        lifecycle_config=settings.lifecycle,
        review_config=settings.review,
        memory_config=settings.memory,
        llm_config=settings.llm,
        pageindex_lite_config=settings.pageindex_lite,
        file_location_config=settings.file_location_binding,
        db_url=db_url,
        archive_service=archive_service,
    )

    schema_registry = init_schema_registry(settings.custom_schema)

    webhook_service: WebhookService | None = None
    if enable_background and settings.webhooks:
        from team_memory.services.webhook import WebhookService as WSClass

        webhook_configs = [w.model_dump() for w in settings.webhooks]
        webhook_service = WSClass(event_bus, webhook_configs)
        logger.info(
            "WebhookService activated with %d target(s)", len(settings.webhooks)
        )

    from team_memory.extensions import ExtensionContext, load_extensions

    ext_ctx = ExtensionContext(
        event_bus=event_bus,
        schema_registry=schema_registry,
        settings=settings,
    )
    load_extensions(ext_ctx)

    if enable_background:
        _register_cache_invalidation(event_bus, service)

    # Register hook registry with usage tracking so MCP tool calls are written to tool_usage_logs
    from team_memory.services.hooks import init_hook_registry
    from team_memory.storage.database import get_session_factory

    init_hook_registry(
        session_factory=get_session_factory(db_url),
        project=getattr(settings, "default_project", None) or "default",
    )

    _instance = AppContext(
        settings=settings,
        db_url=db_url,
        embedding=embedding,
        auth=auth,
        event_bus=event_bus,
        schema_registry=schema_registry,
        search_pipeline=search_pipeline,
        service=service,
        archive_service=archive_service,
        embedding_queue=embedding_queue,
        webhook_service=webhook_service,
        _log_listener=log_listener,
    )

    from team_memory import io_logger

    io_logger._settings_provider = lambda: get_context().settings

    logger.info(
        "AppContext created (background=%s, embedding=%s)",
        enable_background,
        settings.embedding.provider,
    )
    return _instance


def get_context() -> AppContext:
    """Return the cached AppContext, raising if not yet bootstrapped."""
    if _instance is None:
        raise RuntimeError("App not bootstrapped — call bootstrap() first")
    return _instance


def reset_context() -> None:
    """Clear the singleton (for testing)."""
    global _instance
    _instance = None


def _register_cache_invalidation(event_bus: EventBus, service: ExperienceService):
    async def _on_data_change(payload: dict) -> None:
        await service.invalidate_search_cache()
        logger.debug("Cache invalidated by event: %s", payload)

    for evt in (
        Events.EXPERIENCE_CREATED,
        Events.EXPERIENCE_UPDATED,
        Events.EXPERIENCE_DELETED,
        Events.EXPERIENCE_RESTORED,
        Events.EXPERIENCE_MERGED,
        Events.EXPERIENCE_ROLLED_BACK,
        Events.EXPERIENCE_IMPORTED,
        Events.EXPERIENCE_PUBLISHED,
        Events.EXPERIENCE_REVIEWED,
    ):
        event_bus.on(evt, _on_data_change)


async def start_background_tasks(ctx: AppContext) -> None:
    """Start embedding queue worker, stale scanner, and file location cleanup (if enabled)."""
    if ctx.embedding_queue:
        await ctx.embedding_queue.start()
    ctx._stale_scanner_task = asyncio.create_task(_stale_scanner_loop(ctx))
    if ctx.settings.file_location_binding.file_location_cleanup_enabled:
        ctx._file_location_cleanup_task = asyncio.create_task(
            _file_location_cleanup_loop(ctx)
        )
        logger.info(
            "File location cleanup task started (interval=%s h)",
            ctx.settings.file_location_binding.file_location_cleanup_interval_hours,
        )
    logger.info("Background tasks started (embedding queue + stale scanner)")


async def stop_background_tasks(ctx: AppContext) -> None:
    """Gracefully stop background tasks."""
    if ctx.embedding_queue:
        await ctx.embedding_queue.stop()
    if ctx._log_listener is not None:
        try:
            await asyncio.wait_for(
                asyncio.to_thread(ctx._log_listener.stop), timeout=5
            )
        except asyncio.TimeoutError:
            logger.warning("QueueListener.stop() timed out after 5s")
    if ctx._stale_scanner_task:
        ctx._stale_scanner_task.cancel()
        try:
            await ctx._stale_scanner_task
        except asyncio.CancelledError:
            pass
    if ctx._file_location_cleanup_task:
        ctx._file_location_cleanup_task.cancel()
        try:
            await ctx._file_location_cleanup_task
        except asyncio.CancelledError:
            pass
    logger.info("Background tasks stopped")


async def _stale_scanner_loop(ctx: AppContext) -> None:
    """Periodically scan for stale experiences."""
    from team_memory.storage.database import get_session
    from team_memory.storage.repository import ExperienceRepository

    while True:
        interval = ctx.settings.lifecycle.scan_interval_hours
        await asyncio.sleep(interval * 3600)
        try:
            async with get_session(ctx.db_url) as session:
                repo = ExperienceRepository(session)
                months = ctx.settings.lifecycle.stale_months
                stale = await repo.scan_stale(months=months)
                if stale:
                    logger.info(
                        "Stale scanner: found %d stale experiences (unused > %d months)",
                        len(stale),
                        months,
                    )
        except Exception:
            logger.warning("Stale scanner error", exc_info=True)


async def _file_location_cleanup_loop(ctx: AppContext) -> None:
    """Periodically delete expired file location bindings when cleanup is enabled."""
    from team_memory.storage.database import get_session
    from team_memory.storage.repository import ExperienceRepository

    batch_size = 500
    cfg = ctx.settings.file_location_binding
    if not cfg.file_location_cleanup_enabled:
        return
    interval_seconds = cfg.file_location_cleanup_interval_hours * 3600

    while True:
        await asyncio.sleep(interval_seconds)
        if not ctx.settings.file_location_binding.file_location_cleanup_enabled:
            continue
        round_num = 0
        total_deleted = 0
        run_start = datetime.now(timezone.utc)
        try:
            async with get_session(ctx.db_url) as session:
                repo = ExperienceRepository(session)
                while True:
                    round_num += 1
                    round_start = datetime.now(timezone.utc)
                    n = await repo.delete_expired_file_location_bindings(
                        batch_size=batch_size
                    )
                    round_elapsed = (datetime.now(timezone.utc) - round_start).total_seconds()
                    total_deleted += n
                    logger.info(
                        "file_location_cleanup round=%s deleted=%s duration_seconds=%.2f",
                        round_num,
                        n,
                        round_elapsed,
                        extra={"round": round_num, "deleted": n, "duration_seconds": round_elapsed},
                    )
                    if n == 0 or n < batch_size:
                        break
            run_elapsed = (datetime.now(timezone.utc) - run_start).total_seconds()
            logger.info(
                "file_location_cleanup run completed total_deleted=%s total_duration_seconds=%.2f",
                total_deleted,
                run_elapsed,
                extra={"total_deleted": total_deleted, "total_duration_seconds": run_elapsed},
            )
        except asyncio.CancelledError:
            raise
        except Exception:
            logger.warning(
                "file_location_cleanup error (round=%s, total_deleted=%s)",
                round_num,
                total_deleted,
                exc_info=True,
            )

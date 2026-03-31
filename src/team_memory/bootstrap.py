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
from team_memory.services.archive import ArchiveService
from team_memory.services.event_bus import EventBus, Events
from team_memory.services.experience import ExperienceService

if TYPE_CHECKING:
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
    """JSON Lines formatter per docs/design-docs/logging-format.md."""

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
    """True if TEAM_MEMORY_DEBUG=1 or team_memory logger effective level is DEBUG."""
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
            logging.getLogger("team_memory.bootstrap").warning(
                "io_log queue full, dropping log record"
            )


def _configure_logging(settings: Settings) -> QueueListener | None:
    """Configure team_memory loggers per config.LOG_FORMAT."""
    is_debug = _is_debug_mode()
    use_json = getattr(settings, "log_format", "human") == "json"
    root = logging.getLogger("team_memory")
    root.setLevel(logging.INFO)
    root.propagate = False
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
    search_pipeline: SearchPipeline
    service: ExperienceService
    archive_service: ArchiveService
    webhook_service: WebhookService | None = None
    _log_listener: QueueListener | None = None


_instance: AppContext | None = None


def _create_embedding_provider(settings: Settings) -> EmbeddingProvider:
    """Create the configured embedding provider with automatic degradation.

    Priority: configured provider -> Ollama -> OpenAI -> generic -> local.
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

    logger.warning("Configured embedding provider '%s' unavailable, trying fallbacks...", provider)
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
    ``enable_background=True`` creates WebhookService;
    ``False`` skips it (for lightweight MCP stdio usage).
    """
    global _instance
    if _instance is not None:
        return _instance

    settings = load_settings(config_path)
    log_listener = _configure_logging(settings)
    db_url = _resolve_db_url(settings)
    embedding = _create_embedding_provider(settings)
    auth = _configure_auth(settings)

    from team_memory.services.search_pipeline import SearchPipeline

    search_pipeline = SearchPipeline(
        embedding_provider=embedding,
        search_config=settings.search,
        retrieval_config=settings.retrieval,
        cache_config=settings.cache,
        llm_config=settings.llm,
        tag_synonyms=getattr(settings, "tag_synonyms", None) or {},
        db_url=db_url,
    )

    event_bus = EventBus()

    archive_service = ArchiveService(
        embedding_provider=embedding,
        db_url=db_url,
    )

    service = ExperienceService(
        embedding_provider=embedding,
        auth_provider=auth,
        search_pipeline=search_pipeline,
        event_bus=event_bus,
        lifecycle_config=settings.lifecycle,
        llm_config=settings.llm,
        db_url=db_url,
        archive_service=archive_service,
    )

    webhook_service: WebhookService | None = None
    if enable_background and settings.webhooks:
        from team_memory.services.webhook import WebhookService as WSClass

        webhook_configs = [w.model_dump() for w in settings.webhooks]
        webhook_service = WSClass(event_bus, webhook_configs)
        logger.info(
            "WebhookService activated with %d target(s)", len(settings.webhooks)
        )

    if enable_background:
        _register_cache_invalidation(event_bus, service)

    # Register hook registry
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
        search_pipeline=search_pipeline,
        service=service,
        archive_service=archive_service,
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
        raise RuntimeError("App not bootstrapped -- call bootstrap() first")
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
    ):
        event_bus.on(evt, _on_data_change)


async def start_background_tasks(ctx: AppContext) -> None:
    """Start background tasks (placeholder for future use)."""
    logger.info("Background tasks started")


async def stop_background_tasks(ctx: AppContext) -> None:
    """Gracefully stop background tasks."""
    if ctx._log_listener is not None:
        try:
            await asyncio.wait_for(
                asyncio.to_thread(ctx._log_listener.stop), timeout=5
            )
        except asyncio.TimeoutError:
            logger.warning("QueueListener.stop() timed out after 5s")
    logger.info("Background tasks stopped")

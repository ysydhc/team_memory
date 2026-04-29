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
from team_memory.embedding.base import ConcurrencyLimitedEmbedding, EmbeddingProvider
from team_memory.services.archive import ArchiveService
from team_memory.services.evaluation import EvaluationService
from team_memory.services.event_bus import EventBus, Events
from team_memory.services.experience import ExperienceService
from team_memory.services.intent_router import DefaultIntentRouter, IntentRouter
from team_memory.services.search_orchestrator import SearchOrchestrator

if TYPE_CHECKING:
    from team_memory.services.janitor import MemoryJanitor
    from team_memory.services.janitor_scheduler import JanitorScheduler
    from team_memory.services.search_pipeline import SearchPipeline

logger = logging.getLogger("team_memory.bootstrap")


_LOG_RECORD_STD_ATTRS = frozenset(
    {
        "name",
        "msg",
        "args",
        "created",
        "filename",
        "funcName",
        "levelname",
        "levelno",
        "lineno",
        "module",
        "msecs",
        "pathname",
        "process",
        "processName",
        "relativeCreated",
        "stack_info",
        "exc_info",
        "exc_text",
        "thread",
        "threadName",
        "message",
        "taskName",
    }
)

# Sensitive keys to redact in extra (see tests/test_logging_json.py).
_SENSITIVE_KEYS = frozenset(
    {
        "api_key",
        "apikey",
        "api-key",
        "password",
        "secret",
        "token",
        "authorization",
        "auth_header",
    }
)


def _redact_sensitive(extra: dict) -> dict:
    """Redact sensitive keys in extra dict. Returns new dict."""
    result = {}
    for k, v in extra.items():
        key_lower = k.lower().replace("-", "_")
        is_sensitive = key_lower in _SENSITIVE_KEYS or any(
            s in key_lower for s in ("secret", "token", "password")
        )
        result[k] = "***" if is_sensitive else v
    return result


class _JsonFormatter(logging.Formatter):
    """JSON Lines formatter.

    JSON log shape: timestamp, level, logger, message, optional extra (redacted).
    """

    def format(self, record: logging.LogRecord) -> str:
        ts = (
            datetime.fromtimestamp(record.created, tz=timezone.utc).strftime(
                "%Y-%m-%dT%H:%M:%S.%f"
            )[:-3]
            + "Z"
        )
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
        handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s [%(name)s] %(message)s"))
    root.addHandler(handler)

    log_listener: QueueListener | None = None
    if getattr(settings, "logging", None) and getattr(settings.logging, "log_file_enabled", False):
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
    search_orchestrator: SearchOrchestrator
    evaluation_service: EvaluationService | None = None
    intent_router: IntentRouter = None  # type: ignore[assignment]
    janitor: MemoryJanitor | None = None
    janitor_scheduler: JanitorScheduler | None = None
    entity_extractor: object | None = None  # EntityExtractor (lazy import)
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
                api_key=api_key,
                model=cfg.openai.model,
                dim=cfg.openai.dimension,
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


def _validate_embedding_dimension(settings: Settings) -> None:
    """Warn if configured embedding dimension doesn't match the DB schema.

    The database vector columns are defined as Vector(DB_VECTOR_DIM).
    If the embedding provider produces vectors of a different size,
    inserts will fail or search quality will degrade.
    """
    from team_memory.storage.models import DB_VECTOR_DIM

    configured_dim = settings.embedding.dimension
    if configured_dim != DB_VECTOR_DIM:
        logger.warning(
            "Embedding dimension mismatch: provider configured for %d dims "
            "but database schema uses %d. Search quality may be degraded. "
            "To fix: create a migration to change vector(%d) to vector(%d) "
            "and re-embed all records.",
            configured_dim,
            DB_VECTOR_DIM,
            DB_VECTOR_DIM,
            configured_dim,
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
            auth.register_key(env_key, settings.auth.user, "admin")
            logger.debug(
                "Registered API key from TEAM_MEMORY_API_KEY for user %s", settings.auth.user
            )
    return auth


def _resolve_db_url(settings: Settings) -> str:
    return os.environ.get("TEAM_MEMORY_DB_URL") or settings.database.url


def bootstrap(
    config_path: str | None = None,
    enable_background: bool = True,
) -> AppContext:
    """Module-level singleton factory.

    First call creates and caches, subsequent calls return the same instance.
    ``enable_background=True`` registers cache invalidation handlers;
    ``False`` skips them (for lightweight MCP stdio usage).
    """
    global _instance
    if _instance is not None:
        return _instance

    settings = load_settings(config_path)
    log_listener = _configure_logging(settings)
    db_url = _resolve_db_url(settings)
    embedding = _create_embedding_provider(settings)
    embedding = ConcurrencyLimitedEmbedding(embedding, max_concurrent=10)
    _validate_embedding_dimension(settings)
    auth = _configure_auth(settings)

    from team_memory.reranker.factory import create_reranker
    from team_memory.services.search_pipeline import SearchPipeline

    rerank_cfg = settings.reranker
    rerank_enabled = rerank_cfg.provider != "none"
    reranker_provider = create_reranker(rerank_cfg, settings.llm) if rerank_enabled else None

    search_pipeline = SearchPipeline(
        embedding_provider=embedding,
        search_config=settings.search,
        retrieval_config=settings.retrieval,
        cache_config=settings.cache,
        llm_config=settings.llm,
        tag_synonyms=getattr(settings, "tag_synonyms", None) or {},
        db_url=db_url,
        reranker_provider=reranker_provider,
        rerank_enabled=rerank_enabled,
        reranker_signature=rerank_cfg.cache_signature(),
        rerank_max_document_chars=rerank_cfg.max_document_chars,
    )

    event_bus = EventBus()

    archive_service = ArchiveService(
        embedding_provider=embedding,
        db_url=db_url,
        event_bus=event_bus,
    )

    intent_router = DefaultIntentRouter()

    evaluation_service = EvaluationService(db_url=db_url, embedding_provider=embedding)

    search_orchestrator = SearchOrchestrator(
        search_pipeline=search_pipeline,
        embedding_provider=embedding,
        db_url=db_url,
        intent_router=intent_router,
        evaluation_service=evaluation_service,
    )

    service = ExperienceService(
        embedding_provider=embedding,
        auth_provider=auth,
        event_bus=event_bus,
        lifecycle_config=settings.lifecycle,
        llm_config=settings.llm,
        db_url=db_url,
        archive_service=archive_service,
    )

    if enable_background:
        _register_cache_invalidation(event_bus, search_orchestrator)
        _register_pattern_extraction(event_bus, embedding, settings.llm, db_url)
        entity_extractor = _register_entity_extraction(event_bus, settings.llm, db_url)
    else:
        entity_extractor = None

    # Initialize janitor services
    janitor = None
    janitor_scheduler = None

    if settings.janitor.enabled:
        from team_memory.services.janitor import MemoryJanitor
        from team_memory.services.janitor_scheduler import JanitorScheduler

        janitor = MemoryJanitor(db_url=db_url, config=settings.janitor)
        janitor_scheduler = JanitorScheduler(janitor=janitor, config=settings.janitor)

        # Start scheduler if background tasks are enabled
        if enable_background:
            import asyncio

            try:
                # Try to start scheduler in current event loop if available
                loop = asyncio.get_running_loop()
                loop.create_task(janitor_scheduler.start())
                logger.info("Janitor scheduler started in background")
            except RuntimeError:
                # No running event loop, scheduler will be started later
                logger.debug("No running event loop, janitor scheduler will start later")

    _instance = AppContext(
        settings=settings,
        db_url=db_url,
        embedding=embedding,
        auth=auth,
        event_bus=event_bus,
        search_pipeline=search_pipeline,
        service=service,
        archive_service=archive_service,
        search_orchestrator=search_orchestrator,
        evaluation_service=evaluation_service,
        intent_router=intent_router,
        janitor=janitor,
        janitor_scheduler=janitor_scheduler,
        entity_extractor=entity_extractor,
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


def _register_cache_invalidation(event_bus: EventBus, search_orchestrator: SearchOrchestrator):
    async def _on_data_change(payload: dict) -> None:
        await search_orchestrator.invalidate_cache()
        logger.debug("Cache invalidated by event: %s", payload)

    for evt in (
        Events.EXPERIENCE_CREATED,
        Events.EXPERIENCE_UPDATED,
        Events.EXPERIENCE_DELETED,
        Events.EXPERIENCE_RESTORED,
    ):
        event_bus.on(evt, _on_data_change)


def _register_pattern_extraction(
    event_bus: EventBus,
    embedding: EmbeddingProvider,
    llm_config: object,
    db_url: str,
) -> None:
    """Subscribe to ARCHIVE_CREATED to enqueue pattern extraction as a background task."""
    from team_memory.services import task_runner

    async def _handle_pattern_extraction(payload: dict) -> None:
        raw = payload.get("raw_conversation", "")
        user_id = payload.get("user_id", "")
        if not raw or not user_id:
            return
        try:
            from team_memory.services.pattern_extractor import PatternExtractor
            from team_memory.services.personal_memory import PersonalMemoryService

            pm_svc = PersonalMemoryService(embedding_provider=embedding, db_url=db_url)
            extractor = PatternExtractor()
            count = await extractor.extract_and_save(
                conversation=raw,
                user_id=user_id,
                llm_config=llm_config,
                pm_service=pm_svc,
            )
            if count:
                logger.info("Extracted %d patterns for user=%s", count, user_id)
        except Exception:
            logger.warning("Pattern extraction failed", exc_info=True)
            raise  # let task_runner handle retry

    task_runner.register_handler("pattern_extraction", _handle_pattern_extraction)

    async def _on_archive_created(payload: dict) -> None:
        raw = payload.get("raw_conversation", "")
        user_id = payload.get("user_id", "")
        if not raw or not user_id:
            return
        await task_runner.enqueue("pattern_extraction", payload, db_url=db_url)
        # Attempt immediate execution (best-effort, like the old create_task)
        try:
            await task_runner.poll_and_execute(db_url, batch_size=1)
        except Exception:
            logger.debug("Immediate task execution failed; will retry later", exc_info=True)

    event_bus.on(Events.ARCHIVE_CREATED, _on_archive_created)


async def start_background_tasks(ctx: AppContext) -> None:
    """Start background tasks (placeholder for future use)."""
    logger.info("Background tasks started")


def _register_entity_extraction(
    event_bus: EventBus,
    llm_config: object,
    db_url: str,
) -> object:
    """Subscribe to EXPERIENCE_CREATED to fire async entity extraction."""
    import asyncio

    from team_memory.services.entity_extractor import EntityExtractor

    extractor = EntityExtractor(llm_config=llm_config, db_url=db_url)

    async def _on_experience_created(payload: dict) -> None:
        exp_id = payload.get("experience_id", "")
        title = payload.get("title", "")
        status = payload.get("status", "")
        # Only extract for published experiences (covers both CREATED and UPDATED)
        if not exp_id or status not in ("published", "promoted"):
            return
        # Fire-and-forget: don't await directly so we don't block the event bus
        asyncio.get_event_loop().create_task(
            extractor.extract_and_persist(
                experience_id=exp_id,
                title=title,
                description=payload.get("description", ""),
                solution=payload.get("solution", ""),
                tags=payload.get("tags") or [],
                project=payload.get("project", "default"),
            )
        )

    event_bus.on(Events.EXPERIENCE_CREATED, _on_experience_created)
    event_bus.on(Events.EXPERIENCE_UPDATED, _on_experience_created)
    return extractor


async def stop_background_tasks(ctx: AppContext) -> None:
    """Gracefully stop background tasks."""
    # Stop janitor scheduler first
    if ctx.janitor_scheduler is not None:
        try:
            await ctx.janitor_scheduler.stop()
            logger.info("Janitor scheduler stopped")
        except Exception as e:
            logger.warning("Error stopping janitor scheduler: %s", e)

    if ctx._log_listener is not None:
        try:
            await asyncio.wait_for(asyncio.to_thread(ctx._log_listener.stop), timeout=5)
        except asyncio.TimeoutError:
            logger.warning("QueueListener.stop() timed out after 5s")
    from team_memory.storage.database import close_db

    await close_db()
    logger.info("Background tasks stopped")

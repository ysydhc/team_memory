"""Unified application bootstrap — single source of truth for initialization.

Provides AppContext (a dataclass holding all runtime components) and a
module-level singleton factory so that both the FastAPI web process and
the MCP stdio process share the same objects when running in-process.
"""

from __future__ import annotations

import asyncio
import logging
import os
from dataclasses import dataclass
from typing import TYPE_CHECKING

from team_memory.auth.provider import AuthProvider, create_auth_provider
from team_memory.config import Settings, load_settings
from team_memory.embedding.base import EmbeddingProvider
from team_memory.schemas import init_schema_registry
from team_memory.services.event_bus import EventBus, Events
from team_memory.services.experience import ExperienceService

if TYPE_CHECKING:
    from team_memory.schemas import SchemaRegistry
    from team_memory.services.embedding_queue import EmbeddingQueue
    from team_memory.services.search_pipeline import SearchPipeline
    from team_memory.services.webhook import WebhookService

logger = logging.getLogger("team_memory.bootstrap")


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
    embedding_queue: EmbeddingQueue | None = None
    webhook_service: WebhookService | None = None
    _stale_scanner_task: asyncio.Task | None = None


_instance: AppContext | None = None


def _create_embedding_provider(settings: Settings) -> EmbeddingProvider:
    cfg = settings.embedding
    if cfg.provider == "ollama":
        from team_memory.embedding.ollama_provider import OllamaEmbedding

        return OllamaEmbedding(
            model=cfg.ollama.model,
            dim=cfg.ollama.dimension,
            base_url=cfg.ollama.base_url,
        )
    elif cfg.provider == "openai":
        from team_memory.embedding.openai_provider import OpenAIEmbedding

        api_key = cfg.openai.api_key or os.environ.get("OPENAI_API_KEY", "")
        return OpenAIEmbedding(
            api_key=api_key,
            model=cfg.openai.model,
            dim=cfg.openai.dimension,
        )
    else:
        from team_memory.embedding.local_provider import LocalEmbedding

        return LocalEmbedding(
            model_name=cfg.local.model_name,
            device=cfg.local.device,
            dim=cfg.local.dimension,
        )


def _configure_auth(settings: Settings) -> AuthProvider:
    from team_memory.auth.provider import ApiKeyAuth

    auth = create_auth_provider(settings.auth.type)
    if isinstance(auth, ApiKeyAuth):
        if settings.auth.api_key:
            auth.register_key(settings.auth.api_key, settings.auth.user, "admin")
        env_key = os.environ.get("TEAM_MEMORY_API_KEY", "")
        if env_key:
            user_name = os.environ.get("TEAM_MEMORY_USER", "admin")
            auth.register_key(env_key, user_name, "admin")
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
        db_url=db_url,
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

    _instance = AppContext(
        settings=settings,
        db_url=db_url,
        embedding=embedding,
        auth=auth,
        event_bus=event_bus,
        schema_registry=schema_registry,
        search_pipeline=search_pipeline,
        service=service,
        embedding_queue=embedding_queue,
        webhook_service=webhook_service,
    )

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
    """Start embedding queue worker and stale scanner."""
    if ctx.embedding_queue:
        await ctx.embedding_queue.start()
    ctx._stale_scanner_task = asyncio.create_task(_stale_scanner_loop(ctx))
    logger.info("Background tasks started (embedding queue + stale scanner)")


async def stop_background_tasks(ctx: AppContext) -> None:
    """Gracefully stop background tasks."""
    if ctx.embedding_queue:
        await ctx.embedding_queue.stop()
    if ctx._stale_scanner_task:
        ctx._stale_scanner_task.cancel()
        try:
            await ctx._stale_scanner_task
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

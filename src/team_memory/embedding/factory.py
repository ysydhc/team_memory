"""Factory function to create an EmbeddingProvider from configuration."""

from __future__ import annotations

import logging
import os

from team_memory.config import Settings
from team_memory.embedding.base import EmbeddingProvider

logger = logging.getLogger("team_memory.embedding.factory")


def create_embedding_provider(settings: Settings) -> EmbeddingProvider:
    """Create the configured embedding provider with automatic degradation.

    Priority: configured provider -> Ollama -> OpenAI -> generic -> local.
    """
    cfg = settings.embedding
    provider = cfg.provider

    def _try_openai() -> EmbeddingProvider | None:
        api_key = cfg.openai.api_key or os.environ.get("OPENAI_API_KEY", "")
        if api_key:
            from team_memory.embedding.openai_provider import OpenAIEmbedding

            kwargs = {
                "api_key": api_key,
                "model": cfg.openai.model,
                "dim": cfg.openai.dimension,
            }
            if cfg.openai.base_url:
                kwargs["base_url"] = cfg.openai.base_url

            return OpenAIEmbedding(**kwargs)
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
        "ollama": lambda: _make_ollama(cfg),
        "openai": _try_openai,
        "generic": _try_generic,
        "local": _try_local,
    }

    if provider not in factories:
        raise ValueError(
            f"Unknown embedding provider '{provider}'. "
            f"Must be one of: {list(factories.keys())}. "
            f"Set embedding.provider in config.development.yaml."
        )
    result = factories[provider]()
    if result is not None:
        logger.info("Embedding provider: %s (configured)", provider)
        return result

    logger.warning(
        "Configured embedding provider '%s' unavailable, trying fallbacks...", provider
    )
    fallback_order = ["ollama", "openai", "generic", "local"]
    for fb in fallback_order:
        if fb == provider:
            continue
        result = factories[fb]()
        if result is not None:
            logger.warning("Embedding provider degraded to: %s", fb)
            return result

    raise RuntimeError(
        "No embedding provider available. Install sentence-transformers, start Ollama, or set OPENAI_API_KEY."
    )

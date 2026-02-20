"""Factory function to create a RerankerProvider based on configuration.

Reads the reranker config section and instantiates the appropriate provider.
Falls back to NoopRerankerProvider if configuration is invalid.
"""

from __future__ import annotations

import logging

from team_memory.config import LLMConfig, RerankerConfig
from team_memory.reranker.base import RerankerProvider
from team_memory.reranker.noop_provider import NoopRerankerProvider

logger = logging.getLogger("team_memory.reranker.factory")


def create_reranker(
    reranker_config: RerankerConfig,
    llm_config: LLMConfig | None = None,
) -> RerankerProvider:
    """Create a RerankerProvider instance from configuration.

    Args:
        reranker_config: Reranker configuration section.
        llm_config: Global LLM config (for ollama_llm provider fallback values).

    Returns:
        A RerankerProvider instance ready to use.
    """
    provider = reranker_config.provider

    if provider == "none":
        logger.info("Reranker: disabled (provider=none)")
        return NoopRerankerProvider()

    if provider == "ollama_llm":
        from team_memory.reranker.ollama_llm_provider import (
            OllamaLLMRerankerProvider,
        )

        cfg = reranker_config.ollama_llm
        # Fall back to global LLM config if not specified
        model = cfg.model or (llm_config.model if llm_config else "gpt-oss:20b-cloud")
        base_url = cfg.base_url or (llm_config.base_url if llm_config else "http://localhost:11434")

        logger.info(
            "Reranker: ollama_llm (model=%s, base_url=%s, top_k=%d)",
            model,
            base_url,
            cfg.top_k,
        )
        return OllamaLLMRerankerProvider(
            model=model,
            base_url=base_url,
            top_k=cfg.top_k,
            batch_size=cfg.batch_size,
            prompt_template=cfg.prompt_template,
        )

    if provider == "cross_encoder":
        from team_memory.reranker.cross_encoder_provider import (
            CrossEncoderRerankerProvider,
        )

        cfg = reranker_config.cross_encoder
        logger.info(
            "Reranker: cross_encoder (model=%s, device=%s)",
            cfg.model_name,
            cfg.device,
        )
        return CrossEncoderRerankerProvider(
            model_name=cfg.model_name,
            device=cfg.device,
            top_k=cfg.top_k,
        )

    if provider == "jina":
        from team_memory.reranker.jina_provider import JinaRerankerProvider

        cfg = reranker_config.jina
        logger.info("Reranker: jina (model=%s)", cfg.model)
        return JinaRerankerProvider(
            api_key=cfg.api_key,
            model=cfg.model,
            top_k=cfg.top_k,
        )

    logger.warning("Unknown reranker provider '%s', falling back to noop", provider)
    return NoopRerankerProvider()

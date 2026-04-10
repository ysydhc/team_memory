"""Reranker provider configuration (search pipeline server-side reranking)."""

from __future__ import annotations

import hashlib
from typing import Literal

from pydantic import BaseModel, Field


class RerankerOllamaLLMConfig(BaseModel):
    """Ollama / OpenAI-compatible chat reranker."""

    model: str | None = None
    base_url: str | None = None
    top_k: int = 10
    batch_size: int = 5
    prompt_template: str | None = None


class RerankerCrossEncoderConfig(BaseModel):
    """Local cross-encoder (sentence-transformers)."""

    model_name: str = "cross-encoder/ms-marco-MiniLM-L6-v2"
    device: str = "cpu"
    top_k: int = 10


class RerankerJinaConfig(BaseModel):
    """Jina Rerank API."""

    api_key: str = ""
    model: str = "jina-reranker-v2-base-multilingual"
    top_k: int = 10


class RerankerConfig(BaseModel):
    """Top-level reranker section (see team_memory.reranker.factory)."""

    provider: Literal["none", "ollama_llm", "cross_encoder", "jina"] = "none"
    max_document_chars: int = Field(
        default=8000,
        description="Max chars per document text sent to the reranker (title+problem+solution).",
    )
    ollama_llm: RerankerOllamaLLMConfig = Field(default_factory=RerankerOllamaLLMConfig)
    cross_encoder: RerankerCrossEncoderConfig = Field(default_factory=RerankerCrossEncoderConfig)
    jina: RerankerJinaConfig = Field(default_factory=RerankerJinaConfig)

    def cache_signature(self) -> str:
        """Stable short string for search cache keys (provider + meaningful params).

        Excludes secrets (e.g. Jina api_key); model/top_k are included so cache
        invalidates when rerank configuration changes.
        """
        if self.provider == "none":
            return "none"
        if self.provider == "ollama_llm":
            c = self.ollama_llm
            raw = f"ollama_llm|{c.model or ''}|{c.base_url or ''}|{c.top_k}|{c.batch_size}"
            return "ollama_llm:" + hashlib.sha256(raw.encode()).hexdigest()[:16]
        if self.provider == "cross_encoder":
            c = self.cross_encoder
            raw = f"cross_encoder|{c.model_name}|{c.device}|{c.top_k}"
            return "ce:" + hashlib.sha256(raw.encode()).hexdigest()[:16]
        if self.provider == "jina":
            c = self.jina
            raw = f"jina|{c.model}|{c.top_k}"
            return "jina:" + hashlib.sha256(raw.encode()).hexdigest()[:16]
        return "unknown"

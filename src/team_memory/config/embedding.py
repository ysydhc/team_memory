"""Embedding provider configuration."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class OpenAIEmbeddingConfig(BaseModel):
    """OpenAI embedding API configuration."""

    api_key: str = ""
    model: str = "text-embedding-3-small"
    dimension: int = 1536


class OllamaEmbeddingConfig(BaseModel):
    """Ollama embedding configuration."""

    # Explicit :latest avoids ambiguity with Ollama tags (name is always tagged in /api/tags).
    model: str = "nomic-embed-text:latest"
    base_url: str = "http://localhost:11434"
    dimension: int = 768


class LocalEmbeddingConfig(BaseModel):
    """Local embedding model configuration."""

    model_name: str = "BAAI/bge-m3"
    device: str = "cpu"
    dimension: int = 1024


class GenericEmbeddingConfig(BaseModel):
    """Generic OpenAI-compatible embedding endpoint configuration."""

    base_url: str = "http://localhost:8080/v1"
    api_key: str = ""
    model: str = "text-embedding-3-small"
    dimension: int = 1536


class EmbeddingConfig(BaseModel):
    """Embedding service configuration."""

    provider: Literal["ollama", "openai", "local", "generic"] = "ollama"
    ollama: OllamaEmbeddingConfig = Field(default_factory=OllamaEmbeddingConfig)
    openai: OpenAIEmbeddingConfig = Field(default_factory=OpenAIEmbeddingConfig)
    local: LocalEmbeddingConfig = Field(default_factory=LocalEmbeddingConfig)
    generic: GenericEmbeddingConfig = Field(default_factory=GenericEmbeddingConfig)

    @property
    def dimension(self) -> int:
        """Return the dimension of the active embedding provider."""
        if self.provider == "ollama":
            return self.ollama.dimension
        if self.provider == "openai":
            return self.openai.dimension
        if self.provider == "generic":
            return self.generic.dimension
        return self.local.dimension

"""Generic OpenAI-compatible embedding provider.

Supports any endpoint that implements the OpenAI /v1/embeddings API:
vLLM, LiteLLM, Cursor proxy, Cloudflare Workers AI, etc.
"""

from __future__ import annotations

import os

from team_memory.embedding.openai_provider import OpenAIEmbedding


class GenericEmbedding(OpenAIEmbedding):
    """Thin wrapper that accepts a custom ``base_url`` while reusing
    the OpenAI provider's request logic."""

    def __init__(
        self,
        base_url: str = "http://localhost:8080/v1",
        api_key: str | None = None,
        model: str = "text-embedding-3-small",
        dim: int = 1536,
    ):
        resolved_key = api_key or os.environ.get("EMBEDDING_API_KEY", "")
        super().__init__(
            api_key=resolved_key,
            model=model,
            dim=dim,
            base_url=base_url,
        )

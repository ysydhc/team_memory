"""Ollama embedding provider implementation.

Uses Ollama's local API for text embedding. Zero cost, zero network dependency
(once the model is downloaded). Default model: nomic-embed-text (768 dimensions).

Requires Ollama running locally: https://ollama.com
"""

from __future__ import annotations

import httpx

from team_memory.embedding.base import EmbeddingProvider


class OllamaEmbedding(EmbeddingProvider):
    """Embedding provider using Ollama's local embedding API.

    Default model: nomic-embed-text (768 dimensions).
    """

    def __init__(
        self,
        model: str = "nomic-embed-text",
        dim: int = 768,
        base_url: str = "http://localhost:11434",
    ):
        self._model = model
        self._dimension = dim
        self._base_url = base_url.rstrip("/")

    @property
    def dimension(self) -> int:
        return self._dimension

    async def encode(self, texts: list[str]) -> list[list[float]]:
        """Encode texts via Ollama Embeddings API.

        Ollama's /api/embed endpoint supports batch input.

        Args:
            texts: List of texts to encode.

        Returns:
            List of embedding vectors.

        Raises:
            httpx.HTTPStatusError: If the API request fails.
            ConnectionError: If Ollama is not running.
        """
        try:
            async with httpx.AsyncClient(timeout=60.0) as client:
                response = await client.post(
                    f"{self._base_url}/api/embed",
                    json={
                        "model": self._model,
                        "input": texts,
                    },
                )
                response.raise_for_status()
                data = response.json()
        except httpx.ConnectError:
            raise ConnectionError(
                f"Cannot connect to Ollama at {self._base_url}. "
                "Make sure Ollama is running: https://ollama.com"
            )

        return data["embeddings"]

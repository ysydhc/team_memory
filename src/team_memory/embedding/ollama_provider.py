"""Ollama embedding provider implementation.

Uses Ollama's local API for text embedding. Zero cost, zero network dependency
(once the model is downloaded). Default model: qwen3-embedding:0.6b (1024 dimensions).

Requires Ollama running locally: https://ollama.com
"""

from __future__ import annotations

import httpx

from team_memory.embedding.base import EmbeddingProvider


class OllamaEmbedding(EmbeddingProvider):
    """Embedding provider using Ollama's local embedding API.

    Default model: qwen3-embedding:0.6b (1024 dimensions).
    """

    def __init__(
        self,
        model: str = "qwen3-embedding:0.6b",
        dim: int = 1024,
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
        Retries up to 3 times on 429/5xx errors.

        Args:
            texts: List of texts to encode.

        Returns:
            List of embedding vectors.

        Raises:
            httpx.HTTPStatusError: If the API request fails after retries.
            ConnectionError: If Ollama is not running.
        """
        last_error: Exception | None = None
        for attempt in range(3):
            try:
                async with httpx.AsyncClient(timeout=60.0) as client:
                    response = await client.post(
                        f"{self._base_url}/api/embed",
                        json={
                            "model": self._model,
                            "input": texts,
                            "keep_alive": "30m",  # Keep model loaded for 30 min / 模型常驻 30 分钟
                        },
                    )
                    response.raise_for_status()
                    data = response.json()
                return data["embeddings"]
            except httpx.HTTPStatusError as e:
                last_error = e
                # Retry on 404 (model not loaded yet), 429, and 5xx
                # 404 重试：模型可能刚卸载需要重新加载
                if e.response.status_code in (404, 429, 500, 502, 503, 504) and attempt < 2:
                    import asyncio
                    await asyncio.sleep(2 ** attempt)
                    continue
                raise
            except httpx.ConnectError:
                raise ConnectionError(
                    f"Cannot connect to Ollama at {self._base_url}. "
                    "Make sure Ollama is running: https://ollama.com"
                )
        raise last_error  # type: ignore[misc]

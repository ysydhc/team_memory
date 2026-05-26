"""OpenAI embedding provider implementation."""

from __future__ import annotations

import logging

import httpx

from team_memory.embedding.base import EmbeddingProvider

logger = logging.getLogger(__name__)


class EmbeddingTimeoutError(Exception):
    """Embedding API could not be reached after all retries."""


class OpenAIEmbedding(EmbeddingProvider):
    """Embedding provider using OpenAI's text-embedding API.

    Uses text-embedding-3-small by default (1536 dimensions).
    """

    def __init__(
        self,
        api_key: str,
        model: str = "text-embedding-3-small",
        dim: int = 1536,
        base_url: str = "https://api.openai.com/v1",
        timeout: float = 10.0,
        max_retries: int = 1,
    ):
        self._api_key = api_key
        self._model = model
        self._dimension = dim
        self._base_url = base_url.rstrip("/")
        self._timeout = timeout
        self._max_retries = max_retries

    @property
    def dimension(self) -> int:
        return self._dimension

    async def encode(self, texts: list[str]) -> list[list[float]]:
        """Encode texts via OpenAI Embeddings API.

        Args:
            texts: List of texts to encode.

        Returns:
            List of embedding vectors.

        Raises:
            httpx.HTTPStatusError: If the API request fails.
            ValueError: If the API key is not set.
        """
        if not self._api_key:
            raise ValueError(
                "OpenAI API key is not set. Set OPENAI_API_KEY environment variable or configure it in config.yaml."
            )

        import asyncio as _asyncio

        last_exc: Exception | None = None
        for attempt in range(self._max_retries + 1):
            try:
                async with httpx.AsyncClient(timeout=self._timeout) as client:
                    response = await client.post(
                        f"{self._base_url}/embeddings",
                        headers={
                            "Authorization": f"Bearer {self._api_key}",
                            "Content-Type": "application/json",
                        },
                        json={
                            "input": texts,
                            "model": self._model,
                            "encoding_format": "float",
                        },
                    )
                    response.raise_for_status()
                    data = response.json()

                # Sort by index to ensure order matches input
                embeddings_data = sorted(data["data"], key=lambda x: x["index"])
                return [item["embedding"] for item in embeddings_data]

            except (httpx.TimeoutException, httpx.ConnectError) as e:
                last_exc = e
                if attempt < self._max_retries:
                    await _asyncio.sleep(2.0 * (attempt + 1))
                    continue
            except httpx.HTTPStatusError as e:
                if e.response.status_code >= 500 and attempt < self._max_retries:
                    last_exc = e
                    await _asyncio.sleep(2.0 * (attempt + 1))
                    continue
                raise

        raise EmbeddingTimeoutError(
            f"Embedding API unreachable after {self._max_retries + 1} attempts: {last_exc}"
        ) from last_exc

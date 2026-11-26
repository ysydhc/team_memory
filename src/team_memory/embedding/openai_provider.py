"""OpenAI embedding provider implementation."""

from __future__ import annotations

import logging
import time

import httpx

from team_memory.embedding.base import EmbeddingProvider
from team_memory.services.analysis.usage_ledger import estimate_tokens, record_usage_async

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
        timeout: float = 5.0,
        max_retries: int = 1,
    ):
        self._api_key = api_key
        self._model = model
        self._dimension = dim
        self._base_url = base_url.rstrip("/")
        self._timeout = timeout
        self._max_retries = max_retries
        self._client: httpx.AsyncClient | None = None
        self._client_loop: object | None = None

    def _get_client(self) -> httpx.AsyncClient:
        """Return a pooled AsyncClient bound to the current event loop."""
        import asyncio as _asyncio

        try:
            loop = _asyncio.get_running_loop()
        except RuntimeError:
            loop = None

        if self._client is None or self._client.is_closed or self._client_loop is not loop:
            self._client = httpx.AsyncClient(
                timeout=self._timeout,
                limits=httpx.Limits(max_connections=20, max_keepalive_connections=10),
            )
            self._client_loop = loop
        return self._client

    async def aclose(self) -> None:
        """Close the pooled HTTP client if open."""
        if self._client is not None and not self._client.is_closed:
            await self._client.aclose()
        self._client = None
        self._client_loop = None

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

        start = time.monotonic()
        est_prompt = estimate_tokens(texts)
        last_exc: Exception | None = None
        for attempt in range(self._max_retries + 1):
            try:
                client = self._get_client()
                response = await client.post(
                    f"{self._base_url}/embeddings",
                    headers={
                        "Authorization": f"Bearer {self._api_key}",
                        "x-api-key": self._api_key,
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

                usage = data.get("usage") if isinstance(data, dict) else None
                prompt_tokens = (
                    int(usage.get("prompt_tokens") or usage.get("total_tokens") or est_prompt)
                    if isinstance(usage, dict)
                    else est_prompt
                )
                record_usage_async(
                    caller="embedding",
                    provider="openai",
                    model=self._model,
                    prompt_tokens=prompt_tokens,
                    completion_tokens=0,
                    latency_ms=int((time.monotonic() - start) * 1000),
                    success=True,
                )

                # Sort by index to ensure order matches input
                embeddings_data = sorted(data["data"], key=lambda x: x["index"])
                return [item["embedding"] for item in embeddings_data]

            except (httpx.TimeoutException, httpx.ConnectError) as e:
                last_exc = e
                if attempt < self._max_retries:
                    await _asyncio.sleep(0.5 * (attempt + 1))
                    continue
            except httpx.HTTPStatusError as e:
                if e.response.status_code >= 500 and attempt < self._max_retries:
                    last_exc = e
                    await _asyncio.sleep(0.5 * (attempt + 1))
                    continue
                record_usage_async(
                    caller="embedding",
                    provider="openai",
                    model=self._model,
                    prompt_tokens=est_prompt,
                    completion_tokens=0,
                    latency_ms=int((time.monotonic() - start) * 1000),
                    success=False,
                )
                raise

        record_usage_async(
            caller="embedding",
            provider="openai",
            model=self._model,
            prompt_tokens=est_prompt,
            completion_tokens=0,
            latency_ms=int((time.monotonic() - start) * 1000),
            success=False,
        )
        raise EmbeddingTimeoutError(
            f"Embedding API unreachable after {self._max_retries + 1} attempts: {last_exc}"
        ) from last_exc

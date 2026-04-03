"""Abstract base class for embedding providers."""

from __future__ import annotations

import asyncio
from abc import ABC, abstractmethod


class EmbeddingProvider(ABC):
    """Abstract embedding provider interface.

    All embedding providers must implement this interface, ensuring
    the system can switch between local models and API services.
    """

    @abstractmethod
    async def encode(self, texts: list[str]) -> list[list[float]]:
        """Encode a list of texts into embedding vectors.

        Args:
            texts: List of text strings to encode.

        Returns:
            List of embedding vectors, one per input text.
            Each vector has length equal to self.dimension.
        """
        ...

    @property
    @abstractmethod
    def dimension(self) -> int:
        """Return the dimensionality of the embedding vectors."""
        ...

    async def encode_single(self, text: str) -> list[float]:
        """Convenience method to encode a single text.

        Args:
            text: A single text string.

        Returns:
            Embedding vector for the text.
        """
        results = await self.encode([text])
        return results[0]


class ConcurrencyLimitedEmbedding(EmbeddingProvider):
    """Wraps an EmbeddingProvider with a concurrency semaphore.

    Prevents overwhelming the embedding backend (Ollama, OpenAI, etc.)
    when many coroutines request embeddings concurrently.
    """

    def __init__(self, provider: EmbeddingProvider, max_concurrent: int = 10) -> None:
        self._provider = provider
        self._semaphore = asyncio.Semaphore(max_concurrent)

    async def encode(self, texts: list[str]) -> list[list[float]]:
        async with self._semaphore:
            return await self._provider.encode(texts)

    async def encode_single(self, text: str) -> list[float]:
        async with self._semaphore:
            return await self._provider.encode_single(text)

    @property
    def dimension(self) -> int:
        return self._provider.dimension

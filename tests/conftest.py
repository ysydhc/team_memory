"""Shared pytest fixtures for team_doc tests."""

from __future__ import annotations

import uuid
from unittest.mock import AsyncMock

import pytest

from team_doc.auth.provider import ApiKeyAuth, NoAuth, User
from team_doc.embedding.base import EmbeddingProvider


# ============================================================
# Mock Embedding Provider
# ============================================================


class MockEmbeddingProvider(EmbeddingProvider):
    """A deterministic mock embedding provider for testing.

    Generates consistent vectors based on text content so that
    similar texts produce similar vectors.
    """

    def __init__(self, dimension: int = 8):
        self._dimension = dimension

    @property
    def dimension(self) -> int:
        return self._dimension

    async def encode(self, texts: list[str]) -> list[list[float]]:
        """Generate deterministic mock embeddings.

        Creates a simple hash-based vector for each text.
        Similar texts will have somewhat similar vectors.
        """
        return [self._text_to_vector(t) for t in texts]

    def _text_to_vector(self, text: str) -> list[float]:
        """Convert text to a deterministic vector.

        Uses a hash-spreading approach so that:
        - Same text always produces the same vector
        - Similar texts produce somewhat similar vectors
        - Works well at any dimensionality (8, 1536, etc.)
        """
        import hashlib

        values = [0.0] * self._dimension

        # Spread character information across all dimensions
        for i, ch in enumerate(text):
            # Use hash to distribute each character across multiple dimensions
            seed = hashlib.md5(f"{ch}:{i}".encode()).digest()
            for j in range(min(4, self._dimension)):
                idx = (seed[j] + seed[j + 4]) % self._dimension
                values[idx] += ord(ch) / 1000.0

        # Also add word-level features for better semantic similarity
        words = text.lower().split()
        for word in words:
            word_hash = hashlib.md5(word.encode()).digest()
            for j in range(min(8, self._dimension)):
                idx = word_hash[j] % self._dimension
                values[idx] += 0.1

        # Normalize to unit vector
        magnitude = sum(v * v for v in values) ** 0.5
        if magnitude > 0:
            values = [v / magnitude for v in values]
        return values


@pytest.fixture
def mock_embedding() -> MockEmbeddingProvider:
    """Provide a mock embedding provider."""
    return MockEmbeddingProvider(dimension=8)


@pytest.fixture
def no_auth() -> NoAuth:
    """Provide a NoAuth provider."""
    return NoAuth()


@pytest.fixture
def api_key_auth() -> ApiKeyAuth:
    """Provide an ApiKeyAuth provider with a test key."""
    auth = ApiKeyAuth()
    auth.register_key("test_key_123", "test_user", "admin")
    return auth

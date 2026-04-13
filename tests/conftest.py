"""Shared pytest fixtures for team_memory tests."""

from __future__ import annotations

from contextlib import nullcontext
from unittest.mock import AsyncMock, patch

import pytest

from team_memory.auth.provider import ApiKeyAuth, NoAuth
from team_memory.embedding.base import EmbeddingProvider

# ============================================================
# Shared MCP Server Mock Helpers
# ============================================================

LITE_PATCH_BASE = "team_memory.server"
_OPS_PATCH_BASE = "team_memory.services.memory_operations"
_P = LITE_PATCH_BASE  # short alias
_O = _OPS_PATCH_BASE  # MCP orchestration (memory_operations)


def _patch_user(username: str = "admin"):
    """Patch _get_current_user to return a fixed username."""
    return patch(f"{_P}._get_current_user", new_callable=AsyncMock, return_value=username)


def _patch_expansion():
    """No-op: UserExpansion removed in MVP simplification."""
    return nullcontext()


def _patch_personal():
    """Patch _try_extract_and_save_personal_memory to no-op."""
    return patch(
        f"{_O}._try_extract_and_save_personal_memory",
        new_callable=AsyncMock,
    )


def _setup_session_cm(mock_get_session):
    """Configure async context manager for get_session mock.

    Args:
        mock_get_session: The patched get_session mock object.

    Returns:
        A mock AsyncSession ready for use in tests.
    """
    mock_session = AsyncMock()
    mock_get_session.return_value.__aenter__ = AsyncMock(
        return_value=mock_session,
    )
    mock_get_session.return_value.__aexit__ = AsyncMock(return_value=False)
    return mock_session


# ============================================================
# Mock Embedding Provider
# ============================================================


class MockEmbeddingProvider(EmbeddingProvider):
    """A deterministic mock embedding provider for testing.

    Generates consistent vectors based on text content so that
    similar texts produce similar vectors.
    """

    def __init__(self, dimension: int = 768):
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
    """Provide a mock embedding provider (768 dims, matching Ollama nomic-embed-text)."""
    return MockEmbeddingProvider(dimension=768)


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

"""Shared pytest fixtures for team_memory tests."""

from __future__ import annotations

from contextlib import nullcontext
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from team_memory.auth.provider import ApiKeyAuth, NoAuth
from team_memory.embedding.base import EmbeddingProvider

# ============================================================
# Shared MCP Server Mock Helpers
# ============================================================

LITE_PATCH_BASE = "team_memory.server"
_OPS_PATCH_BASE = "team_memory.services.memory.memory_operations"
_P = LITE_PATCH_BASE  # short alias
_O = _OPS_PATCH_BASE  # MCP orchestration (memory_operations)


def make_mock_app_context(
    *,
    settings: MagicMock | None = None,
    service: MagicMock | None = None,
    search_orchestrator=None,
    archive_service=None,
    db_url: str = "sqlite://",
    embedding=None,
) -> MagicMock:
    """Build a MagicMock mimicking bootstrap AppContext fields used by MCP tools."""

    ctx = MagicMock()
    ctx.settings = MagicMock() if settings is None else settings
    ctx.service = MagicMock() if service is None else service
    ctx.search_orchestrator = MagicMock() if search_orchestrator is None else search_orchestrator
    ctx.archive_service = MagicMock() if archive_service is None else archive_service
    ctx.db_url = db_url
    ctx.embedding = MagicMock() if embedding is None else embedding
    return ctx


def _patch_app_context(mock_ctx: MagicMock):
    """Patch ``team_memory.server._app_context`` (bootstrap get_context alias) for tests."""

    return patch(f"{LITE_PATCH_BASE}._app_context", return_value=mock_ctx)


def make_mcp_tool_settings_stub() -> MagicMock:
    """MagicMock ``settings`` with numeric ``settings.mcp.*`` attrs for MCP ``_guard_output`` tests."""

    s = MagicMock()
    s.mcp.truncate_solution_at = 2000
    s.mcp.max_output_tokens = 4000
    s.mcp.max_content_chars = 200_000
    s.mcp.max_tags = 20
    s.mcp.max_tag_length = 50
    s.mcp.profile_max_strings_per_side = 20
    s.mcp.max_archive_solution_doc_chars = 64_000
    return s


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


# --- Pipeline Acceptance Test Fixtures / Pipeline 验收测试固件 ---

import pytest_asyncio  # noqa: E402, I001

from team_memory.daemon.buffers.conversation_buffer import ConversationBuffer  # noqa: E402


# Note: mock_embedding fixture already defined above (line 121)
# 注意：mock_embedding 固件已在上方定义（第 121 行）


@pytest_asyncio.fixture  # noqa: E402
async def fresh_conversation_buffer():
    """Fresh in-memory ConversationBuffer per test / 每次测试新建内存 ConversationBuffer."""
    buf = ConversationBuffer(":memory:")
    await buf.__aenter__()
    yield buf
    await buf.__aexit__(None, None, None)


@pytest.fixture
def mock_llm_result():
    """Deterministic LLM refinement result / 确定性 LLM 精炼结果."""
    return {
        "title": "A sufficiently long title for quality",
        "problem": "This is a problem description that is long enough to pass the minimum length check",
        "solution": "This is a solution description that is long enough to pass the minimum length check",
        "tags": ["test"],
        "experience_type": "bugfix",
    }

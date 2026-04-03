"""Tests for embedding providers."""

import asyncio

import httpx
import pytest

from team_memory.embedding.base import ConcurrencyLimitedEmbedding, EmbeddingProvider
from team_memory.embedding.openai_provider import OpenAIEmbedding


class TestEmbeddingProviderInterface:
    """Test that EmbeddingProvider is a proper abstract base class."""

    def test_cannot_instantiate_abstract(self):
        with pytest.raises(TypeError):
            EmbeddingProvider()  # type: ignore

    def test_concrete_implementation(self):
        class DummyEmbedding(EmbeddingProvider):
            @property
            def dimension(self) -> int:
                return 3

            async def encode(self, texts: list[str]) -> list[list[float]]:
                return [[1.0, 2.0, 3.0] for _ in texts]

        provider = DummyEmbedding()
        assert provider.dimension == 3

    @pytest.mark.asyncio
    async def test_encode_single(self):
        class DummyEmbedding(EmbeddingProvider):
            @property
            def dimension(self) -> int:
                return 2

            async def encode(self, texts: list[str]) -> list[list[float]]:
                return [[float(len(t)), 0.0] for t in texts]

        provider = DummyEmbedding()
        result = await provider.encode_single("hello")
        assert result == [5.0, 0.0]


class TestOpenAIEmbedding:
    """Test OpenAI embedding provider."""

    def test_dimension(self):
        provider = OpenAIEmbedding(api_key="test", dim=1536)
        assert provider.dimension == 1536

    def test_custom_dimension(self):
        provider = OpenAIEmbedding(api_key="test", dim=512)
        assert provider.dimension == 512

    @pytest.mark.asyncio
    async def test_missing_api_key_raises(self):
        provider = OpenAIEmbedding(api_key="")
        with pytest.raises(ValueError, match="API key is not set"):
            await provider.encode(["hello"])

    @pytest.mark.asyncio
    async def test_encode_calls_api(self):
        """Test that encode calls the OpenAI API correctly."""
        mock_response = {
            "data": [
                {"index": 0, "embedding": [0.1, 0.2, 0.3]},
                {"index": 1, "embedding": [0.4, 0.5, 0.6]},
            ],
            "model": "text-embedding-3-small",
            "usage": {"prompt_tokens": 10, "total_tokens": 10},
        }

        provider = OpenAIEmbedding(
            api_key="sk-test-123",
            model="text-embedding-3-small",
            dim=3,
            base_url="https://api.openai.com/v1",
        )

        # We'll use a transport mock instead of httpx_mock fixture
        transport = httpx.MockTransport(
            lambda request: httpx.Response(
                200,
                json=mock_response,
            )
        )

        # Monkey-patch httpx.AsyncClient to use our mock transport
        original_init = httpx.AsyncClient.__init__

        def patched_init(self_client, **kwargs):
            kwargs["transport"] = transport
            original_init(self_client, **kwargs)

        httpx.AsyncClient.__init__ = patched_init
        try:
            results = await provider.encode(["hello", "world"])
            assert len(results) == 2
            assert results[0] == [0.1, 0.2, 0.3]
            assert results[1] == [0.4, 0.5, 0.6]
        finally:
            httpx.AsyncClient.__init__ = original_init

    @pytest.mark.asyncio
    async def test_encode_preserves_order(self):
        """Test that results are sorted by index even if API returns out of order."""
        mock_response = {
            "data": [
                {"index": 1, "embedding": [0.4, 0.5]},
                {"index": 0, "embedding": [0.1, 0.2]},
            ],
            "model": "text-embedding-3-small",
            "usage": {"prompt_tokens": 10, "total_tokens": 10},
        }

        provider = OpenAIEmbedding(
            api_key="sk-test-123",
            dim=2,
        )

        transport = httpx.MockTransport(lambda request: httpx.Response(200, json=mock_response))

        original_init = httpx.AsyncClient.__init__

        def patched_init(self_client, **kwargs):
            kwargs["transport"] = transport
            original_init(self_client, **kwargs)

        httpx.AsyncClient.__init__ = patched_init
        try:
            results = await provider.encode(["first", "second"])
            # index 0 should come first
            assert results[0] == [0.1, 0.2]
            assert results[1] == [0.4, 0.5]
        finally:
            httpx.AsyncClient.__init__ = original_init


# ------------------------------------------------------------------
# ConcurrencyLimitedEmbedding
# ------------------------------------------------------------------


class _DummyEmbedding(EmbeddingProvider):
    """Minimal concrete provider for testing the concurrency wrapper."""

    def __init__(self, dim: int = 3) -> None:
        self._dim = dim
        self.call_count = 0

    @property
    def dimension(self) -> int:
        return self._dim

    async def encode(self, texts: list[str]) -> list[list[float]]:
        self.call_count += 1
        return [[1.0] * self._dim for _ in texts]


class TestConcurrencyLimitedEmbedding:
    """Tests for the semaphore-based concurrency wrapper."""

    def test_dimension_delegates(self) -> None:
        inner = _DummyEmbedding(dim=768)
        wrapper = ConcurrencyLimitedEmbedding(inner, max_concurrent=5)
        assert wrapper.dimension == 768

    @pytest.mark.asyncio
    async def test_encode_delegates(self) -> None:
        inner = _DummyEmbedding(dim=2)
        wrapper = ConcurrencyLimitedEmbedding(inner, max_concurrent=5)
        result = await wrapper.encode(["hello", "world"])
        assert len(result) == 2
        assert result[0] == [1.0, 1.0]
        assert inner.call_count == 1

    @pytest.mark.asyncio
    async def test_encode_single_delegates(self) -> None:
        inner = _DummyEmbedding(dim=4)
        wrapper = ConcurrencyLimitedEmbedding(inner, max_concurrent=5)
        result = await wrapper.encode_single("hi")
        assert len(result) == 4
        assert result == [1.0, 1.0, 1.0, 1.0]

    @pytest.mark.asyncio
    async def test_semaphore_limits_concurrency(self) -> None:
        """Ensure no more than max_concurrent calls run at the same time."""
        max_concurrent = 2
        peak_concurrent = 0
        current_concurrent = 0

        class SlowEmbedding(EmbeddingProvider):
            @property
            def dimension(self) -> int:
                return 1

            async def encode(self, texts: list[str]) -> list[list[float]]:
                nonlocal peak_concurrent, current_concurrent
                current_concurrent += 1
                peak_concurrent = max(peak_concurrent, current_concurrent)
                await asyncio.sleep(0.05)
                current_concurrent -= 1
                return [[0.0] for _ in texts]

        inner = SlowEmbedding()
        wrapper = ConcurrencyLimitedEmbedding(inner, max_concurrent=max_concurrent)

        # Launch more tasks than the semaphore allows
        tasks = [wrapper.encode(["text"]) for _ in range(6)]
        await asyncio.gather(*tasks)

        assert peak_concurrent <= max_concurrent

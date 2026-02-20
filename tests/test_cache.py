"""Tests for search result and embedding caching (D1 refactored)."""

from unittest.mock import AsyncMock

import pytest

from team_memory.services.cache import MemoryCacheBackend, SearchCache, create_cache_backend

# ======================== MemoryCacheBackend ========================


class TestMemoryCacheBackend:
    @pytest.mark.asyncio
    async def test_basic_put_get(self):
        cache = MemoryCacheBackend(max_size=10, ttl_seconds=300)
        await cache.put("key1", "value1")
        assert await cache.get("key1") == "value1"

    @pytest.mark.asyncio
    async def test_missing_key_returns_none(self):
        cache = MemoryCacheBackend(max_size=10, ttl_seconds=300)
        assert await cache.get("nonexistent") is None

    @pytest.mark.asyncio
    async def test_max_size_eviction(self):
        cache = MemoryCacheBackend(max_size=3, ttl_seconds=300)
        await cache.put("a", 1)
        await cache.put("b", 2)
        await cache.put("c", 3)
        await cache.put("d", 4)  # Should evict "a"

        assert await cache.get("a") is None
        assert await cache.get("b") == 2
        assert await cache.get("d") == 4
        assert await cache.size() == 3

    @pytest.mark.asyncio
    async def test_lru_order(self):
        cache = MemoryCacheBackend(max_size=3, ttl_seconds=300)
        await cache.put("a", 1)
        await cache.put("b", 2)
        await cache.put("c", 3)

        # Access "a" to make it most recently used
        await cache.get("a")

        await cache.put("d", 4)  # Should evict "b" (least recently used)
        assert await cache.get("a") == 1
        assert await cache.get("b") is None
        assert await cache.get("d") == 4

    @pytest.mark.asyncio
    async def test_ttl_expiration(self):
        import time

        # Very short TTL
        cache = MemoryCacheBackend(max_size=10, ttl_seconds=0)
        await cache.put("key1", "value1")
        # Should be expired immediately (TTL=0)
        time.sleep(0.01)
        assert await cache.get("key1") is None

    @pytest.mark.asyncio
    async def test_clear(self):
        cache = MemoryCacheBackend(max_size=10, ttl_seconds=300)
        await cache.put("a", 1)
        await cache.put("b", 2)
        await cache.clear()
        assert await cache.size() == 0
        assert await cache.get("a") is None


# ======================== create_cache_backend ========================


class TestCacheFactory:
    def test_create_memory_backend(self):
        backend = create_cache_backend(backend="memory", max_size=50, ttl_seconds=60)
        assert isinstance(backend, MemoryCacheBackend)

    def test_create_redis_backend(self):
        # Should not raise even if Redis is unavailable (lazy connection)
        from team_memory.services.cache import RedisCacheBackend

        backend = create_cache_backend(
            backend="redis",
            redis_url="redis://localhost:9999/0",
        )
        assert isinstance(backend, RedisCacheBackend)

    def test_default_is_memory(self):
        backend = create_cache_backend()
        assert isinstance(backend, MemoryCacheBackend)


# ======================== SearchCache ========================


class TestSearchCache:
    @pytest.mark.asyncio
    async def test_disabled_cache_returns_none(self):
        cache = SearchCache(enabled=False)
        await cache.put("query", None, {"results": []})
        assert await cache.get("query") is None

    @pytest.mark.asyncio
    async def test_basic_result_caching(self):
        cache = SearchCache(enabled=True, ttl_seconds=300)
        data = {"results": [{"title": "test"}]}
        await cache.put("my query", None, data)
        assert await cache.get("my query") == data

    @pytest.mark.asyncio
    async def test_cache_with_tags(self):
        cache = SearchCache(enabled=True, ttl_seconds=300)
        data1 = {"results": [{"title": "no tags"}]}
        data2 = {"results": [{"title": "with tags"}]}

        await cache.put("query", None, data1)
        await cache.put("query", ["python"], data2)

        assert await cache.get("query", None) == data1
        assert await cache.get("query", ["python"]) == data2

    @pytest.mark.asyncio
    async def test_cache_key_normalization(self):
        cache = SearchCache(enabled=True, ttl_seconds=300)
        data = {"test": True}
        await cache.put("  Hello World  ", None, data)
        assert await cache.get("hello world") == data

    @pytest.mark.asyncio
    async def test_embedding_cache(self):
        cache = SearchCache(enabled=True, ttl_seconds=300, embedding_cache_size=10)

        mock_provider = AsyncMock()
        mock_provider.encode_single = AsyncMock(return_value=[0.1, 0.2, 0.3])

        # First call: should compute
        result1 = await cache.get_or_compute_embedding("test text", mock_provider)
        assert result1 == [0.1, 0.2, 0.3]
        assert mock_provider.encode_single.call_count == 1

        # Second call: should use cache
        result2 = await cache.get_or_compute_embedding("test text", mock_provider)
        assert result2 == [0.1, 0.2, 0.3]
        assert mock_provider.encode_single.call_count == 1  # Not called again

    @pytest.mark.asyncio
    async def test_embedding_cache_disabled(self):
        cache = SearchCache(enabled=False)

        mock_provider = AsyncMock()
        mock_provider.encode_single = AsyncMock(return_value=[0.1, 0.2])

        await cache.get_or_compute_embedding("test", mock_provider)
        await cache.get_or_compute_embedding("test", mock_provider)
        assert mock_provider.encode_single.call_count == 2  # Called twice

    @pytest.mark.asyncio
    async def test_clear(self):
        cache = SearchCache(enabled=True)
        await cache.put("q1", None, {"data": 1})
        await cache.clear()
        assert await cache.get("q1") is None

    @pytest.mark.asyncio
    async def test_stats(self):
        cache = SearchCache(enabled=True)
        await cache.put("q1", None, {"data": 1})
        stats = await cache.stats
        assert stats["enabled"] is True
        assert stats["result_cache_size"] == 1
        assert stats["backend"] == "memory"

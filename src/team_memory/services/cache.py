"""Search result and embedding caching with pluggable backends.

Supports two backends:
  - "memory": In-process LRU cache (default). Zero dependencies. Lost on restart.
  - "redis": Redis-backed cache. Survives restarts, shared across instances.

Both backends expose the same CacheBackend interface, so SearchCache
doesn't need to know which one it's using.

Cache is invalidated when experiences are created/updated/deleted
(via the EventBus, see D5).
"""

from __future__ import annotations

import hashlib
import json
import logging
import time
from abc import ABC, abstractmethod
from collections import OrderedDict
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from team_memory.embedding.base import EmbeddingProvider

logger = logging.getLogger("team_memory.cache")


# ============================================================
# Abstract Cache Backend
# ============================================================


class CacheBackend(ABC):
    """Abstract cache backend interface.

    All implementations must provide async get/put/clear/size operations.
    """

    @abstractmethod
    async def get(self, key: str) -> Any | None:
        """Get a value by key. Returns None if missing or expired."""
        ...

    @abstractmethod
    async def put(self, key: str, value: Any, ttl: int | None = None) -> None:
        """Store a value with optional TTL override."""
        ...

    @abstractmethod
    async def clear(self) -> None:
        """Clear all cached entries."""
        ...

    @abstractmethod
    async def size(self) -> int:
        """Return current number of cached entries."""
        ...


# ============================================================
# Memory Backend (default)
# ============================================================


class MemoryCacheBackend(CacheBackend):
    """In-process LRU cache with TTL expiration.

    Items are evicted when:
    - The cache exceeds max_size (least recently used item is evicted).
    - An item's TTL has expired (checked on access).
    """

    def __init__(self, max_size: int = 100, ttl_seconds: int = 300):
        self._max_size = max_size
        self._ttl = ttl_seconds
        self._cache: OrderedDict[str, tuple[float, Any]] = OrderedDict()

    async def get(self, key: str) -> Any | None:
        if key not in self._cache:
            return None

        timestamp, value = self._cache[key]
        if time.monotonic() - timestamp > self._ttl:
            del self._cache[key]
            return None

        self._cache.move_to_end(key)
        return value

    async def put(self, key: str, value: Any, ttl: int | None = None) -> None:
        if key in self._cache:
            self._cache.move_to_end(key)
        self._cache[key] = (time.monotonic(), value)

        while len(self._cache) > self._max_size:
            self._cache.popitem(last=False)

    async def clear(self) -> None:
        self._cache.clear()

    async def size(self) -> int:
        return len(self._cache)


# ============================================================
# Redis Backend (optional)
# ============================================================


class RedisCacheBackend(CacheBackend):
    """Redis-backed cache using redis.asyncio.

    Stores values as JSON-serialized strings with Redis native TTL.
    Falls back gracefully if Redis is unavailable.
    """

    def __init__(self, redis_url: str, ttl_seconds: int = 300, prefix: str = "td:"):
        self._redis_url = redis_url
        self._ttl = ttl_seconds
        self._prefix = prefix
        self._redis = None

    async def _get_client(self):
        """Lazy-initialize Redis connection."""
        if self._redis is None:
            try:
                import redis.asyncio as aioredis

                self._redis = aioredis.from_url(
                    self._redis_url,
                    decode_responses=True,
                    socket_connect_timeout=5,
                )
                # Test connection
                await self._redis.ping()
                logger.info("Redis cache connected: %s", self._redis_url)
            except Exception:
                logger.warning(
                    "Redis unavailable at %s, cache operations will be skipped",
                    self._redis_url,
                    exc_info=True,
                )
                self._redis = None
        return self._redis

    def _key(self, key: str) -> str:
        return f"{self._prefix}{key}"

    async def get(self, key: str) -> Any | None:
        client = await self._get_client()
        if client is None:
            return None
        try:
            raw = await client.get(self._key(key))
            if raw is None:
                return None
            return json.loads(raw)
        except Exception:
            logger.debug("Redis GET error for key %s", key, exc_info=True)
            return None

    async def put(self, key: str, value: Any, ttl: int | None = None) -> None:
        client = await self._get_client()
        if client is None:
            return
        try:
            serialized = json.dumps(value)
            await client.setex(self._key(key), ttl or self._ttl, serialized)
        except Exception:
            logger.debug("Redis SET error for key %s", key, exc_info=True)

    async def clear(self) -> None:
        client = await self._get_client()
        if client is None:
            return
        try:
            # Delete all keys with our prefix
            cursor = 0
            while True:
                cursor, keys = await client.scan(
                    cursor=cursor, match=f"{self._prefix}*", count=100
                )
                if keys:
                    await client.delete(*keys)
                if cursor == 0:
                    break
            logger.info("Redis cache cleared (prefix=%s)", self._prefix)
        except Exception:
            logger.warning("Redis CLEAR error", exc_info=True)

    async def size(self) -> int:
        client = await self._get_client()
        if client is None:
            return 0
        try:
            cursor = 0
            count = 0
            while True:
                cursor, keys = await client.scan(
                    cursor=cursor, match=f"{self._prefix}*", count=100
                )
                count += len(keys)
                if cursor == 0:
                    break
            return count
        except Exception:
            return 0


# ============================================================
# Backend Factory
# ============================================================


def create_cache_backend(
    backend: str = "memory",
    redis_url: str = "redis://localhost:6379/0",
    max_size: int = 100,
    ttl_seconds: int = 300,
) -> CacheBackend:
    """Create a cache backend based on configuration.

    Args:
        backend: "memory" or "redis".
        redis_url: Redis connection URL (only used when backend="redis").
        max_size: Maximum entries for memory backend.
        ttl_seconds: TTL for cached entries.

    Returns:
        A CacheBackend instance.
    """
    if backend == "redis":
        logger.info("Using Redis cache backend: %s", redis_url)
        return RedisCacheBackend(
            redis_url=redis_url,
            ttl_seconds=ttl_seconds,
        )
    else:
        logger.info("Using in-memory cache backend (max_size=%d, ttl=%ds)", max_size, ttl_seconds)
        return MemoryCacheBackend(max_size=max_size, ttl_seconds=ttl_seconds)


# ============================================================
# SearchCache (high-level, uses CacheBackend)
# ============================================================


class SearchCache:
    """Combined cache for embeddings and search results.

    Uses a pluggable CacheBackend for storage. Provides high-level
    methods for caching search results and embeddings.
    """

    def __init__(
        self,
        max_size: int = 100,
        ttl_seconds: int = 300,
        embedding_cache_size: int = 200,
        enabled: bool = True,
        backend: str = "memory",
        redis_url: str = "redis://localhost:6379/0",
    ):
        self.enabled = enabled
        self._backend = backend
        self._result_cache = create_cache_backend(
            backend=backend,
            redis_url=redis_url,
            max_size=max_size,
            ttl_seconds=ttl_seconds,
        )
        self._embedding_cache = create_cache_backend(
            backend=backend,
            redis_url=redis_url,
            max_size=embedding_cache_size,
            ttl_seconds=ttl_seconds * 2,
        )

    @staticmethod
    def _make_key(
        query: str, tags: list[str] | None = None, project: str | None = None
    ) -> str:
        """Create a cache key from query, tags, and project."""
        parts = [query.strip().lower()]
        if tags:
            parts.extend(sorted(tags))
        if project:
            parts.append(f"project:{project.strip().lower()}")
        raw = "|".join(parts)
        return hashlib.md5(raw.encode()).hexdigest()

    async def get(
        self,
        query: str,
        tags: list[str] | None = None,
        project: str | None = None,
    ) -> Any | None:
        """Get cached search results."""
        if not self.enabled:
            return None
        key = self._make_key(query, tags, project)
        result = await self._result_cache.get(key)
        if result is not None:
            logger.debug("Cache hit for query: %s", query[:50])
        return result

    async def put(
        self,
        query: str,
        tags: list[str] | None,
        value: Any,
        project: str | None = None,
    ) -> None:
        """Cache search results."""
        if not self.enabled:
            return
        key = self._make_key(query, tags, project)
        await self._result_cache.put(key, value)

    async def get_or_compute_embedding(
        self,
        text: str,
        embedding_provider: "EmbeddingProvider",
    ) -> list[float]:
        """Get embedding from cache or compute it.

        Args:
            text: Text to encode.
            embedding_provider: Provider to use if cache misses.

        Returns:
            The embedding vector.
        """
        key = hashlib.md5(text.strip().lower().encode()).hexdigest()

        if self.enabled:
            cached = await self._embedding_cache.get(key)
            if cached is not None:
                logger.debug("Embedding cache hit for: %s", text[:50])
                return cached

        # Compute embedding
        embedding = await embedding_provider.encode_single(text)

        if self.enabled:
            await self._embedding_cache.put(key, embedding)

        return embedding

    async def clear(self) -> None:
        """Clear all caches. Call after data mutations."""
        await self._result_cache.clear()
        await self._embedding_cache.clear()
        logger.info("Search cache cleared")

    @property
    async def stats(self) -> dict:
        """Return cache statistics."""
        result_size = await self._result_cache.size()
        embedding_size = await self._embedding_cache.size()
        return {
            "enabled": self.enabled,
            "backend": self._backend,
            "result_cache_size": result_size,
            "embedding_cache_size": embedding_size,
        }

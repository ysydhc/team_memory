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

import asyncio
import hashlib
import json
import logging
import time
from abc import ABC, abstractmethod
from collections import OrderedDict, defaultdict
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from team_memory.embedding.base import EmbeddingProvider

from team_memory import io_logger

logger = logging.getLogger("team_memory.cache")

# Rate-limited Redis warning to avoid log storms
_last_redis_warning: float = 0.0
_REDIS_WARNING_INTERVAL = 300  # seconds


def _warn_redis_once(msg: str, *args: object) -> None:
    """Log a Redis warning at most once per _REDIS_WARNING_INTERVAL seconds."""
    global _last_redis_warning
    now = time.monotonic()
    if now - _last_redis_warning >= _REDIS_WARNING_INTERVAL:
        logger.warning(msg, *args)
        _last_redis_warning = now


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
            _warn_redis_once("Redis GET error for key %s", key)
            return None

    async def put(self, key: str, value: Any, ttl: int | None = None) -> None:
        client = await self._get_client()
        if client is None:
            return
        try:
            serialized = json.dumps(value)
            await client.setex(self._key(key), ttl or self._ttl, serialized)
        except Exception:
            _warn_redis_once("Redis SET error for key %s", key)

    async def clear(self) -> None:
        client = await self._get_client()
        if client is None:
            return
        try:
            # Delete all keys with our prefix
            cursor = 0
            while True:
                cursor, keys = await client.scan(cursor=cursor, match=f"{self._prefix}*", count=100)
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
                cursor, keys = await client.scan(cursor=cursor, match=f"{self._prefix}*", count=100)
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
        self._locks: dict[str, asyncio.Lock] = defaultdict(asyncio.Lock)
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
        query: str,
        tags: list[str] | None = None,
        project: str | None = None,
        current_user: str | None = None,
        include_archives: bool = False,
    ) -> str:
        """Create a cache key from query, tags, project, and optionally current_user.

        When per-user expansion is used, current_user must be included to avoid
        cross-user cache pollution. include_archives differentiates archive-inclusive results.
        """
        parts = [query.strip().lower()]
        if tags:
            parts.extend(sorted(tags))
        if project:
            parts.append(f"project:{project.strip().lower()}")
        if current_user and str(current_user).strip().lower() != "anonymous":
            parts.append(f"user:{current_user.strip().lower()}")
        if include_archives:
            parts.append("archives:1")
        raw = "|".join(parts)
        return hashlib.md5(raw.encode()).hexdigest()

    async def get(
        self,
        query: str,
        tags: list[str] | None = None,
        project: str | None = None,
        current_user: str | None = None,
        include_archives: bool = False,
    ) -> Any | None:
        """Get cached search results."""
        if not self.enabled:
            return None
        key = self._make_key(query, tags, project, current_user, include_archives)
        result = await self._result_cache.get(key)
        query_preview = (query or "")[:50]
        if result is not None:
            io_logger.log_internal("cache_check", {"hit": True, "query_preview": query_preview})
            logger.debug("Search cache hit for: %s", query_preview)
        else:
            io_logger.log_internal("cache_check", {"hit": False, "query_preview": query_preview})
            logger.debug("Search cache miss for: %s", query_preview)
        return result

    async def put(
        self,
        query: str,
        tags: list[str] | None,
        value: Any,
        project: str | None = None,
        current_user: str | None = None,
        include_archives: bool = False,
    ) -> None:
        """Cache search results."""
        if not self.enabled:
            return
        key = self._make_key(query, tags, project, current_user, include_archives)
        await self._result_cache.put(key, value)

    async def get_or_compute_embedding(
        self,
        text: str,
        embedding_provider: "EmbeddingProvider",
    ) -> list[float]:
        """Get embedding from cache or compute it.

        Uses per-key locking to prevent cache stampede: when multiple
        concurrent callers request the same embedding, only the first
        caller computes it; the rest wait and reuse the cached result.

        Args:
            text: Text to encode.
            embedding_provider: Provider to use if cache misses.

        Returns:
            The embedding vector.
        """
        key = hashlib.md5(text.strip().lower().encode()).hexdigest()
        text_preview = (text or "")[:50]

        # Fast path: cache hit (no lock needed)
        if self.enabled:
            cached = await self._embedding_cache.get(key)
            if cached is not None:
                io_logger.log_internal("embedding", {"hit": True, "text_preview": text_preview})
                logger.debug("Embedding cache hit for: %s", text_preview)
                return cached
            logger.debug("Embedding cache miss for: %s", text_preview)

        # Slow path: acquire per-key lock to prevent stampede
        async with self._locks[key]:
            # Double-check after acquiring lock
            if self.enabled:
                cached = await self._embedding_cache.get(key)
                if cached is not None:
                    io_logger.log_internal("embedding", {"hit": True, "text_preview": text_preview})
                    logger.debug("Embedding cache hit (after lock) for: %s", text_preview)
                    # Clean up lock to prevent memory leak
                    self._locks.pop(key, None)
                    return cached

            # Compute embedding
            t0 = time.monotonic()
            embedding = await embedding_provider.encode_single(text)
            duration_ms = int((time.monotonic() - t0) * 1000)

            if self.enabled:
                await self._embedding_cache.put(key, embedding)

            io_logger.log_internal(
                "embedding",
                {"hit": False, "text_preview": text_preview, "duration_ms": duration_ms},
                duration_ms=float(duration_ms),
            )

            # Clean up lock to prevent memory leak
            self._locks.pop(key, None)
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

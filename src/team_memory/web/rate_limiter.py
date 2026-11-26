"""Multi-tenant Redis sliding-window rate limiter with memory fallback.

Rate-limit key resolution priority:
  1. ``tenant:<tenant_id>`` (when tenant is established or X-Tenant-Id is sent with credentials)
  2. ``apikey:<sha256[:16]>`` (when Bearer token or api_key cookie is provided)
  3. ``ip:<client_ip>`` (unauthenticated fallback)

Uses a Redis sorted-set Lua script for atomic sliding-window accounting and
gracefully degrades to an in-memory sliding-window limiter with periodic
expiration sweeps (preventing unbounded dict growth) whenever Redis is
unavailable.
"""

from __future__ import annotations

import hashlib
import logging
import math
import time
import uuid
from dataclasses import dataclass
from typing import Any

logger = logging.getLogger("team_memory.web.rate_limit")

# Atomic sliding-window check in Redis using a ZSET.
# KEYS[1] = rate limit key
# ARGV[1] = now (float seconds)
# ARGV[2] = window (float seconds)
# ARGV[3] = limit (int)
# ARGV[4] = unique member id
# Returns: {allowed (0|1), current_count, retry_after_seconds}
_SLIDING_WINDOW_LUA = """
local key = KEYS[1]
local now = tonumber(ARGV[1])
local window = tonumber(ARGV[2])
local limit = tonumber(ARGV[3])
local member = ARGV[4]

redis.call('ZREMRANGEBYSCORE', key, '-inf', now - window)
local count = redis.call('ZCARD', key)
if count >= limit then
    local oldest = redis.call('ZRANGE', key, 0, 0, 'WITHSCORES')
    local retry_after = 1
    if oldest and oldest[2] then
        retry_after = math.max(1, math.ceil(tonumber(oldest[2]) + window - now))
    end
    return {0, count, retry_after}
end

redis.call('ZADD', key, now, member)
redis.call('PEXPIRE', key, math.ceil(window * 1000) + 1000)
return {1, count + 1, 0}
"""


@dataclass(slots=True)
class RateLimitDecision:
    """Result of a single rate-limit check."""

    allowed: bool
    limit: int
    remaining: int
    retry_after: int
    scope_key: str
    backend: str  # "redis" or "memory"


def extract_rate_limit_key(request: Any) -> str:
    """Resolve rate-limit bucket key: tenant_id > api_key hash > client IP."""
    state = getattr(request, "state", None)
    state_tenant = getattr(state, "tenant_id", None) or getattr(state, "mcp_tenant_id", None)
    if state_tenant:
        return f"tenant:{state_tenant}"

    headers = getattr(request, "headers", None) or {}
    cookies = getattr(request, "cookies", None) or {}

    auth_header = headers.get("authorization") or headers.get("Authorization") or ""
    api_key = auth_header[7:].strip() if auth_header.startswith("Bearer ") else ""
    if not api_key:
        api_key = (cookies.get("api_key") or "").strip()

    # Only trust X-Tenant-Id when accompanied by auth credentials to prevent
    # anonymous callers from spoofing random tenant IDs to bypass IP limits.
    requested_tenant = (
        headers.get("x-tenant-id") or headers.get("X-Tenant-Id") or ""
    ).strip()
    if requested_tenant and api_key:
        return f"tenant:{requested_tenant}"

    if api_key:
        key_hash = hashlib.sha256(api_key.encode("utf-8")).hexdigest()[:16]
        return f"apikey:{key_hash}"

    client = getattr(request, "client", None)
    client_ip = getattr(client, "host", None) or "unknown"
    return f"ip:{client_ip}"


class InMemorySlidingWindowLimiter:
    """In-memory sliding-window limiter with periodic expiration sweep."""

    def __init__(
        self,
        *,
        cleanup_interval: float = 60.0,
        max_keys: int = 10_000,
    ) -> None:
        self._buckets: dict[str, list[float]] = {}
        self._cleanup_interval = cleanup_interval
        self._max_keys = max_keys
        self._last_cleanup = time.monotonic()

    def _sweep_expired(self, now: float, window: float) -> None:
        cutoff = now - window
        expired_keys = [
            k
            for k, ts_list in self._buckets.items()
            if not ts_list or ts_list[-1] <= cutoff
        ]
        for k in expired_keys:
            self._buckets.pop(k, None)
        self._last_cleanup = now

        # Hard cap safeguard against adversarial cardinality bursts
        if len(self._buckets) > self._max_keys:
            sorted_keys = sorted(
                self._buckets.keys(),
                key=lambda k: self._buckets[k][-1] if self._buckets[k] else 0.0,
            )
            excess = len(self._buckets) - self._max_keys
            for k in sorted_keys[:excess]:
                self._buckets.pop(k, None)

    def check(
        self,
        scope_key: str,
        *,
        limit: int,
        window: float = 60.0,
        now: float | None = None,
    ) -> RateLimitDecision:
        ts = time.monotonic() if now is None else now
        if (ts - self._last_cleanup) >= self._cleanup_interval or len(self._buckets) > self._max_keys:
            self._sweep_expired(ts, window)

        cutoff = ts - window
        timestamps = self._buckets.get(scope_key)
        if timestamps:
            timestamps = [t for t in timestamps if t > cutoff]
            if timestamps:
                self._buckets[scope_key] = timestamps
            else:
                self._buckets.pop(scope_key, None)
                timestamps = []
        else:
            timestamps = []

        if len(timestamps) >= limit:
            oldest = timestamps[0]
            retry_after = max(1, math.ceil(oldest + window - ts))
            return RateLimitDecision(
                allowed=False,
                limit=limit,
                remaining=0,
                retry_after=retry_after,
                scope_key=scope_key,
                backend="memory",
            )

        timestamps.append(ts)
        self._buckets[scope_key] = timestamps
        return RateLimitDecision(
            allowed=True,
            limit=limit,
            remaining=max(0, limit - len(timestamps)),
            retry_after=0,
            scope_key=scope_key,
            backend="memory",
        )

    def reset(self) -> None:
        self._buckets.clear()
        self._last_cleanup = time.monotonic()

    @property
    def bucket_count(self) -> int:
        return len(self._buckets)


class SlidingWindowRateLimiter:
    """Redis-backed sliding-window rate limiter with graceful memory fallback."""

    def __init__(
        self,
        redis_url: str | None = None,
        *,
        redis_client: Any = None,
        key_prefix: str = "tm:ratelimit",
        retry_cooldown_seconds: float = 30.0,
        cleanup_interval: float = 60.0,
        max_memory_keys: int = 10_000,
    ) -> None:
        self._redis_url = redis_url
        self._redis = redis_client
        self._key_prefix = key_prefix
        self._retry_cooldown = retry_cooldown_seconds
        self._redis_disabled_until: float = 0.0
        self._memory = InMemorySlidingWindowLimiter(
            cleanup_interval=cleanup_interval,
            max_keys=max_memory_keys,
        )

    def _get_redis_client(self) -> Any:
        if self._redis is not None:
            return self._redis
        if not self._redis_url:
            return None
        try:
            import redis.asyncio as aioredis

            self._redis = aioredis.from_url(
                self._redis_url,
                socket_connect_timeout=0.5,
                socket_timeout=0.5,
                decode_responses=True,
            )
            return self._redis
        except Exception as e:
            logger.warning("Failed to initialize Redis rate limiter client: %s", e)
            return None

    async def check(
        self,
        scope_key: str,
        *,
        limit: int,
        window: float = 60.0,
    ) -> RateLimitDecision:
        """Check whether ``scope_key`` is within ``limit`` per ``window`` seconds."""
        now_mono = time.monotonic()
        if now_mono >= self._redis_disabled_until:
            client = self._get_redis_client()
            if client is not None:
                try:
                    now_wall = time.time()
                    redis_key = f"{self._key_prefix}:{scope_key}"
                    member = f"{now_wall:.6f}:{uuid.uuid4().hex[:8]}"
                    res = await client.eval(
                        _SLIDING_WINDOW_LUA,
                        1,
                        redis_key,
                        str(now_wall),
                        str(window),
                        str(limit),
                        member,
                    )
                    allowed = int(res[0]) == 1
                    count = int(res[1])
                    retry_after = int(res[2])
                    return RateLimitDecision(
                        allowed=allowed,
                        limit=limit,
                        remaining=max(0, limit - count) if allowed else 0,
                        retry_after=retry_after,
                        scope_key=scope_key,
                        backend="redis",
                    )
                except Exception as e:
                    self._redis_disabled_until = now_mono + self._retry_cooldown
                    logger.warning(
                        "Redis rate limiter unavailable (%s); degrading to in-memory limiter for %.0fs",
                        type(e).__name__,
                        self._retry_cooldown,
                    )

        return self._memory.check(scope_key, limit=limit, window=window)

    def reset_memory(self) -> None:
        self._memory.reset()
        self._redis_disabled_until = 0.0

    @property
    def memory_limiter(self) -> InMemorySlidingWindowLimiter:
        return self._memory

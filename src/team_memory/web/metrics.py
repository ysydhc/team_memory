"""Optional Prometheus metrics endpoint and built-in analytics.

If prometheus-client is not installed, /metrics returns 501.
Built-in analytics are always available via /api/v1/analytics/*.
"""

from __future__ import annotations

import logging
import time
from collections import defaultdict

logger = logging.getLogger("team_memory.metrics")

# ============================================================
# In-memory counters for built-in analytics
# ============================================================
_counters: dict[str, int] = defaultdict(int)
_latencies: list[tuple[float, float]] = []  # (timestamp, duration_ms)
_MAX_LATENCY_SAMPLES = 1000


def inc_counter(name: str, value: int = 1) -> None:
    """Increment an in-memory counter."""
    _counters[name] += value


def record_latency(duration_ms: float) -> None:
    """Record a request latency sample."""
    _latencies.append((time.time(), duration_ms))
    if len(_latencies) > _MAX_LATENCY_SAMPLES:
        _latencies.pop(0)


def get_counters() -> dict[str, int]:
    """Get all counters."""
    return dict(_counters)


def get_avg_latency(window_seconds: int = 300) -> float:
    """Get average latency for recent requests (default 5 min window)."""
    cutoff = time.time() - window_seconds
    recent = [d for t, d in _latencies if t >= cutoff]
    return sum(recent) / len(recent) if recent else 0.0


def get_latency_percentiles(window_seconds: int = 300) -> dict:
    """Get p50, p95, p99 latency percentiles."""
    cutoff = time.time() - window_seconds
    recent = sorted([d for t, d in _latencies if t >= cutoff])
    if not recent:
        return {"p50": 0, "p95": 0, "p99": 0}
    n = len(recent)
    return {
        "p50": recent[int(n * 0.5)],
        "p95": recent[int(n * 0.95)] if n > 1 else recent[-1],
        "p99": recent[int(n * 0.99)] if n > 1 else recent[-1],
    }


# ============================================================
# Optional Prometheus integration
# ============================================================
_prometheus_available = False

try:
    from prometheus_client import CONTENT_TYPE_LATEST, Counter, Histogram, generate_latest
    _prometheus_available = True

    REQUEST_COUNT = Counter(
        "team_memory_requests_total",
        "Total HTTP requests",
        ["method", "endpoint", "status"],
    )
    REQUEST_LATENCY = Histogram(
        "team_memory_request_duration_seconds",
        "Request duration in seconds",
        ["method", "endpoint"],
        buckets=(0.01, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0),
    )
    SEARCH_COUNT = Counter(
        "team_memory_search_total",
        "Total search queries",
        ["source"],
    )
    CACHE_HIT = Counter(
        "team_memory_cache_hits_total",
        "Cache hits",
    )
    CACHE_MISS = Counter(
        "team_memory_cache_misses_total",
        "Cache misses",
    )

except ImportError:
    logger.info("prometheus-client not installed. /metrics endpoint disabled.")


def is_prometheus_available() -> bool:
    return _prometheus_available


def get_prometheus_metrics() -> tuple[bytes, str]:
    """Generate Prometheus metrics output.

    Returns:
        Tuple of (metrics_bytes, content_type)
    """
    if not _prometheus_available:
        raise RuntimeError("prometheus-client not installed")
    return generate_latest(), CONTENT_TYPE_LATEST

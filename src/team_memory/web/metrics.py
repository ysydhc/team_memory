"""Optional Prometheus metrics endpoint and built-in analytics.

With prometheus-client: full Prometheus text format (counters, histograms, gauges).
Without: /metrics still returns 200 with hand-written text from in-memory counters.
Built-in analytics are always available via /api/v1/analytics/*.
"""

from __future__ import annotations

import logging
import re
import time
from collections import defaultdict

logger = logging.getLogger("team_memory.metrics")

# ============================================================
# In-memory counters for built-in analytics
# ============================================================
_counters: dict[str, int] = defaultdict(int)
_latencies: list[tuple[float, float]] = []  # (timestamp, duration_ms)
_MAX_LATENCY_SAMPLES = 1000

# UUID pattern for path normalization (reduce cardinality)
_UUID_PATTERN = re.compile(
    r"[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}"
)


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


def normalize_endpoint(path: str) -> str:
    """Replace UUIDs in path with {id} to limit label cardinality."""
    return _UUID_PATTERN.sub("{id}", path)


# ============================================================
# Optional Prometheus integration
# ============================================================
_prometheus_available = False

try:
    from prometheus_client import (
        CONTENT_TYPE_LATEST,
        Counter,
        Gauge,
        Histogram,
        generate_latest,
    )
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
    EMBEDDING_QUEUE_PENDING = Gauge(
        "team_memory_embedding_queue_pending",
        "Number of embedding tasks pending in queue",
    )
    EXPERIENCE_TOTAL = Gauge(
        "team_memory_experiences_total",
        "Total number of experiences in the database",
    )

except ImportError:
    REQUEST_COUNT = None
    REQUEST_LATENCY = None
    SEARCH_COUNT = None
    CACHE_HIT = None
    CACHE_MISS = None
    EMBEDDING_QUEUE_PENDING = None
    EXPERIENCE_TOTAL = None
    logger.info("prometheus-client not installed. /metrics uses fallback text.")


def is_prometheus_available() -> bool:
    return _prometheus_available


def record_request(method: str, endpoint: str, status: int, duration_seconds: float) -> None:
    """Record one HTTP request for Prometheus and in-memory analytics."""
    inc_counter("requests", 1)
    record_latency(duration_seconds * 1000.0)
    if _prometheus_available and REQUEST_COUNT is not None and REQUEST_LATENCY is not None:
        status_class = f"{status // 100}xx"
        REQUEST_COUNT.labels(method=method, endpoint=endpoint, status=status_class).inc()
        REQUEST_LATENCY.labels(method=method, endpoint=endpoint).observe(duration_seconds)


def set_scrape_gauges(experience_total: int = 0, embedding_queue_pending: int = 0) -> None:
    """Set gauge values before generating Prometheus output (scrape-time)."""
    if not _prometheus_available:
        return
    if EMBEDDING_QUEUE_PENDING is not None:
        EMBEDDING_QUEUE_PENDING.set(embedding_queue_pending)
    if EXPERIENCE_TOTAL is not None:
        EXPERIENCE_TOTAL.set(experience_total)


def get_prometheus_metrics() -> tuple[bytes, str]:
    """Generate Prometheus metrics output.

    Returns:
        Tuple of (metrics_bytes, content_type)
    """
    if not _prometheus_available:
        raise RuntimeError("prometheus-client not installed")
    return generate_latest(), CONTENT_TYPE_LATEST


def get_metrics_text_fallback() -> tuple[bytes, str]:
    """Return minimal Prometheus-format text from in-memory counters when no prometheus_client."""
    lines = [
        "# TYPE team_memory_requests_total counter",
        "# HELP team_memory_requests_total Total HTTP requests (fallback)",
        f"team_memory_requests_total {_counters.get('requests', 0)}",
    ]
    if _counters.get("search", 0) > 0:
        lines.extend([
            "# TYPE team_memory_search_total counter",
            "# HELP team_memory_search_total Total search queries (fallback)",
            f"team_memory_search_total {_counters.get('search', 0)}",
        ])
    body = "\n".join(lines) + "\n"
    return body.encode("utf-8"), "text/plain; charset=utf-8; version=0.0.4"

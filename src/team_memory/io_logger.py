"""I/O logger for MCP, service, and pipeline layers.

L0 layer: depends only on config and logging.
See docs/plans/2025-03-10-logging-system-design.md for design details.
"""

from __future__ import annotations

import json
import logging
import os
from collections.abc import Callable
from typing import Any

logger = logging.getLogger("team_memory.io")

# Node ID -> minimum detail level at which this node is logged.
# Order: mcp < service < pipeline < full.
_DETAIL_LEVELS: dict[str, str] = {
    # MCP layer (tool names)
    "search": "mcp",
    "tm_search": "mcp",
    "save": "mcp",
    "tm_save": "mcp",
    "feedback": "mcp",
    "tm_feedback": "mcp",
    "learn": "mcp",
    "tm_learn": "mcp",
    "solve": "mcp",
    "tm_solve": "mcp",
    "suggest": "mcp",
    "tm_suggest": "mcp",
    "preflight": "mcp",
    "tm_preflight": "mcp",
    "task": "mcp",
    "tm_task": "mcp",
    "invalidate_search_cache": "mcp",
    "tm_invalidate_search_cache": "mcp",
    # Pipeline layer (SearchPipeline steps)
    "query_expansion": "pipeline",
    "vector_search": "pipeline",
    "fts_search": "pipeline",
    "rerank": "pipeline",
    # Internal steps, only at full
    "cache_check": "full",
}

_DETAIL_RANK = {"mcp": 0, "service": 1, "pipeline": 2, "full": 3}

_SENSITIVE_KEYS = frozenset(
    {"api_key", "password", "secret", "token", "authorization"}
)

_settings_provider: Callable[[], Any] | None = None


def _mask_sensitive(obj: dict) -> dict:
    """Return a copy with sensitive key values replaced by '***'."""
    out = {}
    for k, v in obj.items():
        key_lower = k.lower() if isinstance(k, str) else ""
        if (
            key_lower in _SENSITIVE_KEYS
            or "password" in key_lower
            or "secret" in key_lower
        ):
            out[k] = "***"
        else:
            out[k] = v
    return out


def _should_log_node(detail: str, node_id: str) -> bool:
    """Return True if the node should be logged at the given detail level."""
    node_level = _DETAIL_LEVELS.get(node_id, "full")
    detail_rank = _DETAIL_RANK.get(detail, -1)
    node_rank = _DETAIL_RANK.get(node_level, 3)
    return detail_rank >= node_rank


def is_io_enabled() -> bool:
    """Return True if I/O logging is enabled. Reads from provider or env each call, no cache."""
    if _settings_provider is not None:
        try:
            settings = _settings_provider()
            if hasattr(settings, "logging") and hasattr(settings.logging, "log_io_enabled"):
                return bool(settings.logging.log_io_enabled)
        except Exception:
            pass
    v = os.environ.get("TEAM_MEMORY_LOGGING__LOG_IO_ENABLED") or os.environ.get(
        "TEAM_MEMORY_LOG_IO_ENABLED", ""
    )
    return str(v).lower() in ("1", "true", "yes", "on")


def get_detail_level() -> str:
    """Return configured log_io_detail. Reads from provider or env each call."""
    if _settings_provider is not None:
        try:
            settings = _settings_provider()
            if hasattr(settings, "logging") and hasattr(settings.logging, "log_io_detail"):
                return str(settings.logging.log_io_detail)
        except Exception:
            pass
    v = os.environ.get("TEAM_MEMORY_LOGGING__LOG_IO_DETAIL") or os.environ.get(
        "TEAM_MEMORY_LOG_IO_DETAIL", "mcp"
    )
    return v if v in _DETAIL_RANK else "mcp"


def get_truncate() -> int:
    """Return configured log_io_truncate. Reads from provider or env each call."""
    if _settings_provider is not None:
        try:
            settings = _settings_provider()
            if hasattr(settings, "logging") and hasattr(settings.logging, "log_io_truncate"):
                return int(settings.logging.log_io_truncate)
        except Exception:
            pass
    v = os.environ.get("TEAM_MEMORY_LOGGING__LOG_IO_TRUNCATE") or os.environ.get(
        "TEAM_MEMORY_LOG_IO_TRUNCATE", "300"
    )
    try:
        return max(0, int(v))
    except (ValueError, TypeError):
        return 300


def log_mcp_io(tool_name: str, kind: str, payload: str | dict) -> None:
    """Log MCP tool input or output. Masks sensitive keys, truncates, logs via logger.info."""
    if not is_io_enabled():
        return
    detail = get_detail_level()
    if not _should_log_node(detail, tool_name):
        return
    if isinstance(payload, dict):
        payload = _mask_sensitive(payload)
        try:
            text = json.dumps(payload, ensure_ascii=False, default=str)
        except Exception:
            text = repr(payload)
    else:
        text = str(payload)
    truncate = get_truncate()
    if truncate > 0 and len(text) > truncate:
        text = text[:truncate] + "..."
    logger.info("[io] %s %s: %s", tool_name, kind, text)


def log_internal(
    node_id: str, payload: dict | str, duration_ms: float | None = None
) -> None:
    """Log internal node I/O. Logs only if _should_log_node and is_io_enabled."""
    if not is_io_enabled():
        return
    if not _should_log_node(get_detail_level(), node_id):
        return
    if isinstance(payload, dict):
        payload = _mask_sensitive(payload)
        try:
            text = json.dumps(payload, ensure_ascii=False, default=str)
        except Exception:
            text = repr(payload)
    else:
        text = str(payload)
    truncate = get_truncate()
    if truncate > 0 and len(text) > truncate:
        text = text[:truncate] + "..."
    if duration_ms is not None:
        logger.info("[io] %s (%.1fms): %s", node_id, duration_ms, text)
    else:
        logger.info("[io] %s: %s", node_id, text)

"""Auto Dream worker: offline semantic memory consolidation, bi-temporal evolution, and distillation."""

from __future__ import annotations

from team_memory.daemon.integration.auto_dream.state import (
    _CACHE_DIR,
    _CONSOLIDATE_SYSTEM_PROMPT,
    _DISTILL_SYSTEM_PROMPT,
    _LAST_DREAM_FILE,
    _LAST_DREAM_STATS_FILE,
    _LOCK_FILE,
    _estimate_tokens,
    _is_protected,
    _NonBlockingFileLock,
    _parse_dt,
    _read_last_dream,
    _truncate_to_tokens,
    _utcnow,
    _write_last_dream,
    _write_last_dream_stats,
    read_last_dream_stats,
)
from team_memory.daemon.integration.auto_dream.worker import AutoDreamWorker
from team_memory.daemon.integration.tm_sink import _daemon_context

__all__ = [
    "AutoDreamWorker",
    "_CACHE_DIR",
    "_CONSOLIDATE_SYSTEM_PROMPT",
    "_DISTILL_SYSTEM_PROMPT",
    "_LAST_DREAM_FILE",
    "_LAST_DREAM_STATS_FILE",
    "_LOCK_FILE",
    "_NonBlockingFileLock",
    "_daemon_context",
    "_estimate_tokens",
    "_is_protected",
    "_parse_dt",
    "_read_last_dream",
    "_truncate_to_tokens",
    "_utcnow",
    "_write_last_dream",
    "_write_last_dream_stats",
    "read_last_dream_stats",
]

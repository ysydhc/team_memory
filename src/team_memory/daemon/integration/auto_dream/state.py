"""Shared constants, prompts, persistence, and locking helpers for AutoDream."""

from __future__ import annotations

import fcntl
import json
import logging
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from team_memory.daemon.integration.tm_sink import _daemon_context as _default_daemon_context

logger = logging.getLogger("daemon.auto_dream")

# Default cache directory and state files
_CACHE_DIR = Path(os.path.expanduser("~/.cache/tm-pipeline"))
_LAST_DREAM_FILE = _CACHE_DIR / "last_dream.txt"
_LAST_DREAM_STATS_FILE = _CACHE_DIR / "last_dream_stats.json"
_LOCK_FILE = _CACHE_DIR / "auto_dream.lock"

_CONSOLIDATE_SYSTEM_PROMPT = """\
You are AutoDream, an offline semantic memory consolidation engine for a software engineering knowledge base.
Given a cluster of related technical experiences (each with id, title, problem, solution, tags, created_at,
quality_tier, is_pinned, epistemic_tier, staleness_status, evidence_anchors, recall_count, used_count),
choose ONE action:

1. "MERGE": Experiences are near-duplicates or complementary facets of the SAME stable solution/fact.
   - Rewrite into ONE compact, high-signal experience (DO NOT concatenate with "---" or repeat boilerplate).
   - Temporal normalization: convert relative time phrases ("yesterday", "last week", "recently", "today",
     "昨天", "上周", "最近", "今天") into absolute dates (YYYY-MM-DD) anchored to each item's created_at.
   - Choose `primary_id` (MUST prefer is_pinned=true, quality_tier="Gold", or fresh epistemic_tier="verified";
     otherwise highest quality/adoption). Put remaining IDs in `secondary_ids`.
2. "SUPERSEDE": Experiences conflict or a newer/fresh experience replaces an older or `possibly_stale` approach
   (e.g., library migration, changed config, fixed misconception, updated API, deleted/refactored anchored files).
   - Pay special attention to items marked `staleness_status="possibly_stale"`: prioritize verifying whether a newer
     `fresh` experience in the cluster supersedes them.
   - Choose `winner_id` (the accurate/current experience) and `superseded_ids` (obsolete/contradicted ones;
     NEVER supersede an item with is_pinned=true, quality_tier="Gold", or fresh epistemic_tier="verified").
3. "KEEP_BOTH": Experiences apply to different scopes, constraints, or distinct problems — keep untouched.

Respond with ONLY valid JSON matching this schema:
{
  "action": "MERGE" | "SUPERSEDE" | "KEEP_BOTH",
  "reason": "concise explanation",
  "primary_id": "<id, required for MERGE>",
  "secondary_ids": ["<id>", ...],
  "merged": {
    "title": "<concise title>",
    "problem": "<distilled problem>",
    "solution": "<distilled, non-redundant solution with absolute dates>",
    "tags": ["tag1", "tag2"]
  },
  "winner_id": "<id, required for SUPERSEDE>",
  "superseded_ids": ["<id>", ...]
}"""

_DISTILL_SYSTEM_PROMPT = """\
You are AutoDream's knowledge distiller. Synthesize the given cluster of related experiences
from the same project into a concise, actionable Project Playbook.
Rules:
- Only distill if the cluster forms a coherent reusable operational/architectural theme;
  otherwise set should_distill=false.
- Focus on invariants, recommended workflow, pitfalls to avoid, and verified commands/configs.
- Convert relative dates to YYYY-MM-DD based on created_at.
- Keep the solution strictly concise (under 1200 tokens, bullet points, zero fluff).
- Respond with ONLY valid JSON:
{
  "should_distill": true | false,
  "reason": "brief explanation",
  "title": "[Playbook] <project>: <topic>",
  "problem": "<when to consult this playbook>",
  "solution": "<compact structured markdown playbook>",
  "tags": ["playbook", "auto-dream", ...]
}"""


def _pkg():
    return sys.modules.get("team_memory.daemon.integration.auto_dream")


def _get_daemon_context():
    fn = getattr(_pkg(), "_daemon_context", _default_daemon_context)
    return fn()


def _get_last_dream_file() -> Path:
    return getattr(_pkg(), "_LAST_DREAM_FILE", _LAST_DREAM_FILE)


def _get_last_dream_stats_file() -> Path:
    return getattr(_pkg(), "_LAST_DREAM_STATS_FILE", _LAST_DREAM_STATS_FILE)


def _get_lock_file() -> Path:
    return getattr(_pkg(), "_LOCK_FILE", _LOCK_FILE)


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


def _parse_dt(val: Any) -> datetime | None:
    """Safely parse datetime or ISO string into a timezone-aware UTC datetime."""
    if val is None:
        return None
    if isinstance(val, datetime):
        return val if val.tzinfo is not None else val.replace(tzinfo=timezone.utc)
    if isinstance(val, str):
        text = val.strip()
        if not text:
            return None
        try:
            dt = datetime.fromisoformat(text.replace("Z", "+00:00"))
            return dt if dt.tzinfo is not None else dt.replace(tzinfo=timezone.utc)
        except ValueError:
            return None
    return None


def _estimate_tokens(text: str) -> int:
    """Rough cross-lingual token estimator (CJK ~1.2 tok/char, ASCII ~0.28 tok/char)."""
    if not text:
        return 0
    cjk = sum(1 for ch in text if "\u4e00" <= ch <= "\u9fff")
    other = len(text) - cjk
    return max(1, int(cjk * 1.2 + other * 0.28))


def _truncate_to_tokens(text: str, max_tokens: int) -> str:
    """Truncate text so its estimated token count does not exceed max_tokens."""
    if not text or max_tokens <= 0 or _estimate_tokens(text) <= max_tokens:
        return text
    # Binary-shrink by line or char boundary
    ratio = max_tokens / max(1, _estimate_tokens(text))
    target_len = max(100, int(len(text) * ratio * 0.92))
    truncated = text[:target_len].rstrip()
    if "\n" in truncated:
        truncated = truncated.rsplit("\n", 1)[0].rstrip()
    return f"{truncated}\n\n_(truncated to <= {max_tokens} tokens)_"


def _is_protected(exp: dict[str, Any]) -> bool:
    """Return True if an experience must not be auto-overwritten or superseded."""
    if exp.get("is_pinned"):
        return True
    tier = str(exp.get("quality_tier") or "")
    if tier == "Gold":
        return True
    epistemic = str(exp.get("epistemic_tier") or "auto_extract").lower()
    staleness = str(exp.get("staleness_status") or "fresh").lower()
    return epistemic == "verified" and staleness == "fresh"


def _read_last_dream() -> datetime | None:
    """Read last dream timestamp from file. Returns None if never run."""
    target = _get_last_dream_file()
    try:
        if target.exists():
            text = target.read_text().strip()
            if text:
                return _parse_dt(text)
    except Exception:
        logger.debug("Failed to read last_dream file", exc_info=True)
    return None


def _write_last_dream(ts: datetime) -> None:
    """Persist last dream timestamp atomically."""
    target = _get_last_dream_file()
    try:
        target.parent.mkdir(parents=True, exist_ok=True)
        tmp = target.with_suffix(".tmp")
        tmp.write_text(ts.isoformat())
        tmp.replace(target)
    except Exception:
        logger.warning("Failed to write last_dream file", exc_info=True)


def read_last_dream_stats() -> dict[str, Any] | None:
    """Read last persisted AutoDream audit summary from JSON file."""
    target = _get_last_dream_stats_file()
    try:
        if target.exists():
            raw = target.read_text(encoding="utf-8").strip()
            if raw:
                data = json.loads(raw)
                if isinstance(data, dict):
                    return data
    except Exception:
        logger.debug("Failed to read last_dream_stats file", exc_info=True)
    return None


def _write_last_dream_stats(stats: dict[str, Any]) -> None:
    """Persist AutoDream audit summary atomically."""
    target = _get_last_dream_stats_file()
    try:
        target.parent.mkdir(parents=True, exist_ok=True)
        tmp = target.with_suffix(".tmp")
        tmp.write_text(json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8")
        tmp.replace(target)
    except Exception:
        logger.warning("Failed to write last_dream_stats file", exc_info=True)


class _NonBlockingFileLock:
    """Cross-process non-blocking file lock using fcntl.flock."""

    def __init__(self, lock_path: Path) -> None:
        self._lock_path = lock_path
        self._fd: int | None = None
        self.acquired: bool = False

    def __enter__(self) -> bool:
        try:
            self._lock_path.parent.mkdir(parents=True, exist_ok=True)
            self._fd = os.open(str(self._lock_path), os.O_CREAT | os.O_RDWR, 0o600)
            fcntl.flock(self._fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
            self.acquired = True
            return True
        except BlockingIOError:
            self.acquired = False
            if self._fd is not None:
                os.close(self._fd)
                self._fd = None
            return False
        except OSError:
            logger.debug("File lock unavailable for %s, proceeding with in-process lock", self._lock_path)
            self.acquired = True
            return True

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        if self._fd is not None:
            try:
                fcntl.flock(self._fd, fcntl.LOCK_UN)
            except OSError:
                pass
            try:
                os.close(self._fd)
            except OSError:
                pass
            self._fd = None

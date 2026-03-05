"""MCP tool input/output debug logging.

Only active when TEAM_MEMORY_DEBUG=1 or TEAM_MEMORY_MCP_DEBUG=1.
Logs input/output with 300-char console truncation; full content written to
files under a Chat-isolated directory, max 20 files per Chat. For local
debugging only; do not enable in production.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import re
import tempfile
import uuid
from datetime import datetime, timezone
from pathlib import Path

logger = logging.getLogger("team_memory.mcp_debug")

TRUNCATE_AT = 300
MAX_FILES_PER_CHAT = 20
SENSITIVE_KEYS = frozenset(
    {"api_key", "password", "secret", "token", "authorization"}
)


def is_mcp_debug_enabled() -> bool:
    """Return True if MCP debug logging is enabled (env checked at call time)."""
    v = os.environ.get("TEAM_MEMORY_DEBUG") or os.environ.get("TEAM_MEMORY_MCP_DEBUG")
    if not v:
        return False
    return str(v).lower() in ("1", "true", "yes", "on")


def _chat_id() -> str:
    """Return Chat/session id for directory isolation; default 'default' if unset."""
    raw = os.environ.get("TEAM_MEMORY_MCP_CHAT_ID") or os.environ.get("CURSOR_CHAT_ID")
    if not raw:
        return "default"
    # Sanitize: alphanumeric, underscore, hyphen only
    return re.sub(r"[^\w\-]", "_", str(raw))[:64] or "default"


def _mask_sensitive(obj: dict) -> dict:
    """Return a copy with sensitive key values replaced by '***'."""
    out = {}
    for k, v in obj.items():
        key_lower = k.lower() if isinstance(k, str) else ""
        if key_lower in SENSITIVE_KEYS or "password" in key_lower or "secret" in key_lower:
            out[k] = "***"
        else:
            out[k] = v
    return out


def _serialize_payload(payload: str | dict) -> str:
    """Serialize payload to string; if dict, mask sensitive keys then JSON dump."""
    if isinstance(payload, dict):
        payload = _mask_sensitive(payload)
        try:
            return json.dumps(payload, ensure_ascii=False, default=str)
        except Exception:
            return repr(payload)
    return str(payload)


def _log_dir() -> Path | None:
    """Return base log directory (create if needed); None if not writable."""
    explicit = os.environ.get("TEAM_MEMORY_MCP_LOG_DIR")
    if explicit:
        p = Path(explicit)
    else:
        # Prefer .debug/mcp_logs under cwd; fallback to system temp
        cwd = Path.cwd()
        candidate = cwd / ".debug" / "mcp_logs"
        if candidate.exists() and os.access(candidate, os.W_OK):
            return candidate
        if not candidate.exists():
            try:
                candidate.mkdir(parents=True, mode=0o700, exist_ok=True)
                if os.access(candidate, os.W_OK):
                    return candidate
            except OSError:
                pass
        p = Path(tempfile.gettempdir()) / "team_memory_mcp_logs"
    try:
        p.mkdir(parents=True, mode=0o700, exist_ok=True)
        if os.access(p, os.W_OK):
            return p
    except OSError:
        pass
    return None


def _ensure_chat_dir(base: Path, chat_id: str) -> Path | None:
    """Return Chat subdir under base; create if needed."""
    sub = base / chat_id
    try:
        sub.mkdir(mode=0o700, exist_ok=True)
        return sub if os.access(sub, os.W_OK) else None
    except OSError:
        return None


def _trim_to_max_files(dir_path: Path, max_files: int) -> None:
    """Remove oldest files in dir_path so that at most max_files remain. Never raises."""
    try:
        files = list(dir_path.glob("*.txt"))
        if len(files) <= max_files:
            return
        by_mtime = sorted(files, key=lambda f: f.stat().st_mtime)
        for f in by_mtime[: len(files) - max_files]:
            try:
                f.unlink()
            except OSError:
                pass
    except OSError:
        pass


def _write_payload_sync(log_dir: Path, tool_name: str, kind: str, payload: str) -> str | None:
    """Write full payload to a new file; return path or None. Runs in thread. Never raises."""
    try:
        _trim_to_max_files(log_dir, MAX_FILES_PER_CHAT)
        ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        short_uuid = uuid.uuid4().hex[:8]
        safe_tool = re.sub(r"[^\w\-]", "_", tool_name)[:32]
        name = f"{safe_tool}_{ts}_{short_uuid}_{kind}.txt"
        path = log_dir / name
        path.write_text(payload, encoding="utf-8")
        path.chmod(0o600)
        return str(path)
    except OSError:
        return None


def log_mcp_io(tool_name: str, kind: str, payload: str | dict) -> None:
    """Log MCP tool input or output: console truncation at 300 chars; optional file write.

    kind is 'in' or 'out'. payload is serialized (dict is masked then JSON-dumped).
    All exceptions are caught; never affects caller.
    """
    try:
        if not is_mcp_debug_enabled():
            return
        text = _serialize_payload(payload) if isinstance(payload, dict) else str(payload)
        n = len(text)
        if n <= TRUNCATE_AT:
            logger.info("[MCP debug] %s %s (%d chars): %s", tool_name, kind, n, text)
            return
        preview = text[:TRUNCATE_AT] + "..."
        base = _log_dir()
        chat_id = _chat_id()
        file_path: str | None = None
        if base:
            chat_dir = _ensure_chat_dir(base, chat_id)
            if chat_dir:
                file_path = _write_payload_sync(chat_dir, tool_name, kind, text)
        if file_path:
            logger.info(
                "[MCP debug] %s %s (%d chars): %s [完整内容已写入文件: %s]",
                tool_name,
                kind,
                n,
                preview,
                file_path,
            )
        else:
            if not base:
                logger.warning(
                    "[MCP debug] mcp_logs 目录不可写，仅打印前 %d 字", TRUNCATE_AT
                )
            logger.info("[MCP debug] %s %s (%d chars): %s", tool_name, kind, n, preview)
    except Exception as e:
        logger.warning("[MCP debug] log_mcp_io failed: %s", e, exc_info=False)


async def log_mcp_io_async(tool_name: str, kind: str, payload: str | dict) -> None:
    """Async wrapper: run log_mcp_io in thread to avoid blocking event loop."""
    try:
        await asyncio.to_thread(log_mcp_io, tool_name, kind, payload)
    except Exception as e:
        logger.warning("[MCP debug] log_mcp_io_async failed: %s", e, exc_info=False)

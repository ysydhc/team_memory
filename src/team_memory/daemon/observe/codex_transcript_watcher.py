"""OpenAI Codex CLI rollout transcript scanner & session importer.

Scans ~/.codex/sessions/YYYY/MM/DD/rollout-*.jsonl (or a specific file/dir),
normalizes turns & tool calls via ``parse_codex_transcript``, filters trivial
sessions, and queues substantive engineering trajectories into
``ConversationBuffer`` (drafts.db) for ``RefinementWorker``.
"""

from __future__ import annotations

import logging
import os
import re
from collections import Counter
from pathlib import Path
from typing import Any

from team_memory.hooks.core.transcript import parse_codex_transcript

logger = logging.getLogger("tm_daemon.codex_watcher")

DEFAULT_CODEX_SESSIONS_DIR = Path.home() / ".codex" / "sessions"

_UUID_SUFFIX_RE = re.compile(r"([0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12})$", re.IGNORECASE)

_TECH_KEYWORDS = (
    "```",
    "error",
    "exception",
    "traceback",
    "pytest",
    "fix",
    "bug",
    "refactor",
    "architecture",
    "修复",
    "报错",
    "架构",
    "落地方案",
    "需求",
    "优化",
    "排障",
    "实现",
)


def _infer_project_from_codex_parsed(parsed: dict[str, Any], default_project: str = "") -> str:
    """Infer project name from Codex workspace_roots or tool target_files."""
    if default_project:
        return default_project
    from team_memory.project_resolver import infer_project_name

    for ws in parsed.get("workspace_roots") or []:
        if isinstance(ws, str) and ws.strip() and os.path.exists(ws):
            inferred = infer_project_name(ws)
            if inferred and inferred not in ("default", "codex"):
                return inferred

    counts: Counter[str] = Counter()
    for tool in parsed.get("recent_tools") or []:
        for file_path in tool.get("target_files") or []:
            if not isinstance(file_path, str) or not os.path.isabs(file_path):
                continue
            target_dir = file_path if os.path.isdir(file_path) else str(Path(file_path).parent)
            if os.path.exists(target_dir):
                inferred = infer_project_name(target_dir)
                if inferred and inferred not in ("default", "codex"):
                    counts[inferred] += 1
    if counts:
        return counts.most_common(1)[0][0]

    # Fallback: use basename of first workspace_root even if path no longer exists
    for ws in parsed.get("workspace_roots") or []:
        if isinstance(ws, str) and ws.strip():
            from team_memory.project_resolver import canonicalize_project_name

            slug = canonicalize_project_name(Path(ws).name)
            if slug and slug not in ("default", "codex"):
                return slug
    return "codex"


def evaluate_codex_trajectory(parsed: dict[str, Any]) -> tuple[bool, str]:
    """Evaluate whether a parsed Codex rollout transcript has engineering value to distill."""
    turns = parsed.get("turns") or []
    recent_tools = parsed.get("recent_tools") or []

    user_turns = [t for t in turns if t.get("role") == "user" and t.get("content")]
    asst_turns = [t for t in turns if t.get("role") == "assistant" and t.get("content")]

    if not user_turns or not asst_turns:
        return False, "missing_user_or_assistant_turns"

    total_chars = sum(len(str(t.get("content") or "")) for t in turns)
    if total_chars < 200 and not recent_tools:
        return False, "too_short"

    if recent_tools:
        return True, "has_tool_trajectory"

    combined_lower = " ".join(str(t.get("content") or "") for t in turns).lower()
    if any(kw in combined_lower for kw in _TECH_KEYWORDS):
        return True, "has_technical_content"

    return False, "low_technical_density"


def _session_id_from_path(path: Path) -> str:
    stem = path.stem
    m = _UUID_SUFFIX_RE.search(stem)
    if m:
        return m.group(1)
    return stem


def discover_codex_transcripts(
    sessions_dir: str | Path | None = None,
    file_path: str | Path | None = None,
    since_ts: float | None = None,
) -> list[tuple[str, Path]]:
    """Return list of ``(session_id, rollout_jsonl_path)`` for Codex CLI sessions."""
    if file_path:
        p = Path(file_path).expanduser().resolve()
        if p.is_file():
            return [(_session_id_from_path(p), p)]
        if p.is_dir():
            sessions_dir = p
        else:
            return []

    root = Path(sessions_dir or DEFAULT_CODEX_SESSIONS_DIR).expanduser()
    if not root.is_dir():
        return []

    results: list[tuple[str, Path]] = []
    for jsonl in sorted(root.rglob("*.jsonl")):
        if not jsonl.is_file():
            continue
        if since_ts is not None:
            try:
                if jsonl.stat().st_mtime < since_ts:
                    continue
            except OSError:
                continue
        results.append((_session_id_from_path(jsonl), jsonl))
    return results


async def import_codex_sessions(
    *,
    buf: Any | None = None,
    sessions_dir: str | Path | None = None,
    file_path: str | Path | None = None,
    project: str = "",
    since_ts: float | None = None,
    dry_run: bool = False,
) -> dict[str, Any]:
    """Scan and import Codex CLI rollout transcripts into ``ConversationBuffer`` for refinement."""
    discovered = discover_codex_transcripts(
        sessions_dir=sessions_dir,
        file_path=file_path,
        since_ts=since_ts,
    )
    imported_sessions: list[dict[str, Any]] = []
    skipped_sessions: list[dict[str, Any]] = []

    for raw_sid, t_path in discovered:
        parsed = parse_codex_transcript(str(t_path))
        sid_core = str(parsed.get("session_id") or raw_sid)
        session_id = f"codex-{sid_core}" if not sid_core.startswith("codex-") else sid_core

        valuable, reason = evaluate_codex_trajectory(parsed)
        if not valuable:
            skipped_sessions.append({"session_id": session_id, "path": str(t_path), "reason": reason})
            continue

        resolved_proj = _infer_project_from_codex_parsed(parsed, default_project=project)
        turns = parsed.get("turns") or []

        if not dry_run and buf is not None:
            existing = await buf.get_session(session_id)
            if existing and existing.get("status") in (
                "published",
                "needs_refinement",
                "refining",
                "publishing",
            ):
                skipped_sessions.append(
                    {"session_id": session_id, "path": str(t_path), "reason": f"already_{existing.get('status')}"}
                )
                continue

            await buf.ensure_session(session_id, resolved_proj)
            for turn in turns:
                role = turn.get("role", "user")
                content = str(turn.get("content") or "")
                tool_calls = turn.get("tool_calls") or []

                if role == "assistant" and tool_calls:
                    tool_lines = []
                    for tc in tool_calls[:5]:
                        t_name = tc.get("tool", "")
                        t_cmd = tc.get("cmd", "")
                        t_files = ", ".join(tc.get("target_files") or [])
                        detail = t_cmd or t_files
                        tool_lines.append(f"[{t_name}] {detail}".strip())
                    if tool_lines:
                        content = f"{content}\n\n[Tools Executed]:\n" + "\n".join(tool_lines)

                if content.strip():
                    await buf.add_message(session_id, role, content.strip(), project=resolved_proj)

            await buf.mark_needs_refinement(session_id)

        imported_sessions.append(
            {
                "session_id": session_id,
                "project": resolved_proj,
                "turns_count": len(turns),
                "tools_count": len(parsed.get("recent_tools") or []),
                "path": str(t_path),
            }
        )

    return {
        "source": "codex",
        "discovered": len(discovered),
        "imported": len(imported_sessions),
        "skipped": len(skipped_sessions),
        "sessions": imported_sessions,
        "skipped_details": skipped_sessions,
        "dry_run": dry_run,
    }

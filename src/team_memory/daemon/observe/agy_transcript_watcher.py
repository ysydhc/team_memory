"""Antigravity (AGY) brain transcript scanner & session importer.

Scans ~/.gemini/antigravity/brain/<conversation-id>/.system_generated/logs/transcript.jsonl
(or a specific transcript.jsonl file), normalizes turns & tool calls, filters trivial sessions,
and queues substantive engineering trajectories into ConversationBuffer for RefinementWorker.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any

from team_memory.hooks.core.transcript import parse_antigravity_transcript

logger = logging.getLogger("tm_daemon.agy_watcher")

DEFAULT_AGY_BRAIN_DIR = Path.home() / ".gemini" / "antigravity" / "brain"

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


_IGNORED_PATH_SEGMENTS = (
    "/.gemini/",
    "/.claude/",
    "/.cursor/",
    "/.config/",
    "/tmp/",
    "/private/tmp/",
    "/var/folders/",
)


def _is_internal_or_temp_path(file_path: str) -> bool:
    norm = file_path.replace("\\", "/")
    return any(seg in norm for seg in _IGNORED_PATH_SEGMENTS)


def _infer_project_from_tools(recent_tools: list[dict[str, Any]], default_project: str = "") -> str:
    """Infer project name from workspace file paths touched by tool calls."""
    if default_project:
        return default_project
    from collections import Counter

    from team_memory.project_resolver import infer_project_name

    counts: Counter[str] = Counter()
    for tool in recent_tools:
        for file_path in tool.get("target_files") or []:
            if not isinstance(file_path, str) or not os.path.isabs(file_path):
                continue
            if _is_internal_or_temp_path(file_path):
                continue
            target_dir = file_path if os.path.isdir(file_path) else str(Path(file_path).parent)
            inferred = infer_project_name(target_dir)
            if inferred and inferred not in ("default", "antigravity", "brain", "scratch"):
                counts[inferred] += 1

    if counts:
        return counts.most_common(1)[0][0]
    return "antigravity"


def evaluate_agy_trajectory(parsed: dict[str, Any]) -> tuple[bool, str]:
    """Evaluate whether a parsed Antigravity transcript has engineering value to distill.

    Returns:
        (is_valuable, reason)
    """
    turns = parsed.get("turns") or []
    recent_tools = parsed.get("recent_tools") or []

    user_turns = [t for t in turns if t.get("role") == "user" and t.get("content")]
    asst_turns = [t for t in turns if t.get("role") == "assistant" and (t.get("content") or t.get("thinking"))]

    if not user_turns or not asst_turns:
        return False, "missing_user_or_assistant_turns"

    total_chars = sum(len(str(t.get("content") or "")) + len(str(t.get("thinking") or "")) for t in turns)
    if total_chars < 200 and not recent_tools:
        return False, "too_short"

    if recent_tools:
        return True, "has_tool_trajectory"

    combined_lower = " ".join(str(t.get("content") or "") for t in turns).lower()
    if any(kw in combined_lower for kw in _TECH_KEYWORDS):
        return True, "has_technical_content"

    return False, "low_technical_density"


def discover_agy_transcripts(
    brain_dir: str | Path | None = None,
    file_path: str | Path | None = None,
    since_ts: float | None = None,
) -> list[tuple[str, Path]]:
    """Return list of (conversation_id, transcript_path) for Antigravity sessions."""
    if file_path:
        p = Path(file_path).expanduser().resolve()
        if p.is_file():
            # Try to infer conversation_id from parent brain/<conv_id>/...
            conv_id = p.stem
            for parent in p.parents:
                if parent.parent.name == "brain":
                    conv_id = parent.name
                    break
            return [(conv_id, p)]
        if p.is_dir():
            brain_dir = p
        else:
            return []

    root = Path(brain_dir or DEFAULT_AGY_BRAIN_DIR).expanduser()
    if not root.is_dir():
        return []

    results: list[tuple[str, Path]] = []
    for conv_dir in sorted(root.iterdir()):
        if not conv_dir.is_dir():
            continue
        transcript = conv_dir / ".system_generated" / "logs" / "transcript.jsonl"
        if not transcript.is_file():
            continue
        if since_ts is not None:
            try:
                if transcript.stat().st_mtime < since_ts:
                    continue
            except OSError:
                continue
        results.append((conv_dir.name, transcript))
    return results


async def import_agy_sessions(
    *,
    buf: Any | None = None,
    brain_dir: str | Path | None = None,
    file_path: str | Path | None = None,
    project: str = "",
    since_ts: float | None = None,
    dry_run: bool = False,
) -> dict[str, Any]:
    """Scan and import Antigravity transcripts into ConversationBuffer for refinement.

    Returns summary stats dict.
    """
    discovered = discover_agy_transcripts(brain_dir=brain_dir, file_path=file_path, since_ts=since_ts)
    imported_sessions: list[dict[str, Any]] = []
    skipped_sessions: list[dict[str, Any]] = []

    for conv_id, t_path in discovered:
        session_id = f"agy-{conv_id}" if not conv_id.startswith("agy-") else conv_id
        parsed = parse_antigravity_transcript(str(t_path))
        valuable, reason = evaluate_agy_trajectory(parsed)
        if not valuable:
            skipped_sessions.append({"session_id": session_id, "path": str(t_path), "reason": reason})
            continue

        resolved_proj = _infer_project_from_tools(parsed.get("recent_tools") or [], default_project=project)
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
                thinking = str(turn.get("thinking") or "")
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
                if role == "assistant" and thinking and len(content) < 400:
                    content = f"{content}\n\n[Reasoning]: {thinking}".strip()

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
        "source": "antigravity",
        "discovered": len(discovered),
        "imported": len(imported_sessions),
        "skipped": len(skipped_sessions),
        "sessions": imported_sessions,
        "skipped_details": skipped_sessions,
        "dry_run": dry_run,
    }


async def sync_agy_mcp_recalls(
    *,
    response_buffer: Any,
    brain_dir: str | Path | None = None,
    file_path: str | Path | None = None,
    since_ts: float | None = None,
    dry_run: bool = False,
) -> dict[str, Any]:
    """Scan Antigravity transcripts and buffer MCP recall pairs into ResponseBuffer.

    Extracts ``mcp_recalls`` (search_log_id + result_ids + post-recall assistant response)
    from ``GENERIC`` steps in Antigravity transcripts so faithfulness evaluation can judge
    recall utilization even when PostToolUse/Stop hooks did not fire live.
    """
    discovered = discover_agy_transcripts(brain_dir=brain_dir, file_path=file_path, since_ts=since_ts)
    recalls_found = 0
    buffered = 0
    already_buffered = 0
    skipped_empty = 0

    for _conv_id, t_path in discovered:
        parsed = parse_antigravity_transcript(str(t_path))
        mcp_recalls = parsed.get("mcp_recalls") or []
        for rec in mcp_recalls:
            s_log_id = str(rec.get("search_log_id") or "").strip()
            asst_resp = str(rec.get("assistant_response") or "").strip()
            r_ids = rec.get("result_ids") or []
            if not s_log_id:
                continue
            recalls_found += 1
            if not asst_resp or not r_ids:
                skipped_empty += 1
                continue
            if dry_run or response_buffer is None:
                buffered += 1
                continue
            if await response_buffer.has_search_log(s_log_id):
                already_buffered += 1
                continue
            from team_memory.daemon.refinement.query_noise import classify_noise_rule

            query_text = (
                str(rec.get("query") or rec.get("user_input") or "(agy mcp recall)").strip()
                or "(agy mcp recall)"
            )
            if classify_noise_rule(query_text) is not None:
                query_text = "(agy mcp recall)"
            row_id = await response_buffer.add(
                query=query_text[:500],
                agent_response=asst_resp[:10000],
                result_ids=r_ids,
                search_log_id=s_log_id,
            )
            if row_id is not None:
                buffered += 1

    return {
        "discovered_transcripts": len(discovered),
        "recalls_found": recalls_found,
        "buffered": buffered,
        "already_buffered": already_buffered,
        "skipped_empty": skipped_empty,
        "dry_run": dry_run,
    }

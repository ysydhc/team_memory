"""Transcript file reader — shared utility for reading JSONL transcripts.

Supports:
1. Cursor/Claude Code transcript format:
   {"role": "assistant", "message": {"content": [{"type": "text", "text": "..."}]}}
2. Antigravity (AGY) brain transcript format (transcript.jsonl / transcript_full.jsonl):
   {"step_index": 1, "type": "USER_INPUT" | "PLANNER_RESPONSE", "content": "...", "tool_calls": [...]}
"""

from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Any

from team_memory.hooks.core.mcp_callback import extract_result_ids

_USER_REQ_RE = re.compile(r"<USER_REQUEST>\s*(.*?)\s*</USER_REQUEST>", re.DOTALL)
_META_TAG_RE = re.compile(
    r"<(ADDITIONAL_METADATA|CONTEXT_SUMMARY)>.*?</\1>",
    re.DOTALL,
)


def resolve_antigravity_transcript_path(transcript_path: str, conversation_id: str = "") -> str:
    """Return existing transcript_path or fallback to ~/.gemini/antigravity/brain/<conv_id>/.../transcript.jsonl."""
    if transcript_path and os.path.isfile(transcript_path):
        return transcript_path
    conv_id = (conversation_id or "").strip()
    if conv_id.startswith("agy-"):
        conv_id = conv_id[4:]
    if conv_id:
        candidate = (
            Path.home()
            / ".gemini"
            / "antigravity"
            / "brain"
            / conv_id
            / ".system_generated"
            / "logs"
            / "transcript.jsonl"
        )
        if candidate.is_file():
            return str(candidate)
    return transcript_path


def clean_antigravity_user_input(raw: str) -> str:
    """Extract clean user prompt text from Antigravity USER_INPUT step content."""
    if not raw:
        return ""
    match = _USER_REQ_RE.search(raw)
    if match:
        extracted = match.group(1).strip()
        if extracted:
            return extracted
    cleaned = _META_TAG_RE.sub("", raw).strip()
    return cleaned


def _resolve_full_transcript_line(transcript_path: str, line_idx: int) -> dict[str, Any] | None:
    """Read a specific 0-based line from sibling transcript_full.jsonl when truncated."""
    p = Path(transcript_path)
    full_path = p.with_name("transcript_full.jsonl")
    if not full_path.is_file():
        return None
    try:
        with full_path.open("r", encoding="utf-8") as f:
            for idx, line in enumerate(f):
                if idx == line_idx:
                    line = line.strip()
                    return json.loads(line) if line else None
    except Exception:
        return None
    return None


def _extract_agy_tool_summary(tc: dict[str, Any]) -> dict[str, Any]:
    """Normalize an Antigravity tool_call entry into a compact observation dict."""
    name = tc.get("name") or tc.get("tool_name") or tc.get("function", {}).get("name") or "unknown"
    args = tc.get("args") or tc.get("arguments") or tc.get("input") or {}
    if isinstance(args, str):
        try:
            args = json.loads(args)
        except Exception:
            args = {"raw": args[:200]}
    if not isinstance(args, dict):
        args = {}

    cmd = str(args.get("CommandLine") or args.get("command") or args.get("cmd") or "")
    target_files: list[str] = []
    for key in (
        "Path",
        "TargetFile",
        "file_path",
        "path",
        "SearchPath",
        "SearchDirectory",
        "DirectoryPath",
        "Cwd",
    ):
        val = args.get(key)
        if isinstance(val, str) and val:
            target_files.append(val)

    return {
        "tool": str(name),
        "cmd": cmd[:300],
        "target_files": target_files,
        "exit_code": tc.get("exit_code"),
    }


def parse_antigravity_transcript(transcript_path: str, since_step: int = -1) -> dict[str, Any]:
    """Parse Antigravity transcript.jsonl into normalized turns and tool observations.

    Args:
        transcript_path: Path to transcript.jsonl.
        since_step: Only include steps with step_index > since_step.

    Returns:
        Dict with keys:
          - turns: list of normalized turn dicts
          - last_user_input: latest user prompt string
          - last_assistant_response: synthesized response from latest turn
          - recent_tools: list of normalized tool observations
          - max_step_index: highest step_index seen
    """
    empty_result: dict[str, Any] = {
        "turns": [],
        "last_user_input": "",
        "last_assistant_response": "",
        "recent_tools": [],
        "mcp_recalls": [],
        "last_turn_mcp_recalls": [],
        "max_step_index": since_step,
    }
    if not transcript_path or not os.path.isfile(transcript_path):
        return empty_result

    turns: list[dict[str, Any]] = []
    recent_tools: list[dict[str, Any]] = []
    mcp_recalls: list[dict[str, Any]] = []
    current_turn_recalls: list[dict[str, Any]] = []
    last_user_input = ""
    last_recall_query = ""
    current_turn_responses: list[str] = []
    current_turn_thinking: list[str] = []
    max_step_index = since_step
    full_lines_cache: list[str] | None = None

    def _get_full_entry(idx: int) -> dict[str, Any] | None:
        nonlocal full_lines_cache
        if full_lines_cache is None:
            full_path = Path(transcript_path).with_name("transcript_full.jsonl")
            if full_path.is_file():
                try:
                    full_lines_cache = full_path.read_text(encoding="utf-8").splitlines()
                except Exception:
                    full_lines_cache = []
            else:
                full_lines_cache = []
        if 0 <= idx < len(full_lines_cache):
            line_str = full_lines_cache[idx].strip()
            if line_str:
                try:
                    return json.loads(line_str)
                except Exception:
                    return None
        return _resolve_full_transcript_line(transcript_path, idx)

    try:
        with open(transcript_path, "r", encoding="utf-8") as f:
            for line_idx, raw_line in enumerate(f):
                raw_line = raw_line.strip()
                if not raw_line:
                    continue
                try:
                    entry = json.loads(raw_line)
                except Exception:
                    continue

                step_idx = entry.get("step_index")
                if isinstance(step_idx, int):
                    if step_idx <= since_step:
                        continue
                    if step_idx > max_step_index:
                        max_step_index = step_idx

                truncated = entry.get("truncated_fields") or []
                if ("content" in truncated or "tool_calls" in truncated) and isinstance(truncated, list):
                    full_entry = _get_full_entry(line_idx)
                    if full_entry:
                        entry = full_entry

                step_type = entry.get("type", "")
                created_at = str(entry.get("created_at") or "")
                eff_step_idx = step_idx if isinstance(step_idx, int) else line_idx

                if step_type == "USER_INPUT":
                    user_text = clean_antigravity_user_input(str(entry.get("content") or ""))
                    if user_text:
                        turn_resp = "\n\n".join(current_turn_responses).strip()
                        if turn_resp:
                            for rec in current_turn_recalls:
                                if not rec.get("assistant_response"):
                                    rec["assistant_response"] = turn_resp
                        last_user_input = user_text
                        last_recall_query = ""
                        current_turn_responses = []
                        current_turn_thinking = []
                        current_turn_recalls = []
                        turns.append(
                            {
                                "step_index": eff_step_idx,
                                "role": "user",
                                "content": user_text,
                                "thinking": "",
                                "tool_calls": [],
                                "created_at": created_at,
                            }
                        )
                elif step_type == "GENERIC":
                    raw_content = str(entry.get("content") or "")
                    if "search_log_id" in raw_content:
                        s_log_id, r_ids = extract_result_ids(raw_content)
                        if s_log_id:
                            rec_entry: dict[str, Any] = {
                                "step_index": eff_step_idx,
                                "search_log_id": s_log_id,
                                "result_ids": r_ids,
                                "query": last_recall_query or last_user_input,
                                "user_input": last_user_input,
                                "assistant_response": "",
                                "created_at": created_at,
                                "_post_responses": [],
                            }
                            mcp_recalls.append(rec_entry)
                            current_turn_recalls.append(rec_entry)
                elif step_type == "PLANNER_RESPONSE":
                    content = str(entry.get("content") or "").strip()
                    thinking = str(entry.get("thinking") or "").strip()
                    raw_tools = entry.get("tool_calls") or []
                    for tc in raw_tools:
                        if not isinstance(tc, dict):
                            continue
                        tc_args = tc.get("args") or tc.get("arguments") or tc.get("input") or {}
                        if isinstance(tc_args, str):
                            try:
                                tc_args = json.loads(tc_args)
                            except Exception:
                                tc_args = {}
                        if isinstance(tc_args, dict):
                            nested = tc_args.get("Arguments") or tc_args.get("arguments") or {}
                            if isinstance(nested, str):
                                try:
                                    nested = json.loads(nested)
                                except Exception:
                                    nested = {}
                            q_cand = tc_args.get("query") or (nested.get("query") if isinstance(nested, dict) else "")
                            if q_cand and isinstance(q_cand, str) and q_cand.strip():
                                last_recall_query = q_cand.strip()
                    norm_tools = [
                        _extract_agy_tool_summary(tc) for tc in raw_tools if isinstance(tc, dict)
                    ]
                    recent_tools.extend(norm_tools)

                    if content:
                        current_turn_responses.append(content)
                        for rec in current_turn_recalls:
                            rec["_post_responses"].append(content)
                    if thinking:
                        current_turn_thinking.append(thinking[:400])

                    if content or thinking or norm_tools:
                        turns.append(
                            {
                                "step_index": eff_step_idx,
                                "role": "assistant",
                                "content": content,
                                "thinking": thinking[:600],
                                "tool_calls": norm_tools,
                                "created_at": created_at,
                            }
                        )
    except Exception:
        return empty_result

    last_assistant = "\n\n".join(current_turn_responses).strip()
    if not last_assistant and turns:
        for t in reversed(turns):
            if t["role"] == "assistant" and t.get("content"):
                last_assistant = t["content"]
                break

    if last_assistant and current_turn_thinking and len(last_assistant) < 300:
        thinking_excerpt = current_turn_thinking[-1]
        last_assistant = f"{last_assistant}\n\n[Reasoning Summary]: {thinking_excerpt}"

    for rec in mcp_recalls:
        post_resps = rec.pop("_post_responses", [])
        if post_resps:
            rec["assistant_response"] = "\n\n".join(post_resps).strip()
        elif not rec.get("assistant_response"):
            # Fallback: find any subsequent assistant turns in the transcript
            subseq = [
                str(t.get("content") or "").strip()
                for t in turns
                if t.get("role") == "assistant"
                and t.get("step_index", -1) > rec.get("step_index", -1)
                and str(t.get("content") or "").strip()
            ]
            if subseq:
                rec["assistant_response"] = "\n\n".join(subseq[:3]).strip()
            elif last_assistant:
                rec["assistant_response"] = last_assistant

    return {
        "turns": turns,
        "last_user_input": last_user_input,
        "last_assistant_response": last_assistant,
        "recent_tools": recent_tools,
        "mcp_recalls": mcp_recalls,
        "last_turn_mcp_recalls": current_turn_recalls,
        "max_step_index": max_step_index,
    }


def read_last_user_from_antigravity_transcript(transcript_path: str) -> str:
    """Read the latest clean USER_INPUT prompt from an Antigravity transcript.jsonl."""
    parsed = parse_antigravity_transcript(transcript_path)
    return str(parsed.get("last_user_input") or "")


def read_last_assistant_from_transcript(transcript_path: str) -> str:
    """Read the last assistant message from a JSONL transcript file.

    Supports both Cursor/Claude role-based JSONL and Antigravity step-based transcript.jsonl.
    """
    if not transcript_path or not os.path.isfile(transcript_path):
        return ""
    try:
        last_assistant = ""
        has_agy_steps = False
        with open(transcript_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                entry = json.loads(line)
                if entry.get("role") == "assistant":
                    msg = entry.get("message", {})
                    content_parts = msg.get("content", [])
                    if isinstance(content_parts, list):
                        texts = [p.get("text", "") for p in content_parts if p.get("type") == "text"]
                        if texts:
                            last_assistant = "\n".join(texts)
                    elif isinstance(content_parts, str):
                        last_assistant = content_parts
                elif entry.get("type") in ("USER_INPUT", "PLANNER_RESPONSE"):
                    has_agy_steps = True
        if last_assistant:
            return last_assistant
        if has_agy_steps:
            return str(parse_antigravity_transcript(transcript_path).get("last_assistant_response") or "")
        return ""
    except Exception:
        return ""

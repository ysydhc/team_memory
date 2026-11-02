"""MCP callback — 共享的 recall 检测和 callback 报告逻辑。

消除 claude_post_tool_use / cursor_after_mcp / hermes_post_tool_call 中
~65 行的重复代码。
"""

from __future__ import annotations

import json
import re

from team_memory.hooks.core.daemon_http import post_hook_fire_and_forget

RECALL_PATTERNS = ["recall", "memory_recall", "team_memory", "mcp__team_memory"]
_SEARCH_LOG_ID_RE = re.compile(r'"search_log_id"\s*:\s*"([0-9a-fA-F-]{36})"')


def is_recall_tool(tool_name: str, tool_args: dict | None = None) -> bool:
    """Check if a tool name (or call_mcp_tool wrapper) matches a TM recall pattern."""
    if any(p in tool_name.lower() for p in RECALL_PATTERNS):
        return True
    if tool_name == "call_mcp_tool" and isinstance(tool_args, dict):
        nested_tool = str(tool_args.get("ToolName") or tool_args.get("tool_name") or "")
        nested_server = str(tool_args.get("ServerName") or tool_args.get("server_name") or "")
        if any(p in nested_tool.lower() for p in ("recall", "memory_recall")):
            return True
        if "team_memory" in nested_server.lower() and "recall" in nested_tool.lower():
            return True
    return False


def _parse_embedded_recall_json(raw_text: str) -> dict | None:
    """Try to decode a JSON dict containing 'search_log_id' from a string."""
    idx = raw_text.find('"search_log_id"')
    if idx == -1:
        return None
    decoder = json.JSONDecoder()
    # Try candidate '{' positions preceding "search_log_id"
    search_pos = 0
    while True:
        brace_pos = raw_text.find("{", search_pos, idx)
        if brace_pos == -1:
            break
        try:
            obj, _ = decoder.raw_decode(raw_text[brace_pos:])
            if isinstance(obj, dict) and obj.get("search_log_id"):
                return obj
        except Exception:
            pass
        search_pos = brace_pos + 1
    return None


def extract_result_ids(raw_response: str | dict) -> tuple[str, list[dict]]:
    """Extract search_log_id and result_ids from a tool response.

    Args:
        raw_response: Tool response as JSON string or dict.

    Returns:
        Tuple of (search_log_id, result_ids list).
    """
    resp_data: dict | None = None
    if isinstance(raw_response, dict):
        resp_data = raw_response
    elif isinstance(raw_response, str) and raw_response:
        try:
            parsed = json.loads(raw_response)
            if isinstance(parsed, dict):
                resp_data = parsed
        except json.JSONDecodeError:
            resp_data = _parse_embedded_recall_json(raw_response)
            if resp_data is None:
                m = _SEARCH_LOG_ID_RE.search(raw_response)
                if m:
                    return m.group(1), []

    if not isinstance(resp_data, dict):
        return "", []

    search_log_id = str(resp_data.get("search_log_id") or "")
    results = resp_data.get("results", [])
    if not isinstance(results, list):
        results = []
    result_ids = [
        {
            "id": str(r.get("group_id") or r.get("id", "")),
            "score": r.get("score", r.get("similarity", 0)),
        }
        for r in results[:10]
        if isinstance(r, dict) and (r.get("group_id") or r.get("id"))
    ]
    return search_log_id, result_ids


def send_mcp_callback(
    session_id: str,
    tool_use_id: str,
    search_log_id: str,
    tool_name: str,
    result_ids: list[dict],
) -> None:
    """Report MCP recall results to daemon (best-effort). 向 daemon 报告 MCP recall 结果。"""
    if not search_log_id:
        return
    payload = {
        "session_id": session_id,
        "tool_use_id": tool_use_id,
        "search_log_id": search_log_id,
        "tool_name": tool_name,
        "result_ids": result_ids,
    }
    post_hook_fire_and_forget("mcp_callback", payload, timeout=5.0)

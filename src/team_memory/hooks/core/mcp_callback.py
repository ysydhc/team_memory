"""MCP callback — 共享的 recall 检测和 callback 报告逻辑。

消除 claude_post_tool_use / cursor_after_mcp / hermes_post_tool_call 中
~65 行的重复代码。
"""

from __future__ import annotations

import json

from team_memory.hooks.core.daemon_http import post_hook_fire_and_forget

RECALL_PATTERNS = ["recall", "memory_recall", "team_memory", "mcp__team_memory"]


def is_recall_tool(tool_name: str) -> bool:
    """Check if a tool name matches a TM recall pattern."""
    return any(p in tool_name.lower() for p in RECALL_PATTERNS)


def extract_result_ids(raw_response: str | dict) -> tuple[str, list[dict]]:
    """Extract search_log_id and result_ids from a tool response.

    Args:
        raw_response: Tool response as JSON string or dict.

    Returns:
        Tuple of (search_log_id, result_ids list).
    """
    try:
        if isinstance(raw_response, str):
            resp_data = json.loads(raw_response)
        elif isinstance(raw_response, dict):
            resp_data = raw_response
        else:
            return "", []

        search_log_id = resp_data.get("search_log_id", "")
        results = resp_data.get("results", [])
        result_ids = [
            {
                "id": str(r.get("group_id") or r.get("id", "")),
                "score": r.get("score", r.get("similarity", 0)),
            }
            for r in results[:10]
        ]
        return search_log_id, result_ids
    except (json.JSONDecodeError, AttributeError):
        return "", []


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

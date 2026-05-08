"""Cursor afterMCPExecution hook — reports MCP recall results to daemon.

Fires after the AI successfully invokes an MCP tool.
When the tool is a TM recall tool, extracts search_log_id from the
response and registers it with the daemon for faithfulness correlation.

Input (from Cursor):
  {
    "conversation_id": "...",
    "generation_id": "...",
    "tool_name": "memory_recall",
    "tool_input": {...},
    "tool_output": "...",   ← MCP 返回的内容
    "tool_use_id": "...",
    "duration": 123
  }

Output: {} (no modification to tool result)
"""
import json
import sys

import httpx

DAEMON_URL = "http://127.0.0.1:3901"

# Tool name patterns that indicate a TM recall
RECALL_PATTERNS = ["recall", "memory_recall", "team_memory"]


def is_recall_tool(tool_name: str) -> bool:
    return any(p in tool_name.lower() for p in RECALL_PATTERNS)


def main() -> None:
    try:
        input_data = json.loads(sys.stdin.read())
    except Exception:
        print("{}")
        sys.stdout.flush()
        return

    tool_name = input_data.get("tool_name", "")
    if not is_recall_tool(tool_name):
        print("{}")
        sys.stdout.flush()
        return

    # Cursor uses conversation_id, not session_id
    session_id = input_data.get("conversation_id", "")
    tool_use_id = input_data.get("tool_use_id", "")
    tool_output = input_data.get("tool_output", "")

    # Extract search_log_id from tool_output
    search_log_id = ""
    result_ids = []

    try:
        if isinstance(tool_output, str):
            resp_data = json.loads(tool_output)
        elif isinstance(tool_output, dict):
            resp_data = tool_output
        else:
            resp_data = {}

        search_log_id = resp_data.get("search_log_id", "")
        results = resp_data.get("results", [])
        if results:
            result_ids = [
                {"id": str(r.get("group_id") or r.get("id", "")),
                 "score": r.get("score", r.get("similarity", 0))}
                for r in results[:10]
            ]
    except (json.JSONDecodeError, AttributeError):
        pass

    if not search_log_id:
        print("{}")
        sys.stdout.flush()
        return

    # Report to daemon
    payload = {
        "session_id": session_id,
        "tool_use_id": tool_use_id,
        "search_log_id": search_log_id,
        "tool_name": tool_name,
        "result_ids": result_ids,
    }

    try:
        httpx.post(
            f"{DAEMON_URL}/hooks/mcp_callback",
            json=payload,
            timeout=5.0,
        )
    except Exception:
        pass  # best-effort

    print("{}")
    sys.stdout.flush()


if __name__ == "__main__":
    main()

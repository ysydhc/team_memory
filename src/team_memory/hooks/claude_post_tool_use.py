"""Claude Code PostToolUse hook — reports MCP recall results to daemon.

Fires after the AI successfully invokes a tool (including MCP tools).
When the tool is a TM recall tool, extracts search_log_id from the
response and registers it with the daemon for faithfulness correlation.

Input (from Claude Code):
  {
    "session_id": "...",
    "tool_name": "mcp__team_memory__recall",
    "tool_input": {...},
    "tool_response": "...",
    "tool_use_id": "..."
  }

Output: {} (no modification to tool result)
"""
import json
import sys

import httpx

DAEMON_URL = "http://127.0.0.1:3901"

# Tool name patterns that indicate a TM recall
RECALL_PATTERNS = ["recall", "memory_recall", "mcp__team_memory"]


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

    session_id = input_data.get("session_id", "")
    tool_use_id = input_data.get("tool_use_id", "")
    tool_response = input_data.get("tool_response", "")

    # Extract search_log_id from tool_response
    search_log_id = ""
    result_ids = []

    # tool_response may be a JSON string or a dict (depending on Claude Code version)
    try:
        if isinstance(tool_response, str):
            resp_data = json.loads(tool_response)
        elif isinstance(tool_response, dict):
            resp_data = tool_response
        else:
            resp_data = {}

        search_log_id = resp_data.get("search_log_id", "")
        # Extract result_ids if available
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
        # No search_log_id — maybe the recall didn't write a log
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
        pass  # best-effort, don't block AI

    print("{}")
    sys.stdout.flush()


if __name__ == "__main__":
    main()

"""Hermes post_tool_call hook — reports MCP recall results to daemon.

Fires after the AI successfully invokes a tool (including MCP tools).
When the tool is a TM recall tool, extracts search_log_id from the
result and registers it with the daemon for faithfulness correlation.

Input (from Hermes, via stdin JSON):
  {
    "hook_event_name": "post_tool_call",
    "tool_name": "memory_recall",
    "tool_input": {...},
    "session_id": "sess_abc123",
    "extra": {
      "tool_call_id": "...",
      "result": "...",
      "duration_ms": 42
    }
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

    session_id = input_data.get("session_id", "")
    extra = input_data.get("extra", {})
    tool_call_id = extra.get("tool_call_id", "")
    result_raw = extra.get("result", "")

    # Extract search_log_id from tool result
    search_log_id = ""
    result_ids = []

    try:
        if isinstance(result_raw, str):
            resp_data = json.loads(result_raw)
        elif isinstance(result_raw, dict):
            resp_data = result_raw
        else:
            resp_data = {}

        search_log_id = resp_data.get("search_log_id", "")
        results = resp_data.get("results", [])
        if results:
            result_ids = [
                {
                    "id": str(r.get("group_id") or r.get("id", "")),
                    "score": r.get("score", r.get("similarity", 0)),
                }
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
        "tool_use_id": tool_call_id,
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

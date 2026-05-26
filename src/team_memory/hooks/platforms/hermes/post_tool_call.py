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

from team_memory.hooks.core.mcp_callback import (
    extract_result_ids,
    is_recall_tool,
    send_mcp_callback,
)
from team_memory.hooks.core.output import print_empty
from team_memory.hooks.core.stdin import parse_stdin_json


def main() -> None:
    input_data = parse_stdin_json()
    if not input_data:
        print_empty()
        return

    tool_name = input_data.get("tool_name", "")
    if not is_recall_tool(tool_name):
        print_empty()
        return

    session_id = input_data.get("session_id", "")
    extra = input_data.get("extra", {})
    tool_call_id = extra.get("tool_call_id", "")
    result_raw = extra.get("result", "")

    search_log_id, result_ids = extract_result_ids(result_raw)
    send_mcp_callback(session_id, tool_call_id, search_log_id, tool_name, result_ids)
    print_empty()


if __name__ == "__main__":
    main()

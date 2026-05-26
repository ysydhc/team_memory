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
    tool_use_id = input_data.get("tool_use_id", "")
    tool_response = input_data.get("tool_response", "")

    search_log_id, result_ids = extract_result_ids(tool_response)
    send_mcp_callback(session_id, tool_use_id, search_log_id, tool_name, result_ids)
    print_empty()


if __name__ == "__main__":
    main()

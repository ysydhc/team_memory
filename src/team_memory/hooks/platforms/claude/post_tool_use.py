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

from team_memory.hooks.core.daemon_http import post_hook_fire_and_forget
from team_memory.hooks.core.mcp_callback import (
    extract_result_ids,
    is_recall_tool,
    send_mcp_callback,
)
from team_memory.hooks.core.output import print_empty
from team_memory.hooks.core.stdin import parse_stdin_json

_CODING_TOOLS: set[str] = {
    "Bash",
    "Edit",
    "Write",
    "MultiEdit",
}


def main() -> None:
    input_data = parse_stdin_json()
    if not input_data:
        print_empty()
        return

    tool_name = input_data.get("tool_name", "")
    session_id = input_data.get("session_id", "")
    tool_use_id = input_data.get("tool_use_id", "")
    tool_response = input_data.get("tool_response", "")

    if is_recall_tool(tool_name):
        search_log_id, result_ids = extract_result_ids(tool_response)
        send_mcp_callback(session_id, tool_use_id, search_log_id, tool_name, result_ids)
        print_empty()
        return

    if tool_name in _CODING_TOOLS and session_id:
        tool_input = input_data.get("tool_input") or {}
        if not isinstance(tool_input, dict):
            tool_input = {}
        command = str(tool_input.get("command") or "")
        target_file = str(tool_input.get("file_path") or tool_input.get("path") or "")
        resp_str = str(tool_response or "")
        lowered = resp_str.lower()
        succeeded = not ("error:" in lowered or "traceback" in lowered or "exit code: 1" in lowered)
        exit_code = 0 if succeeded and tool_name == "Bash" else (1 if not succeeded and tool_name == "Bash" else None)
        post_hook_fire_and_forget(
            "tool_observation",
            {
                "conversation_id": session_id,
                "tool_name": tool_name,
                "command": command[:500],
                "target_file": target_file[:500],
                "exit_code": exit_code,
                "succeeded": succeeded,
                "error_excerpt": resp_str[:500] if not succeeded else "",
                "source_platform": "claude",
            },
        )

    print_empty()


if __name__ == "__main__":
    main()

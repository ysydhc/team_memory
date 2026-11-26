"""OpenAI Codex CLI PostToolUse hook — reports MCP recall results & coding tool observations.

Input (from Codex CLI stdin):
  {
    "session_id": "...",
    "tool_name": "exec_command" | "mcp__team_memory__memory_recall" | ...,
    "tool_input": {...},
    "tool_response": "...",
    "tool_use_id": "..."
  }

Output: {} (no modification)
"""

from __future__ import annotations

from team_memory.hooks.core.daemon_http import post_hook_fire_and_forget
from team_memory.hooks.core.mcp_callback import (
    extract_result_ids,
    is_recall_tool,
    send_mcp_callback,
)
from team_memory.hooks.core.output import print_empty
from team_memory.hooks.core.stdin import parse_stdin_json

_CODING_TOOLS: set[str] = {
    "shell",
    "shell_command",
    "exec_command",
    "apply_patch",
    "Bash",
    "Edit",
    "Write",
}


def main() -> None:
    input_data = parse_stdin_json()
    if not input_data:
        print_empty()
        return

    tool_name = str(input_data.get("tool_name") or input_data.get("name") or "")
    session_id = str(
        input_data.get("session_id") or input_data.get("thread_id") or input_data.get("conversation_id") or ""
    )
    tool_use_id = str(input_data.get("tool_use_id") or input_data.get("call_id") or "")
    tool_response = input_data.get("tool_response") or input_data.get("output") or ""

    if is_recall_tool(tool_name):
        search_log_id, result_ids = extract_result_ids(tool_response)
        send_mcp_callback(session_id, tool_use_id, search_log_id, tool_name, result_ids)
        print_empty()
        return

    if tool_name in _CODING_TOOLS and session_id:
        tool_input = input_data.get("tool_input") or input_data.get("arguments") or {}
        if not isinstance(tool_input, dict):
            tool_input = {}
        raw_cmd = tool_input.get("cmd") or tool_input.get("command") or ""
        command = " ".join(str(x) for x in raw_cmd) if isinstance(raw_cmd, list) else str(raw_cmd)
        target_file = str(tool_input.get("file_path") or tool_input.get("path") or tool_input.get("workdir") or "")
        resp_str = str(tool_response or "")
        lowered = resp_str.lower()
        succeeded = not ("error:" in lowered or "traceback" in lowered or "exit code: 1" in lowered)
        is_shell = tool_name in ("shell", "shell_command", "exec_command", "Bash")
        exit_code = 0 if (succeeded and is_shell) else (1 if (not succeeded and is_shell) else None)
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
                "source_platform": "codex",
            },
        )

    print_empty()


if __name__ == "__main__":
    main()

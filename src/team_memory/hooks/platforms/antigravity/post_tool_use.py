"""Antigravity PostToolUse hook — captures recall callbacks & coding/command observations.

Input (from Antigravity stdin):
  {
    "conversationId": "...",
    "workspacePaths": ["/path/to/workspace"],
    "toolCall": {"name": "run_command", "args": {"CommandLine": "pytest"}},
    "stepIdx": 3,
    "error": "..." (optional)
  }

Output:
  {}
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
from team_memory.hooks.core.transcript import (
    parse_antigravity_transcript,
    resolve_antigravity_transcript_path,
)

_OBSERVED_CODING_TOOLS = {
    "run_command",
    "edit_file",
    "write_file",
    "multi_replace_file_content",
    "replace_file_content",
    "Bash",
    "Edit",
    "Write",
}


def main() -> None:
    input_data = parse_stdin_json()
    if not input_data:
        print_empty()
        return

    tool_call = input_data.get("toolCall") or {}
    tool_name = tool_call.get("name") or input_data.get("tool_name") or ""
    tool_args = tool_call.get("args") or input_data.get("tool_input") or {}
    if not isinstance(tool_args, dict):
        tool_args = {}

    conversation_id = input_data.get("conversationId") or input_data.get("conversation_id") or ""
    workspace_roots = input_data.get("workspacePaths") or input_data.get("workspace_roots") or []
    error_text = str(input_data.get("error") or "")

    if is_recall_tool(tool_name, tool_args):
        search_log_id, result_ids = extract_result_ids(input_data.get("tool_response", ""))
        if not search_log_id:
            t_path = resolve_antigravity_transcript_path(
                input_data.get("transcriptPath") or input_data.get("transcript_path") or "",
                conversation_id,
            )
            if t_path:
                parsed = parse_antigravity_transcript(t_path)
                recalls = parsed.get("mcp_recalls") or []
                if recalls:
                    latest = recalls[-1]
                    search_log_id = str(latest.get("search_log_id") or "")
                    result_ids = latest.get("result_ids") or []
        tool_use_id = str(input_data.get("tool_use_id") or input_data.get("stepIdx") or "")
        eff_tool = (
            str(tool_args.get("ToolName") or tool_args.get("tool_name") or tool_name)
            if tool_name == "call_mcp_tool"
            else tool_name
        )
        send_mcp_callback(conversation_id, tool_use_id, search_log_id, eff_tool, result_ids)
        print_empty()
        return

    if tool_name in _OBSERVED_CODING_TOOLS and conversation_id:
        cmd = str(tool_args.get("CommandLine") or tool_args.get("command") or tool_args.get("cmd") or "")
        target_files: list[str] = []
        for k in ("Path", "TargetFile", "file_path", "path"):
            v = tool_args.get(k)
            if isinstance(v, str) and v:
                target_files.append(v)

        exit_code = input_data.get("exit_code")
        if exit_code is None and error_text:
            exit_code = 1
        elif exit_code is None and cmd:
            exit_code = 0

        post_hook_fire_and_forget(
            "tool_observation",
            {
                "conversation_id": conversation_id,
                "workspace_roots": workspace_roots if isinstance(workspace_roots, list) else [],
                "step_index": input_data.get("stepIdx", 0),
                "tool_name": tool_name,
                "command_summary": cmd[:300],
                "target_files": target_files,
                "exit_code": exit_code,
                "error_excerpt": error_text[:500],
                "succeeded": not bool(error_text) and (exit_code in (None, 0)),
                "source_platform": "antigravity",
            },
        )

    print_empty()


if __name__ == "__main__":
    main()

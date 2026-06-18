"""Cursor afterAgentResponse hook — forwards to TM Daemon.

Maps Cursor's field names to what the daemon expects.
Supports both old format (agent_response field) and new format (transcript_path).

Daemon expects:
  {"conversation_id": "...", "response_text": "...", "workspace_roots": [...]}
"""

import os

from team_memory.hooks.core.daemon_http import post_hook
from team_memory.hooks.core.output import print_cursor_action
from team_memory.hooks.core.stdin import parse_stdin_json
from team_memory.hooks.core.transcript import read_last_assistant_from_transcript


def main() -> None:
    input_data = parse_stdin_json()
    if not input_data:
        print_cursor_action({"action": "error", "message": "invalid input"})
        return

    # Try direct fields first (old Cursor format)
    response_text = (
        input_data.get("response_text", "")
        or input_data.get("agent_response", "")
        or input_data.get("content", "")
        or input_data.get("response", "")
        or ""
    )

    # Fallback: read from transcript_path (new Cursor format)
    if not response_text:
        transcript_path = input_data.get("transcript_path", "")
        response_text = read_last_assistant_from_transcript(transcript_path)

    conversation_id = input_data.get("conversation_id", "") or input_data.get("session_id", "")
    workspace_roots = input_data.get("workspace_roots", [])
    if not workspace_roots:
        cwd = input_data.get("cwd", "") or input_data.get("workspace", "")
        if cwd:
            workspace_roots = [cwd]
        else:
            workspace_roots = [os.getcwd()]

    if not response_text:
        print_cursor_action({"action": "ok", "message": "no response to process"})
        return

    payload = {
        "conversation_id": conversation_id,
        "response_text": response_text,
        "workspace_roots": workspace_roots,
        "model": input_data.get("model", ""),
    }

    try:
        resp = post_hook("after_response", payload)
        print(resp.text)
    except Exception as e:
        from httpx import ConnectError

        if isinstance(e, ConnectError):
            print_cursor_action({"action": "ok", "message": "daemon not running"})
        else:
            print_cursor_action({"action": "error", "message": str(e)})


if __name__ == "__main__":
    main()

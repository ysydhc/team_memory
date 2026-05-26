"""Cursor afterAgentResponse hook — forwards to TM Daemon.

Maps Cursor's field names to what the daemon expects.
Cursor sends different field names than Hermes/Claude Code,
so we normalize the payload here.

Cursor afterAgentResponse stdin format:
  {"conversation_id": "...", "agent_response": "...", ...}

Daemon expects:
  {"conversation_id": "...", "response_text": "...", "workspace_roots": [...]}

Cursor 字段映射: agent_response → response_text
"""

import os

from team_memory.hooks.core.daemon_http import post_hook
from team_memory.hooks.core.output import print_cursor_action
from team_memory.hooks.core.stdin import parse_stdin_json


def main() -> None:
    input_data = parse_stdin_json()
    if not input_data:
        print_cursor_action({"action": "error", "message": "invalid input"})
        return

    response_text = (
        input_data.get("response_text", "")
        or input_data.get("agent_response", "")
        or input_data.get("content", "")
        or input_data.get("response", "")
        or input_data.get("prompt", "")
        or ""
    )
    conversation_id = input_data.get("conversation_id", "") or input_data.get("session_id", "")
    workspace_roots = input_data.get("workspace_roots", [])
    if not workspace_roots:
        cwd = input_data.get("cwd", "") or input_data.get("workspace", "")
        if cwd:
            workspace_roots = [cwd]
        else:
            workspace_roots = [os.getcwd()]

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

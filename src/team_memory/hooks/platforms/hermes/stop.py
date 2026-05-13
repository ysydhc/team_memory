"""Hermes Stop hook - forwards assistant response to TM Daemon after_response."""

from team_memory.hooks.core.daemon_http import post_hook_fire_and_forget
from team_memory.hooks.core.output import print_empty
from team_memory.hooks.core.stdin import parse_stdin_json


def main():
    data = parse_stdin_json()
    if not data:
        print_empty()
        return
    session_id = data.get("session_id", "")
    response = data.get("response", "") or data.get("response_text", "")
    cwd = data.get("cwd", "") or data.get("working_directory", "")
    if not response:
        print_empty()
        return
    payload = {
        "conversation_id": session_id,
        "response_text": response,
        "workspace_roots": [cwd] if cwd else [],
    }
    post_hook_fire_and_forget("after_response", payload)
    print_empty()


if __name__ == "__main__":
    main()

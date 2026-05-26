"""Hermes UserPromptSubmit hook - forwards to TM Daemon before_prompt."""

from team_memory.hooks.core.daemon_http import post_hook
from team_memory.hooks.core.output import print_empty, print_hermes_context
from team_memory.hooks.core.stdin import parse_stdin_json


def main():
    data = parse_stdin_json()
    if not data:
        print_empty()
        return
    extra = data.get("extra", {}) or {}
    session_id = data.get("session_id", "")
    prompt = (data.get("prompt", "")
              or data.get("user_message", "")
              or extra.get("user_message", "")
              or "")
    cwd = (data.get("cwd", "")
           or data.get("working_directory", "")
           or "")
    if not prompt:
        print_empty()
        return
    payload = {
        "conversation_id": session_id,
        "prompt": prompt,
        "workspace_roots": [cwd] if cwd else [],
    }
    try:
        resp = post_hook("before_prompt", payload)
        result = resp.json()
        context = result.get("context", "")
        print_hermes_context(context)
    except Exception:
        print_empty()


if __name__ == "__main__":
    main()

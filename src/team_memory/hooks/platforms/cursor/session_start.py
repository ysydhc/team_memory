"""Cursor sessionStart hook — forwards to TM Daemon."""

from team_memory.hooks.core.daemon_http import post_hook
from team_memory.hooks.core.output import print_cursor_action
from team_memory.hooks.core.stdin import parse_stdin_json


def main() -> None:
    input_data = parse_stdin_json()
    if not input_data:
        print_cursor_action({"action": "error", "message": "invalid input"})
        return

    try:
        resp = post_hook("session_start", input_data)
        print(resp.text)
    except Exception as e:
        from httpx import ConnectError

        if isinstance(e, ConnectError):
            print_cursor_action({"action": "ok", "message": "daemon not running"})
        else:
            print_cursor_action({"action": "error", "message": str(e)})


if __name__ == "__main__":
    main()

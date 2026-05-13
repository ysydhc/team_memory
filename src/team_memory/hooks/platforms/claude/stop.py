"""Claude Code Stop hook — captures agent response for TM draft.

Fires when Claude finishes responding.

Input (from Claude Code 2.1.112):
  {
    "session_id": "...",
    "last_assistant_message": "...",
    "cwd": "/path/to/project",
    "transcript_path": "..."
  }

Output:
  {} (no blocking)
"""

from team_memory.hooks.core.daemon_http import post_hook_fire_and_forget
from team_memory.hooks.core.output import print_empty
from team_memory.hooks.core.stdin import parse_stdin_json


def main() -> None:
    input_data = parse_stdin_json()
    if not input_data:
        print_empty()
        return

    session_id = input_data.get("session_id", "")
    workspace = input_data.get("cwd", "") or input_data.get("workspace", "") or input_data.get("workingDirectory", "")
    response_text = input_data.get("last_assistant_message", "")

    if not response_text:
        transcript = input_data.get("transcript", [])
        for msg in reversed(transcript):
            if msg.get("role") == "assistant":
                response_text = msg.get("content", "")
                break

    if not response_text:
        print_empty()
        return

    payload = {
        "conversation_id": session_id,
        "response_text": response_text,
        "workspace_roots": [workspace] if workspace else [],
    }

    post_hook_fire_and_forget("after_response", payload)
    print_empty()


if __name__ == "__main__":
    main()

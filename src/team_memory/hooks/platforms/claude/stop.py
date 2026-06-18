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


def _debug_log(msg: str) -> None:
    import os
    from datetime import datetime

    log_path = os.path.expanduser("~/.cache/tm-pipeline/stop_hook_debug.log")
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    with open(log_path, "a") as f:
        f.write(f"[{datetime.now().isoformat()}] {msg}\n")


def main() -> None:
    input_data = parse_stdin_json()
    _debug_log(f"input_keys={list(input_data.keys()) if input_data else 'EMPTY'}")
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

    # Fallback: read from transcript_path (version-adaptive)
    if not response_text:
        transcript_path = input_data.get("transcript_path", "")
        if transcript_path:
            from team_memory.hooks.core.transcript import read_last_assistant_from_transcript

            response_text = read_last_assistant_from_transcript(transcript_path)

    if not response_text:
        print_empty()
        return

    payload = {
        "conversation_id": session_id,
        "response_text": response_text,
        "workspace_roots": [workspace] if workspace else [],
    }

    try:
        post_hook_fire_and_forget("after_response", payload)
        _debug_log(f"post_hook OK session={session_id[:12]} text_len={len(response_text)}")
    except Exception as e:
        _debug_log(f"post_hook FAILED: {e}")
    print_empty()


if __name__ == "__main__":
    main()

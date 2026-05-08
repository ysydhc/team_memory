"""Claude Code Stop hook — captures agent response for TM draft.

Fires when Claude finishes responding. We use this instead of PostToolUse
because we want the full response text, not individual tool outputs.

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
import json
import sys

import httpx

DAEMON_URL = "http://127.0.0.1:3901"


def main() -> None:
    try:
        input_data = json.load(sys.stdin)
    except Exception:
        print("{}")
        return

    session_id = input_data.get("session_id", "")
    # Claude Code 2.1.112 uses "cwd" not "workspace"
    workspace = input_data.get("cwd", "") or input_data.get("workspace", "") or input_data.get("workingDirectory", "")
    # Claude Code 2.1.112 uses "last_assistant_message" not "transcript"
    response_text = input_data.get("last_assistant_message", "")

    # Fallback: try to extract from transcript array if present
    if not response_text:
        transcript = input_data.get("transcript", [])
        for msg in reversed(transcript):
            if msg.get("role") == "assistant":
                response_text = msg.get("content", "")
                break

    if not response_text:
        print("{}")
        return

    payload = {
        "conversation_id": session_id,
        "response_text": response_text,
        "workspace_roots": [workspace] if workspace else [],
    }

    try:
        httpx.post(f"{DAEMON_URL}/hooks/after_response", json=payload, timeout=10.0)
    except Exception:
        pass

    print("{}")


if __name__ == "__main__":
    main()

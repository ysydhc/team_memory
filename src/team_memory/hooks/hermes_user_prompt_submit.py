"""Hermes UserPromptSubmit hook - forwards to TM Daemon before_prompt."""
import json, sys, os, httpx

DAEMON_URL = "http://127.0.0.1:3901"

def main():
    try:
        data = json.load(sys.stdin)
    except Exception:
        print("{}")
        return
    session_id = data.get("session_id", "")
    prompt = data.get("prompt", "") or data.get("user_message", "")
    cwd = data.get("cwd", "") or data.get("working_directory", "")
    if not prompt:
        print("{}")
        return
    payload = {
        "conversation_id": session_id,
        "prompt": prompt,
        "workspace_roots": [cwd] if cwd else [],
    }
    try:
        httpx.post(f"{DAEMON_URL}/hooks/before_prompt", json=payload, timeout=10.0)
    except Exception:
        pass
    print("{}")

if __name__ == "__main__":
    main()

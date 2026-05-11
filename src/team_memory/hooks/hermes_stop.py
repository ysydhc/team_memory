"""Hermes Stop hook - forwards assistant response to TM Daemon after_response."""
import json, sys, os, httpx

DAEMON_URL = "http://127.0.0.1:3901"

def main():
    try:
        data = json.load(sys.stdin)
    except Exception:
        print("{}")
        return
    session_id = data.get("session_id", "")
    response = data.get("response", "") or data.get("response_text", "")
    cwd = data.get("cwd", "") or data.get("working_directory", "")
    if not response:
        print("{}")
        return
    payload = {
        "conversation_id": session_id,
        "response_text": response,
        "workspace_roots": [cwd] if cwd else [],
    }
    try:
        httpx.post(f"{DAEMON_URL}/hooks/after_response", json=payload, timeout=10.0)
    except Exception:
        pass
    print("{}")

if __name__ == "__main__":
    main()

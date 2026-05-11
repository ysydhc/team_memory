"""Cursor beforeSubmitPrompt hook — forwards to TM Daemon and injects context."""
import json
import sys

import httpx

DAEMON_URL = "http://127.0.0.1:3901"


def main() -> None:
    try:
        input_data = json.load(sys.stdin)
    except Exception:
        print(json.dumps({"action": "error", "message": "invalid input"}))
        return

    try:
        resp = httpx.post(
            f"{DAEMON_URL}/hooks/before_prompt",
            json=input_data,
            timeout=10.0,
        )
        result = resp.json()
        context = result.get("context", "")
        # Inject TM context into Cursor's prompt
        if context:
            print(json.dumps({"action": "ok", "context": context}))
        else:
            print(resp.text)
    except httpx.ConnectError:
        print(json.dumps({"action": "ok", "message": "daemon not running"}))
    except Exception as e:
        print(json.dumps({"action": "error", "message": str(e)}))


if __name__ == "__main__":
    main()

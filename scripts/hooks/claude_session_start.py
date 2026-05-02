"""Claude Code SessionStart hook — forwards to TM Daemon.

Input (from Claude Code):
  {"session_id": "...", "workspace": "/path/to/project"}

Output:
  {"additionalContext": "<retrieved context>"} or {}
"""
import json
import sys

import httpx

DAEMON_URL = "http://127.0.0.1:3901"


def main() -> None:
    try:
        input_data = json.load(sys.stdin)
    except Exception:
        # No valid input, silently pass
        sys.exit(0)

    # Extract workspace from Claude Code input
    workspace = input_data.get("workspace", "") or input_data.get("workingDirectory", "")
    session_id = input_data.get("session_id", "")

    if not workspace:
        sys.exit(0)

    payload = {
        "workspace_roots": [workspace] if workspace else [],
        "conversation_id": session_id,
    }

    try:
        resp = httpx.post(f"{DAEMON_URL}/hooks/session_start", json=payload, timeout=10.0)
        result = resp.json()
        # Daemon returns additional_context with profile + experiences
        add_ctx = result.get("additional_context", {})
        experiences = add_ctx.get("relevant_experiences", [])
        profile = add_ctx.get("profile", {})
        static = profile.get("static", [])
        dynamic = profile.get("dynamic", [])

        parts = []
        if experiences:
            for exp in experiences[:5]:
                title = exp.get("title", "Untitled")
                eid = exp.get("id", "")[:8]
                parts.append(f"[mem:{eid}] {title}")
        if static:
            parts.append(f"Static context: {json.dumps(static)}")
        if dynamic:
            parts.append(f"Dynamic context: {json.dumps(dynamic)}")

        if parts:
            context = "\n".join(parts)
            print(json.dumps({"additionalContext": f"[mem:context] project={result.get('project', '?')}\n{context}"}))
        else:
            print("{}")
    except httpx.ConnectError:
        print("{}")
    except Exception:
        print("{}")


if __name__ == "__main__":
    main()

"""Claude Code SessionStart hook — forwards to TM Daemon.

Input (from Claude Code):
  {"session_id": "...", "workspace": "/path/to/project"}

Output:
  {"additionalContext": "<retrieved context>"} or {}
"""

import json

from team_memory.hooks.core.daemon_http import post_hook
from team_memory.hooks.core.output import print_claude_context, print_empty
from team_memory.hooks.core.stdin import parse_stdin_json


def main() -> None:
    input_data = parse_stdin_json()
    if not input_data:
        return

    workspace = input_data.get("workspace", "") or input_data.get("workingDirectory", "")
    session_id = input_data.get("session_id", "")

    if not workspace:
        return

    payload = {
        "workspace_roots": [workspace] if workspace else [],
        "conversation_id": session_id,
    }

    try:
        resp = post_hook("session_start", payload)
        result = resp.json()
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
            print_claude_context(f"[mem:context] project={result.get('project', '?')}\n{context}")
        else:
            print_empty()
    except Exception:
        print_empty()


if __name__ == "__main__":
    main()

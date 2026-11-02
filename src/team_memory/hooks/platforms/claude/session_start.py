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

    workspace = (
        input_data.get("cwd", "")
        or input_data.get("workspace", "")
        or input_data.get("workingDirectory", "")
    )
    session_id = input_data.get("session_id", "")

    if not workspace:
        print_empty()
        return

    payload = {
        "workspace_roots": [workspace] if workspace else [],
        "conversation_id": session_id,
    }

    try:
        resp = post_hook("session_start", payload)
        result = resp.json()
        top_l0 = result.get("l0_manifest", "")
        if isinstance(top_l0, str) and top_l0.strip():
            print_claude_context(top_l0.strip())
            return

        add_ctx = result.get("additional_context", {})
        if isinstance(add_ctx, str):
            if add_ctx.strip():
                print_claude_context(add_ctx.strip())
            else:
                print_empty()
            return

        if not isinstance(add_ctx, dict):
            print_empty()
            return

        l0_manifest = add_ctx.get("l0_manifest", "")
        experiences = add_ctx.get("relevant_experiences", [])
        profile = add_ctx.get("profile", {}) or {}
        static = profile.get("static", [])
        dynamic = profile.get("dynamic", [])

        parts = []
        if l0_manifest:
            parts.append(l0_manifest)
        elif experiences:
            for exp in experiences[:5]:
                exp_id = str(exp.get("id", "") or "")[:8]
                title = exp.get("title", "Untitled")
                prefix = f"[tm://exp/{exp_id}] " if exp_id else ""
                parts.append(f"{prefix}{title}")
        if static:
            parts.append(f"Static context: {json.dumps(static)}")
        if dynamic:
            parts.append(f"Dynamic context: {json.dumps(dynamic)}")

        if parts:
            context = "\n".join(parts)
            if l0_manifest:
                print_claude_context(context)
            else:
                print_claude_context(f"TM context project={result.get('project', '?')}\n{context}")
        else:
            print_empty()
    except Exception:
        print_empty()


if __name__ == "__main__":
    main()

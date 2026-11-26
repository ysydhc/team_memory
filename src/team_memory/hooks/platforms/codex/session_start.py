"""OpenAI Codex CLI SessionStart hook — injects L0 project memory manifest.

Input (from Codex CLI stdin):
  {
    "session_id": "...",
    "cwd": "/path/to/project",
    "workspace_roots": ["/path/to/project"]
  }

Output:
  {"additionalContext": "<L0 manifest>"} or {}
"""

from __future__ import annotations

import json

from team_memory.hooks.core.daemon_http import post_hook
from team_memory.hooks.core.output import print_claude_context, print_empty
from team_memory.hooks.core.stdin import parse_stdin_json


def _extract_workspaces(input_data: dict) -> list[str]:
    ws = input_data.get("workspace_roots") or input_data.get("workspacePaths") or []
    if isinstance(ws, str):
        ws = [ws]
    single = input_data.get("cwd") or input_data.get("workspace") or input_data.get("workingDirectory")
    if single and single not in ws:
        ws = [single, *ws]
    return [str(p) for p in ws if p]


def main() -> None:
    input_data = parse_stdin_json()
    if not input_data:
        print_empty()
        return

    workspace_roots = _extract_workspaces(input_data)
    session_id = input_data.get("session_id") or input_data.get("thread_id") or input_data.get("conversation_id") or ""
    if not workspace_roots:
        print_empty()
        return

    payload = {
        "workspace_roots": workspace_roots,
        "conversation_id": session_id,
        "source_platform": "codex",
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
        if l0_manifest:
            print_claude_context(l0_manifest)
            return

        experiences = add_ctx.get("relevant_experiences", [])
        profile = add_ctx.get("profile", {}) or {}
        parts: list[str] = []
        for exp in experiences[:5]:
            exp_id = str(exp.get("id", "") or "")[:8]
            title = exp.get("title", "Untitled")
            prefix = f"[tm://exp/{exp_id}] " if exp_id else ""
            parts.append(f"{prefix}{title}")
        if profile.get("static"):
            parts.append(f"Static context: {json.dumps(profile['static'])}")
        if profile.get("dynamic"):
            parts.append(f"Dynamic context: {json.dumps(profile['dynamic'])}")

        if parts:
            context = "\n".join(parts)
            print_claude_context(f"TM context project={result.get('project', '?')}\n{context}")
        else:
            print_empty()
    except Exception:
        print_empty()


if __name__ == "__main__":
    main()

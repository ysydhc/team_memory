"""Antigravity SessionStart hook — injects L0 project memory manifest.

Input (from Antigravity stdin):
  {
    "conversationId": "...",
    "workspacePaths": ["/path/to/workspace"],
    "transcriptPath": ".../transcript.jsonl",
    "modelName": "..."
  }

Output:
  {"injectSteps": [{"systemMessage": {"systemMessage": "<L0 manifest>"}}]} or {}
"""

from __future__ import annotations

import json

from team_memory.hooks.core.daemon_http import post_hook
from team_memory.hooks.core.output import print_antigravity_system_message, print_empty
from team_memory.hooks.core.stdin import parse_stdin_json


def _extract_workspaces(input_data: dict) -> list[str]:
    ws = input_data.get("workspacePaths") or input_data.get("workspace_roots") or []
    if isinstance(ws, str):
        ws = [ws]
    single = input_data.get("workspace") or input_data.get("cwd")
    if single and single not in ws:
        ws = [single, *ws]
    return [str(p) for p in ws if p]


def main() -> None:
    input_data = parse_stdin_json()
    if not input_data:
        print_empty()
        return

    workspace_roots = _extract_workspaces(input_data)
    conversation_id = input_data.get("conversationId") or input_data.get("conversation_id") or ""
    if not workspace_roots:
        print_empty()
        return

    payload = {
        "workspace_roots": workspace_roots,
        "conversation_id": conversation_id,
        "source_platform": "antigravity",
    }

    try:
        resp = post_hook("session_start", payload)
        result = resp.json()
        l0_manifest = result.get("l0_manifest", "")
        add_ctx = result.get("additional_context", {})

        if l0_manifest:
            print_antigravity_system_message(l0_manifest)
            return

        if isinstance(add_ctx, str) and add_ctx.strip():
            print_antigravity_system_message(add_ctx.strip())
            return

        if isinstance(add_ctx, dict):
            manifest = add_ctx.get("l0_manifest", "")
            if manifest:
                print_antigravity_system_message(manifest)
                return
            experiences = add_ctx.get("relevant_experiences", [])
            profile = add_ctx.get("profile", {}) or {}
            parts = []
            for exp in experiences[:5]:
                exp_id = str(exp.get("id", "") or "")[:8]
                title = exp.get("title", "Untitled")
                prefix = f"[tm://exp/{exp_id}] " if exp_id else ""
                parts.append(f"{prefix}{title}")
            if profile.get("static"):
                parts.append(f"Static context: {json.dumps(profile['static'])}")
            if parts:
                proj = result.get("project", "?")
                print_antigravity_system_message(f"[TM L0 Manifest | project={proj}]\n" + "\n".join(parts))
                return

        print_empty()
    except Exception:
        print_empty()


if __name__ == "__main__":
    main()

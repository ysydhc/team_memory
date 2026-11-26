"""OpenAI Codex CLI UserPromptSubmit hook — retrieves L1 memories for user prompt.

Input (from Codex CLI stdin):
  {
    "session_id": "...",
    "prompt": "user prompt text",
    "cwd": "/path/to/project"
  }

Output:
  {"additionalContext": "<L1 retrieved context>"} or {}
"""

from __future__ import annotations

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

    prompt = str(input_data.get("prompt") or input_data.get("user_prompt") or "")
    session_id = str(
        input_data.get("session_id") or input_data.get("thread_id") or input_data.get("conversation_id") or ""
    )
    workspace_roots = _extract_workspaces(input_data)

    payload = {
        "prompt": prompt,
        "workspace_roots": workspace_roots,
        "conversation_id": session_id,
        "source_platform": "codex",
    }

    try:
        resp = post_hook("before_prompt", payload)
        result = resp.json()
        l1_context = result.get("l1_context") or result.get("context") or ""
        if isinstance(l1_context, str) and l1_context.strip():
            print_claude_context(l1_context.strip())
            return

        results = result.get("results", [])
        if results:
            context_parts: list[str] = []
            for r in results[:3]:
                if r.get("l1_summary"):
                    context_parts.append(str(r["l1_summary"]))
                else:
                    exp_id = str(r.get("id", "") or "")[:8]
                    uri = f"tm://exp/{exp_id}" if exp_id else "tm://exp"
                    title = r.get("title", "Untitled")
                    body = (
                        r.get("summary")
                        or r.get("solution")
                        or r.get("description")
                        or r.get("problem")
                        or r.get("content")
                        or ""
                    )
                    cleaned = " ".join(str(body).split())[:180]
                    context_parts.append(f"- [{uri}] {title}: {cleaned}" if cleaned else f"- [{uri}] {title}")
            print_claude_context("[TM L1 Context]\n" + "\n".join(context_parts))
        else:
            print_empty()
    except Exception:
        print_empty()


if __name__ == "__main__":
    main()

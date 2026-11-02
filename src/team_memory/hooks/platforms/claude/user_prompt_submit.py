"""Claude Code UserPromptSubmit hook — retrieves memories for user prompt.

Fires when user submits a prompt, before Claude processes it.
Always triggers (no keyword filter) to ensure T1 timestamp is recorded
for MCP recall correlation in after_response.

Input (from Claude Code):
  {"session_id": "...", "prompt": "user text", "workspace": "/path"}

Output:
  {"additionalContext": "<retrieved context>"} or {}
"""

import json
import sys

from team_memory.hooks.core.daemon_http import post_hook
from team_memory.hooks.core.output import print_claude_context, print_empty


def main() -> None:
    with open("/tmp/tm-hook-trace.log", "a") as f:
        f.write(f"[{__file__.split('/')[-1]}] started at {__import__('datetime').datetime.now().isoformat()}\n")

    raw_stdin = sys.stdin.read()
    with open("/tmp/tm-hook-trace.log", "a") as f:
        f.write(f"  stdin_preview={raw_stdin[:200]!r}\n")

    try:
        input_data = json.loads(raw_stdin)
    except Exception as e:
        with open("/tmp/tm-hook-trace.log", "a") as f:
            f.write(f"  JSON_PARSE_ERROR: {e}\n")
        print_empty()
        return

    prompt = input_data.get("prompt", "")
    session_id = input_data.get("session_id", "")
    workspace = input_data.get("cwd", "") or input_data.get("workspace", "") or input_data.get("workingDirectory", "")

    with open("/tmp/tm-hook-trace.log", "a") as f:
        f.write(f"  prompt={prompt[:50]!r} workspace={workspace!r}\n")

    payload = {
        "prompt": prompt,
        "workspace_roots": [workspace] if workspace else [],
        "conversation_id": session_id,
    }

    try:
        resp = post_hook("before_prompt", payload)
        result = resp.json()
        results = result.get("results", [])
        l1_context = result.get("l1_context") or result.get("context") or ""
        with open("/tmp/tm-hook-trace.log", "a") as f:
            f.write(f"  daemon_resp_results={len(results)}\n")
        if l1_context:
            print_claude_context(l1_context)
        elif results:
            context_parts = []
            for r in results[:3]:
                if r.get("l1_summary"):
                    context_parts.append(r["l1_summary"])
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
            context = "[TM L1 Context]\n" + "\n".join(context_parts)
            print_claude_context(context)
        else:
            print_empty()
    except Exception as e:
        with open("/tmp/tm-hook-trace.log", "a") as f:
            f.write(f"  DAEMON_ERROR: {type(e).__name__}: {e}\n")
        print_empty()


if __name__ == "__main__":
    main()

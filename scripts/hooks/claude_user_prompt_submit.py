"""Claude Code UserPromptSubmit hook — retrieves memories for user prompt.

Fires when user submits a prompt, before Claude processes it.

Input (from Claude Code):
  {"session_id": "...", "prompt": "user text", "workspace": "/path"}

Output:
  {"additionalContext": "<retrieved context>"} or {}
"""
import json
import sys

import httpx

DAEMON_URL = "http://127.0.0.1:3901"

# Trigger keywords for retrieval
TRIGGER_KEYWORDS = ["之前", "上次", "以前", "经验", "遇到过", "怎么解决", "历史", "记得"]


def should_trigger(prompt: str) -> bool:
    return any(kw in prompt for kw in TRIGGER_KEYWORDS)


def main() -> None:
    # Trace: confirm hook is actually executed by Claude Code
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
        print("{}")
        sys.stdout.flush()
        return

    prompt = input_data.get("prompt", "")
    session_id = input_data.get("session_id", "")
    # Claude Code 2.1.112 uses "cwd" not "workspace"
    workspace = input_data.get("cwd", "") or input_data.get("workspace", "") or input_data.get("workingDirectory", "")

    with open("/tmp/tm-hook-trace.log", "a") as f:
        f.write(f"  prompt={prompt[:50]!r} workspace={workspace!r} trigger={should_trigger(prompt)}\n")

    # Only trigger on keyword matches
    if not should_trigger(prompt):
        print("{}")
        sys.stdout.flush()
        return

    payload = {
        "prompt": prompt,
        "workspace_roots": [workspace] if workspace else [],
        "conversation_id": session_id,
    }

    try:
        resp = httpx.post(f"{DAEMON_URL}/hooks/before_prompt", json=payload, timeout=10.0)
        result = resp.json()
        results = result.get("results", [])
        with open("/tmp/tm-hook-trace.log", "a") as f:
            f.write(f"  daemon_resp_results={len(results)}\n")
        if results:
            context_parts = []
            for r in results[:5]:
                title = r.get("title", "Untitled")
                content = r.get("content", "")[:500]
                rid = r.get("id", "")[:8]
                context_parts.append(f"[mem:{rid}] {title}\n{content}")
            context = "\n---\n".join(context_parts)
            out = json.dumps({"additionalContext": context})
            print(out)
        else:
            print("{}")
    except httpx.ConnectError as e:
        with open("/tmp/tm-hook-trace.log", "a") as f:
            f.write(f"  DAEMON_CONNECT_ERROR: {e}\n")
        print("{}")
    except Exception as e:
        with open("/tmp/tm-hook-trace.log", "a") as f:
            f.write(f"  DAEMON_OTHER_ERROR: {type(e).__name__}: {e}\n")
        print("{}")
    sys.stdout.flush()


if __name__ == "__main__":
    main()

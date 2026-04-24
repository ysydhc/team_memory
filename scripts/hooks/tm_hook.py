#!/usr/bin/env python3
"""tm-hook — TM Daemon 命令行客户端。

Hermes Agent 通过 terminal 工具调用此脚本，实现记忆自动读写。

用法:
  tm-hook.py session-start [--project NAME] [--workspace PATH]
  tm-hook.py before-prompt MESSAGE [--project NAME] [--workspace PATH]
  tm-hook.py after-response CONVERSATION_ID RESPONSE_TEXT [--project NAME] [--workspace PATH]
  tm-hook.py session-end CONVERSATION_ID
  tm-hook.py recall QUERY [--project NAME] [--max-results N]
  tm-hook.py save --title TITLE --problem PROBLEM --solution SOLUTION [--project NAME] [--tags TAG1,TAG2]
  tm-hook.py status

所有命令失败时静默退出（exit 0 + stderr 输出），不会阻断 Agent 流程。
"""
from __future__ import annotations

import argparse
import json
import sys
import urllib.parse
import urllib.request
import urllib.error

DAEMON_URL = "http://127.0.0.1:3901"


def _post(endpoint: str, payload: dict) -> dict | None:
    """POST JSON to daemon, return parsed response or None on failure."""
    url = f"{DAEMON_URL}{endpoint}"
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        url, data=data, headers={"Content-Type": "application/json"}
    )
    try:
        with urllib.request.urlopen(req, timeout=5) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except urllib.error.URLError:
        # Daemon not running — silent fallback
        return None
    except Exception as e:
        print(f"tm-hook error: {e}", file=sys.stderr)
        return None


def _get(endpoint: str, params: dict | None = None) -> dict | list | None:
    """GET from daemon, return parsed response or None on failure."""
    query = ""
    if params:
        encoded = urllib.parse.urlencode({k: v for k, v in params.items() if v is not None})
        if encoded:
            query = "?" + encoded
    url = f"{DAEMON_URL}{endpoint}{query}"
    try:
        with urllib.request.urlopen(url, timeout=5) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except urllib.error.URLError:
        return None
    except Exception as e:
        print(f"tm-hook error: {e}", file=sys.stderr)
        return None


def _format_recall_results(results: list[dict]) -> str:
    """Format recall results for agent consumption."""
    if not results:
        return "No relevant memories found."

    lines = []
    for i, r in enumerate(results, 1):
        title = r.get("title", "Untitled")
        problem = r.get("problem", "")
        solution = r.get("solution", "")
        tags = r.get("tags", [])
        project = r.get("project", "")

        lines.append(f"[{i}] {title}")
        if project:
            lines.append(f"    project: {project}")
        if tags:
            lines.append(f"    tags: {', '.join(tags) if isinstance(tags, list) else tags}")
        if problem:
            lines.append(f"    problem: {problem[:200]}")
        if solution:
            lines.append(f"    solution: {solution[:300]}")
        lines.append("")

    return "\n".join(lines)


def cmd_status(_args: argparse.Namespace) -> None:
    """Check daemon status."""
    result = _get("/status")
    if result is None:
        print("TM Daemon: not running")
    else:
        print(f"TM Daemon: {result.get('status', 'unknown')} (mode={result.get('tm_mode', 'unknown')})")


def cmd_session_start(args: argparse.Namespace) -> None:
    """Retrieve project context at session start."""
    workspace = args.workspace or ""
    payload = {
        "workspace_roots": [workspace] if workspace else [],
    }
    result = _post("/hooks/session_start", payload)
    if result is None:
        return  # Daemon not running, silent

    context = result.get("additional_context", "")
    project = result.get("project", "")
    if context:
        print(f"[mem:context] project={project}")
        print(context)
    else:
        print(f"[mem:context] project={project} — no relevant context")


def cmd_before_prompt(args: argparse.Namespace) -> None:
    """Retrieve relevant memories before user prompt."""
    workspace = args.workspace or ""
    payload = {
        "prompt": args.message,
        "workspace_roots": [workspace] if workspace else [],
        "project": args.project or "",
    }
    result = _post("/hooks/before_prompt", payload)
    if result is None:
        return

    results = result.get("results", [])
    project = result.get("project", "")
    if results:
        print(f"[mem:recall] project={project} — {len(results)} results")
        print(_format_recall_results(results))
    else:
        # No results is normal, don't clutter output
        pass


def cmd_after_response(args: argparse.Namespace) -> None:
    """Capture agent response for draft pipeline."""
    workspace = args.workspace or ""
    payload = {
        "conversation_id": args.conversation_id,
        "prompt": args.response_text,
        "workspace_roots": [workspace] if workspace else [],
        "project": args.project or "",
    }
    result = _post("/hooks/after_response", payload)
    if result is None:
        return

    action = result.get("action", "")
    convergence = result.get("convergence", False)

    # Only print if something interesting happened
    if convergence and action == "published":
        print(f"[mem:publish] draft published (convergence detected)")
    elif action == "draft_saved":
        pass  # Normal, silent


def cmd_session_end(args: argparse.Namespace) -> None:
    """Flush remaining drafts at session end."""
    payload = {
        "conversation_id": args.conversation_id,
    }
    result = _post("/hooks/session_end", payload)
    if result is None:
        return

    flushed = result.get("flushed", False)
    if flushed:
        print(f"[mem:flush] remaining draft published")


def cmd_recall(args: argparse.Namespace) -> None:
    """Direct recall query."""
    params = {
        "query": args.query,
        "project": args.project,
        "max_results": str(args.max_results),
    }
    result = _get("/recall", params)
    if result is None:
        print("TM Daemon not running, cannot recall.")
        return

    if isinstance(result, list):
        print(_format_recall_results(result))
    else:
        print("No results.")


def cmd_save(args: argparse.Namespace) -> None:
    """Direct save experience."""
    payload = {
        "title": args.title,
        "problem": args.problem,
        "solution": args.solution,
        "project": args.project or "",
        "tags": args.tags.split(",") if args.tags else [],
    }
    # Use the draft/save + draft/publish flow via daemon
    result = _post("/draft/save", payload)
    if result is None:
        print("TM Daemon not running, cannot save.")
        return

    draft_id = result.get("id", "")
    if draft_id:
        # Auto-publish
        pub_result = _post("/draft/publish", {"draft_id": draft_id})
        if pub_result:
            print(f"[mem:saved] published: {args.title}")
        else:
            print(f"[mem:saved] draft saved (not published): {args.title}")
    else:
        print(f"[mem:saved] draft saved: {args.title}")


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="tm-hook",
        description="TM Daemon CLI client for Hermes Agent integration",
    )
    sub = parser.add_subparsers(dest="command")

    # status
    sub.add_parser("status", help="Check daemon status")

    # session-start
    p_ss = sub.add_parser("session-start", help="Retrieve project context")
    p_ss.add_argument("--project", default="")
    p_ss.add_argument("--workspace", default="")

    # before-prompt
    p_bp = sub.add_parser("before-prompt", help="Retrieve memories for user prompt")
    p_bp.add_argument("message", help="User's prompt text")
    p_bp.add_argument("--project", default="")
    p_bp.add_argument("--workspace", default="")

    # after-response
    p_ar = sub.add_parser("after-response", help="Capture agent response")
    p_ar.add_argument("conversation_id", help="Conversation/session ID")
    p_ar.add_argument("response_text", help="Agent's response text")
    p_ar.add_argument("--project", default="")
    p_ar.add_argument("--workspace", default="")

    # session-end
    p_se = sub.add_parser("session-end", help="Flush remaining drafts")
    p_se.add_argument("conversation_id", help="Conversation/session ID")

    # recall
    p_rc = sub.add_parser("recall", help="Direct recall query")
    p_rc.add_argument("query", help="Search query")
    p_rc.add_argument("--project", default="")
    p_rc.add_argument("--max-results", type=int, default=5)

    # save
    p_sv = sub.add_parser("save", help="Save experience directly")
    p_sv.add_argument("--title", required=True)
    p_sv.add_argument("--problem", required=True)
    p_sv.add_argument("--solution", required=True)
    p_sv.add_argument("--project", default="")
    p_sv.add_argument("--tags", default="")

    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        return

    commands = {
        "status": cmd_status,
        "session-start": cmd_session_start,
        "before-prompt": cmd_before_prompt,
        "after-response": cmd_after_response,
        "session-end": cmd_session_end,
        "recall": cmd_recall,
        "save": cmd_save,
    }

    handler = commands.get(args.command)
    if handler:
        handler(args)


if __name__ == "__main__":
    main()

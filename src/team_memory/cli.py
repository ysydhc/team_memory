"""tm-cli -- command-line interface for Team Memory.

Subcommands:
    archive      Create or update an archive (POST /api/v1/archives)
    upload       Upload a file attachment to an archive
    save         Save team knowledge (POST /api/v1/mcp/save)
    recall       Search team knowledge (POST /api/v1/mcp/recall)
    context      Profile + relevant knowledge (POST /api/v1/mcp/context)
    get-archive  Fetch full archive L2 by id (GET /api/v1/mcp/archive/{id})
    feedback     Rate an experience (POST /api/v1/mcp/feedback)

Usage:
    python -m team_memory.cli archive --title "..." --solution-doc "..." [options]
    python -m team_memory.cli upload --archive-id <id> --file <path> [options]
    python -m team_memory.cli save --title "..." --problem "..." --solution "..." [options]
    python -m team_memory.cli recall [--query "..."] [options]
    python -m team_memory.cli context [--file-paths "a.py,b.py"] [options]
    python -m team_memory.cli get-archive --id <uuid> [options]
    python -m team_memory.cli feedback --experience-id <uuid> --rating 5 [options]
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

import httpx


def _get_base_url() -> str:
    return os.environ.get("TM_BASE_URL", "http://localhost:9111")


def _get_api_key() -> str:
    key = os.environ.get("TEAM_MEMORY_API_KEY", "")
    if not key:
        print("Error: TEAM_MEMORY_API_KEY environment variable is not set.", file=sys.stderr)
        print("Set it with: export TEAM_MEMORY_API_KEY=your-key", file=sys.stderr)
        sys.exit(1)
    return key


def _headers() -> dict[str, str]:
    return {
        "Authorization": f"Bearer {_get_api_key()}",
        "Content-Type": "application/json",
    }


def _print_response_json(resp: httpx.Response) -> None:
    print(json.dumps(resp.json(), indent=2, ensure_ascii=False))


def _post_mcp_json(path: str, body: dict[str, Any], *, timeout: float = 60.0) -> None:
    url = f"{_get_base_url()}{path}"
    try:
        resp = httpx.post(url, json=body, headers=_headers(), timeout=timeout)
        resp.raise_for_status()
        _print_response_json(resp)
    except httpx.HTTPStatusError as e:
        print(f"Error {e.response.status_code}: {e.response.text}", file=sys.stderr)
        sys.exit(1)
    except httpx.ConnectError:
        print(f"Error: Cannot connect to {url}. Is the server running?", file=sys.stderr)
        sys.exit(1)


def _get_mcp_json(
    path: str, *, params: dict[str, str] | None = None, timeout: float = 30.0
) -> None:
    url = f"{_get_base_url()}{path}"
    try:
        resp = httpx.get(url, headers=_headers(), params=params or None, timeout=timeout)
        resp.raise_for_status()
        _print_response_json(resp)
    except httpx.HTTPStatusError as e:
        print(f"Error {e.response.status_code}: {e.response.text}", file=sys.stderr)
        sys.exit(1)
    except httpx.ConnectError:
        print(f"Error: Cannot connect to {url}. Is the server running?", file=sys.stderr)
        sys.exit(1)


def cmd_save(args: argparse.Namespace) -> None:
    """POST /api/v1/mcp/save — direct fields or long-form content."""
    has_direct = bool(args.title or args.problem or args.solution)
    has_content = bool(args.content)
    if has_content and has_direct:
        print(
            "Error: --content cannot be used together with --title/--problem/--solution.",
            file=sys.stderr,
        )
        sys.exit(1)
    if not has_content and not (args.title and args.problem and args.solution):
        print(
            "Error: use direct mode (--title, --problem, --solution) or --content.",
            file=sys.stderr,
        )
        sys.exit(1)

    body: dict[str, Any] = {}
    if has_content:
        body["content"] = args.content
    else:
        body["title"] = args.title
        body["problem"] = args.problem
        body["solution"] = args.solution
    if args.tags:
        body["tags"] = [t.strip() for t in args.tags.split(",") if t.strip()]
    if args.scope:
        body["scope"] = args.scope
    if args.experience_type:
        body["experience_type"] = args.experience_type
    if args.project:
        body["project"] = args.project
    if args.group_key:
        body["group_key"] = args.group_key

    _post_mcp_json("/api/v1/mcp/save", body, timeout=120.0)


def cmd_recall(args: argparse.Namespace) -> None:
    """POST /api/v1/mcp/recall."""
    body: dict[str, Any] = {}
    if args.query is not None:
        body["query"] = args.query
    if args.problem is not None:
        body["problem"] = args.problem
    if args.file_path is not None:
        body["file_path"] = args.file_path
    if args.language is not None:
        body["language"] = args.language
    if args.framework is not None:
        body["framework"] = args.framework
    if args.tags:
        body["tags"] = [t.strip() for t in args.tags.split(",") if t.strip()]
    body["max_results"] = args.max_results
    if args.project is not None:
        body["project"] = args.project
    if args.include_archives:
        body["include_archives"] = True
    if args.include_user_profile:
        body["include_user_profile"] = True

    _post_mcp_json("/api/v1/mcp/recall", body, timeout=120.0)


def cmd_context(args: argparse.Namespace) -> None:
    """POST /api/v1/mcp/context."""
    body: dict[str, Any] = {}
    if args.file_paths:
        body["file_paths"] = [p.strip() for p in args.file_paths.split(",") if p.strip()]
    if args.task_description is not None:
        body["task_description"] = args.task_description
    if args.project is not None:
        body["project"] = args.project

    _post_mcp_json("/api/v1/mcp/context", body, timeout=120.0)


def cmd_get_archive(args: argparse.Namespace) -> None:
    """GET /api/v1/mcp/archive/{archive_id}."""
    archive_id = args.archive_id
    params: dict[str, str] = {}
    if args.project is not None:
        params["project"] = args.project
    path = f"/api/v1/mcp/archive/{archive_id}"
    _get_mcp_json(path, params=params or None)


def cmd_feedback(args: argparse.Namespace) -> None:
    """POST /api/v1/mcp/feedback."""
    body: dict[str, Any] = {
        "experience_id": args.experience_id,
        "rating": args.rating,
    }
    if args.comment is not None:
        body["comment"] = args.comment

    _post_mcp_json("/api/v1/mcp/feedback", body)


def cmd_archive(args: argparse.Namespace) -> None:
    """Create or update an archive via POST /api/v1/archives."""
    body: dict = {
        "title": args.title,
        "solution_doc": args.solution_doc,
    }
    if args.content_type:
        body["content_type"] = args.content_type
    if args.value_summary:
        body["value_summary"] = args.value_summary
    if args.tags:
        body["tags"] = [t.strip() for t in args.tags.split(",") if t.strip()]
    if args.overview_file:
        body["overview"] = Path(args.overview_file).read_text(encoding="utf-8")
    elif args.overview:
        body["overview"] = args.overview
    if args.solution_file:
        body["solution_doc"] = Path(args.solution_file).read_text(encoding="utf-8")
    if args.summary:
        body["conversation_summary"] = args.summary
    if args.linked_experience_ids:
        body["linked_experience_ids"] = [
            s.strip() for s in args.linked_experience_ids.split(",") if s.strip()
        ]
    if args.project:
        body["project"] = args.project
    if args.scope:
        body["scope"] = args.scope
    if args.scope_ref:
        body["scope_ref"] = args.scope_ref

    url = f"{_get_base_url()}/api/v1/archives"
    try:
        resp = httpx.post(url, json=body, headers=_headers(), timeout=30.0)
        resp.raise_for_status()
        data = resp.json()
        item = data.get("item") if isinstance(data.get("item"), dict) else data
        action = item.get("action", "unknown")
        archive_id = item.get("archive_id", "?")
        print(f"Archive {action}: {archive_id}")
        if action == "updated" and item.get("previous_updated_at"):
            print(f"  (previous version from {item['previous_updated_at']})")
    except httpx.HTTPStatusError as e:
        print(f"Error {e.response.status_code}: {e.response.text}", file=sys.stderr)
        sys.exit(1)
    except httpx.ConnectError:
        print(f"Error: Cannot connect to {url}. Is the server running?", file=sys.stderr)
        sys.exit(1)


def cmd_upload(args: argparse.Namespace) -> None:
    """Upload a file attachment to an archive."""
    file_path = Path(args.file)
    if not file_path.is_file():
        print(f"Error: File not found: {file_path}", file=sys.stderr)
        sys.exit(1)

    url = f"{_get_base_url()}/api/v1/archives/{args.archive_id}/attachments/upload"
    params: dict[str, str] = {}
    if getattr(args, "project", None):
        params["project"] = args.project
    headers = {"Authorization": f"Bearer {_get_api_key()}"}
    files = {"file": (file_path.name, file_path.open("rb"), "application/octet-stream")}
    data = {"kind": args.kind or "file"}
    if args.snippet:
        data["note"] = args.snippet

    try:
        resp = httpx.post(
            url,
            headers=headers,
            files=files,
            data=data,
            params=params or None,
            timeout=60.0,
        )
        resp.raise_for_status()
        result = resp.json()
        print(f"Uploaded: {result.get('id', '?')}")
        if result.get("download_api_path"):
            print(f"  Download: {_get_base_url()}{result['download_api_path']}")
    except httpx.HTTPStatusError as e:
        print(f"Error {e.response.status_code}: {e.response.text}", file=sys.stderr)
        sys.exit(1)
    except httpx.ConnectError:
        print(f"Error: Cannot connect to {url}. Is the server running?", file=sys.stderr)
        sys.exit(1)


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="tm-cli",
        description="Team Memory CLI — archives, uploads, and MCP HTTP helpers",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    # archive subcommand
    p_arch = sub.add_parser("archive", help="Create or update an archive")
    p_arch.add_argument("--title", required=True, help="Archive title")
    p_arch.add_argument("--solution-doc", default="", help="L2 solution document text")
    p_arch.add_argument("--content-type", default="session_archive", help="Content type")
    p_arch.add_argument("--value-summary", help="One-line value summary")
    p_arch.add_argument("--tags", help="Comma-separated tags")
    p_arch.add_argument("--overview", help="L1 overview text")
    p_arch.add_argument("--overview-file", help="Read L1 overview from file")
    p_arch.add_argument("--solution-file", help="Read L2 solution from file")
    p_arch.add_argument("--summary", help="Conversation summary")
    p_arch.add_argument("--linked-experience-ids", help="Comma-separated experience UUIDs")
    p_arch.add_argument("--project", help="Project scope")
    p_arch.add_argument("--scope", default="session", help="Archive scope")
    p_arch.add_argument("--scope-ref", help="Scope reference")
    p_arch.set_defaults(func=cmd_archive)

    # upload subcommand
    p_up = sub.add_parser("upload", help="Upload a file attachment to an archive")
    p_up.add_argument("--archive-id", required=True, help="Target archive UUID")
    p_up.add_argument("--file", required=True, help="File path to upload")
    p_up.add_argument("--kind", default="file", help="Attachment kind")
    p_up.add_argument("--snippet", help="Optional snippet/note")
    p_up.add_argument(
        "--project",
        help="Project scope (query param; align with memory_archive_upsert / MCP env)",
    )
    p_up.set_defaults(func=cmd_upload)

    # save subcommand
    p_save = sub.add_parser("save", help="Save team knowledge (MCP save)")
    p_save.add_argument("--title", help="Title (direct mode)")
    p_save.add_argument("--problem", help="Problem (direct mode)")
    p_save.add_argument("--solution", help="Solution (direct mode)")
    p_save.add_argument(
        "--content",
        help="Long text for LLM parse mode (mutually exclusive with title/problem/solution)",
    )
    p_save.add_argument("--tags", help="Comma-separated tags")
    p_save.add_argument(
        "--scope",
        choices=("project", "personal"),
        default="project",
        help="Scope (default: project)",
    )
    p_save.add_argument(
        "--experience-type",
        help="Experience type (e.g. general, feature, bugfix)",
    )
    p_save.add_argument("--project", help="Project scope")
    p_save.add_argument("--group-key", help="Group key for deduplication")
    p_save.set_defaults(func=cmd_save)

    # recall subcommand
    p_recall = sub.add_parser("recall", help="Search team knowledge (MCP recall)")
    p_recall.add_argument("--query", help="Exploratory search query")
    p_recall.add_argument("--problem", help="Problem-focused recall")
    p_recall.add_argument("--file-path", help="File path for suggestions")
    p_recall.add_argument("--language", help="Language hint")
    p_recall.add_argument("--framework", help="Framework hint")
    p_recall.add_argument("--tags", help="Comma-separated tags")
    p_recall.add_argument("--max-results", type=int, default=5, help="Max results (default: 5)")
    p_recall.add_argument("--project", help="Project scope")
    p_recall.add_argument(
        "--include-archives",
        action="store_true",
        help="Include archive previews in results",
    )
    p_recall.add_argument(
        "--include-user-profile",
        action="store_true",
        help="Attach user profile to response",
    )
    p_recall.set_defaults(func=cmd_recall)

    # context subcommand
    p_ctx = sub.add_parser("context", help="Profile + relevant knowledge (MCP context)")
    p_ctx.add_argument("--file-paths", help="Comma-separated file paths")
    p_ctx.add_argument("--task-description", help="Current task description")
    p_ctx.add_argument("--project", help="Project scope")
    p_ctx.set_defaults(func=cmd_context)

    # get-archive subcommand
    p_ga = sub.add_parser("get-archive", help="Fetch full archive L2 by id (MCP get_archive)")
    p_ga.add_argument("--id", required=True, dest="archive_id", help="Archive UUID")
    p_ga.add_argument("--project", help="Project scope (query param)")
    p_ga.set_defaults(func=cmd_get_archive)

    # feedback subcommand
    p_fb = sub.add_parser("feedback", help="Rate an experience (MCP feedback)")
    p_fb.add_argument("--experience-id", required=True, help="Experience UUID")
    p_fb.add_argument("--rating", type=int, required=True, help="Rating 1–5")
    p_fb.add_argument("--comment", help="Optional comment")
    p_fb.set_defaults(func=cmd_feedback)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()

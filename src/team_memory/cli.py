"""tm-cli -- command-line interface for Team Memory archive operations.

Usage:
    python -m team_memory.cli archive --title "..." --solution-doc "..." [options]
    python -m team_memory.cli upload --archive-id <id> --file <path> [options]
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

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
        description="Team Memory CLI -- archive and upload operations",
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

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()

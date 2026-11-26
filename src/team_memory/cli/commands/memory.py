"""Memory, archive, upload, and session import subcommands for tm-cli."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

from team_memory.cli.http_client import (
    _get_api_key,
    _get_base_url,
    _get_mcp_json,
    _get_project,
    _get_queue_db_path,
    _headers,
    _httpx,
    _post_mcp_json,
)


def cmd_save(args: argparse.Namespace) -> None:
    """POST /api/v1/mcp/save — direct fields or long-form content."""
    has_direct = bool(args.title and (args.problem or args.solution))
    has_content = bool(args.content)
    if not has_direct and not has_content:
        print(
            "Error: use direct mode (--title + --problem/--solution) or --content.",
            file=sys.stderr,
        )
        sys.exit(1)

    body: dict[str, Any] = {}
    if args.title:
        body["title"] = args.title
    if args.problem:
        body["problem"] = args.problem
    if args.solution:
        body["solution"] = args.solution
    if args.content:
        body["content"] = args.content
    if args.tags:
        body["tags"] = [t.strip() for t in args.tags.split(",") if t.strip()]
    if args.scope:
        body["scope"] = args.scope
    if args.experience_type:
        body["experience_type"] = args.experience_type
    project = _get_project(args.project)
    if project:
        body["project"] = project
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
    project = _get_project(args.project)
    if project:
        body["project"] = project
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
    project = _get_project(args.project)
    if project:
        body["project"] = project

    _post_mcp_json("/api/v1/mcp/context", body, timeout=120.0)


def cmd_get_archive(args: argparse.Namespace) -> None:
    """GET /api/v1/mcp/archive/{archive_id}."""
    archive_id = args.archive_id
    params: dict[str, str] = {}
    project = _get_project(args.project)
    if project:
        params["project"] = project
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


def cmd_import(args: argparse.Namespace) -> None:
    """Import and distill historical sessions from ChatGPT or Antigravity."""
    import asyncio
    from datetime import datetime

    source = args.source
    since_ts: float | None = None
    if getattr(args, "since", None):
        try:
            since_ts = datetime.fromisoformat(args.since).timestamp()
        except ValueError:
            print(f"Error: Invalid --since date format '{args.since}' (expected YYYY-MM-DD)", file=sys.stderr)
            sys.exit(1)

    dry_run = bool(getattr(args, "dry_run", False))
    project = getattr(args, "project", None) or ""

    if source == "chatgpt":
        if not getattr(args, "file", None):
            print("Error: --file <path> is required when --source chatgpt is used", file=sys.stderr)
            sys.exit(1)

        from team_memory.daemon.refinement.chatgpt_importer import import_chatgpt_sessions

        async def _run_chatgpt() -> None:
            from team_memory.daemon.buffers.conversation_buffer import ConversationBuffer

            if dry_run:
                res = await import_chatgpt_sessions(
                    file_path=args.file,
                    project=project,
                    since_ts=since_ts,
                    dry_run=True,
                )
            else:
                async with ConversationBuffer(db_path=_get_queue_db_path()) as buf:
                    res = await import_chatgpt_sessions(
                        buf=buf,
                        file_path=args.file,
                        project=project,
                        since_ts=since_ts,
                        dry_run=False,
                    )
            print(
                f"[chatgpt {'dry-run' if dry_run else 'import'}] "
                f"Total: {res['total_conversations']} | Imported: {res['imported']} | Skipped: {res['skipped']}"
            )
            for s in res.get("sessions", [])[:10]:
                print(f"  - [{s['project']}] {s.get('title', s['session_id'])} (score={s.get('score')})")

        asyncio.run(_run_chatgpt())

    elif source == "antigravity":
        from team_memory.daemon.observe.agy_transcript_watcher import import_agy_sessions

        async def _run_agy() -> None:
            from team_memory.daemon.buffers.conversation_buffer import ConversationBuffer

            if dry_run:
                res = await import_agy_sessions(
                    file_path=getattr(args, "file", None),
                    project=project,
                    since_ts=since_ts,
                    dry_run=True,
                )
            else:
                async with ConversationBuffer(db_path=_get_queue_db_path()) as buf:
                    res = await import_agy_sessions(
                        buf=buf,
                        file_path=getattr(args, "file", None),
                        project=project,
                        since_ts=since_ts,
                        dry_run=False,
                    )
            print(
                f"[antigravity {'dry-run' if dry_run else 'import'}] "
                f"Discovered: {res['discovered']} | Imported: {res['imported']} | Skipped: {res['skipped']}"
            )
            for s in res.get("sessions", [])[:10]:
                print(f"  - [{s['project']}] {s['session_id']} ({s['turns_count']} turns, {s['tools_count']} tools)")

        asyncio.run(_run_agy())

    elif source == "codex":
        from team_memory.daemon.observe.codex_transcript_watcher import import_codex_sessions

        async def _run_codex() -> None:
            from team_memory.daemon.buffers.conversation_buffer import ConversationBuffer

            if dry_run:
                res = await import_codex_sessions(
                    file_path=getattr(args, "file", None),
                    project=project,
                    since_ts=since_ts,
                    dry_run=True,
                )
            else:
                async with ConversationBuffer(db_path=_get_queue_db_path()) as buf:
                    res = await import_codex_sessions(
                        buf=buf,
                        file_path=getattr(args, "file", None),
                        project=project,
                        since_ts=since_ts,
                        dry_run=False,
                    )
            print(
                f"[codex {'dry-run' if dry_run else 'import'}] "
                f"Discovered: {res['discovered']} | Imported: {res['imported']} | Skipped: {res['skipped']}"
            )
            for s in res.get("sessions", [])[:10]:
                print(f"  - [{s['project']}] {s['session_id']} ({s['turns_count']} turns, {s['tools_count']} tools)")

        asyncio.run(_run_codex())


def cmd_archive(args: argparse.Namespace) -> None:
    """Create or update an archive via POST /api/v1/archives."""
    hx = _httpx()
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
        body["linked_experience_ids"] = [s.strip() for s in args.linked_experience_ids.split(",") if s.strip()]
    project = _get_project(args.project)
    if project:
        body["project"] = project
    if args.scope:
        body["scope"] = args.scope
    if args.scope_ref:
        body["scope_ref"] = args.scope_ref

    url = f"{_get_base_url()}/api/v1/archives"
    try:
        resp = hx.post(url, json=body, headers=_headers(), timeout=30.0)
        resp.raise_for_status()
        data = resp.json()
        item = data.get("item") if isinstance(data.get("item"), dict) else data
        action = item.get("action", "unknown")
        archive_id = item.get("archive_id", "?")
        print(f"Archive {action}: {archive_id}")
        if action == "updated" and item.get("previous_updated_at"):
            print(f"  (previous version from {item['previous_updated_at']})")
    except hx.HTTPStatusError as e:
        print(f"Error {e.response.status_code}: {e.response.text}", file=sys.stderr)
        sys.exit(1)
    except hx.ConnectError:
        print(f"Error: Cannot connect to {url}. Is the server running?", file=sys.stderr)
        sys.exit(1)


def cmd_upload(args: argparse.Namespace) -> None:
    """Upload a file attachment to an archive."""
    hx = _httpx()
    file_path = Path(args.file)
    if not file_path.is_file():
        print(f"Error: File not found: {file_path}", file=sys.stderr)
        sys.exit(1)

    url = f"{_get_base_url()}/api/v1/archives/{args.archive_id}/attachments/upload"
    params: dict[str, str] = {}
    project = _get_project(getattr(args, "project", None))
    if project:
        params["project"] = project
    headers = {"Authorization": f"Bearer {_get_api_key()}"}
    files = {"file": (file_path.name, file_path.open("rb"), "application/octet-stream")}
    data = {"kind": args.kind or "file"}
    if args.snippet:
        data["note"] = args.snippet

    try:
        resp = hx.post(
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
    except hx.HTTPStatusError as e:
        print(f"Error {e.response.status_code}: {e.response.text}", file=sys.stderr)
        sys.exit(1)
    except hx.ConnectError:
        print(f"Error: Cannot connect to {url}. Is the server running?", file=sys.stderr)
        sys.exit(1)

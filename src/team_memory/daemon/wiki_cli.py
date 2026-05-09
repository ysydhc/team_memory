#!/usr/bin/env python3
"""Wiki Compile CLI — compile PG experiences into structured markdown wiki.

Usage:
    # Incremental compile (only new/changed experiences)
    python -m team_memory.daemon.wiki_cli compile [--full]

    # Show wiki status
    python -m team_memory.daemon.wiki_cli status

    # Show uncompiled experience IDs
    python -m team_memory.daemon.wiki_cli uncompiled
"""
from __future__ import annotations

import argparse
import asyncio
import os
import sys

from team_memory.project_resolver import find_project_root, load_dotenv_if_exists

load_dotenv_if_exists(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))


async def _fetch_published_experiences() -> list[dict]:
    """Fetch all published experiences from PG via team_memory service."""
    try:
        from team_memory.config import load_settings
        from team_memory.storage.database import get_session
        from team_memory.storage.models import Experience

        settings = load_settings()
        db_url = str(settings.database.url)

        experiences = []
        async with get_session(db_url) as session:
            from sqlalchemy import select
            stmt = (
                select(Experience)
                .where(Experience.exp_status == "published")
                .order_by(Experience.updated_at.desc())
            )
            result = await session.execute(stmt)
            rows = result.scalars().all()
            experiences = [exp.to_dict() for exp in rows]

        # Close engine to release connections
        from team_memory.storage.database import close_db
        await close_db()

        return experiences
    except Exception as e:
        print(f"[ERROR] Failed to fetch experiences from PG: {e}", file=sys.stderr)
        return []


def _default_wiki_root() -> str:
    """Default wiki root: <project_root>/wiki/"""
    start = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    return os.path.join(find_project_root(start), "wiki")


async def cmd_compile(full: bool = False, wiki_root: str | None = None) -> None:
    """Compile experiences into wiki pages."""
    from team_memory.daemon.wiki_compiler import WikiCompiler

    if wiki_root is None:
        wiki_root = _default_wiki_root()

    print("Fetching published experiences from PG...")
    experiences = await _fetch_published_experiences()
    if not experiences:
        print("No published experiences found.")
        return

    print(f"Found {len(experiences)} published experiences.")

    # Get PG URL for entity/embedding data
    from team_memory.config import load_settings as _load_settings
    settings = _load_settings()
    pg_url = str(settings.database.url)

    compiler = WikiCompiler(wiki_root=wiki_root)
    async with compiler:
        if full:
            print("Running full rebuild...")
            result = await compiler.full_rebuild(experiences, pg_url=pg_url)
        else:
            print("Running incremental compile...")
            result = await compiler.compile_incremental(experiences, pg_url=pg_url)

    print("\nCompile result:")
    print(f"  Created:  {result.created}")
    print(f"  Updated:  {result.updated}")
    print(f"  Skipped:  {result.skipped}")
    print(f"  Deleted:  {result.deleted}")
    print(f"  Errors:   {result.errors}")
    print(f"\nWiki root: {wiki_root}")


async def cmd_status(wiki_root: str | None = None) -> None:
    """Show wiki compilation status."""
    from team_memory.daemon.wiki_compiler import WikiCompiler

    if wiki_root is None:
        wiki_root = _default_wiki_root()

    compiler = WikiCompiler(wiki_root=wiki_root)
    async with compiler:
        stats = await compiler.get_stats()

    print("Wiki Status")
    print(f"  Root:     {wiki_root}")
    print(f"  Total:    {stats['total']} compiled pages")

    if stats["by_project"]:
        print("\n  By Project:")
        for proj, count in sorted(stats["by_project"].items(), key=lambda x: -x[1]):
            print(f"    {proj or '(none)':30s} {count:4d}")

    if stats["by_status"]:
        print("\n  By Status:")
        for status, count in stats["by_status"].items():
            print(f"    {status:20s} {count:4d}")

    # Check PG total
    pg_experiences = await _fetch_published_experiences()
    pg_count = len(pg_experiences)
    compiled_count = stats["total"]
    if compiled_count < pg_count:
        print(f"\n  ⚠ {pg_count - compiled_count} experiences not yet compiled")
    elif compiled_count == pg_count:
        print(f"\n  ✓ All {pg_count} experiences compiled")


async def cmd_uncompiled(wiki_root: str | None = None) -> None:
    """List experience IDs that are not yet compiled."""
    from team_memory.daemon.wiki_compiler import WikiCompiler

    if wiki_root is None:
        wiki_root = _default_wiki_root()

    experiences = await _fetch_published_experiences()
    all_ids = [str(exp["id"]) for exp in experiences]

    compiler = WikiCompiler(wiki_root=wiki_root)
    async with compiler:
        uncompiled = await compiler.get_uncompiled(all_ids)

    if uncompiled:
        print(f"{len(uncompiled)} uncompiled experience IDs:")
        for eid in uncompiled:
            print(f"  {eid}")
    else:
        print("All experiences are compiled.")


def main() -> None:
    parser = argparse.ArgumentParser(description="TM Wiki Compiler CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # compile
    compile_parser = subparsers.add_parser(
        "compile", help="Compile experiences into wiki pages",
    )
    compile_parser.add_argument(
        "--full", action="store_true",
        help="Full rebuild (clear cache first)",
    )
    compile_parser.add_argument(
        "--wiki-root", type=str, default=None,
        help="Wiki root directory",
    )

    # status
    status_parser = subparsers.add_parser("status", help="Show wiki compilation status")
    status_parser.add_argument("--wiki-root", type=str, default=None, help="Wiki root directory")

    # uncompiled
    uncompiled_parser = subparsers.add_parser(
        "uncompiled", help="List uncompiled experience IDs",
    )
    uncompiled_parser.add_argument(
        "--wiki-root", type=str, default=None,
        help="Wiki root directory",
    )

    args = parser.parse_args()

    if args.command == "compile":
        asyncio.run(cmd_compile(full=args.full, wiki_root=args.wiki_root))
    elif args.command == "status":
        asyncio.run(cmd_status(wiki_root=args.wiki_root))
    elif args.command == "uncompiled":
        asyncio.run(cmd_uncompiled(wiki_root=args.wiki_root))
    else:
        parser.print_help()


if __name__ == "__main__":
    main()

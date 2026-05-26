#!/usr/bin/env python3
"""Graph backfill CLI — ingest existing experiences and archives into Graphiti.

Usage:
    # Dry run
    python -m team_memory.daemon.backfill.graph_backfill --dry-run

    # Execute
    python -m team_memory.daemon.backfill.graph_backfill

    # With limit
    python -m team_memory.daemon.backfill.graph_backfill --limit 10
"""

from __future__ import annotations

import argparse
import asyncio
import os
import time

from team_memory.project_resolver import load_dotenv_if_exists

load_dotenv_if_exists(
    os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
)


async def run_backfill(dry_run: bool = False, limit: int = 0) -> None:
    """Backfill missing graph nodes and edges."""
    from sqlalchemy import text

    from team_memory.config import load_settings
    from team_memory.storage.database import get_session
    from team_memory.services.infra.task_runner import enqueue

    print("Loading settings...")
    settings = load_settings()
    db_url = str(settings.database.url)

    # Backfill experiences
    print("\n--- Backfilling Experiences to Graph ---")
    async with get_session(db_url) as session:
        result = await session.execute(
            text(
                """
            SELECT id, title, description, solution, tags
            FROM experiences
            WHERE exp_status = 'published'
            ORDER BY created_at ASC
        """
            )
        )
        rows = result.fetchall()

    if not rows:
        print("No published experiences found.")
    else:
        total = len(rows)
        if limit > 0:
            rows = rows[:limit]

        print(
            f"Found {total} published experiences{f' (processing {len(rows)})' if limit > 0 else ''}."
        )

        if dry_run:
            for row in rows[:10]:
                print(f"  {str(row[0])[:8]}: {(row[1] or '')[:60]}")
            if total > 10:
                print(f"  ... and {total - 10} more")
        else:
            succeeded = 0
            failed = 0
            start = time.monotonic()

            for i, row in enumerate(rows):
                exp_id = row[0]
                title = row[1] or ""
                description = row[2] or ""
                solution = row[3] or ""
                tags = row[4] or []

                content = f"{title}\n{description}\n{solution or ''}"
                if tags:
                    content += f"\n{' '.join(str(t) for t in tags)}"

                try:
                    await enqueue(
                        "graphiti_ingest",
                        {
                            "record_id": str(exp_id),
                            "content": content,
                            "source_type": "experience",
                        },
                        db_url=db_url,
                    )
                    succeeded += 1
                except Exception as e:
                    failed += 1
                    print(f"  Failed to submit task for {str(exp_id)[:8]}: {e}")

                if (i + 1) % 10 == 0 or i + 1 == len(rows):
                    elapsed = time.monotonic() - start
                    rate = (i + 1) / elapsed if elapsed > 0 else 0
                    print(
                        f"  Progress: {i + 1}/{len(rows)} ({rate:.1f}/s) — failed: {failed}"
                    )

            elapsed = time.monotonic() - start
            print(
                f"Experiences graph backfill tasks submitted in {elapsed:.1f}s: Succeeded {succeeded}, Failed {failed}"
            )

    # Backfill archives
    print("\n--- Backfilling Archives to Graph ---")
    async with get_session(db_url) as session:
        result = await session.execute(
            text(
                """
            SELECT id, title, overview, solution_doc
            FROM archives
            ORDER BY created_at ASC
        """
            )
        )
        archive_rows = result.fetchall()

    if not archive_rows:
        print("No archives found.")
    else:
        total = len(archive_rows)
        if limit > 0:
            archive_rows = archive_rows[:limit]

        print(
            f"Found {total} archives{f' (processing {len(archive_rows)})' if limit > 0 else ''}."
        )

        if not dry_run:
            succeeded = 0
            failed = 0
            start = time.monotonic()

            for i, row in enumerate(archive_rows):
                arch_id = row[0]
                title = row[1] or ""
                overview = row[2] or ""
                solution_doc = row[3] or ""

                content = f"{title}\n{overview}\n{solution_doc}"

                try:
                    await enqueue(
                        "graphiti_ingest",
                        {
                            "record_id": str(arch_id),
                            "content": content,
                            "source_type": "archive",
                        },
                        db_url=db_url,
                    )
                    succeeded += 1
                except Exception as e:
                    failed += 1
                    print(f"  Failed to submit task for {str(arch_id)[:8]}: {e}")

                if (i + 1) % 10 == 0 or i + 1 == len(archive_rows):
                    elapsed = time.monotonic() - start
                    rate = (i + 1) / elapsed if elapsed > 0 else 0
                    print(
                        f"  Progress: {i + 1}/{len(archive_rows)} ({rate:.1f}/s) — failed: {failed}"
                    )

            elapsed = time.monotonic() - start
            print(
                f"Archives graph backfill tasks submitted in {elapsed:.1f}s: Succeeded {succeeded}, Failed {failed}"
            )

    # Backfill personal memories
    print("\n--- Backfilling Personal Memories to Graph ---")
    async with get_session(db_url) as session:
        result = await session.execute(
            text(
                """
            SELECT id, content
            FROM personal_memories
            ORDER BY created_at ASC
        """
            )
        )
        pm_rows = result.fetchall()

    if not pm_rows:
        print("No personal memories found.")
    else:
        total = len(pm_rows)
        if limit > 0:
            pm_rows = pm_rows[:limit]

        print(
            f"Found {total} personal memories{f' (processing {len(pm_rows)})' if limit > 0 else ''}."
        )

        if not dry_run:
            succeeded = 0
            failed = 0
            start = time.monotonic()

            for i, row in enumerate(pm_rows):
                pm_id = row[0]
                content = row[1] or ""

                try:
                    await enqueue(
                        "graphiti_ingest",
                        {
                            "record_id": str(pm_id),
                            "content": content,
                            "source_type": "personal_memory",
                        },
                        db_url=db_url,
                    )
                    succeeded += 1
                except Exception as e:
                    failed += 1
                    print(f"  Failed to submit task for {str(pm_id)[:8]}: {e}")

                if (i + 1) % 10 == 0 or i + 1 == len(pm_rows):
                    elapsed = time.monotonic() - start
                    rate = (i + 1) / elapsed if elapsed > 0 else 0
                    print(
                        f"  Progress: {i + 1}/{len(pm_rows)} ({rate:.1f}/s) — failed: {failed}"
                    )

            elapsed = time.monotonic() - start
            print(
                f"Personal memories graph backfill tasks submitted in {elapsed:.1f}s: Succeeded {succeeded}, Failed {failed}"
            )


def main() -> None:
    print("Starting main...")
    parser = argparse.ArgumentParser(
        description="Backfill missing embeddings for published experiences",
    )
    parser.add_argument("--dry-run", action="store_true", help="Show count only")
    parser.add_argument(
        "--limit", type=int, default=0, help="Max experiences to process"
    )
    args = parser.parse_args()

    print("Running backfill...")
    asyncio.run(run_backfill(dry_run=args.dry_run, limit=args.limit))
    print("Main finished.")


if __name__ == "__main__":
    main()

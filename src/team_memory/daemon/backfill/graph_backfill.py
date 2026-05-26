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


async def run_backfill(
    dry_run: bool = False,
    limit: int = 0,
    chunk_size: int = 50,
    resume_from: str | None = None,
) -> None:
    """Backfill missing graph nodes with chunked task submission.

    Each record becomes a background task (task_runner.enqueue).
    Chunking controls how many tasks are submitted per batch to avoid
    overwhelming the task queue during large backfills.
    """
    from team_memory.config import load_settings
    from team_memory.services.infra.task_runner import enqueue
    from team_memory.services.maintenance.batch_ops import BatchOpsService

    print(f"Chunk size: {chunk_size}, resume from: {resume_from or 'start'}")

    print("Loading settings...")
    settings = load_settings()
    db_url = str(settings.database.url)
    svc = BatchOpsService(db_url=db_url)

    # Backfill experiences
    print("\n--- Backfilling Experiences to Graph ---")
    rows = await svc.list_published_experiences(limit=chunk_size if limit == 0 else limit)

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
                print(f"  {str(row['id'])[:8]}: {(row.get('title') or '')[:60]}")
            if total > 10:
                print(f"  ... and {total - 10} more")
        else:
            succeeded = 0
            failed = 0
            start = time.monotonic()

            for i, row in enumerate(rows):
                exp_id = row["id"]
                title = row.get("title") or ""
                description = row.get("description") or ""
                solution = row.get("solution") or ""
                tags = row.get("tags") or []

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
    archive_rows = await svc.list_published_archives(limit=chunk_size if limit == 0 else limit)

    if not archive_rows:
        print("No archives found.")
    else:
        total = len(archive_rows)
        if limit > 0:
            archive_rows = archive_rows[:limit]

        print(
            f"Found {total} archives{f' (processing {len(archive_rows)})' if limit > 0 else ''}."
        )

        if dry_run:
            for row in archive_rows[:10]:
                print(f"  {str(row['id'])[:8]}: {(row.get('title') or '')[:60]}")
            if total > 10:
                print(f"  ... and {total - 10} more")
        else:
            succeeded = 0
            failed = 0
            start = time.monotonic()

            for i, row in enumerate(archive_rows):
                arch_id = row["id"]
                title = row.get("title") or ""
                overview = row.get("overview") or ""
                solution_doc = row.get("solution_doc") or ""

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
    pm_rows = await svc.list_personal_memories_missing_embedding(limit=chunk_size if limit == 0 else limit)

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
                pm_id = row["id"]
                content = row.get("content") or ""

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
                f"Personal memories graph backfill tasks submitted in"
                f" {elapsed:.1f}s: Succeeded {succeeded}, Failed {failed}"
            )


def main() -> None:
    print("Starting main...")
    parser = argparse.ArgumentParser(
        description="Backfill missing embeddings for published experiences",
    )
    parser.add_argument("--dry-run", action="store_true", help="Show count only")
    parser.add_argument(
        "--limit", type=int, default=0, help="Max records (0=all)"
    )
    parser.add_argument(
        "--chunk-size", type=int, default=50,
        help="Submit N tasks per chunk. Smaller chunks survive sleep interruptions better."
    )
    parser.add_argument(
        "--resume-from", type=str, default=None,
        help="Resume from record ID (skip already submitted)"
    )
    args = parser.parse_args()

    print("Running backfill...")
    asyncio.run(run_backfill(
        dry_run=args.dry_run,
        limit=args.limit,
        chunk_size=args.chunk_size,
        resume_from=args.resume_from,
    ))
    print("Main finished.")


if __name__ == "__main__":
    main()

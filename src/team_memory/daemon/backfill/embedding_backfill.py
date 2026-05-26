#!/usr/bin/env python3
"""Embedding backfill CLI — generate embeddings for experiences missing them.

Usage:
    # Dry run
    python -m team_memory.daemon.backfill.embedding_backfill --dry-run

    # Execute
    python -m team_memory.daemon.backfill.embedding_backfill

    # With limit
    python -m team_memory.daemon.backfill.embedding_backfill --limit 10
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
    """Backfill missing embeddings."""
    from sqlalchemy import text

    from team_memory.config import load_settings
    from team_memory.embedding.factory import create_embedding_provider
    from team_memory.storage.database import get_session

    print("Loading settings...")
    settings = load_settings()
    db_url = str(settings.database.url)
    print("Creating embedding provider...")
    embedding_provider = create_embedding_provider(settings)
    print(f"Embedding provider created: {embedding_provider.__class__.__name__}")

    # Backfill experiences
    print("\n--- Backfilling Experiences ---")
    async with get_session(db_url) as session:
        result = await session.execute(
            text(
                """
            SELECT id, title, description, solution, tags
            FROM experiences
            WHERE exp_status = 'published' AND embedding IS NULL
            ORDER BY created_at ASC
        """
            )
        )
        rows = result.fetchall()

    if not rows:
        print("All published experiences have embeddings.")
    else:
        total = len(rows)
        if limit > 0:
            rows = rows[:limit]

        print(
            f"Found {total} experiences without embedding{f' (processing {len(rows)})' if limit > 0 else ''}."
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

                embed_text = f"{title}\n{description}\n{solution or ''}"
                if tags:
                    embed_text += f"\n{' '.join(str(t) for t in tags)}"
                embed_text = embed_text[:4000]

                try:
                    embedding = await embedding_provider.encode_single(embed_text)
                    if embedding is None:
                        failed += 1
                        continue

                    async with get_session(db_url) as session:
                        await session.execute(
                            text(
                                "UPDATE experiences SET embedding = :embedding WHERE id = :id"
                            ),
                            {"embedding": str(embedding), "id": exp_id},
                        )
                        await session.commit()

                    succeeded += 1
                except Exception as e:
                    failed += 1
                    print(f"  Failed {str(exp_id)[:8]}: {e}")

                if (i + 1) % 10 == 0 or i + 1 == len(rows):
                    elapsed = time.monotonic() - start
                    rate = (i + 1) / elapsed if elapsed > 0 else 0
                    print(
                        f"  Progress: {i + 1}/{len(rows)} ({rate:.1f}/s) — failed: {failed}"
                    )

            elapsed = time.monotonic() - start
            print(
                f"Experiences backfill complete in {elapsed:.1f}s: Succeeded {succeeded}, Failed {failed}"
            )

    # Backfill archives
    print("\n--- Backfilling Archives ---")
    async with get_session(db_url) as session:
        result = await session.execute(
            text(
                """
            SELECT id, title, overview, solution_doc
            FROM archives
            WHERE embedding IS NULL
            ORDER BY created_at ASC
        """
            )
        )
        archive_rows = result.fetchall()

    if not archive_rows:
        print("All archives have embeddings.")
    else:
        total = len(archive_rows)
        if limit > 0:
            archive_rows = archive_rows[:limit]

        print(
            f"Found {total} archives without embedding{f' (processing {len(archive_rows)})' if limit > 0 else ''}."
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

                embed_text = f"{title}\n{overview}\n{solution_doc}"[:4000]

                try:
                    embedding = await embedding_provider.encode_single(embed_text)
                    if embedding is None:
                        failed += 1
                        continue

                    async with get_session(db_url) as session:
                        await session.execute(
                            text(
                                "UPDATE archives SET embedding = :embedding WHERE id = :id"
                            ),
                            {"embedding": str(embedding), "id": arch_id},
                        )
                        await session.commit()

                    succeeded += 1
                except Exception as e:
                    failed += 1
                    print(f"  Failed {str(arch_id)[:8]}: {e}")

                if (i + 1) % 10 == 0 or i + 1 == len(archive_rows):
                    elapsed = time.monotonic() - start
                    rate = (i + 1) / elapsed if elapsed > 0 else 0
                    print(
                        f"  Progress: {i + 1}/{len(archive_rows)} ({rate:.1f}/s) — failed: {failed}"
                    )

            elapsed = time.monotonic() - start
            print(
                f"Archives backfill complete in {elapsed:.1f}s: Succeeded {succeeded}, Failed {failed}"
            )

    # Backfill personal memories
    print("\n--- Backfilling Personal Memories ---")
    async with get_session(db_url) as session:
        result = await session.execute(
            text(
                """
            SELECT id, content
            FROM personal_memories
            WHERE embedding IS NULL
            ORDER BY created_at ASC
        """
            )
        )
        pm_rows = result.fetchall()

    if not pm_rows:
        print("All personal memories have embeddings.")
    else:
        total = len(pm_rows)
        if limit > 0:
            pm_rows = pm_rows[:limit]

        print(
            f"Found {total} personal memories without embedding{f' (processing {len(pm_rows)})' if limit > 0 else ''}."
        )

        if not dry_run:
            succeeded = 0
            failed = 0
            start = time.monotonic()

            for i, row in enumerate(pm_rows):
                pm_id = row[0]
                content = row[1] or ""

                embed_text = content[:4000]

                try:
                    embedding = await embedding_provider.encode_single(embed_text)
                    if embedding is None:
                        failed += 1
                        continue

                    async with get_session(db_url) as session:
                        await session.execute(
                            text(
                                "UPDATE personal_memories SET embedding = :embedding WHERE id = :id"
                            ),
                            {"embedding": str(embedding), "id": pm_id},
                        )
                        await session.commit()

                    succeeded += 1
                except Exception as e:
                    failed += 1
                    print(f"  Failed {str(pm_id)[:8]}: {e}")

                if (i + 1) % 10 == 0 or i + 1 == len(pm_rows):
                    elapsed = time.monotonic() - start
                    rate = (i + 1) / elapsed if elapsed > 0 else 0
                    print(
                        f"  Progress: {i + 1}/{len(pm_rows)} ({rate:.1f}/s) — failed: {failed}"
                    )

            elapsed = time.monotonic() - start
            print(
                f"Personal memories backfill complete in {elapsed:.1f}s: Succeeded {succeeded}, Failed {failed}"
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

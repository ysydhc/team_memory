#!/usr/bin/env python3
"""Embedding backfill CLI — generate embeddings for experiences missing them.

Usage:
    # Dry run
    python scripts/daemon/embedding_backfill.py --dry-run

    # Execute
    python scripts/daemon/embedding_backfill.py

    # With limit
    python scripts/daemon/embedding_backfill.py --limit 10
"""
from __future__ import annotations

import argparse
import asyncio
import os
import sys
import time

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.abspath(os.path.join(_SCRIPT_DIR, "..", ".."))
sys.path.insert(0, os.path.join(_PROJECT_ROOT, "scripts"))  # noqa: E402
sys.path.insert(0, os.path.join(_PROJECT_ROOT, "src"))  # noqa: E402

from dotenv import load_dotenv  # noqa: E402

load_dotenv(os.path.join(_PROJECT_ROOT, ".env"))


async def run_backfill(dry_run: bool = False, limit: int = 0) -> None:
    """Backfill missing embeddings."""
    from sqlalchemy import text

    from team_memory.bootstrap import _create_embedding_provider
    from team_memory.config import load_settings
    from team_memory.storage.database import get_session

    settings = load_settings()
    db_url = str(settings.database.url)
    embedding_provider = _create_embedding_provider(settings)

    # Find experiences without embeddings
    async with get_session(db_url) as session:
        result = await session.execute(text("""
            SELECT id, title, description, solution, tags
            FROM experiences
            WHERE exp_status = 'published' AND embedding IS NULL
            ORDER BY created_at ASC
        """))
        rows = result.fetchall()

    if not rows:
        print("All published experiences have embeddings.")
        return

    total = len(rows)
    if limit > 0:
        rows = rows[:limit]

    print(f"Found {total} experiences without embedding"
          f"{f' (processing {len(rows)})' if limit > 0 else ''}.")

    if dry_run:
        for row in rows[:10]:
            print(f"  {str(row[0])[:8]}: {(row[1] or '')[:60]}")
        if total > 10:
            print(f"  ... and {total - 10} more")
        return

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

            # Update PG
            async with get_session(db_url) as session:
                await session.execute(text("""
                    UPDATE experiences SET embedding = :embedding WHERE id = :id
                """), {"embedding": str(embedding), "id": exp_id})
                await session.commit()

            succeeded += 1
        except Exception as e:
            failed += 1
            print(f"  Failed {exp_id[:8]}: {e}")

        # Progress
        if (i + 1) % 10 == 0 or i + 1 == len(rows):
            elapsed = time.monotonic() - start
            rate = (i + 1) / elapsed if elapsed > 0 else 0
            print(f"  Progress: {i + 1}/{len(rows)} ({rate:.1f}/s) — failed: {failed}")

    elapsed = time.monotonic() - start
    print(f"\nBackfill complete in {elapsed:.1f}s:")
    print(f"  Succeeded: {succeeded}")
    print(f"  Failed:    {failed}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Backfill missing embeddings for published experiences",
    )
    parser.add_argument("--dry-run", action="store_true", help="Show count only")
    parser.add_argument("--limit", type=int, default=0, help="Max experiences to process")
    args = parser.parse_args()

    asyncio.run(run_backfill(dry_run=args.dry_run, limit=args.limit))


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Backfill FTS (full-text search) column for existing experiences (P1-6).

Populates Experience.fts for rows where fts IS NULL so that search_by_fts can
match them. Uses PostgreSQL to_tsvector('simple', ...) on title, description,
solution, root_cause, code_snippets.

Usage:
    python scripts/migrate_fts.py              # run migration
    python scripts/migrate_fts.py --dry-run    # count only, no writes
    python scripts/migrate_fts.py --batch-size 100 --limit 500
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger("migrate_fts")


async def main(args: argparse.Namespace) -> None:
    from sqlalchemy import func, select, text

    from team_memory.config import load_settings
    from team_memory.storage.database import get_session
    from team_memory.storage.models import Experience

    settings = load_settings()
    db_url = settings.database.url

    async with get_session(db_url) as session:
        count_q = (
            select(func.count())
            .select_from(Experience)
            .where(Experience.is_deleted == False)  # noqa: E712
            .where(Experience.fts.is_(None))
        )
        total = (await session.execute(count_q)).scalar() or 0

    if total == 0:
        logger.info("No experiences with fts IS NULL (is_deleted=false). Nothing to do.")
        return

    max_process = total if args.limit is None else min(total, args.limit)
    logger.info("Found %d experiences to backfill (will process up to %d)", total, max_process)

    if args.dry_run:
        logger.info("Dry run — no changes made.")
        return

    batch_size = args.batch_size
    processed = 0
    errors = 0
    start = time.monotonic()

    while processed + errors < max_process:
        async with get_session(db_url) as session:
            # Select a batch of ids (next batch excludes already-updated rows)
            q = (
                select(Experience.id)
                .where(Experience.is_deleted == False)  # noqa: E712
                .where(Experience.fts.is_(None))
                .limit(batch_size)
            )
            result = await session.execute(q)
            batch_ids = [row[0] for row in result.fetchall()]

        if not batch_ids:
            break

        async with get_session(db_url) as session:
            stmt = text("""
                UPDATE experiences e
                SET fts = to_tsvector('simple',
                    coalesce(e.title, '') || ' ' ||
                    coalesce(e.description, '') || ' ' ||
                    coalesce(e.solution, '') || ' ' ||
                    coalesce(e.root_cause, '') || ' ' ||
                    coalesce(e.code_snippets, '')
                )
                WHERE e.id = ANY(:ids)
            """)
            try:
                await session.execute(stmt, {"ids": batch_ids})
                processed += len(batch_ids)
                logger.info("Updated batch: %d rows (total so far: %d)", len(batch_ids), processed)
            except Exception as e:
                errors += len(batch_ids)
                logger.exception("Batch update failed: %s", e)

        if args.limit is not None and processed >= args.limit:
            break

    elapsed = time.monotonic() - start
    logger.info("Done. Processed=%d, errors=%d, elapsed=%.1fs", processed, errors, elapsed)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Backfill FTS column for experiences")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Only count rows to update, do not write",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=200,
        help="Batch size for UPDATE (default 200)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Max number of rows to update (default: all)",
    )
    args = parser.parse_args()
    asyncio.run(main(args))

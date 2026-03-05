#!/usr/bin/env python3
"""Backfill fts_title_text, fts_desc_text, fts_solution_text for FTS scheme C.

Populates jieba-tokenized columns for existing experiences. The DB trigger
will then update the fts tsvector with setweight (title A, desc B, solution C).

Usage:
    python scripts/migrate_fts_jieba.py              # run migration
    python scripts/migrate_fts_jieba.py --dry-run    # count only, no writes
    python scripts/migrate_fts_jieba.py --batch-size 100 --limit 500
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
logger = logging.getLogger("migrate_fts_jieba")


async def main(args: argparse.Namespace) -> None:
    from sqlalchemy import select

    from team_memory.config import load_settings
    from team_memory.services.tokenizer import tokenize
    from team_memory.storage.database import get_session
    from team_memory.storage.models import Experience

    settings = load_settings()
    db_url = settings.database.url

    # Count rows missing jieba tokenized columns (any of the three is null)
    async with get_session(db_url) as session:
        from sqlalchemy import func

        count_q = (
            select(func.count())
            .select_from(Experience)
            .where(Experience.is_deleted == False)  # noqa: E712
            .where(
                (Experience.fts_title_text.is_(None))
                | (Experience.fts_desc_text.is_(None))
                | (Experience.fts_solution_text.is_(None))
            )
        )
        total = (await session.execute(count_q)).scalar() or 0

    if total == 0:
        logger.info(
            "All experiences already have fts_title_text, fts_desc_text, "
            "fts_solution_text. Nothing to do."
        )
        return

    max_process = total if args.limit is None else min(total, args.limit)
    logger.info(
        "Found %d experiences to backfill (will process up to %d)", total, max_process
    )

    if args.dry_run:
        logger.info("Dry run — no changes made.")
        return

    batch_size = args.batch_size
    processed = 0
    errors = 0
    start = time.monotonic()

    while processed + errors < max_process:
        async with get_session(db_url) as session:
            fetch_q = (
                select(Experience.id, Experience.title, Experience.description, Experience.solution)
                .where(Experience.is_deleted == False)  # noqa: E712
                .where(
                    (Experience.fts_title_text.is_(None))
                    | (Experience.fts_desc_text.is_(None))
                    | (Experience.fts_solution_text.is_(None))
                )
                .limit(batch_size)
            )
            result = await session.execute(fetch_q)
            batch = result.all()

        if not batch:
            break

        async with get_session(db_url) as session:
            from sqlalchemy import text

            for exp_id, title, desc, sol in batch:
                try:
                    ft = tokenize(title or "") or None
                    fd = tokenize(desc or "") or None
                    fs = tokenize(sol or "") or None
                    await session.execute(
                        text("""
                            UPDATE experiences
                            SET fts_title_text = :ft, fts_desc_text = :fd, fts_solution_text = :fs
                            WHERE id = :id
                        """),
                        {"id": exp_id, "ft": ft, "fd": fd, "fs": fs},
                    )
                    processed += 1
                except Exception as e:
                    errors += 1
                    logger.exception("Update failed for %s: %s", exp_id, e)

        if processed % batch_size == 0 and processed > 0:
            logger.info("Processed %d rows so far", processed)

        if args.limit is not None and processed >= args.limit:
            break

    elapsed = time.monotonic() - start
    logger.info("Done. Processed=%d, errors=%d, elapsed=%.1fs", processed, errors, elapsed)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Backfill jieba-tokenized FTS columns for experiences"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Only count rows to update, do not write",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=200,
        help="Batch size for processing (default 200)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Max number of rows to update (default: all)",
    )
    args = parser.parse_args()
    asyncio.run(main(args))

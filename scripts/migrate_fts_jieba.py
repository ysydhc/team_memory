"""Backfill PostgreSQL FTS tsvector column using TeamMemory technical tokenizer.

Recomputes weighted `experiences.fts` for all rows (across all tenants via
`bypass_rls=True`) using `team_memory.utils.tokenizer.tokenize()` so that
write-side indexing and query-side `to_tsquery` share the exact same
Chinese + technical identifier (camelCase, snake_case, hyphenated, version)
tokenization.

Usage:
    python scripts/migrate_fts_jieba.py --dry-run
    python scripts/migrate_fts_jieba.py --batch-size 200
"""

from __future__ import annotations

import argparse
import asyncio
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from sqlalchemy import text  # noqa: E402

from team_memory.config import load_settings  # noqa: E402
from team_memory.storage.database import close_db, get_session  # noqa: E402
from team_memory.utils.tokenizer import tokenize  # noqa: E402


async def _inspect_fts_tables(db_url: str) -> list[tuple[str, str]]:
    """Return all public tables that currently have a tsvector column."""
    async with get_session(db_url, bypass_rls=True) as session:
        res = await session.execute(
            text(
                """
                SELECT table_name, column_name
                FROM information_schema.columns
                WHERE table_schema = 'public'
                  AND data_type = 'tsvector'
                ORDER BY table_name, column_name
                """
            )
        )
        return [(str(r[0]), str(r[1])) for r in res.all()]


async def backfill_experiences_fts(
    db_url: str,
    *,
    dry_run: bool = False,
    batch_size: int = 200,
    include_deleted: bool = True,
) -> dict[str, int | float]:
    """Recompute `experiences.fts` for all rows in batches."""
    t0 = time.perf_counter()
    where_sql = "" if include_deleted else "WHERE is_deleted = false"

    async with get_session(db_url, bypass_rls=True) as session:
        total_res = await session.execute(text(f"SELECT count(*) FROM experiences {where_sql}"))
        total = int(total_res.scalar() or 0)
        active_res = await session.execute(
            text("SELECT count(*) FROM experiences WHERE is_deleted = false")
        )
        active_count = int(active_res.scalar() or 0)

    if dry_run:
        return {
            "total_rows": total,
            "active_rows": active_count,
            "updated_rows": 0,
            "elapsed_s": round(time.perf_counter() - t0, 3),
        }

    updated = 0
    offset = 0
    update_sql = text(
        """
        UPDATE experiences
        SET fts =
            setweight(to_tsvector('simple', :t_title), 'A') ||
            setweight(to_tsvector('simple', :t_desc), 'B') ||
            setweight(to_tsvector('simple', :t_sol), 'C')
        WHERE id = :exp_id
        """
    )

    while True:
        async with get_session(db_url, bypass_rls=True) as session:
            rows_res = await session.execute(
                text(
                    f"""
                    SELECT id, title, description, solution, tags
                    FROM experiences
                    {where_sql}
                    ORDER BY created_at ASC, id ASC
                    LIMIT :lim OFFSET :off
                    """
                ),
                {"lim": batch_size, "off": offset},
            )
            rows = rows_res.all()
            if not rows:
                break

            params = []
            for exp_id, title, description, solution, tags in rows:
                tags_str = " ".join(str(t) for t in (tags or []) if t)
                desc_and_tags = f"{description or ''} {tags_str}".strip()
                params.append(
                    {
                        "exp_id": exp_id,
                        "t_title": tokenize(title or ""),
                        "t_desc": tokenize(desc_and_tags),
                        "t_sol": tokenize(solution or ""),
                    }
                )

            await session.execute(update_sql, params)
            updated += len(rows)
            offset += len(rows)
            print(f"  [migrate-fts-jieba] backfilled {updated}/{total} experiences...", flush=True)

    return {
        "total_rows": total,
        "active_rows": active_count,
        "updated_rows": updated,
        "elapsed_s": round(time.perf_counter() - t0, 3),
    }


async def _main_async(args: argparse.Namespace) -> int:
    settings = load_settings()
    db_url = str(settings.database.url)
    try:
        fts_tables = await _inspect_fts_tables(db_url)
        print(f"Detected tsvector columns in public schema: {fts_tables}")
        stats = await backfill_experiences_fts(
            db_url,
            dry_run=args.dry_run,
            batch_size=args.batch_size,
            include_deleted=not args.active_only,
        )
        mode = "DRY-RUN" if args.dry_run else "APPLIED"
        print(
            f"[{mode}] experiences FTS backfill: total={stats['total_rows']}, "
            f"active={stats['active_rows']}, updated={stats['updated_rows']}, "
            f"elapsed={stats['elapsed_s']}s"
        )
        return 0
    finally:
        await close_db()


def main() -> int:
    parser = argparse.ArgumentParser(description="Backfill PostgreSQL FTS with technical tokenizer")
    parser.add_argument("--dry-run", action="store_true", help="Preview affected rows without updating")
    parser.add_argument("--batch-size", type=int, default=200, help="Rows per transaction batch (default: 200)")
    parser.add_argument(
        "--active-only",
        action="store_true",
        help="Only backfill non-deleted experiences (default: all rows)",
    )
    args = parser.parse_args()
    return asyncio.run(_main_async(args))


if __name__ == "__main__":
    raise SystemExit(main())

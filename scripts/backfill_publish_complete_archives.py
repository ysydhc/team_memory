#!/usr/bin/env python3
"""One-time backfill: publish draft archives that the completeness judge accepts.

Historically every archive was created as ``status='draft'`` and only promoted
to ``published`` when all linked experiences were published — but most archives
have no linked experiences, so 96%+ stayed draft and were invisible to recall.

This script runs ArchiveCompletenessJudge over existing drafts and flips the
complete ones to ``published``. Idempotent: re-running only touches remaining
drafts. Use --dry-run to preview.

Usage:
    python scripts/backfill_publish_complete_archives.py [--dry-run] [--limit N]
"""

from __future__ import annotations

import argparse
import asyncio
import os

from team_memory.project_resolver import load_dotenv_if_exists

load_dotenv_if_exists(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


async def run(dry_run: bool = False, limit: int = 0) -> None:
    import asyncpg

    from team_memory.daemon.observe.tm_stats import _resolve_dsn
    from team_memory.services.archive.completeness_judge import ArchiveCompletenessJudge

    dsn = _resolve_dsn()
    judge = ArchiveCompletenessJudge()
    conn = await asyncpg.connect(dsn)
    try:
        sql = (
            "SELECT id, title, overview, solution_doc, project FROM archives "
            "WHERE status = 'draft' ORDER BY created_at ASC"
        )
        if limit > 0:
            sql += f" LIMIT {int(limit)}"
        rows = await conn.fetch(sql)
        print(f"Drafts to evaluate: {len(rows)}")

        published = 0
        kept = 0
        for r in rows:
            verdict = await judge.judge(
                title=r["title"] or "",
                overview=r["overview"] or "",
                solution_doc=r["solution_doc"] or "",
            )
            mark = "PUBLISH" if verdict.complete else "keep   "
            print(f"  [{mark}] {(r['title'] or '')[:55]!r} ({r['project']}) — {verdict.reason[:60]}")
            if verdict.complete:
                published += 1
                if not dry_run:
                    await conn.execute(
                        "UPDATE archives SET status = 'published', updated_at = NOW() WHERE id = $1",
                        r["id"],
                    )
            else:
                kept += 1

        print(f"\n{'(dry-run) ' if dry_run else ''}Published: {published}  Kept draft: {kept}")
    finally:
        await conn.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Publish complete draft archives via LLM judge")
    parser.add_argument("--dry-run", action="store_true", help="Preview without writing")
    parser.add_argument("--limit", type=int, default=0, help="Max drafts to evaluate (0=all)")
    args = parser.parse_args()
    asyncio.run(run(dry_run=args.dry_run, limit=args.limit))


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""One-time backfill: flag historical noise queries in search_logs.

The entry-gate noise classifier (rule + LLM) only began tagging search_logs
after it shipped, so rows written earlier have is_noise IS NULL even when they
are obviously noise ("Kill test 1", slash commands, confirmations). This leaves
faithfulness/recall stats polluted by pre-gate history.

This script runs the *rule* classifier (zero LLM cost) over unflagged rows and
sets is_noise=true / noise_reason='rule' on the high-confidence matches. It is
conservative by design — only obvious noise is flagged; ambiguous rows stay
NULL. Idempotent: only touches is_noise IS NULL rows.

Usage:
    python scripts/backfill_search_log_noise.py [--dry-run]
"""

from __future__ import annotations

import argparse
import asyncio
import os

from team_memory.project_resolver import load_dotenv_if_exists

load_dotenv_if_exists(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


async def run(dry_run: bool = False) -> None:
    import asyncpg

    from team_memory.daemon.observe.tm_stats import _resolve_dsn
    from team_memory.daemon.refinement.query_noise import classify_noise_rule

    conn = await asyncpg.connect(_resolve_dsn())
    try:
        rows = await conn.fetch("SELECT id, query FROM search_logs WHERE is_noise IS NULL")
        print(f"Unflagged search_logs: {len(rows)}")

        noise_ids = [r["id"] for r in rows if classify_noise_rule(r["query"] or "") is not None]
        print(f"{'(dry-run) ' if dry_run else ''}Rule-flagged as noise: {len(noise_ids)}")

        if not dry_run and noise_ids:
            await conn.execute(
                "UPDATE search_logs SET is_noise = TRUE, noise_reason = 'rule', updated_at = NOW() "
                "WHERE id = ANY($1::uuid[])",
                noise_ids,
            )
            print(f"Updated {len(noise_ids)} rows.")
    finally:
        await conn.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Backfill is_noise on historical search_logs (rule-only)")
    parser.add_argument("--dry-run", action="store_true", help="Preview without writing")
    args = parser.parse_args()
    asyncio.run(run(dry_run=args.dry_run))


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Entity Dedup CLI — merge duplicate entities across projects.

Usage:
    # Dry run (show what would be merged)
    python -m team_memory.daemon.backfill.entity_dedup --dry-run

    # Execute merge
    python -m team_memory.daemon.backfill.entity_dedup
"""

from __future__ import annotations

import argparse
import asyncio
import os

from team_memory.project_resolver import load_dotenv_if_exists

load_dotenv_if_exists(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))


async def run_dedup(dry_run: bool = False) -> None:
    """Run entity deduplication."""
    from team_memory.config import load_settings
    from team_memory.services.maintenance.batch_ops import BatchOpsService

    settings = load_settings()
    db_url = str(settings.database.url)
    svc = BatchOpsService(db_url=db_url)

    groups = await svc.find_duplicate_entities()
    if not groups:
        print("No duplicate entities found.")
        return

    print(f"Found {len(groups)} duplicate entity groups:\n")

    total_merges = 0
    for group in groups:
        names = group["names"]
        projects = group["projects"]
        counts = group["counts"]

        primary_id = group["ids"][0]
        dup_ids = group["ids"][1:]

        name_display = names[0]
        print(
            f"  '{name_display}' — {len(names)} copies: "
            + ", ".join(f"{n} (proj={p}, count={c})" for n, p, c in zip(names, projects, counts))
        )

        if not dry_run:
            affected = await svc.merge_entity_group(primary_id, dup_ids)
            total_merges += len(dup_ids)
            print(f"    → Merged {len(dup_ids)} duplicates into {primary_id} ({affected} rows affected)")
        else:
            print(f"    → Would merge {len(dup_ids)} duplicates")

    if dry_run:
        print(f"\nDry run: would merge {sum(len(g['ids']) - 1 for g in groups)} entities")
    else:
        print(f"\nMerged {total_merges} duplicate entities")

    # Show final stats
    count = await svc.count_entities()
    print(f"Final entity count: {count}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Deduplicate entities across projects",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be merged without executing",
    )
    args = parser.parse_args()

    asyncio.run(run_dedup(dry_run=args.dry_run))


if __name__ == "__main__":
    main()

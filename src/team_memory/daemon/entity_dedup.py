#!/usr/bin/env python3
"""Entity Dedup CLI — merge duplicate entities across projects.

Usage:
    # Dry run (show what would be merged)
    python scripts/daemon/entity_dedup.py --dry-run

    # Execute merge
    python scripts/daemon/entity_dedup.py
"""
from __future__ import annotations

import argparse
import asyncio
import os
import sys

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.abspath(os.path.join(_SCRIPT_DIR, "..", ".."))
sys.path.insert(0, os.path.join(_PROJECT_ROOT, "scripts"))
sys.path.insert(0, os.path.join(_PROJECT_ROOT, "src"))

from dotenv import load_dotenv

load_dotenv(os.path.join(_PROJECT_ROOT, ".env"))


async def _find_duplicates(db_url: str) -> list[dict]:
    """Find entity groups with same name (case-insensitive) across projects."""
    from sqlalchemy import text

    from team_memory.storage.database import get_session

    groups = []
    async with get_session(db_url) as session:
        # Find names that appear more than once (case-insensitive)
        result = await session.execute(text("""
            SELECT lower(name) as lower_name, array_agg(id ORDER BY source_count DESC) as ids,
                   array_agg(name ORDER BY source_count DESC) as names,
                   array_agg(project ORDER BY source_count DESC) as projects,
                   array_agg(source_count ORDER BY source_count DESC) as counts
            FROM entities
            GROUP BY lower(name)
            HAVING count(*) > 1
            ORDER BY count(*) DESC
        """))
        for row in result.fetchall():
            groups.append({
                "lower_name": row[0],
                "ids": row[1],
                "names": row[2],
                "projects": row[3],
                "counts": row[4],
            })

    from team_memory.storage.database import close_db
    await close_db()
    return groups


async def _merge_group(
    db_url: str,
    primary_id: str,
    duplicate_ids: list[str],
    dry_run: bool = False,
) -> int:
    """Merge duplicate entities into the primary one.

    Returns number of affected rows.
    """
    from sqlalchemy import text

    from team_memory.storage.database import get_session

    affected = 0
    async with get_session(db_url) as session:
        for dup_id in duplicate_ids:
            # 1. Migrate experience_entities links
            if not dry_run:
                result = await session.execute(text("""
                    UPDATE experience_entities
                    SET entity_id = :primary_id
                    WHERE entity_id = :dup_id
                    AND NOT EXISTS (
                        SELECT 1 FROM experience_entities ee2
                        WHERE ee2.experience_id = experience_entities.experience_id
                        AND ee2.entity_id = :primary_id
                    )
                """), {"primary_id": primary_id, "dup_id": dup_id})
                affected += result.rowcount

                # Delete any remaining links to dup (duplicates with primary)
                await session.execute(text("""
                    DELETE FROM experience_entities WHERE entity_id = :dup_id
                """), {"dup_id": dup_id})

                # 2. Migrate relationships — delete dups first to avoid unique constraint
                # Delete relationships where target would become a duplicate
                await session.execute(text("""
                    DELETE FROM relationships r1
                    WHERE r1.source_entity_id = :primary_id
                    AND r1.relation_type IN (
                        SELECT r2.relation_type FROM relationships r2
                        WHERE r2.source_entity_id = :dup_id
                        AND r2.target_entity_id = r1.target_entity_id
                    )
                """), {"primary_id": primary_id, "dup_id": dup_id})
                await session.execute(text("""
                    DELETE FROM relationships r1
                    WHERE r1.target_entity_id = :primary_id
                    AND r1.relation_type IN (
                        SELECT r2.relation_type FROM relationships r2
                        WHERE r2.target_entity_id = :dup_id
                        AND r2.source_entity_id = r1.source_entity_id
                    )
                """), {"primary_id": primary_id, "dup_id": dup_id})
                await session.execute(text("""
                    UPDATE relationships
                    SET source_entity_id = :primary_id
                    WHERE source_entity_id = :dup_id
                """), {"primary_id": primary_id, "dup_id": dup_id})
                await session.execute(text("""
                    UPDATE relationships
                    SET target_entity_id = :primary_id
                    WHERE target_entity_id = :dup_id
                """), {"primary_id": primary_id, "dup_id": dup_id})

                # 3. Delete duplicate entity
                await session.execute(text("""
                    DELETE FROM entities WHERE id = :dup_id
                """), {"dup_id": dup_id})

                # 4. Update primary source_count
                await session.execute(text("""
                    UPDATE entities SET source_count = source_count + :extra
                    WHERE id = :primary_id
                """), {"primary_id": primary_id, "extra": 1})

                affected += 1

        if not dry_run:
            await session.commit()

    return affected


async def run_dedup(dry_run: bool = False) -> None:
    """Run entity deduplication."""
    from team_memory.config import load_settings

    settings = load_settings()
    db_url = str(settings.database.url)

    groups = await _find_duplicates(db_url)
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
            + ", ".join(
                f"{n} (proj={p}, count={c})"
                for n, p, c in zip(names, projects, counts)
            )
        )

        if not dry_run:
            affected = await _merge_group(db_url, primary_id, dup_ids, dry_run)
            total_merges += len(dup_ids)
            print(
                f"    → Merged {len(dup_ids)} duplicates "
                f"into {primary_id} ({affected} rows affected)"
            )
        else:
            print(f"    → Would merge {len(dup_ids)} duplicates")

    if dry_run:
        print(f"\nDry run: would merge {sum(len(g['ids'])-1 for g in groups)} entities")
    else:
        print(f"\nMerged {total_merges} duplicate entities")

    # Show final stats
    from sqlalchemy import text as sql_text

    from team_memory.storage.database import get_session
    async with get_session(db_url) as s:
        r = await s.execute(sql_text("SELECT count(*) FROM entities"))
        print(f"Final entity count: {r.scalar()}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Deduplicate entities across projects",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Show what would be merged without executing",
    )
    args = parser.parse_args()

    asyncio.run(run_dedup(dry_run=args.dry_run))


if __name__ == "__main__":
    main()

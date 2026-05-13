"""One-time migration: move knowledge_card experiences from PG → LLMWiki SQLite.

Reads all experience_type='knowledge_card' (not yet deleted) from PG,
inserts each into LLMWiki (upsert by title+project), then soft-deletes
them in PG (is_deleted=true, deleted_at=NOW()).

Idempotent: can be re-run safely — LLMWiki.add() upserts, and the PG
query only selects is_deleted=false rows.
"""

from __future__ import annotations

import asyncio
import json
import sys
from collections import defaultdict
from pathlib import Path

import asyncpg

# Ensure scripts/daemon is importable
sys.path.insert(0, str(Path(__file__).resolve().parent / "daemon"))

from llm_wiki import LLMWiki  # noqa: E402

# Connection params matching team_memory config
PG_DSN = "postgresql://developer:devpass@localhost:5433/team_memory"


async def main() -> None:
    print("=" * 60)
    print("Migration: knowledge_card (PG) → LLMWiki (SQLite)")
    print("=" * 60)

    # 1. Connect to PG and fetch knowledge cards
    conn = await asyncpg.connect(PG_DSN)
    rows = await conn.fetch(
        """
        SELECT id, title, description, project, tags
        FROM experiences
        WHERE experience_type = 'knowledge_card'
          AND is_deleted = false
        ORDER BY created_at
        """
    )
    print(f"\nFound {len(rows)} knowledge_card entries in PG (is_deleted=false).\n")

    if not rows:
        print("Nothing to migrate. Done.")
        await conn.close()
        return

    # 2. Open LLMWiki and migrate each entry
    wiki = LLMWiki()
    async with wiki:
        migrated_ids: list[str] = []
        by_project: dict[str, int] = defaultdict(int)

        for i, row in enumerate(rows, 1):
            entry_id = str(row["id"])
            title = row["title"] or "(untitled)"
            description = row["description"] or ""
            project = row["project"] or "default"

            # Parse tags — PG stores as jsonb (dict or list)
            raw_tags = row["tags"]
            if isinstance(raw_tags, str):
                try:
                    tags = json.loads(raw_tags)
                except (json.JSONDecodeError, TypeError):
                    tags = []
            elif isinstance(raw_tags, dict):
                tags = list(raw_tags.keys()) if raw_tags else []
            elif isinstance(raw_tags, list):
                tags = raw_tags
            else:
                tags = []

            # Ensure tags is a flat list of strings
            if isinstance(tags, dict):
                tags = list(tags.keys()) if tags else []
            tags = [str(t) for t in tags] if tags else []

            wiki_id = await wiki.add(
                title=title,
                content=description,
                project=project,
                tags=tags if tags else None,
            )
            migrated_ids.append(entry_id)
            by_project[project] += 1
            print(f"  [{i:>2}/{len(rows)}] ✓ {title[:50]:<50} → wiki:{wiki_id[:8]}...")

        # 3. Soft-delete in PG
        print(f"\nSoft-deleting {len(migrated_ids)} entries in PG...")
        await conn.execute(
            """
            UPDATE experiences
            SET is_deleted = true, deleted_at = NOW()
            WHERE id = ANY($1::uuid[])
            """,
            migrated_ids,
        )
        await conn.close()

        # 4. Verify wiki stats
        stats = await wiki.stats()
        print("\n" + "=" * 60)
        print("MIGRATION COMPLETE")
        print("=" * 60)
        print(f"  Migrated: {len(migrated_ids)} entries")
        print("  By project:")
        for proj, count in sorted(by_project.items()):
            print(f"    {proj}: {count}")
        print(f"\n  Wiki stats after migration: {stats}")
        print("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())

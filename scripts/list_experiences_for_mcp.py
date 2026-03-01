#!/usr/bin/env python3
"""List experiences from DB for MCP verification (id, title, tags, query hints).

Outputs JSON array of {id, title, tags, query_hint} for use in building
MCP test queries. Uses project config and DB.

Usage:
    python -m scripts.list_experiences_for_mcp [--limit 20]
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from sqlalchemy import select

from team_memory.config import get_settings
from team_memory.storage.database import get_session
from team_memory.storage.models import Experience


async def main(limit: int) -> None:
    settings = get_settings()
    db_url = settings.database.url
    out = []
    async with get_session(db_url) as session:
        q = (
            select(Experience)
            .where(Experience.is_deleted == False)  # noqa: E712
            .order_by(Experience.updated_at.desc())
            .limit(limit)
        )
        result = await session.execute(q)
        for exp in result.scalars().all():
            title = (exp.title or "")[:80]
            desc = (exp.description or "")[:120]
            tags = list(exp.tags)[:10] if exp.tags else []
            query_hint = title or (desc[:30] + "..." if len(desc) > 30 else desc)
            out.append({
                "id": str(exp.id),
                "title": title,
                "tags": tags,
                "query_hint": query_hint,
            })
    print(json.dumps(out, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=20)
    args = ap.parse_args()
    asyncio.run(main(args.limit))

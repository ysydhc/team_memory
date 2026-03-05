#!/usr/bin/env python3
"""Inspect a single experience to see why it may not appear in tm_search.

Checks: embedding (null or not), embedding_status, fts (null or not),
project, created_by, visibility, exp_status. Use this to verify why
experience 4d44f14a (or any id) is or isn't returned by MCP search.

Usage:
    python scripts/inspect_experience_for_search.py 4d44f14a-af38-4147-8599-331f64dce351
    python scripts/inspect_experience_for_search.py 4d44f14a
"""

from __future__ import annotations

import asyncio
import os
import sys
import uuid

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sqlalchemy import select

from team_memory.config import load_settings
from team_memory.storage.database import get_session
from team_memory.storage.models import Experience


async def main(experience_id: str) -> None:
    try:
        eid = uuid.UUID(experience_id)
    except ValueError:
        print("Error: invalid UUID (use full UUID e.g. 4d44f14a-af38-4147-8599-331f64dce351)")
        return

    settings = load_settings()
    async with get_session(settings.database.url) as session:
        r = await session.execute(
            select(Experience).where(Experience.id == eid)
        )
        exp = r.scalar_one_or_none()

    if exp is None:
        print("Experience not found.")
        return

    has_embedding = exp.embedding is not None
    has_fts = exp.fts is not None
    # Vector search requires: embedding IS NOT NULL AND embedding_status = 'ready'
    in_vector = has_embedding and (exp.embedding_status or "").strip().lower() == "ready"
    # FTS requires: fts @@ tsquery (fts not null to match)
    in_fts = has_fts

    print("--- Experience (id={}) ---".format(exp.id))
    print("title: {}".format((exp.title or "")[:80]))
    print("created_by: {}".format(exp.created_by))
    print("project: {}".format(exp.project))
    print("visibility: {}".format(exp.visibility))
    print("exp_status: {}".format(exp.exp_status))
    print("publish_status: {}".format(exp.publish_status))
    print("embedding_status: {}".format(exp.embedding_status))
    print("has_embedding: {} (in_vector_candidate: {})".format(has_embedding, in_vector))
    print("has_fts: {} (in_fts_candidate: {})".format(has_fts, in_fts))
    print("---")
    print("Can appear in vector search: {}".format(in_vector))
    print("Can appear in FTS search: {}".format(in_fts))
    print("Can appear in hybrid: {}".format(in_vector or in_fts))
    if not in_vector and not in_fts:
        print(
            "-> This experience will NOT be returned by tm_search until "
            "embedding or fts is populated."
        )


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python scripts/inspect_experience_for_search.py <experience_uuid>")
        sys.exit(1)
    asyncio.run(main(sys.argv[1]))

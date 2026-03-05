#!/usr/bin/env python3
"""Print vector and FTS candidate ids for a query to debug why an experience is missing.

Usage:
    TEAM_MEMORY_DEBUG_USER=admin python scripts/debug_search_candidates.py "验证默认可见性：个人经验检索测试" 4d44f14a
    python scripts/debug_search_candidates.py "个人经验检索" 4d44f14a-af38-4147-8599-331f64dce351
"""

from __future__ import annotations

import asyncio
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from team_memory.bootstrap import bootstrap
from team_memory.storage.database import get_session
from team_memory.storage.repository import ExperienceRepository


async def main(query: str, target_id: str) -> None:
    ctx = bootstrap(enable_background=False)
    settings = ctx.settings
    current_user = os.environ.get("TEAM_MEMORY_DEBUG_USER", "admin")
    project = getattr(settings, "default_project", None) or "team_memory"
    db_url = ctx.db_url

    # Get embedding for query (same as pipeline)
    query_embedding = await ctx.embedding.encode_single(query)
    over_fetch = 60  # same as max_results*3 when max_results=20

    async with get_session(db_url) as session:
        repo = ExperienceRepository(session)
        # Vector (grouped, same as pipeline default)
        vector_raw = await repo.search_by_vector_grouped(
            query_embedding=query_embedding,
            max_results=over_fetch,
            min_similarity=0.0,  # no min to see all candidates
            project=project,
            current_user=current_user,
        )
        vector_ids = [r.get("group_id") or r.get("id") for r in vector_raw]

        # FTS
        fts_raw = await repo.search_by_fts(
            query_text=query,
            max_results=over_fetch,
            project=project,
            current_user=current_user,
        )
        fts_ids = [r.get("id") for r in fts_raw]

    def matches(tid: str | None) -> bool:
        if not tid:
            return False
        tid = str(tid).lower()
        t = target_id.lower().strip()
        return tid == t or t in tid or (len(t) == 8 and tid.replace("-", "").startswith(t.replace("-", "")))

    in_vector = any(matches(tid) for tid in vector_ids)
    in_fts = any(matches(tid) for tid in fts_ids)

    print("Query: {}".format(query[:60]))
    print("current_user: {}".format(current_user))
    print("project: {}".format(project))
    print("Vector candidate count: {}".format(len(vector_ids)))
    print("FTS candidate count: {}".format(len(fts_ids)))
    print("Target id (looking for): {}".format(target_id))
    print("Target in vector candidates: {}".format(in_vector))
    print("Target in FTS candidates: {}".format(in_fts))
    if vector_ids and target_id:
        for i, vid in enumerate(vector_ids):
            if vid and target_id in str(vid):
                print("  -> vector rank {}: {}".format(i + 1, vid))
    if fts_ids and target_id:
        for i, fid in enumerate(fts_ids):
            if fid and target_id in str(fid):
                print("  -> fts rank {}: {}".format(i + 1, fid))
    if not in_vector and not in_fts:
        print("First 5 vector ids: {}".format(vector_ids[:5]))
        print("First 5 FTS ids: {}".format(fts_ids[:5]))


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python scripts/debug_search_candidates.py <query> <target_experience_id>")
        sys.exit(1)
    asyncio.run(main(sys.argv[1], sys.argv[2]))

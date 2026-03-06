#!/usr/bin/env python3
"""Run search pipeline with TEAM_MEMORY_SEARCH_DEBUG=1 to trace retrieval.

Usage:
    TEAM_MEMORY_SEARCH_DEBUG=1 python scripts/debug_search_pipeline.py "query"
"""
from __future__ import annotations

import asyncio
import logging
import os
import sys

# Enable debug before importing
os.environ["TEAM_MEMORY_SEARCH_DEBUG"] = "1"

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Configure logging to see SEARCH_DEBUG
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(name)s %(levelname)s %(message)s",
)
logging.getLogger("team_memory.search_pipeline").setLevel(logging.INFO)


async def main(query: str) -> None:
    from team_memory.bootstrap import bootstrap

    ctx = bootstrap(enable_background=False)
    current_user = os.environ.get("TEAM_MEMORY_USER", "anonymous")
    project = getattr(ctx.settings, "default_project", None) or "team_memory"

    print(f"Query: {query}")
    print(f"current_user: {current_user}")
    print(f"project: {project}")
    print("---")

    results = await ctx.service.search(
        query=query,
        max_results=10,
        min_similarity=0.5,
        user_name=current_user,
        project=project,
        grouped=True,
    )

    print(f"Results: {len(results)}")
    for i, r in enumerate(results):
        eid = r.get("group_id") or r.get("id")
        title = (r.get("title") or "")[:50]
        sim = r.get("similarity", 0)
        print(f"  {i+1}. {eid} sim={sim:.4f} {title}")
    has_target = any("4d44f14a" in str(r.get("group_id") or r.get("id")) for r in results)
    print(f"--- 4d44f14a in results: {has_target}")


if __name__ == "__main__":
    q = sys.argv[1] if len(sys.argv) > 1 else "验证默认可见性：个人经验检索测试"
    asyncio.run(main(q))

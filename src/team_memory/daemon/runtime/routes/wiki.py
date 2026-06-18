"""Wiki/knowledge endpoints for the TM Daemon.

POST /wiki/search
POST /wiki/index
GET /wiki/stats
"""

from __future__ import annotations

import logging
from typing import Any

from fastapi import APIRouter, Request

from team_memory.daemon.wiki.llm_wiki import LLMWiki

logger = logging.getLogger("daemon.app")

router = APIRouter()


@router.get("/wiki/search")
async def wiki_search(
    request: Request,
    q: str = "",
    project: str | None = None,
    limit: int = 5,
) -> list[dict[str, Any]]:
    """Search the local LLMWiki knowledge store using FTS5."""
    llm_wiki: LLMWiki | None = getattr(request.app.state, "llm_wiki", None)
    if llm_wiki is None:
        return []
    results = await llm_wiki.search(query=q, project=project, limit=limit)
    logger.info(
        '[READ]  wiki/search → q="%s" project=%s results=%d',
        q[:60],
        project or "?",
        len(results),
    )
    return results


@router.post("/wiki/index")
async def wiki_index(request: Request) -> dict[str, Any]:
    """Index a single file into the wiki compiler."""
    data = await request.json()
    path = data.get("path")
    project = data.get("project", "")
    wiki_compiler = getattr(request.app.state, "wiki_compiler", None)
    if wiki_compiler and path:
        compile_fn = getattr(wiki_compiler, "index_file", None)
        if compile_fn:
            await compile_fn(path, project=project)
    return {"status": "ok"}


@router.get("/wiki/stats")
async def wiki_stats(request: Request) -> dict[str, Any]:
    """Return wiki entry statistics."""
    llm_wiki: LLMWiki | None = getattr(request.app.state, "llm_wiki", None)
    if llm_wiki is None:
        return {"total": 0, "by_project": {}}
    return await llm_wiki.stats()

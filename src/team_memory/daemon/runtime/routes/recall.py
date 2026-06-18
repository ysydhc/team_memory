"""Recall endpoint for the TM Daemon.

GET /recall
"""

from __future__ import annotations

import logging
from typing import Any

from fastapi import APIRouter, Request

from team_memory.daemon.integration.tm_sink import TMSink

logger = logging.getLogger("daemon.app")

router = APIRouter()


@router.get("/recall")
async def recall(
    request: Request,
    query: str | None = None,
    project: str | None = None,
    max_results: int = 5,
) -> list[dict[str, Any]]:
    """Direct recall query."""
    sink_obj: TMSink = request.app.state.sink
    results = await sink_obj.recall(
        query=query,
        project=project,
        max_results=max_results,
    )
    result_list = results if isinstance(results, list) else []
    result_lines = []
    for i, r in enumerate(result_list[:5]):
        tid = r.get("id", "?")
        title = r.get("title", "?")[:50]
        result_lines.append(f'  #{i + 1} "{title}" [{tid[:8]}]')
    extra = "\n" + "\n".join(result_lines) if result_lines else ""
    logger.info(
        '[READ]  recall → query="%s" project=%s results=%d%s',
        (query or "")[:60],
        project or "?",
        len(result_list),
        extra,
    )
    return results

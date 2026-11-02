"""MCP memory tool orchestration — business logic extracted from server.py.

Callers (MCP server) inject ``user`` and serialize/guard JSON output.

This module is a facade: implementations live in ``ops/`` sub-modules.
All public and private names are re-exported here for backward compatibility.
``write_search_log`` remains defined here because existing tests patch its
dependencies (get_session, SearchLogRepository) at this module path.
"""

from __future__ import annotations

import logging

from team_memory.services.memory.ops.archive import (  # noqa: F401
    op_archive_upsert,
    op_get_archive,
)
from team_memory.services.memory.ops.context import op_context  # noqa: F401
from team_memory.services.memory.ops.feedback import (  # noqa: F401
    op_feedback,
    op_submit_response,
)
from team_memory.services.memory.ops.recall import (  # noqa: F401
    _append_user_profile_to_recall_json,
    _maybe_write_search_log,
    _recall_adaptive,
    _recall_search,
    _recall_solve,
    _recall_suggest,
    op_recall,
)
from team_memory.services.memory.ops.save import (  # noqa: F401
    _save_from_content,
    _try_extract_and_save_personal_memory,
    op_draft_publish,
    op_draft_save,
    op_experience_update,
    op_save,
)
from team_memory.storage.database import get_session
from team_memory.storage.search_log_repository import SearchLogRepository

logger = logging.getLogger("team_memory")


async def write_search_log(
    *,
    db_url: str,
    query: str,
    intent_type: str = "unknown",
    project: str = "default",
    source: str = "mcp",
    result_ids: list[dict] | None = None,
    stage_metrics: dict | None = None,
    stage_counts: dict | None = None,
    returned_tokens: int | None = None,
) -> str | None:
    """Write a SearchLog entry after a search completes.

    Failures are logged and do not propagate — search logging is best-effort.

    Returns:
        The search_log_id (UUID string) on success, None on failure.
    """
    try:
        async with get_session(db_url) as session:
            repo = SearchLogRepository(session)
            create_kwargs: dict = {
                "query": query,
                "intent_type": intent_type,
                "project": project,
                "source": source,
                "result_ids": result_ids,
            }
            if stage_metrics is not None:
                create_kwargs["stage_metrics"] = stage_metrics
            if stage_counts is not None:
                create_kwargs["stage_counts"] = stage_counts
            if returned_tokens is not None:
                create_kwargs["returned_tokens"] = returned_tokens
            log = await repo.create(**create_kwargs)
            await session.commit()
            return str(log.id)
    except Exception:
        logger.debug("write_search_log failed (best-effort)", exc_info=True)
        return None

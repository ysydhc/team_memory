from __future__ import annotations

import json
import logging
import uuid as uuid_mod

from sqlalchemy import text

from team_memory.storage.database import get_session
from team_memory.utils.project import resolve_project as _resolve_project

logger = logging.getLogger("team_memory")


async def op_feedback(
    user: str,
    *,
    service: object,
    experience_id: str,
    rating: int,
    comment: str | None = None,
) -> dict:
    """Submit feedback for an experience."""
    if not (1 <= rating <= 5):
        return {
            "error": True,
            "message": "Rating must be between 1 and 5.",
            "code": "validation_error",
        }

    success = await service.feedback(
        experience_id=experience_id,
        rating=rating,
        feedback_by=user,
        comment=comment,
    )

    if success:
        return {"message": "Feedback recorded. Thank you!"}
    return {
        "error": True,
        "message": "Experience not found.",
        "code": "not_found",
    }


async def op_submit_response(
    user: str,
    *,
    db_url: str,
    query: str,
    response: str,
    result_ids: list[str] | None = None,
    project: str | None = None,
) -> dict:
    """Buffer an agent response for faithfulness evaluation (response_buffer row)."""
    logger.debug("op_submit_response: user=%s", user)

    if not str(query or "").strip() or not str(response or "").strip():
        return {
            "error": True,
            "code": "validation_error",
            "message": "query and response must be non-empty.",
        }

    resolved_project = _resolve_project(project)
    truncated_query = (query or "")[:2000]
    truncated_response = (response or "")[:10000]

    payload_rows: list[dict[str, str]] = []
    if result_ids:
        for rid in result_ids:
            try:
                payload_rows.append({"id": str(uuid_mod.UUID(str(rid).strip()))})
            except (ValueError, TypeError, AttributeError):
                return {
                    "error": True,
                    "code": "validation_error",
                    "message": f"Invalid UUID in result_ids: {rid!r}",
                }

    ids_json = json.dumps(payload_rows)
    buffer_id = uuid_mod.uuid4()

    try:
        async with get_session(db_url) as session:
            await session.execute(
                text(
                    """
                    INSERT INTO response_buffer (id, query, agent_response, result_ids, created_at)
                    VALUES (:id, :query, :agent_response, CAST(:result_ids AS json), NOW())
                    """
                ),
                {
                    "id": buffer_id,
                    "query": truncated_query,
                    "agent_response": truncated_response,
                    "result_ids": ids_json,
                },
            )
    except Exception as e:
        logger.exception("op_submit_response failed: %s", e)
        return {
            "error": True,
            "code": "internal_error",
            "message": f"Failed to buffer response: {e}",
        }

    logger.info(
        "response buffered: id=%s project=%s",
        buffer_id,
        resolved_project,
    )
    return {"status": "buffered", "buffer_id": str(buffer_id)}

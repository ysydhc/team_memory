"""Draft management endpoints for the TM Daemon.

POST /draft/save
POST /draft/publish
"""

from __future__ import annotations

import logging
from typing import Any

from fastapi import APIRouter, Request

from team_memory.daemon.integration.tm_sink import TMSink
from team_memory.daemon.runtime.models import DraftPublishPayload, DraftSavePayload

logger = logging.getLogger("daemon.app")

router = APIRouter()


@router.post("/draft/save")
async def draft_save(payload: DraftSavePayload, request: Request) -> dict[str, Any]:
    """Direct draft save."""
    sink_obj: TMSink = request.app.state.sink
    result = await sink_obj.draft_save(
        title=payload.title,
        content=payload.content,
        project=payload.project,
        group_key=payload.group_key,
        conversation_id=payload.conversation_id,
    )
    logger.info(
        '[WRITE] draft_save → title="%s" project=%s',
        payload.title[:40],
        payload.project or "?",
    )
    return result


@router.post("/draft/publish")
async def draft_publish(payload: DraftPublishPayload, request: Request) -> dict[str, Any]:
    """Direct draft publish."""
    sink_obj: TMSink = request.app.state.sink
    result = await sink_obj.draft_publish(
        draft_id=payload.draft_id,
        refined_content=payload.refined_content,
    )
    logger.info(
        "[WRITE] draft_publish → draft_id=%s",
        payload.draft_id[:8],
    )
    return result

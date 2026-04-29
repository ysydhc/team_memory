"""HTTP endpoints that mirror MCP memory_* tools — same orchestration as MCP (memory_operations)."""

from __future__ import annotations

import logging

from fastapi import APIRouter, Depends, HTTPException, Path, Query
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from team_memory.auth.provider import User
from team_memory.services import memory_operations
from team_memory.web.auth_session import get_current_user

logger = logging.getLogger("team_memory.web")

router = APIRouter(prefix="/mcp", tags=["mcp-compat"])


def _http_status_for_result(result: dict) -> int:
    """Map orchestration error dicts to HTTP status codes."""
    if not result.get("error"):
        return 200
    code = result.get("code") or "internal_error"
    if code == "not_found":
        return 404
    if code == "validation_error":
        return 400
    if code == "content_too_long":
        return 413
    if code == "scope_removed":
        return 400
    if code == "embedding_failed":
        return 502
    return 500


def _json_response(result: dict, *, user: str, tool_name: str) -> JSONResponse:
    logger.info("mcp_compat: tool=%s user=%s", tool_name, user)
    return JSONResponse(content=result, status_code=_http_status_for_result(result))


# --- Request bodies (aligned with MCP tool parameters) ---


class McpSaveRequest(BaseModel):
    title: str | None = None
    problem: str | None = None
    solution: str | None = None
    content: str | None = None
    tags: list[str] | None = None
    scope: str = "project"
    experience_type: str | None = None
    project: str | None = None
    group_key: str | None = None


class McpRecallRequest(BaseModel):
    query: str | None = None
    problem: str | None = None
    file_path: str | None = None
    language: str | None = None
    framework: str | None = None
    tags: list[str] | None = None
    max_results: int = Field(5, ge=1)
    project: str | None = None
    include_archives: bool | None = None
    include_user_profile: bool = False


class McpContextRequest(BaseModel):
    file_paths: list[str] | None = None
    task_description: str | None = None
    project: str | None = None


class McpArchiveUpsertRequest(BaseModel):
    title: str
    solution_doc: str
    content_type: str = "session_archive"
    value_summary: str | None = None
    tags: list[str] | None = None
    overview: str | None = None
    conversation_summary: str | None = None
    raw_conversation: str | None = None
    linked_experience_ids: list[str] | None = None
    project: str | None = None
    scope: str = "session"
    scope_ref: str | None = None


class McpFeedbackRequest(BaseModel):
    experience_id: str
    rating: int = Field(..., ge=1, le=5)
    comment: str | None = None


# --- Routes ---


@router.post("/save")
async def mcp_save(
    body: McpSaveRequest,
    user: User = Depends(get_current_user),
):
    """POST /mcp/save — same contract as memory_save."""
    uid = user.name
    try:
        result = await memory_operations.op_save(
            uid,
            title=body.title,
            problem=body.problem,
            solution=body.solution,
            content=body.content,
            tags=body.tags,
            scope=body.scope,
            experience_type=body.experience_type,
            project=body.project,
            group_key=body.group_key,
        )
        return _json_response(result, user=uid, tool_name="save")
    except HTTPException:
        raise
    except Exception as e:
        logger.exception("mcp_compat save failed: %s", e)
        raise HTTPException(status_code=500, detail=str(e)) from e


@router.post("/recall")
async def mcp_recall(
    body: McpRecallRequest,
    user: User = Depends(get_current_user),
):
    """POST /mcp/recall — same contract as memory_recall."""
    uid = user.name
    try:
        result = await memory_operations.op_recall(
            uid,
            query=body.query,
            problem=body.problem,
            file_path=body.file_path,
            language=body.language,
            framework=body.framework,
            tags=body.tags,
            max_results=body.max_results,
            project=body.project,
            include_archives=body.include_archives,
            include_user_profile=body.include_user_profile,
        )
        return _json_response(result, user=uid, tool_name="recall")
    except HTTPException:
        raise
    except Exception as e:
        logger.exception("mcp_compat recall failed: %s", e)
        raise HTTPException(status_code=500, detail=str(e)) from e


@router.post("/context")
async def mcp_context(
    body: McpContextRequest,
    user: User = Depends(get_current_user),
):
    """POST /mcp/context — same contract as memory_context."""
    uid = user.name
    try:
        result = await memory_operations.op_context(
            uid,
            file_paths=body.file_paths,
            task_description=body.task_description,
            project=body.project,
        )
        return _json_response(result, user=uid, tool_name="context")
    except HTTPException:
        raise
    except Exception as e:
        logger.exception("mcp_compat context failed: %s", e)
        raise HTTPException(status_code=500, detail=str(e)) from e


@router.get("/archive/{archive_id}")
async def mcp_get_archive(
    archive_id: str = Path(..., description="Archive UUID"),
    project: str | None = Query(None),
    user: User = Depends(get_current_user),
):
    """GET /mcp/archive/{archive_id} — same contract as memory_get_archive."""
    uid = user.name
    try:
        result = await memory_operations.op_get_archive(
            uid,
            archive_id=archive_id,
            project=project,
        )
        return _json_response(result, user=uid, tool_name="get_archive")
    except HTTPException:
        raise
    except Exception as e:
        logger.exception("mcp_compat get_archive failed: %s", e)
        raise HTTPException(status_code=500, detail=str(e)) from e


@router.post("/archive-upsert")
async def mcp_archive_upsert(
    body: McpArchiveUpsertRequest,
    user: User = Depends(get_current_user),
):
    """POST /mcp/archive-upsert — same contract as memory_archive_upsert."""
    uid = user.name
    try:
        result = await memory_operations.op_archive_upsert(
            uid,
            title=body.title,
            solution_doc=body.solution_doc,
            content_type=body.content_type,
            value_summary=body.value_summary,
            tags=body.tags,
            overview=body.overview,
            conversation_summary=body.conversation_summary,
            raw_conversation=body.raw_conversation,
            linked_experience_ids=body.linked_experience_ids,
            project=body.project,
            scope=body.scope,
            scope_ref=body.scope_ref,
        )
        return _json_response(result, user=uid, tool_name="archive_upsert")
    except HTTPException:
        raise
    except Exception as e:
        logger.exception("mcp_compat archive_upsert failed: %s", e)
        raise HTTPException(status_code=500, detail=str(e)) from e


@router.post("/feedback")
async def mcp_feedback(
    body: McpFeedbackRequest,
    user: User = Depends(get_current_user),
):
    """POST /mcp/feedback — same contract as memory_feedback."""
    uid = user.name
    try:
        result = await memory_operations.op_feedback(
            uid,
            experience_id=body.experience_id,
            rating=body.rating,
            comment=body.comment,
        )
        return _json_response(result, user=uid, tool_name="feedback")
    except HTTPException:
        raise
    except Exception as e:
        logger.exception("mcp_compat feedback failed: %s", e)
        raise HTTPException(status_code=500, detail=str(e)) from e


# ============================================================
# Draft pipeline endpoints (used by Daemon RemoteTMSink)
# ============================================================


class McpDraftSaveRequest(BaseModel):
    title: str
    content: str
    tags: list[str] | None = None
    project: str | None = None
    group_key: str | None = None
    conversation_id: str | None = None
    skip_dedup: bool = False


class McpDraftPublishRequest(BaseModel):
    draft_id: str
    refined_content: str | None = None


@router.post("/draft-save")
async def mcp_draft_save(
    body: McpDraftSaveRequest,
    user: User = Depends(get_current_user),
):
    """POST /mcp/draft-save — same contract as memory_draft_save.

    Used by Daemon RemoteTMSink to persist drafts into team_memory_service.
    """
    uid = user.name
    try:
        result = await memory_operations.op_draft_save(
            uid,
            title=body.title,
            content=body.content,
            tags=body.tags,
            project=body.project,
            group_key=body.group_key,
            conversation_id=body.conversation_id,
        )
        return _json_response(result, user=uid, tool_name="draft_save")
    except HTTPException:
        raise
    except Exception as e:
        logger.exception("mcp_compat draft_save failed: %s", e)
        raise HTTPException(status_code=500, detail=str(e)) from e


@router.post("/draft-publish")
async def mcp_draft_publish(
    body: McpDraftPublishRequest,
    user: User = Depends(get_current_user),
):
    """POST /mcp/draft-publish — same contract as memory_draft_publish.

    Used by Daemon RemoteTMSink to promote a draft to published experience.
    """
    uid = user.name
    try:
        result = await memory_operations.op_draft_publish(
            uid,
            draft_id=body.draft_id,
            refined_content=body.refined_content,
        )
        return _json_response(result, user=uid, tool_name="draft_publish")
    except HTTPException:
        raise
    except Exception as e:
        logger.exception("mcp_compat draft_publish failed: %s", e)
        raise HTTPException(status_code=500, detail=str(e)) from e

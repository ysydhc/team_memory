from __future__ import annotations

import logging
import uuid as uuid_mod

from pydantic import ValidationError

from team_memory.schemas import ArchiveCreateRequest
from team_memory.services.archive import ArchiveUploadError
from team_memory.utils.project import resolve_project as _resolve_project

logger = logging.getLogger("team_memory")


async def op_get_archive(
    user: str,
    *,
    archive_service: object,
    archive_id: str,
    project: str | None = None,
) -> dict:
    """Return L2 dict or error-shaped dict."""
    try:
        aid = uuid_mod.UUID(archive_id.strip())
    except (ValueError, TypeError, AttributeError):
        return {"error": True, "message": "Archive not found", "code": "not_found"}

    resolved = _resolve_project(project)
    archive_svc = archive_service
    try:
        out = await archive_svc.get_archive(aid, viewer=user, project=resolved)
    except Exception as e:
        logger.exception("memory_get_archive failed: %s", e)
        return {"error": True, "message": str(e), "code": "internal_error"}
    if out is None:
        return {"error": True, "message": "Archive not found", "code": "not_found"}
    if "attachments" not in out or out["attachments"] is None:
        out = {**out, "attachments": []}
    if "document_tree_nodes" not in out or out["document_tree_nodes"] is None:
        out = {**out, "document_tree_nodes": []}
    return out


async def op_archive_upsert(
    user: str,
    *,
    archive_service: object,
    settings: object,
    title: str,
    solution_doc: str,
    content_type: str = "session_archive",
    value_summary: str | None = None,
    tags: list[str] | None = None,
    overview: str | None = None,
    conversation_summary: str | None = None,
    raw_conversation: str | None = None,
    linked_experience_ids: list[str] | None = None,
    project: str | None = None,
    scope: str = "session",
    scope_ref: str | None = None,
) -> dict:
    """Upsert archive via ArchiveService; success or error dict."""
    max_sd = int(getattr(settings.mcp, "max_archive_solution_doc_chars", 64_000))
    if len(solution_doc or "") > max_sd:
        return {
            "error": True,
            "message": (
                f"solution_doc exceeds MCP limit ({max_sd} chars). "
                "Shorten/summarize or upload full text as an attachment via HTTP/cli upload."
            ),
            "code": "validation_error",
        }

    try:
        body = ArchiveCreateRequest(
            title=title,
            solution_doc=solution_doc,
            content_type=content_type,
            value_summary=value_summary,
            tags=tags,
            overview=overview,
            conversation_summary=conversation_summary,
            raw_conversation=raw_conversation,
            linked_experience_ids=linked_experience_ids,
            project=project,
            scope=scope,
            scope_ref=scope_ref,
        )
    except ValidationError as e:
        return {"error": True, "message": str(e), "code": "validation_error"}

    resolved = _resolve_project(body.project)

    linked_uuids: list[uuid_mod.UUID] = []
    if body.linked_experience_ids:
        for s in body.linked_experience_ids:
            try:
                linked_uuids.append(uuid_mod.UUID(s))
            except (ValueError, TypeError):
                pass

    archive_svc = archive_service
    try:
        result = await archive_svc.archive_upsert(
            title=body.title,
            solution_doc=body.solution_doc,
            created_by=user,
            project=resolved,
            scope=body.scope,
            scope_ref=body.scope_ref,
            overview=body.overview,
            conversation_summary=body.conversation_summary,
            raw_conversation=body.raw_conversation,
            content_type=body.content_type,
            value_summary=body.value_summary,
            tags=body.tags,
            linked_experience_ids=linked_uuids if linked_uuids else None,
        )
    except ArchiveUploadError as e:
        return {"error": True, "message": e.message, "code": e.error_code}
    except Exception as e:
        logger.exception("memory_archive_upsert failed: %s", e)
        return {"error": True, "message": str(e), "code": "internal_error"}

    item = dict(result)
    archive_id = item.get("archive_id")
    if archive_id is not None and not isinstance(archive_id, str):
        archive_id = str(archive_id)
        item = {**item, "archive_id": archive_id}
    action = item.get("action", "created")
    msg = "Updated successfully" if action == "updated" else "Created successfully"
    return {
        "archive_id": archive_id,
        "action": action,
        "message": msg,
        "item": item,
    }

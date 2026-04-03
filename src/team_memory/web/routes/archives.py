"""Archive (档案馆) browse API — list, L2 detail, upsert, multipart upload, failures."""

from __future__ import annotations

import uuid as _uuid

from fastapi import (
    APIRouter,
    Depends,
    File,
    Form,
    Header,
    HTTPException,
    Query,
    UploadFile,
)
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel

from team_memory.auth.provider import User
from team_memory.bootstrap import get_context
from team_memory.schemas import ArchiveCreateRequest
from team_memory.services.archive import ArchiveUploadError
from team_memory.web.app import _resolve_project
from team_memory.web.auth_session import get_current_user, get_optional_user

router = APIRouter(tags=["archives"])


class UploadFailureResolveBody(BaseModel):
    resolved: bool = True


def _parse_archive_id(archive_id: str) -> _uuid.UUID:
    try:
        return _uuid.UUID(archive_id)
    except (ValueError, TypeError) as e:
        raise HTTPException(status_code=404, detail="Archive not found") from e


def _viewer_info(user: User | None) -> tuple[str | None, str | None]:
    """Extract (viewer_name, viewer_role) from optional User."""
    if user is None:
        return None, None
    return user.name, user.role


@router.get("/archives")
async def list_archives(
    project: str | None = Query(None, description="Project scope"),
    q: str | None = Query(None, description="Keyword filter on title/overview"),
    limit: int = Query(30, ge=1, le=100),
    offset: int = Query(0, ge=0),
    user: User | None = Depends(get_optional_user),
):
    """Paginated archives visible to the current user (same rules as vector search)."""
    ctx = get_context()
    viewer, role = _viewer_info(user)
    resolved = _resolve_project(project)
    items, total = await ctx.archive_service.list_archives(
        viewer=viewer,
        project=resolved,
        q=q,
        limit=limit,
        offset=offset,
        viewer_role=role,
    )
    return {
        "items": items,
        "total": total,
        "limit": limit,
        "offset": offset,
    }


@router.post("/archives")
async def create_or_update_archive(
    body: ArchiveCreateRequest,
    user: User = Depends(get_current_user),
):
    """Create or upsert an archive (title+project dedup). Requires authentication."""
    ctx = get_context()
    linked_uuids: list[_uuid.UUID] = []
    if body.linked_experience_ids:
        for s in body.linked_experience_ids:
            try:
                linked_uuids.append(_uuid.UUID(s))
            except (ValueError, TypeError):
                pass
    result = await ctx.archive_service.archive_upsert(
        title=body.title,
        solution_doc=body.solution_doc,
        created_by=user.name,
        project=_resolve_project(body.project),
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
    # Ensure UUID fields are JSON-serializable
    if "archive_id" in result and not isinstance(result["archive_id"], str):
        result["archive_id"] = str(result["archive_id"])
    is_update = result.get("action") == "updated"
    status_code = 200 if is_update else 201
    message = "Updated successfully" if is_update else "Created successfully"
    return JSONResponse(
        content={"item": result, "message": message},
        status_code=status_code,
    )


@router.get("/archives/{archive_id}")
async def get_archive_detail(
    archive_id: str,
    project: str | None = Query(None),
    user: User | None = Depends(get_optional_user),
):
    """Full archive (L2) when allowed for viewer."""
    aid = _parse_archive_id(archive_id)
    ctx = get_context()
    viewer, role = _viewer_info(user)
    resolved = _resolve_project(project)
    out = await ctx.archive_service.get_archive(
        aid,
        viewer=viewer,
        project=resolved,
        viewer_role=role,
    )
    if out is None:
        raise HTTPException(status_code=404, detail="Archive not found")
    return out


@router.post("/archives/{archive_id}/attachments/upload")
async def upload_archive_attachment(
    archive_id: str,
    file: UploadFile = File(...),
    kind: str = Form("file"),
    note: str | None = Form(None),
    source_path: str | None = Form(None),
    project: str | None = Query(None),
    user: User | None = Depends(get_optional_user),
    x_upload_source: str | None = Header(None, alias="X-Upload-Source"),
):
    """multipart upload of a single file; requires same visibility as L2."""
    aid = _parse_archive_id(archive_id)
    ctx = get_context()
    viewer = user.name if user else None
    resolved = _resolve_project(project)
    body = await file.read()
    src = (x_upload_source or "web").strip().lower()[:20]
    if src not in ("web", "api", "agent"):
        src = "api"
    try:
        return await ctx.archive_service.upload_archive_attachment(
            aid,
            viewer,
            resolved,
            ctx.settings.uploads,
            file_content=body,
            client_filename=file.filename,
            kind=kind,
            snippet=note,
            source=src,
            source_path=source_path,
        )
    except ArchiveUploadError as e:
        raise HTTPException(status_code=e.http_status, detail=e.message) from e


@router.get("/archives/{archive_id}/attachments/{attachment_id}/file")
async def download_archive_attachment(
    archive_id: str,
    attachment_id: str,
    project: str | None = Query(None),
    user: User | None = Depends(get_optional_user),
):
    """Stream file bytes; GET allowed even when POST uploads disabled (§4.1)."""
    aid = _parse_archive_id(archive_id)
    try:
        att_id = _uuid.UUID(attachment_id)
    except (ValueError, TypeError) as e:
        raise HTTPException(status_code=404, detail="Archive not found") from e

    ctx = get_context()
    viewer = user.name if user else None
    resolved = _resolve_project(project)
    pair = await ctx.archive_service.read_archive_attachment_file(
        aid, att_id, viewer, resolved, ctx.settings.uploads
    )
    if pair is None:
        raise HTTPException(status_code=404, detail="Archive not found")
    path, fname = pair
    return FileResponse(
        path,
        filename=fname,
        media_type="application/octet-stream",
    )


@router.get("/archives/{archive_id}/upload-failures")
async def list_archive_upload_failures(
    archive_id: str,
    project: str | None = Query(None),
    limit: int = Query(20, ge=1, le=100),
    include_resolved: bool = Query(False),
    user: User | None = Depends(get_optional_user),
):
    """Failed upload attempts for remediation (curl) — same visibility as L2."""
    aid = _parse_archive_id(archive_id)
    ctx = get_context()
    viewer, role = _viewer_info(user)
    resolved = _resolve_project(project)
    if (
        await ctx.archive_service.get_archive(
            aid,
            viewer=viewer,
            project=resolved,
            viewer_role=role,
        )
        is None
    ):
        raise HTTPException(status_code=404, detail="Archive not found")
    items = await ctx.archive_service.list_upload_failures(
        aid,
        viewer,
        resolved,
        limit=limit,
        include_resolved=include_resolved,
    )
    return {"items": items}


@router.patch("/archives/{archive_id}/upload-failures/{failure_id}")
async def resolve_archive_upload_failure(
    archive_id: str,
    failure_id: str,
    body: UploadFailureResolveBody,
    project: str | None = Query(None),
    user: User | None = Depends(get_optional_user),
):
    if not body.resolved:
        raise HTTPException(status_code=400, detail="Only resolved=true supported")
    aid = _parse_archive_id(archive_id)
    try:
        fid = _uuid.UUID(failure_id)
    except (ValueError, TypeError) as e:
        raise HTTPException(status_code=404, detail="Archive not found") from e
    ctx = get_context()
    viewer = user.name if user else None
    resolved = _resolve_project(project)
    ok = await ctx.archive_service.mark_upload_failure_resolved(aid, fid, viewer, resolved)
    if not ok:
        raise HTTPException(status_code=404, detail="Archive not found")
    return {"ok": True}

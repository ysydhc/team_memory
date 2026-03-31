"""Archive (档案馆) browse API — list, L2 detail, multipart upload, failures."""

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
from fastapi.responses import FileResponse
from pydantic import BaseModel

from team_memory.auth.provider import User
from team_memory.bootstrap import get_context
from team_memory.services.archive import ArchiveUploadError
from team_memory.web.app import _resolve_project, get_optional_user

router = APIRouter(tags=["archives"])


class UploadFailureResolveBody(BaseModel):
    resolved: bool = True


def _parse_archive_id(archive_id: str) -> _uuid.UUID:
    try:
        return _uuid.UUID(archive_id)
    except (ValueError, TypeError) as e:
        raise HTTPException(status_code=404, detail="Archive not found") from e


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
    viewer = user.name if user else None
    resolved = _resolve_project(project)
    items, total = await ctx.archive_service.list_archives(
        viewer=viewer,
        project=resolved,
        q=q,
        limit=limit,
        offset=offset,
    )
    return {
        "items": items,
        "total": total,
        "limit": limit,
        "offset": offset,
        "project": resolved,
    }


@router.get("/archives/{archive_id}")
async def get_archive_detail(
    archive_id: str,
    project: str | None = Query(None),
    user: User | None = Depends(get_optional_user),
):
    """Full archive (L2) when allowed for viewer."""
    aid = _parse_archive_id(archive_id)
    ctx = get_context()
    viewer = user.name if user else None
    resolved = _resolve_project(project)
    out = await ctx.archive_service.get_archive(aid, viewer=viewer, project=resolved)
    if out is None:
        raise HTTPException(status_code=404, detail="Archive not found")
    return out


@router.post("/archives/{archive_id}/attachments/upload")
async def upload_archive_attachment(
    archive_id: str,
    file: UploadFile = File(...),
    kind: str = Form("file"),
    note: str | None = Form(None),
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
    viewer = user.name if user else None
    resolved = _resolve_project(project)
    if await ctx.archive_service.get_archive(aid, viewer=viewer, project=resolved) is None:
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

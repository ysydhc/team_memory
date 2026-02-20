"""Import, export, batch, templates, installables, audit-logs, tags/suggest routes."""

from __future__ import annotations

import logging
import uuid
from pathlib import Path

import httpx
from fastapi import APIRouter, Depends, File, HTTPException, Query, UploadFile
from fastapi.responses import StreamingResponse
from sqlalchemy import select

from team_memory.auth.permissions import require_role
from team_memory.auth.provider import User
from team_memory.services.installable_catalog import (
    InstallableCatalogError,
)
from team_memory.storage.database import get_session
from team_memory.storage.models import AuditLog, Experience
from team_memory.web import app as app_module
from team_memory.web.app import (
    BatchActionRequest,
    InstallableInstallRequest,
    _get_db_url,
    get_current_user,
    get_optional_user,
)

router = APIRouter(tags=["import_export"])


@router.post("/experiences/import")
async def import_experiences(
    file: UploadFile = File(...),
    user: User = Depends(require_role("import")),
):
    """Import experiences from a JSON or CSV file."""
    if not file.filename:
        raise HTTPException(status_code=400, detail="No file provided")

    filename = file.filename.lower()
    if filename.endswith(".json"):
        fmt = "json"
    elif filename.endswith(".csv"):
        fmt = "csv"
    else:
        raise HTTPException(
            status_code=400,
            detail="Unsupported file format. Use .json or .csv",
        )

    content = await file.read()
    data = content.decode("utf-8")

    db_url = _get_db_url()
    async with get_session(db_url) as session:
        result = await app_module._service.import_experiences(
            session=session,
            data=data,
            fmt=fmt,
            created_by=user.name,
        )
        return result


@router.get("/experiences/export")
async def export_experiences(
    format: str = "json",
    tag: str | None = None,
    start: str | None = None,
    end: str | None = None,
    user: User = Depends(get_current_user),
):
    """Export experiences as JSON or CSV."""
    if format not in ("json", "csv"):
        raise HTTPException(status_code=400, detail="format must be 'json' or 'csv'")

    db_url = _get_db_url()
    async with get_session(db_url) as session:
        content = await app_module._service.export_experiences(
            session=session,
            fmt=format,
            tag=tag,
            start=start,
            end=end,
        )

    if format == "json":
        return StreamingResponse(
            iter([content]),
            media_type="application/json",
            headers={"Content-Disposition": "attachment; filename=experiences.json"},
        )
    return StreamingResponse(
        iter([content]),
        media_type="text/csv",
        headers={"Content-Disposition": "attachment; filename=experiences.csv"},
    )


@router.post("/experiences/batch")
async def batch_action(
    req: BatchActionRequest,
    user: User = Depends(require_role("update")),
):
    """Perform batch operations on multiple experiences."""
    if not req.ids:
        raise HTTPException(status_code=400, detail="No IDs provided")
    if req.action not in ("delete", "tag", "publish", "set_scope"):
        raise HTTPException(status_code=400, detail="Invalid action")

    from sqlalchemy import update

    db_url = _get_db_url()
    affected = 0
    async with get_session(db_url) as session:
        uuids = [uuid.UUID(i) for i in req.ids]

        if req.action == "delete":
            result = await session.execute(
                update(Experience)
                .where(Experience.id.in_(uuids))
                .values(is_deleted=True)
            )
            affected = result.rowcount

        elif req.action == "tag" and req.tags:
            exps = await session.execute(
                select(Experience).where(Experience.id.in_(uuids))
            )
            for exp in exps.scalars().all():
                existing = set(exp.tags or [])
                existing.update(req.tags)
                exp.tags = sorted(existing)
                affected += 1

        elif req.action == "publish":
            result = await session.execute(
                update(Experience)
                .where(Experience.id.in_(uuids))
                .values(publish_status="published")
            )
            affected = result.rowcount

        elif req.action == "set_scope" and req.scope:
            result = await session.execute(
                update(Experience)
                .where(Experience.id.in_(uuids))
                .values(scope=req.scope)
            )
            affected = result.rowcount

        await session.commit()

    return {"message": f"Batch {req.action}: {affected} experience(s) affected"}


@router.get("/templates")
async def list_templates(
    user: User | None = Depends(get_optional_user),
):
    """List available workflow templates for experience creation."""
    import yaml as _yaml

    from team_memory.schemas import get_schema_registry

    config_root = Path(__file__).parent.parent.parent.parent / "config"
    template_path = config_root / "templates" / "templates.yaml"
    templates: list[dict] = []
    if template_path.exists():
        with open(template_path) as f:
            data = _yaml.safe_load(f) or {}
        templates = data.get("templates", [])

    registry = get_schema_registry()
    existing_ids = {t.get("id") or t.get("experience_type", "") for t in templates}

    for type_def in registry.get_experience_types():
        if type_def.id in existing_ids:
            for tpl in templates:
                if (tpl.get("id") or tpl.get("experience_type")) == type_def.id:
                    if not tpl.get("severity_options"):
                        if type_def.severity:
                            tpl["severity_options"] = registry.get_severity_levels()
                    if not tpl.get("category_options"):
                        tpl["category_options"] = [
                            c.id for c in registry.get_categories()
                        ]
                    if not tpl.get("progress_states"):
                        tpl["progress_states"] = type_def.progress_states
            continue

        auto_tpl: dict = {
            "id": type_def.id,
            "experience_type": type_def.id,
            "name": type_def.label or type_def.id,
            "description": type_def.label or type_def.id,
            "icon": "📝",
            "core_fields": [
                {"field": "title", "priority": "required"},
                {"field": "problem", "priority": "required"},
                {"field": "solution", "priority": "recommended"},
                {"field": "tags", "priority": "recommended"},
            ],
            "structured_fields": [
                {
                    "field": sf.name,
                    "type": sf.type,
                    "priority": "required" if sf.required else "recommended",
                    "label": sf.label or sf.name,
                    "hint": "",
                }
                for sf in type_def.structured_fields
            ],
            "progress_states": type_def.progress_states,
            "severity_options": (
                registry.get_severity_levels() if type_def.severity else []
            ),
            "category_options": [c.id for c in registry.get_categories()],
            "suggested_tags": [],
            "hints": {
                "title": f"{type_def.label or type_def.id}: [简要描述]",
                "problem": "描述问题背景",
                "solution": "描述解决方案",
            },
        }
        templates.append(auto_tpl)

    return {"templates": templates}


@router.get("/audit-logs")
async def list_audit_logs(
    limit: int = 100,
    action: str | None = None,
    user: User = Depends(require_role("admin")),
):
    """List audit logs (admin only)."""
    db_url = _get_db_url()
    async with get_session(db_url) as session:
        q = select(AuditLog).order_by(AuditLog.created_at.desc()).limit(limit)
        if action:
            q = q.where(AuditLog.action == action)
        result = await session.execute(q)
        logs = result.scalars().all()
        return {"logs": [log.to_dict() for log in logs], "total": len(logs)}


@router.get("/tags/suggest")
async def suggest_tags(
    prefix: str = "",
    limit: int = 20,
    user: User | None = Depends(get_optional_user),
):
    """Suggest tags based on prefix for autocomplete."""
    from sqlalchemy import text as sa_text

    db_url = _get_db_url()
    async with get_session(db_url) as session:
        if prefix:
            result = await session.execute(
                sa_text("""
                    SELECT DISTINCT unnest(tags) as tag
                    FROM experiences WHERE is_deleted = false
                    HAVING unnest(tags) ILIKE :pattern
                    ORDER BY tag LIMIT :limit
                """).bindparams(pattern=f"{prefix}%", limit=limit)
            )
        else:
            result = await session.execute(
                sa_text("""
                    SELECT tag, count(*) as cnt
                    FROM (SELECT unnest(tags) as tag FROM experiences WHERE is_deleted = false) t
                    GROUP BY tag ORDER BY cnt DESC LIMIT :limit
                """).bindparams(limit=limit)
            )
        tags = [row[0] for row in result]

        synonyms = {}
        if app_module._settings:
            synonyms = getattr(app_module._settings, "_tag_synonyms", {})
        return {"tags": tags, "synonyms": synonyms}


@router.get("/installables")
async def list_installables(
    source: str | None = None,
    type: str | None = None,  # noqa: A002
    user: User = Depends(get_current_user),
):
    """List installable rules/prompts from local and registry sources."""
    if source and source not in ("local", "registry"):
        raise HTTPException(status_code=422, detail="source must be local or registry")
    if type and type not in ("rule", "prompt"):  # noqa: A002
        raise HTTPException(status_code=422, detail="type must be rule or prompt")

    try:
        service = app_module._get_catalog_service()
        items = await service.list_items(source=source, item_type=type)
    except InstallableCatalogError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except httpx.HTTPError as exc:
        raise HTTPException(
            status_code=502, detail=f"registry request failed: {exc}"
        ) from exc

    return {"items": [x.model_dump() for x in items], "total": len(items)}


@router.get("/installables/preview")
async def preview_installable(
    id: str = Query(..., min_length=1),  # noqa: A002
    source: str | None = Query(default=None),
    user: User = Depends(get_current_user),
):
    """Preview one installable content by id."""
    if source and source not in ("local", "registry"):
        raise HTTPException(status_code=422, detail="source must be local or registry")

    try:
        service = app_module._get_catalog_service()
        payload = await service.preview(item_id=id, source=source)
    except InstallableCatalogError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except httpx.HTTPError as exc:
        raise HTTPException(
            status_code=502, detail=f"registry request failed: {exc}"
        ) from exc
    return payload


@router.post("/installables/install")
async def install_installable(
    req: InstallableInstallRequest,
    user: User = Depends(require_role("admin")),
):
    """Install one rule/prompt into configured target directory."""

    if req.source and req.source not in ("local", "registry"):
        raise HTTPException(status_code=422, detail="source must be local or registry")

    try:
        service = app_module._get_catalog_service()
        payload = await service.install(item_id=req.id, source=req.source)
    except InstallableCatalogError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except httpx.HTTPError as exc:
        raise HTTPException(
            status_code=502, detail=f"registry request failed: {exc}"
        ) from exc

    log = logging.getLogger("team_memory.web")
    log.info(
        "Installable installed by %s: %s (%s) -> %s",
        user.name,
        req.id,
        req.source or "auto",
        payload.get("target_path"),
    )
    return payload

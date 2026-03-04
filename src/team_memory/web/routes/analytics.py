"""Analytics routes: stats, tags, query-logs, query-stats, analytics overview."""

from __future__ import annotations

import hashlib
import logging
import os
import shutil
from datetime import datetime, timedelta, timezone
from pathlib import Path

import yaml
from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel
from sqlalchemy import Integer, and_, func, select
from sqlalchemy import text as sa_text

from team_memory.auth.provider import User
from team_memory.storage.database import get_session
from team_memory.web import app as app_module
from team_memory.web.app import (
    _get_db_url,
    get_current_user,
    get_optional_user,
)

logger = logging.getLogger(__name__)

router = APIRouter(tags=["analytics"])


def _get_service():
    """Return experience service; raise 503 if not ready."""
    if app_module._service is None:
        raise HTTPException(
            status_code=503,
            detail="Service not ready (bootstrap may not have run)",
        )
    return app_module._service


@router.get("/stats")
async def get_stats(
    project: str | None = None,
    scope: str | None = None,
    user: User | None = Depends(get_optional_user),
):
    """Get experience database statistics. Supports anonymous access."""
    service = _get_service()
    current_user = user.name if user else None
    stats = await service.get_stats(project=project, scope=scope, current_user=current_user)
    return stats


@router.get("/tags")
async def get_tags(
    project: str | None = None,
    scope: str | None = None,
    user: User | None = Depends(get_optional_user),
):
    """Get all tags with counts. Supports anonymous access."""
    service = _get_service()
    current_user = user.name if user else None
    stats = await service.get_stats(project=project, scope=scope, current_user=current_user)
    return {"tags": stats.get("tag_distribution", {})}


@router.get("/projects")
async def list_projects(user: User | None = Depends(get_optional_user)):
    """List distinct project names."""
    service = _get_service()
    projects = await service.list_projects()
    return {"projects": projects}


@router.get("/query-logs")
async def get_query_logs(
    limit: int = 100,
    user: User = Depends(get_current_user),
):
    """Get recent query logs for analytics."""
    service = _get_service()
    logs = await service.get_query_logs(limit=limit)
    return {"logs": logs, "total": len(logs)}


@router.get("/query-stats")
async def get_query_stats(user: User = Depends(get_current_user)):
    """Get query analytics summary."""
    service = _get_service()
    stats = await service.get_query_stats()
    return stats


@router.get("/analytics/overview")
async def analytics_overview(
    days: int = 7,
    user: User = Depends(get_current_user),
):
    """Get analytics overview for dashboard charts."""
    _get_service()  # ensure service is ready
    db_url = _get_db_url()
    async with get_session(db_url) as session:
        # Search volume by day
        search_by_day = await session.execute(
            sa_text("""
                SELECT date_trunc('day', created_at) as day, count(*) as cnt
                FROM query_logs
                WHERE created_at >= now() - interval ':days days'
                GROUP BY day ORDER BY day
            """).bindparams(days=days)
        )
        search_trend = [
            {"date": row[0].isoformat() if row[0] else "", "count": row[1]} for row in search_by_day
        ]

        # Experience growth by day
        exp_by_day = await session.execute(
            sa_text("""
                SELECT date_trunc('day', created_at) as day, count(*) as cnt
                FROM experiences
                WHERE is_deleted = false AND created_at >= now() - interval ':days days'
                GROUP BY day ORDER BY day
            """).bindparams(days=days)
        )
        growth_trend = [
            {"date": row[0].isoformat() if row[0] else "", "count": row[1]} for row in exp_by_day
        ]

        # Tag distribution (top 15)
        tag_dist = await session.execute(
            sa_text("""
                SELECT unnest(tags) as tag, count(*) as cnt
                FROM experiences WHERE is_deleted = false
                GROUP BY tag ORDER BY cnt DESC LIMIT 15
            """)
        )
        tags = [{"tag": row[0], "count": row[1]} for row in tag_dist]

        # Cache stats (from in-memory counters)
        from team_memory.web.metrics import (
            get_avg_latency,
            get_counters,
            get_latency_percentiles,
        )

        counters = get_counters()

        return {
            "search_trend": search_trend,
            "growth_trend": growth_trend,
            "tag_distribution": tags,
            "counters": counters,
            "avg_latency_ms": round(get_avg_latency(), 1),
            "latency_percentiles": get_latency_percentiles(),
        }


@router.get("/analytics/tool-usage")
async def get_tool_usage(
    days: int = 30,
    group_by: str = "tool",  # tool, user, day, api_key
    api_key_name: str | None = None,
    project: list[str] | None = Query(None, description="Filter by project(s); omit for all"),
    user: User = Depends(get_current_user),
):
    """Tool usage analytics: group_by=api_key, filter by api_key_name and project(s)."""
    try:
        db_url = _get_db_url()
        async with get_session(db_url) as session:
            from team_memory.storage.models import ToolUsageLog

            cutoff = datetime.now(timezone.utc) - timedelta(days=days)
            base_filter = ToolUsageLog.created_at >= cutoff
            if api_key_name is not None:
                base_filter = and_(
                    base_filter, ToolUsageLog.api_key_name == api_key_name
                )
            if project:
                base_filter = and_(base_filter, ToolUsageLog.project.in_(project))

            if group_by == "user":
                query = (
                    select(
                        ToolUsageLog.user,
                        func.count().label("count"),
                        func.avg(ToolUsageLog.duration_ms).label("avg_duration"),
                    )
                    .where(base_filter)
                    .group_by(ToolUsageLog.user)
                    .order_by(func.count().desc())
                )
            elif group_by == "day":
                query = (
                    select(
                        func.date_trunc("day", ToolUsageLog.created_at).label("day"),
                        func.count().label("count"),
                    )
                    .where(base_filter)
                    .group_by(func.date_trunc("day", ToolUsageLog.created_at))
                    .order_by(func.date_trunc("day", ToolUsageLog.created_at))
                )
            elif group_by == "api_key":
                # P3-7: group by api_key_name (null -> "(未关联)")
                key_label = func.coalesce(
                    ToolUsageLog.api_key_name, "(未关联)"
                ).label("api_key_name")
                query = (
                    select(
                        key_label,
                        func.count().label("count"),
                        func.avg(ToolUsageLog.duration_ms).label("avg_duration"),
                        func.sum(
                            func.cast(~ToolUsageLog.success, Integer)
                        ).label("errors"),
                    )
                    .where(base_filter)
                    .group_by(ToolUsageLog.api_key_name)
                    .order_by(func.count().desc())
                )
            else:  # group_by == "tool"
                query = (
                    select(
                        ToolUsageLog.tool_name,
                        ToolUsageLog.tool_type,
                        func.count().label("count"),
                        func.avg(ToolUsageLog.duration_ms).label("avg_duration"),
                        func.sum(
                            func.cast(~ToolUsageLog.success, Integer)
                        ).label("errors"),
                    )
                    .where(base_filter)
                    .group_by(ToolUsageLog.tool_name, ToolUsageLog.tool_type)
                    .order_by(func.count().desc())
                )

            result = await session.execute(query)
            rows = result.all()

            if group_by == "user":
                data = [
                    {
                        "user": r[0],
                        "count": r[1],
                        "avg_duration_ms": round(float(r[2] or 0)),
                    }
                    for r in rows
                ]
            elif group_by == "day":
                data = [
                    {"day": r[0].isoformat() if r[0] else None, "count": r[1]}
                    for r in rows
                ]
            elif group_by == "api_key":
                data = [
                    {
                        "api_key_name": r[0],
                        "count": r[1],
                        "avg_duration_ms": round(float(r[2] or 0)),
                        "errors": r[3] or 0,
                    }
                    for r in rows
                ]
            else:
                data = [
                    {
                        "tool_name": r[0],
                        "tool_type": r[1],
                        "count": r[2],
                        "avg_duration_ms": round(float(r[3] or 0)),
                        "errors": r[4] or 0,
                    }
                    for r in rows
                ]

            return {"data": data, "group_by": group_by, "days": days}
    except Exception as exc:
        logger.warning("tool-usage query failed (table may not exist): %s", exc)
        return {"data": [], "group_by": group_by, "days": days}


@router.get("/analytics/tool-usage/summary")
async def get_tool_usage_summary(
    project: list[str] | None = Query(None, description="Filter by project(s); omit for all"),
    user: User = Depends(get_current_user),
):
    """Get tool usage summary: top tools, total calls; optionally filtered by project(s)."""
    try:
        db_url = _get_db_url()
        async with get_session(db_url) as session:
            from team_memory.storage.models import ToolUsageLog

            cutoff = datetime.now(timezone.utc) - timedelta(days=30)
            base_filter = ToolUsageLog.created_at >= cutoff
            if project:
                base_filter = and_(base_filter, ToolUsageLog.project.in_(project))
            top_q = (
                select(
                    ToolUsageLog.tool_name,
                    ToolUsageLog.tool_type,
                    func.count().label("count"),
                )
                .where(base_filter)
                .group_by(ToolUsageLog.tool_name, ToolUsageLog.tool_type)
                .order_by(func.count().desc())
                .limit(10)
            )
            top_result = await session.execute(top_q)
            top_tools = [
                {"tool_name": r[0], "tool_type": r[1], "count": r[2]}
                for r in top_result.all()
            ]

            total_q = (
                select(func.count())
                .select_from(ToolUsageLog)
                .where(base_filter)
            )
            total = (await session.execute(total_q)).scalar_one()

            return {"top_tools": top_tools, "total_calls": total}
    except Exception as exc:
        logger.warning("tool-usage summary failed (table may not exist): %s", exc)
        return {"top_tools": [], "total_calls": 0}


def _extract_summary_from_content(raw: str, max_len: int = 200) -> str:
    """Extract a meaningful summary from skill/rule file content.
    Prefer YAML frontmatter 'description'; else first heading + first paragraph.
    """
    if not raw or not raw.strip():
        return ""
    text = raw.strip()
    # Step 1: try frontmatter description
    if text.startswith("---"):
        parts = text.split("---", 2)
        if len(parts) >= 2 and parts[1].strip():
            try:
                fm = yaml.safe_load(parts[1])
                if isinstance(fm, dict):
                    desc = fm.get("description")
                    if isinstance(desc, str) and desc.strip():
                        s = desc.strip().replace("\n", " ")[:max_len]
                        return s + "…" if len(desc.strip()) > max_len else s
            except Exception:
                pass
        # body is after second ---
        body = parts[2].strip() if len(parts) > 2 else ""
    else:
        body = text
    # Step 2: fallback — first # heading and/or first paragraph
    lines = [ln.strip() for ln in body.splitlines()]
    first_heading = ""
    first_para: list[str] = []
    in_para = False
    for ln in lines:
        if not ln:
            if in_para:
                break
            continue
        if ln.startswith("#"):
            if in_para:
                break
            head = ln.lstrip("#").strip()
            if head and not first_heading:
                first_heading = head
            continue
        in_para = True
        first_para.append(ln)
    para_text = " ".join(first_para).replace("**", "").replace("`", "").strip()
    combined = first_heading
    if para_text and combined:
        combined = combined + " — " + para_text
    elif para_text:
        combined = para_text
    if len(combined) > max_len:
        combined = combined[: max_len - 1] + "…"
    return combined.strip() or ""


def _scan_directory(base: Path, pattern: str = "*") -> list[dict]:
    """Scan a directory for matching files and return metadata."""
    items: list[dict] = []
    if not base.exists():
        return items
    for p in sorted(base.rglob(pattern)):
        if p.is_file() and not p.name.startswith("."):
            rel = str(p.relative_to(base))
            try:
                size = p.stat().st_size
                mtime = datetime.fromtimestamp(p.stat().st_mtime, tz=timezone.utc).isoformat()
            except OSError:
                size = 0
                mtime = None
            display_name = p.stem
            if display_name.upper() == "SKILL" and p.parent != base:
                display_name = p.parent.name
            summary = ""
            if size > 0 and size < 51200:
                try:
                    raw = p.read_text("utf-8", errors="ignore")
                    summary = _extract_summary_from_content(raw, max_len=200)
                except Exception:
                    pass
            items.append({
                "name": display_name,
                "path": rel,
                "full_path": str(p),
                "dir_path": str(p.parent),
                "size_bytes": size,
                "modified_at": mtime,
                "summary": summary or None,
            })
    return items


# Categories that are skill folders (SKILL.md + scripts/references/etc.)
SKILL_FOLDER_CATEGORIES = (
    "claude_skills",
    "cursor_skills",
    "user_claude_skills",
    "user_cursor_skills",
)


def _safe_path_hash(path: Path) -> str:
    """Stable short hash for workspace base path (for cache key)."""
    return hashlib.sha256(str(path.resolve()).encode()).hexdigest()[:12]


def _disabled_cache_key(
    category: str, base_path: Path, file_path: str, *, is_skill_folder: bool
) -> str:
    """Cache key for disabled item: category + workspace hash + identifier."""
    safe = _safe_path_hash(base_path)
    if is_skill_folder:
        skill_dir_name = file_path.split("/")[0]
        return f"{category}__{safe}__{skill_dir_name}"
    return f"{category}__{safe}__{file_path.replace('/', '__')}"


def _disabled_cache_name(category: str, file_path: str) -> str:
    """Legacy cache filename (single file); prefer _disabled_cache_key with is_skill_folder."""
    return category + "__" + file_path.replace("/", "__")


def _get_scan_config() -> dict:
    """Read scan directories config from .team_memory/scan_config.json."""
    import json as json_mod

    cfg_path = Path.home() / ".team_memory" / "scan_config.json"
    if cfg_path.exists():
        try:
            return json_mod.loads(cfg_path.read_text("utf-8"))
        except Exception:
            pass
    return {}


def _save_scan_config(cfg: dict) -> None:
    """Persist scan directories config."""
    import json as json_mod

    cfg_dir = Path.home() / ".team_memory"
    cfg_dir.mkdir(parents=True, exist_ok=True)
    (cfg_dir / "scan_config.json").write_text(
        json_mod.dumps(cfg, ensure_ascii=False, indent=2), "utf-8"
    )


@router.get("/config/scan-dirs")
async def get_scan_dirs_config(user: User | None = Depends(get_optional_user)):
    """Get configured scan directories and project-path mapping."""
    cfg = _get_scan_config()
    return {
        "project_paths": cfg.get("project_paths", {}),
        "extra_scan_dirs": cfg.get("extra_scan_dirs", []),
    }


class ScanDirsConfigUpdate(BaseModel):
    project_paths: dict[str, str] | None = None
    extra_scan_dirs: list[dict] | None = None


@router.put("/config/scan-dirs")
async def update_scan_dirs_config(
    req: ScanDirsConfigUpdate,
    user: User = Depends(get_current_user),
):
    """Update scan directories config."""
    cfg = _get_scan_config()
    if req.project_paths is not None:
        cfg["project_paths"] = req.project_paths
    if req.extra_scan_dirs is not None:
        cfg["extra_scan_dirs"] = req.extra_scan_dirs
    _save_scan_config(cfg)
    return {"message": "扫描目录配置已保存", "config": cfg}


@router.get("/analytics/skills-rules")
async def get_skills_rules_stats(
    workspace: str | None = None,
    project: str | None = None,
    user: User | None = Depends(get_optional_user),
):
    """Scan .claude and .cursor directories for skills/rules files."""
    scan_cfg = _get_scan_config()
    project_paths = scan_cfg.get("project_paths", {})

    if workspace:
        ws = Path(workspace)
    elif project and project in project_paths:
        ws = Path(project_paths[project])
    else:
        ws = Path(os.getcwd())

    scan_targets = {
        "claude_skills": (ws / ".claude" / "skills", "SKILL.md"),
        "cursor_rules": (ws / ".cursor" / "rules", "*.mdc"),
        "cursor_prompts": (ws / ".cursor" / "prompts", "*.md"),
        "cursor_skills": (ws / ".cursor" / "skills-cursor", "SKILL.md"),
    }

    user_home = Path.home()
    user_targets = {
        "user_claude_skills": (user_home / ".claude" / "skills", "SKILL.md"),
        "user_cursor_skills": (user_home / ".cursor" / "skills-cursor", "SKILL.md"),
    }

    extra_dirs = scan_cfg.get("extra_scan_dirs", [])
    extra_targets = {}
    for i, d in enumerate(extra_dirs):
        p = Path(d.get("path", ""))
        pat = d.get("pattern", "*")
        label = d.get("label", f"extra_{i}")
        if p.exists():
            extra_targets[label] = (p, pat)

    result: dict = {"workspace": str(ws), "categories": {}}
    total_files = 0

    all_targets = {**scan_targets, **user_targets, **extra_targets}
    cache_dir = Path.home() / ".team_memory" / "disabled_skills"
    safe_by_path = {}

    for key, (path, pat) in all_targets.items():
        files = _scan_directory(path, pat)
        existing_paths = {f["path"] for f in files}
        if cache_dir.exists():
            safe = safe_by_path.setdefault(str(path), _safe_path_hash(path))
            prefix = f"{key}__{safe}__"
            for cache_item in cache_dir.iterdir():
                name = cache_item.name
                if not name.startswith(prefix):
                    continue
                suffix = name[len(prefix) :]
                if cache_item.is_dir():
                    # Skill folder cache: suffix is skill_dir_name
                    skill_dir_name = suffix
                    rel = f"{skill_dir_name}/SKILL.md"
                    if rel in existing_paths:
                        continue
                    full = path / rel
                    skill_md = cache_item / "SKILL.md"
                    size = 0
                    mtime = None
                    summary = ""
                    if skill_md.exists() and skill_md.is_file():
                        try:
                            size = skill_md.stat().st_size
                            mtime = datetime.fromtimestamp(
                                skill_md.stat().st_mtime, tz=timezone.utc
                            ).isoformat()
                            if size > 0 and size < 51200:
                                raw = skill_md.read_text("utf-8", errors="ignore")
                                summary = _extract_summary_from_content(raw, max_len=200)
                        except (OSError, Exception):
                            pass
                    files.append({
                        "name": skill_dir_name,
                        "path": rel,
                        "full_path": str(full),
                        "dir_path": str(full.parent),
                        "size_bytes": size,
                        "modified_at": mtime,
                        "summary": summary or None,
                        "enabled": False,
                    })
                elif cache_item.is_file():
                    rel = suffix.replace("__", "/")
                    if rel in existing_paths:
                        continue
                    full = path / rel
                    display_name = Path(rel).stem
                    if display_name.upper() == "SKILL" and "/" in rel:
                        display_name = Path(rel).parent.name
                    try:
                        size = cache_item.stat().st_size
                        mtime = datetime.fromtimestamp(
                            cache_item.stat().st_mtime, tz=timezone.utc
                        ).isoformat()
                    except OSError:
                        size = 0
                        mtime = None
                    summary = ""
                    if size > 0 and size < 51200:
                        try:
                            raw = cache_item.read_text("utf-8", errors="ignore")
                            summary = _extract_summary_from_content(raw, max_len=200)
                        except Exception:
                            pass
                    files.append({
                        "name": display_name,
                        "path": rel,
                        "full_path": str(full),
                        "dir_path": str(full.parent),
                        "size_bytes": size,
                        "modified_at": mtime,
                        "summary": summary or None,
                        "enabled": False,
                    })
        total_files += len(files)
        result["categories"][key] = {
            "path": str(path),
            "count": len(files),
            "files": files,
        }

    result["total_files"] = total_files
    result["project_paths"] = project_paths

    for _cat in result["categories"].values():
        for f in _cat.get("files", []):
            if "enabled" not in f:
                fp = f.get("full_path")
                f["enabled"] = bool(fp and Path(fp).exists())

    return result


class ToggleSkillRequest(BaseModel):
    category: str
    file_path: str
    enabled: bool


@router.post("/analytics/skills-rules/toggle")
async def toggle_skill_file(
    req: ToggleSkillRequest,
    project: str | None = None,
    user: User | None = Depends(get_optional_user),
):
    """Enable or disable a skill/rule. Skills (SKILL.md in a folder) move the whole folder."""
    scan_cfg = _get_scan_config()
    project_paths = scan_cfg.get("project_paths", {})
    if project and project in project_paths:
        ws = Path(project_paths[project])
    else:
        ws = Path(os.getcwd())

    scan_targets = {
        "claude_skills": (ws / ".claude" / "skills", "SKILL.md"),
        "cursor_rules": (ws / ".cursor" / "rules", "*.mdc"),
        "cursor_prompts": (ws / ".cursor" / "prompts", "*.md"),
        "cursor_skills": (ws / ".cursor" / "skills-cursor", "SKILL.md"),
    }
    user_home = Path.home()
    user_targets = {
        "user_claude_skills": (user_home / ".claude" / "skills", "SKILL.md"),
        "user_cursor_skills": (user_home / ".cursor" / "skills-cursor", "SKILL.md"),
    }
    extra_dirs = scan_cfg.get("extra_scan_dirs", [])
    extra_targets = {}
    for i, d in enumerate(extra_dirs):
        p = Path(d.get("path", ""))
        pat = d.get("pattern", "*")
        label = d.get("label", f"extra_{i}")
        if p.exists():
            extra_targets[label] = (p, pat)

    all_targets = {**scan_targets, **user_targets, **extra_targets}
    cat_info = all_targets.get(req.category)
    if not cat_info:
        raise HTTPException(status_code=400, detail=f"Unknown category: {req.category}")

    base_path, pat = cat_info
    full_path = base_path / req.file_path
    cache_dir = Path.home() / ".team_memory" / "disabled_skills"
    is_skill_folder = (
        req.category in SKILL_FOLDER_CATEGORIES or pat == "SKILL.md"
    )
    cache_key = _disabled_cache_key(
        req.category, base_path, req.file_path, is_skill_folder=is_skill_folder
    )
    cache_path = cache_dir / cache_key

    if is_skill_folder:
        # Resolve skill dir: directory containing SKILL.md (so we move the whole folder)
        if full_path.exists() and full_path.is_file():
            skill_dir = full_path.parent
        else:
            skill_dir_name = req.file_path.split("/")[0] if "/" in req.file_path else req.file_path
            skill_dir = base_path / skill_dir_name

        if not skill_dir.exists() and req.enabled:
            if cache_path.exists() and cache_path.is_dir():
                cache_dir.mkdir(parents=True, exist_ok=True)
                # copytree(src, dst) requires dst to NOT exist; do not mkdir(skill_dir) first
                shutil.copytree(str(cache_path), str(skill_dir))
                shutil.rmtree(str(cache_path))
                return {"status": "enabled", "file": req.file_path}
            raise HTTPException(
                status_code=404, detail=f"Skill folder not found: {skill_dir}"
            )

        if not req.enabled:
            if skill_dir.exists() and skill_dir.is_dir():
                cache_dir.mkdir(parents=True, exist_ok=True)
                if cache_path.exists():
                    shutil.rmtree(str(cache_path))
                shutil.copytree(str(skill_dir), str(cache_path))
                shutil.rmtree(str(skill_dir))
                return {"status": "disabled", "file": req.file_path}
            raise HTTPException(
                status_code=404, detail=f"Skill folder not found: {skill_dir}"
            )
        return {"status": "already_enabled", "file": req.file_path}

    # Single-file (rules, prompts)
    if not full_path.exists() and req.enabled:
        if cache_path.exists() and cache_path.is_file():
            full_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(str(cache_path), str(full_path))
            os.remove(str(cache_path))
            return {"status": "enabled", "file": req.file_path}
        raise HTTPException(status_code=404, detail=f"File not found: {full_path}")

    if not req.enabled:
        if full_path.exists():
            cache_dir.mkdir(parents=True, exist_ok=True)
            shutil.copy2(str(full_path), str(cache_path))
            os.remove(str(full_path))
            return {"status": "disabled", "file": req.file_path}
        raise HTTPException(status_code=404, detail=f"File not found: {full_path}")
    return {"status": "already_enabled", "file": req.file_path}


def _get_allowed_preview_roots() -> list[Path]:
    """Return list of allowed root paths for preview (workspace + user dirs)."""
    scan_cfg = _get_scan_config()
    project_paths = scan_cfg.get("project_paths", {})
    roots: list[Path] = [Path(os.getcwd())]
    roots.extend(Path(p) for p in project_paths.values() if p)
    roots.append(Path.home() / ".claude" / "skills")
    roots.append(Path.home() / ".cursor" / "skills-cursor")
    roots.append(Path.home() / ".cursor" / "rules")
    return roots


def _resolve_preview_from_cache(file_path: Path) -> str | None:
    """If path is a disabled skill/rule, return content from cache; else None."""
    cache_dir = Path.home() / ".team_memory" / "disabled_skills"
    if not cache_dir.exists():
        return None
    scan_cfg = _get_scan_config()
    project_paths = scan_cfg.get("project_paths", {})
    workspaces = [Path(os.getcwd())] + [Path(p) for p in project_paths.values() if p]
    user_home = Path.home()
    for ws in workspaces:
        scan_targets = {
            "claude_skills": (ws / ".claude" / "skills", "SKILL.md"),
            "cursor_rules": (ws / ".cursor" / "rules", "*.mdc"),
            "cursor_prompts": (ws / ".cursor" / "prompts", "*.md"),
            "cursor_skills": (ws / ".cursor" / "skills-cursor", "SKILL.md"),
        }
        user_targets = {
            "user_claude_skills": (user_home / ".claude" / "skills", "SKILL.md"),
            "user_cursor_skills": (user_home / ".cursor" / "skills-cursor", "SKILL.md"),
        }
        for _cat, (base_path, pat) in {**scan_targets, **user_targets}.items():
            try:
                if not file_path.resolve().is_relative_to(base_path.resolve()):
                    continue
                rel = file_path.relative_to(base_path)
                is_skill = _cat in SKILL_FOLDER_CATEGORIES or pat == "SKILL.md"
                if is_skill and len(rel.parts) >= 2:
                    cache_key = _disabled_cache_key(
                        _cat, base_path, str(rel), is_skill_folder=True
                    )
                    cached_dir = cache_dir / cache_key
                    if cached_dir.exists() and cached_dir.is_dir():
                        inner = Path(*rel.parts[1:])
                        path_in_cache = cached_dir / inner
                        if path_in_cache.exists() and path_in_cache.is_file():
                            return path_in_cache.read_text("utf-8", errors="replace")
                else:
                    cache_key = _disabled_cache_key(
                        _cat, base_path, str(rel), is_skill_folder=False
                    )
                    cached = cache_dir / cache_key
                    if cached.exists() and cached.is_file():
                        return cached.read_text("utf-8", errors="replace")
            except (ValueError, OSError):
                continue
    return None


@router.get("/analytics/skills-rules/preview")
async def preview_skill_file(
    path: str,
    user: User | None = Depends(get_optional_user),
):
    """Read a skill/rule file content for preview. Path must be under allowed dirs."""
    file_path = Path(path).resolve()
    allowed = [r.resolve() for r in _get_allowed_preview_roots()]
    try:
        if not any(file_path.is_relative_to(r) for r in allowed):
            raise HTTPException(status_code=403, detail="Path not allowed")
    except (ValueError, OSError):
        raise HTTPException(status_code=403, detail="Path not allowed") from None

    if file_path.exists() and file_path.is_file():
        try:
            content = file_path.read_text("utf-8")
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e)) from e
    else:
        content = _resolve_preview_from_cache(file_path)
        if content is None:
            raise HTTPException(status_code=404, detail="File not found")
    if len(content) > 50000:
        content = content[:50000] + "\n\n... (truncated)"
    return {"content": content, "path": str(file_path), "name": file_path.name}

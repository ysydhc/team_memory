"""Janitor management and cleanup status routes."""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy import select

from team_memory.auth.provider import User
from team_memory.bootstrap import get_context
from team_memory.storage.models import Experience
from team_memory.web import app as app_module
from team_memory.web.app import _get_db_url, _resolve_project
from team_memory.web.dependencies import require_admin

router = APIRouter(tags=["janitor"])


@router.post("/api/v1/janitor/run")
async def run_janitor(
    project: str | None = None,
    user: User = Depends(require_admin()),
):
    """手动触发清理任务（需admin权限）。

    Args:
        project: 可选的项目名，限制清理范围

    Returns:
        清理任务的执行结果
    """
    context = get_context()

    if context.janitor is None:
        raise HTTPException(
            status_code=503,
            detail="Janitor service not available (may not be initialized)",
        )

    # 解析项目名
    resolved_project = _resolve_project(project) if project else None

    try:
        result = await context.janitor.run_all(resolved_project)
        return {
            "message": "Janitor cleanup completed successfully",
            "project": resolved_project,
            "results": result,
        }
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Janitor cleanup failed: {str(e)}",
        )


@router.get("/api/v1/janitor/status")
async def get_janitor_status(
    user: User = Depends(require_admin()),
):
    """查看清理状态和调度器信息。

    Returns:
        调度器运行状态和最近执行结果
    """
    context = get_context()

    status = {
        "janitor_available": context.janitor is not None,
        "scheduler_available": context.janitor_scheduler is not None,
        "scheduler_running": False,
        "last_run": None,
        "config": {},
    }

    if context.janitor_scheduler:
        status["scheduler_running"] = context.janitor_scheduler.is_running()

        # 获取调度器配置信息
        if hasattr(context.janitor_scheduler, "_config") and context.janitor_scheduler._config:
            config = context.janitor_scheduler._config
            status["config"] = {
                "interval_hours": getattr(config, "janitor_interval_hours", "unknown"),
                "enabled": getattr(config, "janitor_enabled", "unknown"),
            }

    if context.janitor and hasattr(context.janitor, "_config") and context.janitor._config:
        janitor_config = context.janitor._config
        status["janitor_config"] = {
            "protection_period_days": getattr(janitor_config, "protection_period_days", 10),
            "auto_soft_delete_outdated": getattr(
                janitor_config, "auto_soft_delete_outdated", False
            ),
            "purge_soft_deleted_days": getattr(janitor_config, "purge_soft_deleted_days", 30),
            "draft_expiry_days": getattr(janitor_config, "draft_expiry_days", 7),
            "personal_memory_retention_days": getattr(
                janitor_config, "personal_memory_retention_days", 90
            ),
        }

    return status


@router.get("/api/v1/experiences/outdated")
async def list_outdated_experiences(
    limit: int = Query(20, ge=1, le=100),
    offset: int = Query(0, ge=0),
    project: str | None = None,
    user: User = Depends(require_admin()),
):
    """列出Outdated经验。

    Args:
        limit: 返回条数限制
        offset: 偏移量
        project: 可选的项目过滤

    Returns:
        Outdated状态的经验列表
    """
    db_url = _get_db_url()
    resolved_project = _resolve_project(project) if project else None

    async with app_module.get_session(db_url) as session:
        # 构建查询
        query = (
            select(Experience)
            .where(Experience.quality_tier == "Outdated")
            .where(Experience.is_deleted == False)  # noqa: E712
            .order_by(Experience.last_scored_at.desc())
        )

        # 添加项目过滤
        if resolved_project:
            query = query.where(Experience.project == resolved_project)

        # 计算总数
        count_query = (
            select(Experience.id)
            .where(Experience.quality_tier == "Outdated")
            .where(Experience.is_deleted == False)  # noqa: E712
        )
        if resolved_project:
            count_query = count_query.where(Experience.project == resolved_project)

        # 执行查询
        total_result = await session.execute(count_query)
        total = len(total_result.all())

        # 分页查询
        paginated_query = query.limit(limit).offset(offset)
        result = await session.execute(paginated_query)
        experiences = result.scalars().all()

        # 转换为字典格式
        exp_list = []
        for exp in experiences:
            exp_dict = exp.to_dict()
            # 添加一些有用的清理相关信息
            exp_dict["cleanup_info"] = {
                "quality_score": exp.quality_score,
                "quality_tier": exp.quality_tier,
                "last_scored_at": exp.last_scored_at.isoformat() if exp.last_scored_at else None,
                "is_pinned": exp.is_pinned,
            }
            exp_list.append(exp_dict)

        return {
            "items": exp_list,
            "total": total,
            "limit": limit,
            "offset": offset,
            "project": resolved_project,
            "message": f"Found {total} outdated experiences"
            + (f" in project '{resolved_project}'" if resolved_project else ""),
        }

"""Janitor management and cleanup status routes."""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, Query

from team_memory.auth.provider import User
from team_memory.bootstrap import get_context
from team_memory.web.app import _resolve_project
from team_memory.web.dependencies import require_admin

router = APIRouter(tags=["janitor"])


@router.post("/janitor/run")
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


@router.get("/janitor/status")
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
            interval_val = getattr(config, "janitor_interval_hours", None)
            if not isinstance(interval_val, (int, float)) or isinstance(interval_val, bool):
                interval_val = getattr(config, "interval_hours", "unknown")
            enabled_val = getattr(config, "janitor_enabled", None)
            if not isinstance(enabled_val, bool):
                enabled_val = getattr(config, "enabled", "unknown")
            status["config"] = {
                "interval_hours": interval_val,
                "enabled": enabled_val,
            }

    if context.janitor and hasattr(context.janitor, "_config") and context.janitor._config:
        janitor_config = context.janitor._config
        status["janitor_config"] = {
            "protection_period_days": getattr(janitor_config, "protection_period_days", 10),
            "auto_soft_delete_outdated": getattr(janitor_config, "auto_soft_delete_outdated", True),
            "purge_soft_deleted_days": getattr(janitor_config, "purge_soft_deleted_days", 30),
            "draft_expiry_days": getattr(janitor_config, "draft_expiry_days", 7),
            "personal_memory_retention_days": getattr(janitor_config, "personal_memory_retention_days", 90),
        }

    return status


@router.get("/experiences/outdated")
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
    context = get_context()
    return await context.service.list_outdated(
        limit=limit,
        offset=offset,
        project=project,
    )

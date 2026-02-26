"""Analytics routes: stats, tags, query-logs, query-stats, analytics overview."""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy import text as sa_text

from team_memory.auth.provider import User
from team_memory.storage.database import get_session
from team_memory.web import app as app_module
from team_memory.web.app import (
    _get_db_url,
    get_current_user,
    get_optional_user,
)

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
async def get_stats(user: User | None = Depends(get_optional_user)):
    """Get experience database statistics. Supports anonymous access."""
    service = _get_service()
    stats = await service.get_stats()
    return stats


@router.get("/tags")
async def get_tags(user: User | None = Depends(get_optional_user)):
    """Get all tags with counts. Supports anonymous access."""
    service = _get_service()
    stats = await service.get_stats()
    return {"tags": stats.get("tag_distribution", {})}


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
            {"date": row[0].isoformat() if row[0] else "", "count": row[1]}
            for row in search_by_day
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
            {"date": row[0].isoformat() if row[0] else "", "count": row[1]}
            for row in exp_by_day
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

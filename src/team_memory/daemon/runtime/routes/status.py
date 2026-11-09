"""Status and monitoring endpoints for the TM Daemon.

GET /status
GET /status/faithfulness
GET /stats
GET /persona/profile
"""

from __future__ import annotations

import logging
from typing import Any

from fastapi import APIRouter, Request

from team_memory.daemon.runtime.config import DaemonConfig
from team_memory.services.search.search_log_writer import SearchLogWriter

logger = logging.getLogger("daemon.app")

router = APIRouter()


@router.get("/status")
async def get_status(request: Request) -> dict[str, Any]:
    """Return daemon status and TM mode."""
    app = request.app
    heartbeat = getattr(app.state, "heartbeat", None)
    result: dict[str, Any] = {
        "status": "running",
        "tm_mode": getattr(app.state, "tm_mode", "unknown"),
    }
    if heartbeat:
        result["heartbeat"] = heartbeat.to_dict()
    # Worker health — exposes RefinementWorker tick status
    rw = getattr(app.state, "refinement_worker", None)
    if rw:
        consecutive_fails = getattr(rw, "_consecutive_tick_failures", None)
        if consecutive_fails is None:
            fail_cnt = getattr(rw, "_tick_fail_count", 0)
            tick_cnt = getattr(rw, "_tick_count", 0)
            is_healthy = fail_cnt == 0 or (fail_cnt < 5 and tick_cnt > fail_cnt * 2)
        else:
            is_healthy = consecutive_fails < 3
        result["refinement_worker"] = {
            "tick_count": getattr(rw, "_tick_count", 0),
            "fail_count": getattr(rw, "_tick_fail_count", 0),
            "consecutive_failures": consecutive_fails if consecutive_fails is not None else 0,
            "last_tick_ok": getattr(rw, "_last_tick_ok", None),
            "healthy": is_healthy,
        }
    janitor_sched = getattr(app.state, "janitor_scheduler", None)
    dream_worker = getattr(app.state, "dream_worker", None)
    last_janitor_dt = (
        janitor_sched._read_last_run()
        if janitor_sched and hasattr(janitor_sched, "_read_last_run")
        else None
    )
    result["maintenance"] = {
        "janitor_running": bool(janitor_sched and getattr(janitor_sched, "is_running", False)),
        "last_janitor_at": last_janitor_dt.isoformat() if last_janitor_dt else None,
        "auto_dream_running": bool(
            dream_worker and getattr(dream_worker, "_task", None) and not dream_worker._task.done()
        ),
    }
    return result


@router.get("/status/faithfulness")
async def get_faithfulness_status(request: Request, limit: int = 10) -> dict[str, Any]:
    """Return recent faithfulness buffer entries with scores."""
    app = request.app
    response_buffer = getattr(app.state, "response_buffer", None)
    if response_buffer is None:
        return {"error": "response_buffer not initialized"}

    stats = await response_buffer.get_stats(days=7)
    await response_buffer.fetch_pending(limit=0)  # don't fetch pending

    # Get recent entries (all, not just pending)
    try:
        pool = await response_buffer._ensure_pool()
        async with pool.acquire() as conn:
            rows = await conn.fetch(
                """
                SELECT id::text, query, evaluated, faithfulness_score, created_at::text
                FROM response_buffer
                ORDER BY created_at DESC
                LIMIT $1
                """,
                limit,
            )
        recent = [
            {
                "id": str(r["id"]),
                "query": r["query"][:100],
                "evaluated": r["evaluated"],
                "score": r["faithfulness_score"],
                "created_at": r["created_at"],
            }
            for r in rows
        ]
    except Exception:
        recent = []

    return {
        "stats": stats,
        "recent": recent,
    }


@router.get("/stats")
async def get_stats(request: Request, days: int = 7) -> dict[str, Any]:
    """Get evaluation statistics for the last N days."""
    search_log: SearchLogWriter = request.app.state.search_log
    return await search_log.get_stats(days=days)


@router.get("/persona/profile")
async def persona_profile(request: Request, persona: str = "general") -> dict[str, Any]:
    """Return user's personal memory profile for a given persona.

    Useful for debugging/inspection of persona-specific memories.
    """
    app = request.app
    config_obj: DaemonConfig = app.state.config
    user_id = getattr(config_obj.tm, "user", "admin") or "admin"

    try:
        from team_memory.config import load_settings as _load_tm_settings
        from team_memory.embedding.factory import create_embedding_provider
        from team_memory.services.memory.personal_memory import PersonalMemoryService

        _settings = _load_tm_settings()
        embedding = create_embedding_provider(_settings)
        pms = PersonalMemoryService(
            embedding_provider=embedding,
            db_url=str(_settings.database.url),
        )
        profile = await pms.build_profile_for_user(user_id=user_id, persona=persona)
        return {
            "user_id": user_id,
            "persona": persona,
            "static": profile.get("static", []),
            "dynamic": profile.get("dynamic", []),
        }
    except Exception as e:
        logger.warning("persona/profile failed: %s", e, exc_info=True)
        return {
            "user_id": user_id,
            "persona": persona,
            "static": [],
            "dynamic": [],
            "error": str(e),
        }

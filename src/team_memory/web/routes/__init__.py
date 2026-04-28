"""Route modules aggregated for v1 API. Exports mount_all to wire sub-routers."""

from __future__ import annotations

from fastapi import APIRouter

from team_memory.web.routes import (
    archives,
    auth,
    config,
    dedup,
    entities,
    experiences,
    janitor,
    mcp_compat,
    personal_memory,
    search,
)


def mount_all(parent_router: APIRouter) -> None:
    """Include all sub-routers on the given parent (v1 router)."""
    # /dedup/* before /experiences/* — avoids any ambiguity with path params.
    parent_router.include_router(dedup.router)
    parent_router.include_router(archives.router)
    parent_router.include_router(experiences.router)
    parent_router.include_router(entities.router)
    parent_router.include_router(janitor.router)
    parent_router.include_router(auth.router)
    parent_router.include_router(config.router)
    parent_router.include_router(search.router)
    parent_router.include_router(personal_memory.router)
    parent_router.include_router(mcp_compat.router)

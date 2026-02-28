"""Route modules aggregated for v1 API. Exports mount_all to wire sub-routers."""

from __future__ import annotations

from fastapi import APIRouter

from team_memory.web.routes import (
    analytics,
    auth,
    config,
    experiences,
    import_export,
    lifecycle,
    parse,
    schema,
    search,
    tasks,
)


def mount_all(parent_router: APIRouter) -> None:
    """Include all sub-routers on the given parent (v1 router).

    Route order matters: more specific paths (e.g. /experiences/parse-document)
    must be registered before parameterized paths (e.g. /experiences/{id}).
    """
    # Parse routes first (they use /experiences/parse-* before /experiences/{id})
    parent_router.include_router(parse.router)
    parent_router.include_router(experiences.router)
    parent_router.include_router(tasks.router)
    parent_router.include_router(auth.router)
    parent_router.include_router(search.router)
    parent_router.include_router(config.router)
    parent_router.include_router(schema.router)
    parent_router.include_router(lifecycle.router)
    parent_router.include_router(analytics.router)
    parent_router.include_router(import_export.router)

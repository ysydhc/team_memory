"""Route sub-modules for the TM Daemon FastAPI application.

Each sub-module defines an APIRouter that is included by create_app().
"""

from __future__ import annotations

from team_memory.daemon.runtime.routes.debug import router as debug_router
from team_memory.daemon.runtime.routes.drafts import router as drafts_router
from team_memory.daemon.runtime.routes.hooks import router as hooks_router
from team_memory.daemon.runtime.routes.recall import router as recall_router
from team_memory.daemon.runtime.routes.status import router as status_router
from team_memory.daemon.runtime.routes.wiki import router as wiki_router

__all__ = [
    "debug_router",
    "drafts_router",
    "hooks_router",
    "recall_router",
    "status_router",
    "wiki_router",
]

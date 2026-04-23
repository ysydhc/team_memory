"""TM Daemon — FastAPI application skeleton.

Provides HTTP API endpoints for hook callbacks, draft management,
and recall queries. Uses TMSink abstraction for TM access.
"""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI
from pydantic import BaseModel, Field

from daemon.config import DaemonConfig, load_config
from daemon.draft_refiner import DraftRefiner
from daemon.pipeline import (
    process_after_response,
    process_before_prompt,
    process_session_end,
    process_session_start,
)
from daemon.tm_sink import TMSink, create_sink
from hooks.convergence_detector import ConvergenceDetector
from hooks.draft_buffer import DraftBuffer

logger = logging.getLogger("daemon.app")


# ---------------------------------------------------------------------------
# Pydantic request models
# ---------------------------------------------------------------------------


class HookPayload(BaseModel):
    """Payload for hook callback endpoints."""

    conversation_id: str = ""
    prompt: str = ""
    workspace_roots: list[str] = Field(default_factory=list)
    model: str = ""


class DraftSavePayload(BaseModel):
    """Payload for /draft/save endpoint."""

    title: str
    content: str
    project: str | None = None
    group_key: str | None = None
    conversation_id: str | None = None


class DraftPublishPayload(BaseModel):
    """Payload for /draft/publish endpoint."""

    draft_id: str
    refined_content: str | None = None


class RecallQuery(BaseModel):
    """Query params for /recall endpoint."""

    query: str | None = None
    project: str | None = None
    max_results: int = 5


# ---------------------------------------------------------------------------
# App factory
# ---------------------------------------------------------------------------


def create_app(config: DaemonConfig | None = None) -> FastAPI:
    """Create and configure the FastAPI application.

    Args:
        config: Optional DaemonConfig. If None, loaded via load_config().

    Returns:
        Configured FastAPI application instance.
    """
    if config is None:
        config = load_config()

    @asynccontextmanager
    async def lifespan(application: FastAPI):
        """Initialize resources on startup, cleanup on shutdown."""
        # -- Startup --
        sink_config = {
            "mode": config.tm.mode,
            "base_url": config.tm.base_url,
            "user": config.tm.user,
        }
        sink: TMSink = create_sink(sink_config)
        application.state.sink = sink
        application.state.tm_mode = config.tm.mode

        db_path = config.draft.db_path or ":memory:"
        buf = DraftBuffer(db_path)
        await buf.__aenter__()
        application.state.buf = buf

        detector = ConvergenceDetector()
        application.state.detector = detector

        refiner = DraftRefiner(sink=sink, draft_buffer=buf)
        application.state.refiner = refiner

        logger.info("TM Daemon started (mode=%s)", config.tm.mode)
        yield

        # -- Shutdown --
        buf_obj: DraftBuffer | None = getattr(application.state, "buf", None)
        if buf_obj is not None:
            await buf_obj.__aexit__(None, None, None)
        logger.info("TM Daemon shutdown")

    app = FastAPI(title="TM Daemon", lifespan=lifespan)

    # ------------------------------------------------------------------
    # Endpoints
    # ------------------------------------------------------------------

    @app.get("/status")
    async def get_status() -> dict[str, str]:
        """Return daemon status and TM mode."""
        return {
            "status": "running",
            "tm_mode": getattr(app.state, "tm_mode", "unknown"),
        }

    @app.post("/hooks/after_response")
    async def hook_after_response(payload: HookPayload) -> dict[str, Any]:
        """Process agent response (draft pipeline)."""
        result = await process_after_response(
            sink=app.state.sink,
            buf=app.state.buf,
            detector=app.state.detector,
            refiner=app.state.refiner,
            payload=payload.model_dump(),
        )
        return result

    @app.post("/hooks/session_start")
    async def hook_session_start(payload: HookPayload) -> dict[str, Any]:
        """Inject project context at session start."""
        result = await process_session_start(
            sink=app.state.sink,
            payload=payload.model_dump(),
        )
        return result

    @app.post("/hooks/before_prompt")
    async def hook_before_prompt(payload: HookPayload) -> dict[str, Any]:
        """Retrieve relevant memories before prompt."""
        result = await process_before_prompt(
            sink=app.state.sink,
            payload=payload.model_dump(),
        )
        return result

    @app.post("/hooks/session_end")
    async def hook_session_end(payload: HookPayload) -> dict[str, Any]:
        """Flush remaining drafts at session end."""
        result = await process_session_end(
            sink=app.state.sink,
            buf=app.state.buf,
            refiner=app.state.refiner,
            payload=payload.model_dump(),
        )
        return result

    @app.post("/draft/save")
    async def draft_save(payload: DraftSavePayload) -> dict[str, Any]:
        """Direct draft save."""
        sink_obj: TMSink = app.state.sink
        result = await sink_obj.draft_save(
            title=payload.title,
            content=payload.content,
            project=payload.project,
            group_key=payload.group_key,
            conversation_id=payload.conversation_id,
        )
        return result

    @app.post("/draft/publish")
    async def draft_publish(payload: DraftPublishPayload) -> dict[str, Any]:
        """Direct draft publish."""
        sink_obj: TMSink = app.state.sink
        result = await sink_obj.draft_publish(
            draft_id=payload.draft_id,
            refined_content=payload.refined_content,
        )
        return result

    @app.get("/recall")
    async def recall(
        query: str | None = None,
        project: str | None = None,
        max_results: int = 5,
    ) -> list[dict[str, Any]]:
        """Direct recall query."""
        sink_obj: TMSink = app.state.sink
        results = await sink_obj.recall(
            query=query,
            project=project,
            max_results=max_results,
        )
        return results

    return app

"""TM Daemon — FastAPI application skeleton.

Provides HTTP API endpoints for hook callbacks, draft management,
and recall queries. Uses TMSink abstraction for TM access.
"""

from __future__ import annotations

import asyncio
import logging
import os
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI
from pydantic import BaseModel, Field

from daemon.config import DaemonConfig, load_config
from daemon.draft_refiner import DraftRefiner
from daemon.pipeline import (
    _resolve_project,
    process_after_response,
    process_before_prompt,
    process_session_end,
    process_session_start,
)
from daemon.search_log_writer import SearchLogWriter
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
    response_text: str = ""
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
        # When running in local mode, bootstrap the TM AppContext so that
        # LocalTMSink → op_draft_save → _get_service() uses a properly
        # initialised ExperienceService (with Embedding provider ready).
        # Without this, _get_service() falls back to bootstrap() on every
        # call, which can fail or return a service without Embedding — causing
        # draft_save to return {"error": True} with no "id" field.
        if config.tm.mode == "local":
            from team_memory.bootstrap import bootstrap as _tm_bootstrap  # noqa: PLC0415
            # enable_background=False: Janitor scheduler runs exclusively in
            # team_memory_service (Docker). Daemon must not start a duplicate.
            _tm_bootstrap(enable_background=False)
            logger.info("TM AppContext bootstrapped for local mode (background=False)")

        sink_config = {
            "mode": config.tm.mode,
            "base_url": config.tm.base_url,
            "user": config.tm.user,
        }
        sink: TMSink = create_sink(sink_config)
        application.state.sink = sink
        application.state.tm_mode = config.tm.mode
        application.state.config = config

        db_path = config.draft.db_path or ":memory:"
        if db_path != ":memory:":
            os.makedirs(os.path.dirname(db_path), exist_ok=True)
        buf = DraftBuffer(db_path)
        await buf.__aenter__()
        application.state.buf = buf

        detector = ConvergenceDetector()
        application.state.detector = detector

        refiner = DraftRefiner(sink=sink, draft_buffer=buf)
        application.state.refiner = refiner

        # LLM background refinement worker
        from daemon.refinement_worker import RefinementWorker
        refinement_worker = RefinementWorker(config=config, buf=buf, sink=sink)
        refinement_task = refinement_worker.start()
        application.state.refinement_worker = refinement_worker
        application.state.refinement_task = refinement_task

        # Search log writer for evaluation
        search_log = SearchLogWriter()
        application.state.search_log = search_log

        # Start Obsidian vault watcher as background task
        from daemon.watcher import start_watcher
        watcher_task = asyncio.create_task(start_watcher(config, sink, buf))
        application.state.watcher_task = watcher_task

        logger.info("TM Daemon started (mode=%s)", config.tm.mode)
        yield

        # -- Shutdown --
        search_log_obj: SearchLogWriter | None = getattr(application.state, "search_log", None)
        if search_log_obj is not None:
            await search_log_obj.close()

        # Stop refinement worker
        refinement_worker_obj: RefinementWorker | None = getattr(application.state, "refinement_worker", None)
        if refinement_worker_obj is not None:
            refinement_worker_obj.stop()
        refinement_task_obj = getattr(application.state, "refinement_task", None)
        if refinement_task_obj is not None:
            refinement_task_obj.cancel()
            try:
                await asyncio.wait_for(refinement_task_obj, timeout=5.0)
            except (asyncio.TimeoutError, asyncio.CancelledError):
                pass

        watcher_task = getattr(application.state, "watcher_task", None)
        if watcher_task is not None:
            watcher_task.cancel()
            try:
                await asyncio.wait_for(watcher_task, timeout=3.0)
            except (asyncio.TimeoutError, asyncio.CancelledError):
                pass

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
            input_data=payload.model_dump(),
            config=app.state.config,
            sink=app.state.sink,
            buf=app.state.buf,
            detector=app.state.detector,
            refiner=app.state.refiner,
        )
        # Resolve project from workspace_roots for logging
        resolved_project = _resolve_project(payload.workspace_roots, app.state.config) or ""
        action = result.get("action", "")
        convergence = result.get("convergence", False)
        result["project"] = resolved_project
        logger.info(
            "[WRITE] after_response → project=%s action=%s convergence=%s",
            resolved_project, action, convergence,
        )
        # Scan for [mem:xxx] markers to mark search logs as used
        response_text = payload.response_text or payload.prompt
        if response_text:
            search_log: SearchLogWriter = app.state.search_log
            marked = await search_log.mark_used_from_response(
                response_text,
                project=resolved_project or None,
            )
            if marked > 0:
                logger.info("[EVAL]  marked %d search_log(s) as was_used (marker)", marked)

            # Fuzzy match: check unjudged logs for keyword overlap
            eval_config = app.state.config.evaluation
            if eval_config.fuzzy_match_enabled:
                fuzzy_marked = await search_log.mark_used_fuzzy(
                    response_text,
                    project=resolved_project or None,
                    threshold=eval_config.fuzzy_match_threshold,
                )
                if fuzzy_marked > 0:
                    logger.info("[EVAL]  marked %d search_log(s) as was_used (fuzzy)", fuzzy_marked)
        return result

    @app.post("/hooks/session_start")
    async def hook_session_start(payload: HookPayload) -> dict[str, Any]:
        """Inject project context at session start."""
        result = await process_session_start(
            input_data=payload.model_dump(),
            config=app.state.config,
            sink=app.state.sink,
        )
        project = result.get("project", "")
        has_ctx = bool(result.get("additional_context", ""))
        logger.info(
            "[READ]  session_start → project=%s context=%s",
            project, "yes" if has_ctx else "none",
        )
        return result

    @app.post("/hooks/before_prompt")
    async def hook_before_prompt(payload: HookPayload) -> dict[str, Any]:
        """Retrieve relevant memories before prompt."""
        result = await process_before_prompt(
            input_data=payload.model_dump(),
            config=app.state.config,
            sink=app.state.sink,
        )
        project = result.get("project", "")
        all_results = result.get("results", [])
        n_results = len(all_results)
        query_preview = payload.prompt[:60] if payload.prompt else ""
        # Log result titles and ids for debugging
        result_lines = []
        for i, r in enumerate(all_results[:5]):
            tid = r.get("id", "?")
            title = r.get("title", "?")[:50]
            result_lines.append(f"  #{i+1} \"{title}\" [{tid[:8]}]")
        extra = "\n" + "\n".join(result_lines) if result_lines else ""
        logger.info(
            "[READ]  before_prompt → project=%s query=\"%s\" results=%d%s",
            project, query_preview, n_results, extra,
        )
        # Write search log for evaluation
        if query_preview:
            search_log: SearchLogWriter = app.state.search_log
            log_result_ids = [
                {"id": r.get("id", ""), "score": r.get("score", 0)}
                for r in all_results[:5]
            ]
            await search_log.log_search(
                query=query_preview,
                project=project or "default",
                source="daemon",
                result_ids=log_result_ids if log_result_ids else None,
            )
        return result

    @app.post("/hooks/session_end")
    async def hook_session_end(payload: HookPayload) -> dict[str, Any]:
        """Flush remaining drafts at session end."""
        result = await process_session_end(
            input_data=payload.model_dump(),
            config=app.state.config,
            sink=app.state.sink,
            buf=app.state.buf,
            refiner=app.state.refiner,
        )
        flushed = result.get("flushed", False)
        logger.info(
            "[WRITE] session_end → conversation=%s flushed=%s",
            payload.conversation_id[:8] if payload.conversation_id else "?", flushed,
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
        logger.info(
            "[WRITE] draft_save → title=\"%s\" project=%s",
            payload.title[:40], payload.project or "?",
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
        logger.info(
            "[WRITE] draft_publish → draft_id=%s",
            payload.draft_id[:8],
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
        result_list = results if isinstance(results, list) else []
        result_lines = []
        for i, r in enumerate(result_list[:5]):
            tid = r.get("id", "?")
            title = r.get("title", "?")[:50]
            result_lines.append(f"  #{i+1} \"{title}\" [{tid[:8]}]")
        extra = "\n" + "\n".join(result_lines) if result_lines else ""
        logger.info(
            "[READ]  recall → query=\"%s\" project=%s results=%d%s",
            (query or "")[:60], project or "?", len(result_list), extra,
        )
        return results

    @app.get("/stats")
    async def get_stats(days: int = 7) -> dict[str, Any]:
        """Get evaluation statistics for the last N days."""
        search_log: SearchLogWriter = app.state.search_log
        return await search_log.get_stats(days=days)

    return app

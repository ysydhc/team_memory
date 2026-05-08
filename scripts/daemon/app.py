"""TM Daemon — FastAPI application skeleton.

Provides HTTP API endpoints for hook callbacks, draft management,
and recall queries. Uses TMSink abstraction for TM access.
"""

from __future__ import annotations

import asyncio
import logging
import os
from datetime import datetime, timezone
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI
from hooks.convergence_detector import ConvergenceDetector
from daemon.conversation_buffer import ConversationBuffer
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
from daemon.llm_wiki import LLMWiki
from daemon.search_log_writer import SearchLogWriter
from daemon.tm_sink import TMSink, create_sink
from daemon.wiki_compiler import WikiCompiler

logger = logging.getLogger("daemon.app")

# ---------------------------------------------------------------------------
# Alerting helper — thin wrapper around monitoring.alerting.AlertManager
# ---------------------------------------------------------------------------


async def send_daemon_alert(
    title: str,
    body: str,
    severity: str = "critical",
) -> None:
    """Send an alert through the daemon's configured alert channels (feishu etc.).

    Safe to call from anywhere in the daemon; silently no-ops if alert
    manager is not initialized.
    """
    try:
        from monitoring.alerting import AlertManager, AlertSeverity
        alert_mgr: AlertManager | None = getattr(_app_state, "alert_manager", None)
        if alert_mgr is None:
            return
        sev = AlertSeverity(severity)
        await alert_mgr.send_alert(title=title, body=body, severity=sev)
    except Exception:
        logger.debug("Daemon alert send failed (non-critical)", exc_info=True)


# Module-level reference to app.state so send_daemon_alert can access it
# without needing a FastAPI request context.
_app_state: Any = None


# ---------------------------------------------------------------------------
# Pydantic request models
# ---------------------------------------------------------------------------


class HookPayload(BaseModel):
    """Payload for hook callback endpoints.

    Different platforms send the agent response under different field names:
    - Hermes / tm_hook.py: uses ``prompt``
    - Claude Code / claude_stop.py: uses ``response_text``

    The pipeline layer reads both (response_text takes precedence) so new
    platforms can use either convention without touching pipeline code.
    """

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
            # team_memory_service (Docker). Entity extraction is now always
            # registered regardless of this flag (see bootstrap.py).
            _tm_bootstrap(enable_background=False)
            logger.info(
                "TM AppContext bootstrapped for local mode "
                "(background=False, entity extraction=always)"
            )

        sink_config = {
            "mode": config.tm.mode,
            "base_url": config.tm.base_url,
            "user": config.tm.user,
        }
        sink: TMSink = create_sink(sink_config)
        application.state.sink = sink
        application.state.tm_mode = config.tm.mode
        application.state.config = config

        # Start heartbeat
        from monitoring.heartbeat import start_heartbeat
        application.state.heartbeat = start_heartbeat()

        # Initialize alert manager (feishu / wecom / webhook channels)
        try:
            from monitoring.config import load_monitoring_config
            from monitoring.alerting import AlertManager
            mon_config = load_monitoring_config()
            alert_manager = AlertManager.from_config(mon_config)
            application.state.alert_manager = alert_manager
            logger.info(
                "AlertManager initialized (%d channel(s))",
                len(alert_manager.channels),
            )
        except Exception as e:
            logger.warning("AlertManager init failed (alerts disabled): %s", e)
            application.state.alert_manager = None

        # Expose app.state to module-level so send_daemon_alert works
        global _app_state
        _app_state = application.state

        db_path = config.draft.db_path or ":memory:"
        if db_path != ":memory:":
            os.makedirs(os.path.dirname(db_path), exist_ok=True)
        buf = ConversationBuffer(db_path)
        await buf.__aenter__()
        application.state.buf = buf

        detector = ConvergenceDetector()
        application.state.detector = detector

        # LLMWiki — local SQLite knowledge store with FTS5 search
        wiki_db_path_default = os.path.expanduser("~/.cache/tm-pipeline/wiki.db")
        llm_wiki = LLMWiki(db_path=wiki_db_path_default)
        await llm_wiki.__aenter__()
        application.state.llm_wiki = llm_wiki
        logger.info("LLMWiki initialized (db=%s)", wiki_db_path_default)

        # Wiki compiler (must init before DraftRefiner)
        wiki_compiler: WikiCompiler | None = None
        if config.wiki.enabled:
            # Determine wiki_root: explicit config > project root / wiki/
            wiki_root = config.wiki.wiki_root
            if not wiki_root:
                # Default: <project_root>/wiki/
                # Project root = grandparent of scripts/daemon/
                project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
                wiki_root = os.path.join(project_root, "wiki")
            wiki_db_path = config.wiki.db_path or None  # None → WikiCompiler default
            wiki_compiler = WikiCompiler(wiki_root=wiki_root, db_path=wiki_db_path, llm_wiki=llm_wiki)
            await wiki_compiler.__aenter__()
            # Pre-load entity/embedding data for enhanced cross-referencing
            from team_memory.config import load_settings as _load_tm_settings
            _tm_settings = _load_tm_settings()
            pg_url = str(_tm_settings.database.url)
            await wiki_compiler._load_pg_data(pg_url)
            application.state.wiki_compiler = wiki_compiler
            logger.info(
                "WikiCompiler initialized (root=%s, entities=%d, embeddings=%d)",
                wiki_root,
                len(wiki_compiler._exp_entity_map),
                len(wiki_compiler._exp_embeddings),
            )
        else:
            application.state.wiki_compiler = None

        refiner = DraftRefiner(sink=sink, draft_buffer=buf, wiki_compiler=wiki_compiler)
        application.state.refiner = refiner

        # LLM background refinement worker
        from daemon.refinement_worker import RefinementWorker
        refinement_worker = RefinementWorker(config=config, buf=buf, sink=sink)
        refinement_task = refinement_worker.start()
        application.state.refinement_worker = refinement_worker
        application.state.refinement_task = refinement_task

        # Auto Dream worker — periodic experience consolidation & pruning
        from daemon.auto_dream import AutoDreamWorker
        dream_worker = AutoDreamWorker(config=config, sink=sink)
        dream_task = dream_worker.start()
        application.state.dream_worker = dream_worker
        application.state.dream_task = dream_task

        # Search log writer for evaluation
        from team_memory.config import load_settings as _load_tm_settings
        _tm_settings = _load_tm_settings()
        _db_url = str(_tm_settings.database.url).replace("+asyncpg", "")
        search_log = SearchLogWriter(dsn=_db_url)
        application.state.search_log = search_log

        # Recent searches cache for before_prompt → after_response correlation
        application.state.recent_searches = {}

        # Response buffer for faithfulness evaluation
        from daemon.response_buffer import ResponseBuffer
        eval_config = config.evaluation
        response_buffer = ResponseBuffer(
            dsn=_db_url,
            batch_threshold=getattr(eval_config, "faithfulness_batch_threshold", 5),
        )
        application.state.response_buffer = response_buffer
        logger.info("[INIT] response_buffer initialized (threshold=%d)", response_buffer._batch_threshold)

        # Start Obsidian vault watcher as background task
        from daemon.watcher import start_watcher
        watcher_task = asyncio.create_task(start_watcher(config, sink, buf))
        application.state.watcher_task = watcher_task

        logger.info("TM Daemon started (mode=%s)", config.tm.mode)
        yield

        # -- Shutdown --
        # Close LLMWiki
        llm_wiki_obj: LLMWiki | None = getattr(application.state, "llm_wiki", None)
        if llm_wiki_obj is not None:
            await llm_wiki_obj.__aexit__(None, None, None)
            logger.info("LLMWiki shutdown")

        wiki_compiler_obj: WikiCompiler | None = getattr(application.state, "wiki_compiler", None)
        if wiki_compiler_obj is not None:
            await wiki_compiler_obj.__aexit__(None, None, None)
            logger.info("WikiCompiler shutdown")

        search_log_obj: SearchLogWriter | None = getattr(application.state, "search_log", None)
        if search_log_obj is not None:
            await search_log_obj.close()

        # Shutdown response buffer
        response_buffer_obj = getattr(application.state, "response_buffer", None)
        if response_buffer_obj is not None:
            await response_buffer_obj.close()

        # Stop refinement worker
        refinement_worker_obj: RefinementWorker | None = getattr(
            application.state, "refinement_worker", None
        )
        if refinement_worker_obj is not None:
            refinement_worker_obj.stop()
        refinement_task_obj = getattr(application.state, "refinement_task", None)
        if refinement_task_obj is not None:
            refinement_task_obj.cancel()
            try:
                await asyncio.wait_for(refinement_task_obj, timeout=5.0)
            except (asyncio.TimeoutError, asyncio.CancelledError):
                pass

        # Stop auto dream worker
        dream_worker_obj = getattr(application.state, "dream_worker", None)
        if dream_worker_obj is not None:
            dream_worker_obj.stop()
        dream_task_obj = getattr(application.state, "dream_task", None)
        if dream_task_obj is not None:
            dream_task_obj.cancel()
            try:
                await asyncio.wait_for(dream_task_obj, timeout=5.0)
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
    async def get_status() -> dict[str, Any]:
        """Return daemon status and TM mode."""
        heartbeat = getattr(app.state, "heartbeat", None)
        result = {
            "status": "running",
            "tm_mode": getattr(app.state, "tm_mode", "unknown"),
        }
        if heartbeat:
            result["heartbeat"] = heartbeat.to_dict()
        return result

    @app.get("/status/faithfulness")
    async def get_faithfulness_status(limit: int = 10) -> dict[str, Any]:
        """Return recent faithfulness buffer entries with scores."""
        response_buffer = getattr(app.state, "response_buffer", None)
        if response_buffer is None:
            return {"error": "response_buffer not initialized"}

        stats = await response_buffer.get_stats(days=7)
        entries = await response_buffer.fetch_pending(limit=0)  # don't fetch pending

        # Get recent entries (all, not just pending)
        import asyncpg
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
        # Evaluate usage: fuzzy match + faithfulness buffer
        response_text = payload.response_text or payload.prompt
        if response_text:
            search_log: SearchLogWriter = app.state.search_log

            # Look up cached search from before_prompt
            recent_searches: dict = app.state.recent_searches
            key = payload.conversation_id or resolved_project or "default"
            cached_search = recent_searches.pop(key, None)

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

            # Auto-buffer response for faithfulness evaluation
            try:
                response_buffer = getattr(app.state, "response_buffer", None)
                if response_buffer is not None:
                    # Use cached search_log_id from before_prompt (precise correlation)
                    if cached_search:
                        recent_log = {
                            "id": cached_search["log_id"],
                            "query": cached_search["query"],
                            "result_ids": cached_search["result_ids"],
                        }
                    else:
                        # Fallback: find most recent search_log with results
                        recent_log = await search_log.get_recent_with_results(
                            project=resolved_project or None,
                            hours_window=1,
                        )
                    if recent_log:
                        await response_buffer.add(
                            query=recent_log["query"],
                            agent_response=response_text[:10000],
                            result_ids=recent_log["result_ids"],
                            search_log_id=recent_log["id"],
                        )
                        # Check if batch threshold reached
                        if await response_buffer.should_batch():
                            logger.info("[EVAL]  faithfulness batch threshold reached, triggering evaluation")
                            import subprocess, sys as _sys
                            subprocess.Popen(
                                [_sys.executable, "scripts/daemon/faithfulness_batch.py", "--force"],
                                cwd=os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
                                stdout=subprocess.DEVNULL,
                                stderr=subprocess.DEVNULL,
                            )
            except Exception as e:
                logger.info("Faithfulness buffer capture failed: %s", e, exc_info=True)
                await send_daemon_alert(
                    title="Faithfulness Buffer Error",
                    body=f"Faithfulness buffer capture failed: {e}",
                    severity="warning",
                )
        return result

    @app.post("/hooks/session_start")
    async def hook_session_start(payload: HookPayload) -> dict[str, Any]:
        """Inject project context at session start."""
        result = await process_session_start(
            input_data=payload.model_dump(),
            config=app.state.config,
            sink=app.state.sink,
            buf=app.state.buf,
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
            buf=app.state.buf,
            wiki=getattr(app.state, "llm_wiki", None),
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
            log_id = await search_log.log_search(
                query=query_preview,
                project=project or "default",
                source="daemon",
                result_ids=log_result_ids if log_result_ids else None,
            )
            # Cache for after_response correlation
            if log_id:
                recent_searches: dict = app.state.recent_searches
                # Key: conversation_id if available, else project
                key = payload.conversation_id or project or "default"
                recent_searches[key] = {
                    "log_id": log_id,
                    "query": query_preview,
                    "result_ids": log_result_ids,
                    "project": project,
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                }
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

    @app.get("/wiki/search")
    async def wiki_search(
        q: str = "",
        project: str | None = None,
        limit: int = 5,
    ) -> list[dict[str, Any]]:
        """Search the local LLMWiki knowledge store using FTS5."""
        llm_wiki: LLMWiki | None = getattr(app.state, "llm_wiki", None)
        if llm_wiki is None:
            return []
        results = await llm_wiki.search(query=q, project=project, limit=limit)
        logger.info(
            "[READ]  wiki/search → q=\"%s\" project=%s results=%d",
            q[:60], project or "?", len(results),
        )
        return results

    @app.get("/wiki/stats")
    async def wiki_stats() -> dict[str, Any]:
        """Return wiki entry statistics."""
        llm_wiki: LLMWiki | None = getattr(app.state, "llm_wiki", None)
        if llm_wiki is None:
            return {"total": 0, "by_project": {}}
        return await llm_wiki.stats()

    @app.get("/persona/profile")
    async def persona_profile(persona: str = "general") -> dict[str, Any]:
        """Return user's personal memory profile for a given persona.

        Useful for debugging/inspection of persona-specific memories.
        """
        config_obj: DaemonConfig = app.state.config
        user_id = getattr(config_obj.tm, "user", "admin") or "admin"

        try:
            from team_memory.services.personal_memory import PersonalMemoryService
            from team_memory.config import load_settings as _load_tm_settings
            from team_memory.bootstrap import _create_embedding_provider
            _settings = _load_tm_settings()
            embedding = _create_embedding_provider(_settings)
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

    # ------------------------------------------------------------------
    # Alerting endpoints
    # ------------------------------------------------------------------

    @app.post("/alert/test")
    async def alert_test(
        title: str = "TM Daemon Alert Test",
        body: str = "This is a test alert from TM Daemon.",
        severity: str = "info",
    ) -> dict[str, Any]:
        """Send a test alert through all configured channels (feishu etc.).

        Useful for verifying feishu webhook integration at deploy time.
        Query params: title, body, severity (info|warning|critical).
        """
        await send_daemon_alert(title=title, body=body, severity=severity)
        alert_mgr = getattr(app.state, "alert_manager", None)
        n_channels = len(alert_mgr.channels) if alert_mgr else 0
        return {"sent": True, "channels": n_channels, "title": title, "severity": severity}

    return app

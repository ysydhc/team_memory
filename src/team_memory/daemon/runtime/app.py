"""TM Daemon — FastAPI application skeleton.

Provides HTTP API endpoints for hook callbacks, draft management,
and recall queries. Uses TMSink abstraction for TM access.
"""

from __future__ import annotations

import asyncio
import logging
import os
import time
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any

from fastapi import FastAPI, Request
from pydantic import BaseModel, ConfigDict, Field, model_validator

if TYPE_CHECKING:
    from team_memory.daemon.buffers.payload_capture import PayloadCapture  # noqa: F401

from team_memory.daemon.buffers.conversation_buffer import ConversationBuffer
from team_memory.daemon.buffers.draft_buffer import DraftBuffer
from team_memory.daemon.integration.tm_sink import TMSink, create_sink
from team_memory.daemon.observe.search_log_writer import SearchLogWriter
from team_memory.daemon.quality.convergence_detector import ConvergenceDetector
from team_memory.daemon.refinement.draft_refiner import DraftRefiner
from team_memory.daemon.refinement.pipeline import (
    _resolve_project,
    process_after_response,
    process_before_prompt,
    process_session_end,
    process_session_start,
)
from team_memory.daemon.runtime.config import DaemonConfig, load_config
from team_memory.daemon.wiki.llm_wiki import LLMWiki
from team_memory.daemon.wiki.wiki_compiler import WikiCompiler
from team_memory.project_resolver import find_project_root

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
        from team_memory.monitoring.alerting import AlertManager, AlertSeverity

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
# Payload source detection
# ---------------------------------------------------------------------------


def _detect_source(payload: dict) -> str:
    """Detect which platform sent the payload based on field patterns."""
    if "last_assistant_message" in payload or "transcript" in payload:
        return "claude"
    if "agent_response" in payload:
        return "cursor"
    if "hook_event_name" in payload:
        return "hermes"
    if "tool_use_id" in payload:
        return "mcp"
    return "unknown"


# ---------------------------------------------------------------------------
# Pydantic request models
# ---------------------------------------------------------------------------


class HookPayload(BaseModel):
    """Payload for hook callback endpoints.

    Lenient mode: accepts extra fields and logs them for contract derivation.
    宽松模式: 接受额外字段并记录，用于契约推导。
    """

    model_config = ConfigDict(extra="allow")

    conversation_id: str = ""
    prompt: str = ""
    response_text: str = ""
    agent_response: str = ""  # Cursor afterAgentResponse format
    content: str = ""  # Some platforms use content
    workspace_roots: list[str] = Field(default_factory=list)
    model: str = ""

    @model_validator(mode="before")
    @classmethod
    def _log_extra_fields(cls, data: Any) -> Any:
        if isinstance(data, dict):
            known = {
                "conversation_id",
                "prompt",
                "response_text",
                "agent_response",
                "content",
                "workspace_roots",
                "model",
            }
            extra = set(data.keys()) - known
            if extra:
                logger.info("[CONTRACT] HookPayload extra fields: %s", sorted(extra))
        return data


class DraftSavePayload(BaseModel):
    """Payload for /draft/save endpoint."""

    model_config = ConfigDict(extra="allow")

    title: str
    content: str
    project: str | None = None
    group_key: str | None = None
    conversation_id: str | None = None


class DraftPublishPayload(BaseModel):
    """Payload for /draft/publish endpoint."""

    draft_id: str
    refined_content: str | None = None


class MCPCallbackPayload(BaseModel):
    """Payload for /hooks/mcp_callback — PostToolUse hook reports MCP recall results.

    Called by Claude Code PostToolUse / Cursor afterMCPExecution / Hermes post_tool_call
    after the AI successfully invokes a TM recall tool.  The hook extracts the
    ``search_log_id`` from the tool_response and registers it with the daemon so
    that the subsequent ``after_response`` can correlate the MCP search_log with
    the AI's final answer.
    """

    model_config = ConfigDict(extra="allow")

    session_id: str
    tool_use_id: str = ""
    search_log_id: str = ""
    tool_name: str = ""
    result_ids: list[dict] = Field(default_factory=list)


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
            logger.info("TM AppContext bootstrapped for local mode (background=False, entity extraction=always)")

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
        from team_memory.monitoring.heartbeat import start_heartbeat

        application.state.heartbeat = start_heartbeat()

        # Initialize alert manager (feishu / wecom / webhook channels)
        try:
            from team_memory.monitoring.alerting import AlertManager
            from team_memory.monitoring.config import load_monitoring_config

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
        logger.info("[INIT] ConversationBuffer initialized (db=%s)", db_path)

        # Separate DraftBuffer for Obsidian watcher (drafts table, not sessions)
        # 独立的 DraftBuffer 给 Obsidian watcher 使用（drafts 表，非 sessions）
        draft_buf = DraftBuffer(db_path)
        await draft_buf.__aenter__()
        application.state.draft_buf = draft_buf

        # Payload capture for contract derivation
        from team_memory.daemon.buffers.payload_capture import PayloadCapture

        capture_db = config.draft.db_path or ":memory:"
        if capture_db != ":memory:":
            capture_db = capture_db.replace(".db", "_capture.db")
        payload_capture = PayloadCapture(db_path=capture_db)
        await payload_capture.start()
        application.state.payload_capture = payload_capture

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
                start = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
                project_root = find_project_root(start)
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

        # Startup recovery: reset any sessions stuck in "refining" status
        # 启动恢复：重置卡在 refining 状态的 session
        ref_count = await buf.reset_refining_sessions()
        pub_count = await buf.reset_publishing_sessions()
        if ref_count or pub_count:
            logger.info(
                "Reset %d refining, %d publishing sessions (crash recovery)",
                ref_count,
                pub_count,
            )

        # Metrics collector / 指标收集
        from team_memory.daemon.observe.metrics_collector import MetricsCollector

        _pg_url_clean = str(_tm_settings.database.url).replace("+asyncpg", "")
        metrics = MetricsCollector(db_path=db_path, pg_url=_pg_url_clean)
        metrics.cleanup()
        application.state.metrics = metrics

        # In-process health check / 进程内健康检查
        from team_memory.daemon.runtime.health_check import HealthCheck

        _health_check = HealthCheck(
            db_path=db_path,
            pg_url=_pg_url_clean,
            worker_task=None,  # will be set after worker starts
            alert_callback=send_daemon_alert,
        )

        # Pending queue for graceful degradation / 优雅降级队列
        from team_memory.daemon.buffers.pending_queue import PendingQueue

        pending_queue = PendingQueue(db_path=db_path)

        # LLM background refinement worker
        from team_memory.daemon.refinement.refinement_worker import RefinementWorker

        refinement_worker = RefinementWorker(
            config=config,
            buf=buf,
            sink=sink,
            queue=pending_queue,
            metrics=metrics,
            health_check=_health_check,
        )
        refinement_task = refinement_worker.start()
        _health_check._worker_task = refinement_task  # 绑定实际 task
        application.state.refinement_worker = refinement_worker
        application.state.refinement_task = refinement_task

        # Auto Dream worker — periodic experience consolidation & pruning
        from team_memory.daemon.integration.auto_dream import AutoDreamWorker

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
        from team_memory.daemon.buffers.response_buffer import ResponseBuffer

        eval_config = config.evaluation
        response_buffer = ResponseBuffer(
            dsn=_db_url,
            batch_threshold=getattr(eval_config, "faithfulness_batch_threshold", 5),
        )
        application.state.response_buffer = response_buffer
        logger.info("[INIT] response_buffer initialized (threshold=%d)", response_buffer._batch_threshold)

        # Start Obsidian vault watcher as background task
        from team_memory.daemon.observe.watcher import start_watcher

        watcher_task = asyncio.create_task(start_watcher(config, sink, draft_buf))
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
        buf_obj: ConversationBuffer | None = getattr(application.state, "buf", None)
        if buf_obj is not None:
            await buf_obj.__aexit__(None, None, None)
        draft_buf_obj: DraftBuffer | None = getattr(application.state, "draft_buf", None)
        if draft_buf_obj is not None:
            await draft_buf_obj.__aexit__(None, None, None)
        # Shutdown PayloadCapture
        capture_obj = getattr(application.state, "payload_capture", None)
        if capture_obj is not None:
            await capture_obj.stop()

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

    @app.post("/hooks/after_response")
    async def hook_after_response(request: Request) -> dict[str, Any]:
        """Process agent response (draft pipeline)."""
        raw = await request.json()
        capture = getattr(app.state, "payload_capture", None)
        if capture:
            await capture.capture("/hooks/after_response", raw, source=_detect_source(raw))
        payload = HookPayload.model_validate(raw)
        result = await process_after_response(
            input_data=payload.model_dump(),
            config=app.state.config,
            sink=app.state.sink,
            buf=app.state.buf,
            detector=app.state.detector,
            refiner=app.state.refiner,
            metrics=getattr(app.state, "metrics", None),
        )
        # Resolve project from workspace_roots for logging
        resolved_project = _resolve_project(payload.workspace_roots, app.state.config) or ""
        action = result.get("action", "")
        convergence = result.get("convergence", False)
        result["project"] = resolved_project
        logger.info(
            "[WRITE] after_response → project=%s action=%s convergence=%s",
            resolved_project,
            action,
            convergence,
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
                        # Fallback: time-window query for MCP search_logs
                        # when recent_searches is lost (daemon restart, etc.)
                        t_end = datetime.now(timezone.utc)
                        t_start = None

                        # Try ConversationBuffer for T1 (C-2)
                        conv_buf = getattr(app.state, "conversation_buffer", None)
                        if conv_buf and payload.conversation_id:
                            try:
                                t_start = await conv_buf.get_last_user_message_time(payload.conversation_id)
                            except Exception:
                                pass

                        # If no T1 from ConversationBuffer, use 5 min window
                        if not t_start:
                            from datetime import timedelta

                            t_start = t_end - timedelta(minutes=5)

                        # Query unjudged MCP search_logs in window
                        mcp_fallback = await search_log.find_unjudged_in_window(
                            t_start=t_start,
                            t_end=t_end,
                            project=resolved_project or None,
                            source="mcp",
                        )
                        if mcp_fallback:
                            # Buffer each MCP search_log found in window
                            for mcp_log in mcp_fallback:
                                if not await response_buffer.has_search_log(mcp_log["id"]):
                                    await response_buffer.add(
                                        query=mcp_log["query"] or "(mcp recall)",
                                        agent_response=response_text[:10000],
                                        result_ids=mcp_log["result_ids"],
                                        search_log_id=mcp_log["id"],
                                    )
                                    logger.info(
                                        "[EVAL]  fallback: buffered MCP search_log %s (window %s~%s)",
                                        mcp_log["id"][:8],
                                        t_start.strftime("%H:%M:%S"),
                                        t_end.strftime("%H:%M:%S"),
                                    )
                            # Also try daemon search_log fallback
                            recent_log = await search_log.get_recent_with_results(
                                project=resolved_project or None,
                                hours_window=1,
                            )
                        else:
                            # Last resort: find any recent search_log with results
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

                    # Process MCP recall calls registered by PostToolUse hooks
                    mcp_calls = (cached_search or {}).get("mcp_calls", [])
                    for mcp in mcp_calls:
                        if not mcp.get("search_log_id"):
                            continue
                        # Skip if this search_log_id was already buffered (dedup)
                        if await response_buffer.has_search_log(mcp["search_log_id"]):
                            logger.debug(
                                "[EVAL]  mcp search_log %s already buffered, skipping",
                                mcp["search_log_id"][:8],
                            )
                            continue
                        await response_buffer.add(
                            query="(mcp recall)",
                            agent_response=response_text[:10000],
                            result_ids=mcp.get("result_ids", []),
                            search_log_id=mcp["search_log_id"],
                        )
                        logger.info(
                            "[EVAL]  buffered MCP search_log %s for faithfulness",
                            mcp["search_log_id"][:8],
                        )

                    # Check if batch threshold reached
                    if await response_buffer.should_batch():
                        logger.info("[EVAL]  faithfulness batch threshold reached, triggering evaluation")
                        import subprocess
                        import sys as _sys
                        import os as _os

                        subprocess.Popen(
                            [
                                _sys.executable,
                                "-m",
                                "team_memory.daemon.quality.faithfulness_batch",
                                "--force",
                                "--db-url", _db_url,
                            ],
                            cwd=find_project_root(),
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
    async def hook_session_start(request: Request) -> dict[str, Any]:
        """Inject project context at session start."""
        raw = await request.json()
        capture = getattr(app.state, "payload_capture", None)
        if capture:
            await capture.capture("/hooks/session_start", raw, source=_detect_source(raw))
        payload = HookPayload.model_validate(raw)
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
            project,
            "yes" if has_ctx else "none",
        )
        return result

    @app.post("/hooks/before_prompt")
    async def hook_before_prompt(request: Request) -> dict[str, Any]:
        """Retrieve relevant memories before prompt."""
        raw = await request.json()
        capture = getattr(app.state, "payload_capture", None)
        if capture:
            await capture.capture("/hooks/before_prompt", raw, source=_detect_source(raw))
        payload = HookPayload.model_validate(raw)
        result = await process_before_prompt(
            input_data=payload.model_dump(),
            config=app.state.config,
            sink=app.state.sink,
            buf=app.state.buf,
            wiki=getattr(app.state, "llm_wiki", None),
            metrics=getattr(app.state, "metrics", None),
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
            result_lines.append(f'  #{i + 1} "{title}" [{tid[:8]}]')
        extra = "\n" + "\n".join(result_lines) if result_lines else ""
        logger.info(
            '[READ]  before_prompt → project=%s query="%s" results=%d%s',
            project,
            query_preview,
            n_results,
            extra,
        )
        # Write search log for evaluation
        if query_preview:
            search_log: SearchLogWriter = app.state.search_log
            log_result_ids = [{"id": r.get("id", ""), "score": r.get("score", 0)} for r in all_results[:5]]
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
                    "T1": time.monotonic(),  # monotonic timestamp for window matching
                    "T1_iso": datetime.now(timezone.utc).isoformat(),
                    "mcp_calls": [],  # populated by /hooks/mcp_callback
                }
        return result

    @app.post("/hooks/session_end")
    async def hook_session_end(request: Request) -> dict[str, Any]:
        """Flush remaining drafts at session end."""
        raw = await request.json()
        capture = getattr(app.state, "payload_capture", None)
        if capture:
            await capture.capture("/hooks/session_end", raw, source=_detect_source(raw))
        payload = HookPayload.model_validate(raw)
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
            payload.conversation_id[:8] if payload.conversation_id else "?",
            flushed,
        )
        return result

    @app.post("/hooks/mcp_callback")
    async def hook_mcp_callback(request: Request) -> dict[str, Any]:
        """Register an MCP recall result from a PostToolUse hook.

        PostToolUse / afterMCPExecution / post_tool_call hooks call this
        after the AI successfully invokes a TM recall tool.  The daemon
        stores the search_log_id in ``recent_searches[session_id].mcp_calls``
        so that ``after_response`` can correlate it with the AI's answer.
        """
        raw = await request.json()
        capture = getattr(app.state, "payload_capture", None)
        if capture:
            await capture.capture("/hooks/mcp_callback", raw, source=_detect_source(raw))
        payload = MCPCallbackPayload.model_validate(raw)
        recent_searches: dict = app.state.recent_searches
        key = payload.session_id

        if key in recent_searches:
            # Session already has a before_prompt entry — append MCP call
            recent_searches[key]["mcp_calls"].append(
                {
                    "tool_use_id": payload.tool_use_id,
                    "search_log_id": payload.search_log_id,
                    "result_ids": payload.result_ids,
                    "tool_name": payload.tool_name,
                }
            )
            logger.info(
                "[MCP]   mcp_callback → session=%s tool=%s log_id=%s..%s",
                key[:8],
                payload.tool_name,
                payload.search_log_id[:8] if payload.search_log_id else "?",
                payload.tool_use_id[:8] if payload.tool_use_id else "?",
            )
        else:
            # No before_prompt entry yet — create a stub entry
            recent_searches[key] = {
                "log_id": None,
                "query": "",
                "result_ids": [],
                "project": "",
                "T1": time.monotonic(),
                "T1_iso": datetime.now(timezone.utc).isoformat(),
                "mcp_calls": [
                    {
                        "tool_use_id": payload.tool_use_id,
                        "search_log_id": payload.search_log_id,
                        "result_ids": payload.result_ids,
                        "tool_name": payload.tool_name,
                    }
                ],
            }
            logger.info(
                "[MCP]   mcp_callback → session=%s (no before_prompt) tool=%s",
                key[:8],
                payload.tool_name,
            )

        return {"status": "ok", "mcp_calls": len(recent_searches[key]["mcp_calls"])}

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
            '[WRITE] draft_save → title="%s" project=%s',
            payload.title[:40],
            payload.project or "?",
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
            result_lines.append(f'  #{i + 1} "{title}" [{tid[:8]}]')
        extra = "\n" + "\n".join(result_lines) if result_lines else ""
        logger.info(
            '[READ]  recall → query="%s" project=%s results=%d%s',
            (query or "")[:60],
            project or "?",
            len(result_list),
            extra,
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
            '[READ]  wiki/search → q="%s" project=%s results=%d',
            q[:60],
            project or "?",
            len(results),
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

    @app.get("/debug/payloads")
    async def debug_payloads(endpoint: str = "", limit: int = 20) -> dict[str, Any]:
        capture = getattr(app.state, "payload_capture", None)
        if capture is None:
            return {"error": "payload_capture not initialized"}
        if endpoint:
            payloads = await capture.get_payloads(endpoint=endpoint, limit=limit)
            summary = await capture.get_field_summary(endpoint)
            return {"payloads": payloads, "summary": summary}
        payloads = await capture.get_payloads(limit=limit)
        return {"payloads": payloads}

    @app.get("/debug/contracts")
    async def debug_contracts() -> dict[str, Any]:
        capture = getattr(app.state, "payload_capture", None)
        if capture is None:
            return {"error": "payload_capture not initialized"}
        endpoints = [
            "/hooks/before_prompt",
            "/hooks/after_response",
            "/hooks/session_start",
            "/hooks/session_end",
            "/hooks/mcp_callback",
        ]
        result = {}
        for ep in endpoints:
            result[ep] = await capture.get_field_summary(ep)
        result["_hook_payload_schema"] = {
            "fields": list(HookPayload.model_fields.keys()),
            "extra": "allow",
        }
        return result

    return app

"""TM Daemon — FastAPI application skeleton.

Provides HTTP API endpoints for hook callbacks, draft management,
and recall queries. Uses TMSink abstraction for TM access.
"""

from __future__ import annotations

import asyncio
import logging
import os
from contextlib import asynccontextmanager
from typing import TYPE_CHECKING, Any

from fastapi import FastAPI

if TYPE_CHECKING:
    from team_memory.daemon.buffers.payload_capture import PayloadCapture  # noqa: F401

from team_memory.daemon.buffers.conversation_buffer import ConversationBuffer
from team_memory.daemon.buffers.draft_buffer import DraftBuffer
from team_memory.daemon.integration.tm_sink import TMSink, create_sink
from team_memory.daemon.quality.convergence_detector import ConvergenceDetector
from team_memory.daemon.refinement.draft_refiner import DraftRefiner
from team_memory.daemon.runtime.config import DaemonConfig, load_config

# Re-export models for backward compatibility
from team_memory.daemon.runtime.models import (  # noqa: F401
    DraftPublishPayload,
    DraftSavePayload,
    HookPayload,
    MCPCallbackPayload,
    RecallQuery,
)
from team_memory.daemon.wiki.llm_wiki import LLMWiki
from team_memory.daemon.wiki.wiki_compiler import WikiCompiler
from team_memory.project_resolver import find_project_root
from team_memory.services.search.search_log_writer import SearchLogWriter

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
# App factory
# ---------------------------------------------------------------------------


async def _wait_for_deps(config: DaemonConfig, logger: logging.Logger) -> bool:
    """Wait for critical dependencies before serving.

    Reads config (no hardcoded values): checks Docker, PostgreSQL, LLM API,
    and Embedding API.  Loops every 5s for up to 2 minutes.  Returns True
    if all are ready, False if timed out (degraded mode).
    """
    import asyncio as _asyncio
    import time as _time

    max_wait = 120  # 2 minutes
    interval = 5

    # -- Resolve DB URL from config.development.yaml (not daemon config) --
    _db_url = ""
    try:
        from team_memory.config import load_settings as _load_tm

        _tm = _load_tm()
        _db_url = str(_tm.database.url).replace("+asyncpg", "")
    except Exception:
        pass

    # -- Resolve LLM API URL from refinement config --
    _llm_url = ""
    if getattr(config, "refinement", None) and config.refinement.base_url:
        _llm_url = config.refinement.base_url.rstrip("/")

    # -- Resolve Embedding API URL --
    _emb_url = ""
    try:
        from team_memory.config import load_settings as _load_tm2

        _tm2 = _load_tm2()
        _ecfg = _tm2.embedding
        if _ecfg.provider == "openai" and _ecfg.openai.base_url:
            _emb_url = _ecfg.openai.base_url.rstrip("/")
    except Exception:
        pass

    started = _time.monotonic()

    while (_time.monotonic() - started) < max_wait:
        ok: list[str] = []
        fail: list[str] = []

        # --- Docker ---
        try:
            _docker = "/usr/local/bin/docker"
            proc = await _asyncio.create_subprocess_exec(
                _docker,
                "info",
                stdout=_asyncio.subprocess.DEVNULL,
                stderr=_asyncio.subprocess.DEVNULL,
            )
            try:
                await _asyncio.wait_for(proc.wait(), timeout=5)
                if proc.returncode == 0:
                    ok.append("docker")
                else:
                    fail.append("docker")
            except _asyncio.TimeoutError:
                proc.kill()
                fail.append("docker (timeout)")
        except Exception:
            fail.append("docker")

        # --- PostgreSQL ---
        if _db_url:
            try:
                import asyncpg

                conn = await _asyncio.wait_for(
                    asyncpg.connect(_db_url, timeout=5),
                    timeout=6,
                )
                await conn.close()
                ok.append("postgres")
            except Exception:
                fail.append("postgres")
        else:
            fail.append("postgres (no url)")

        # --- LLM API ---
        if _llm_url:
            try:
                import httpx

                async with httpx.AsyncClient(timeout=5.0) as client:
                    resp = await client.get(f"{_llm_url}/models")
                    if 200 <= resp.status_code < 500:
                        ok.append("llm_api")
                    else:
                        fail.append("llm_api")
            except Exception:
                fail.append("llm_api")
        else:
            ok.append("llm_api (skipped)")

        # --- Embedding API ---
        if _emb_url:
            try:
                import httpx

                async with httpx.AsyncClient(timeout=5.0) as client:
                    resp = await client.get(f"{_emb_url}/models")
                    if 200 <= resp.status_code < 500:
                        ok.append("embedding_api")
                    else:
                        fail.append("embedding_api")
            except Exception:
                fail.append("embedding_api")
        else:
            fail.append("embedding_api (no url)")

        # --- Decision ---
        # Docker is a soft check — PostgreSQL reachability is the real gate.
        _hard_fail = [f for f in fail if f not in ("docker",)]
        if not _hard_fail:
            if "docker" in fail:
                logger.info(
                    "[STARTUP] All critical deps ready (%s; docker CLI unavailable but PG reachable) after %.0fs",
                    ", ".join(ok),
                    _time.monotonic() - started,
                )
            else:
                logger.info(
                    "[STARTUP] All dependencies ready (%s) after %.0fs",
                    ", ".join(ok),
                    _time.monotonic() - started,
                )
            return True

        logger.info(
            "[STARTUP] Waiting for deps (ok=%s, fail=%s, elapsed=%.0fs)",
            ", ".join(ok) if ok else "none",
            ", ".join(fail),
            _time.monotonic() - started,
        )
        await _asyncio.sleep(interval)

    logger.warning(
        "[STARTUP] Degraded mode after %.0fs — failed deps: %s",
        _time.monotonic() - started,
        ", ".join(fail) if fail else "unknown",
    )
    return False


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
        # Wait for dependencies before serving (critical for laptop/sleep recovery)
        deps_ready = await _wait_for_deps(config, logger)
        application.state.deps_ready = deps_ready
        if not deps_ready:
            logger.warning(
                "[STARTUP] Degraded mode — some dependencies unavailable. "
                "Refinement/Faithfulness workers will retry on next tick."
            )

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

        # Start heartbeat (with sleep/wake detection)
        from team_memory.monitoring.heartbeat import start_heartbeat

        application.state.heartbeat = start_heartbeat()

        # Register wake-up recovery callback on the heartbeat
        async def _on_wake_from_sleep(sleep_seconds: float) -> None:
            sleep_min = sleep_seconds / 60
            logger.warning(
                "[WAKE]  System slept for %.0f min. Running recovery sequence.",
                sleep_min,
            )

            # 1. Pause workers to avoid cascading failures
            _rw = getattr(application.state, "refinement_worker", None)
            _dw = getattr(application.state, "dream_worker", None)
            if _rw and hasattr(_rw, "pause"):
                _rw.pause()
                logger.info("[WAKE]  RefinementWorker paused")
            if _dw and hasattr(_dw, "pause"):
                _dw.pause()
                logger.info("[WAKE]  AutoDreamWorker paused")

            # 2. Close PG pool so next access rebuilds it
            try:
                import team_memory.storage.database as _db_mod

                _pg_engine = getattr(_db_mod, "_engine", None)
                if _pg_engine:
                    await _pg_engine.dispose()
                    _db_mod._engine = None  # force rebuild on next use
                    logger.info("[WAKE]  PG connection pool disposed")
            except Exception as e:
                logger.warning("[WAKE]  PG pool dispose failed: %s", e)

            # 3. Re-run dependency check
            _cfg = application.state.config
            _ok = await _wait_for_deps(_cfg, logger)
            if _ok:
                logger.info("[WAKE]  All dependencies recovered")
            else:
                logger.warning("[WAKE]  Some dependencies still unavailable")

            # 4. Resume workers
            if _rw and hasattr(_rw, "resume"):
                _rw.resume()
                logger.info("[WAKE]  RefinementWorker resumed")
            if _dw and hasattr(_dw, "resume"):
                _dw.resume()
                logger.info("[WAKE]  AutoDreamWorker resumed")

            import json as _json2
            import time as _time2

            _rec_log = _json2.dumps(
                {
                    "event": "system_wake_recovery_complete",
                    "sleep_seconds": round(sleep_seconds, 1),
                    "sleep_minutes": round(sleep_min, 1),
                    "deps_recovered": _ok,
                    "workers_resumed": True,
                    "recovered_at": _time2.time(),
                },
                ensure_ascii=False,
            )
            logger.info("[WAKE]  %s", _rec_log)

        application.state.heartbeat.on_wake(_on_wake_from_sleep)

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

        # IntentGate — LLM-based intent classification for retrieval routing
        from team_memory.daemon.integration.intent_gate import IntentGate

        intent_gate = IntentGate()
        application.state.intent_gate = intent_gate
        logger.info("[INIT] IntentGate initialized (model=%s)", intent_gate._model)

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
        _health_check._pending_queue = pending_queue

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

        # Start JanitorScheduler inside persistent tm-daemon (with cross-restart timestamp file)
        try:
            from team_memory.daemon.integration.tm_sink import _daemon_context

            _tm_ctx_janitor = _daemon_context()
            if _tm_ctx_janitor and _tm_ctx_janitor.janitor_scheduler is not None:
                if hasattr(_tm_ctx_janitor.janitor_scheduler, "enable_state_persistence"):
                    _tm_ctx_janitor.janitor_scheduler.enable_state_persistence()
                await _tm_ctx_janitor.janitor_scheduler.start()
                application.state.janitor_scheduler = _tm_ctx_janitor.janitor_scheduler
                logger.info("[INIT] janitor_scheduler started in tm-daemon")
        except Exception as exc:
            logger.warning("Failed to start janitor_scheduler in tm-daemon: %s", exc)

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

        # Arena evaluation worker
        from team_memory.daemon.runtime.config import ArenaSettings

        arena_config = getattr(config, "arena", ArenaSettings())
        if arena_config.enabled:
            from team_memory.daemon.quality.arena.runner import ArenaRunner
            from team_memory.daemon.quality.arena.storage import ArenaStorage

            arena_storage = ArenaStorage(
                data_dir=arena_config.data_dir or None,
            )
            arena_runner = ArenaRunner(
                dsn=_db_url,
                storage=arena_storage,
            )
            application.state.arena_runner = arena_runner

            async def _arena_loop() -> None:
                """Background loop: run arena round + production eval."""
                import asyncio as _aio

                try:
                    await arena_runner.ensure_initialized()
                    while True:
                        try:
                            await arena_runner.run_round()
                            await arena_runner.run_production_eval()
                        except Exception as e:
                            logger.warning("Arena round failed: %s", e)

                        interval = (
                            arena_config.converged_interval_minutes
                            if arena_storage.is_converged()
                            else arena_config.interval_minutes
                        )
                        await _aio.sleep(interval * 60)
                except _aio.CancelledError:
                    pass
                finally:
                    await arena_runner.close()

            arena_task = asyncio.create_task(_arena_loop())
            application.state.arena_task = arena_task
            logger.info("[INIT] arena worker started (interval=%dm)", arena_config.interval_minutes)

        # Start Obsidian vault watcher as background task
        from team_memory.daemon.observe.watcher import start_watcher

        watcher_task = asyncio.create_task(start_watcher(config, sink, draft_buf))
        application.state.watcher_task = watcher_task

        # Periodic Antigravity transcript MCP recall sync (fallback when AGY hooks are skipped)
        async def _agy_sync_loop() -> None:
            import time as _time

            from team_memory.daemon.observe.agy_transcript_watcher import sync_agy_mcp_recalls

            try:
                while True:
                    try:
                        sync_res = await sync_agy_mcp_recalls(
                            response_buffer=response_buffer,
                            since_ts=_time.time() - 86400 * 3,
                        )
                        if sync_res.get("buffered", 0) > 0:
                            logger.info(
                                "[EVAL]  agy_sync_loop buffered %d MCP recall(s)",
                                sync_res["buffered"],
                            )
                    except Exception as exc:
                        logger.debug("AGY transcript sync failed: %s", exc)
                    await asyncio.sleep(300)
            except asyncio.CancelledError:
                pass

        agy_sync_task = asyncio.create_task(_agy_sync_loop())
        application.state.agy_sync_task = agy_sync_task

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

        # Stop janitor scheduler if hosted in daemon
        janitor_sched_obj = getattr(application.state, "janitor_scheduler", None)
        if janitor_sched_obj is not None:
            try:
                await janitor_sched_obj.stop()
            except Exception:
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

        # Stop arena worker
        arena_task_obj = getattr(application.state, "arena_task", None)
        if arena_task_obj is not None:
            arena_task_obj.cancel()
            try:
                await asyncio.wait_for(arena_task_obj, timeout=10.0)
            except (asyncio.TimeoutError, asyncio.CancelledError):
                pass

        watcher_task = getattr(application.state, "watcher_task", None)
        if watcher_task is not None:
            watcher_task.cancel()
            try:
                await asyncio.wait_for(watcher_task, timeout=3.0)
            except (asyncio.TimeoutError, asyncio.CancelledError):
                pass

        agy_sync_task_obj = getattr(application.state, "agy_sync_task", None)
        if agy_sync_task_obj is not None:
            agy_sync_task_obj.cancel()
            try:
                await asyncio.wait_for(agy_sync_task_obj, timeout=3.0)
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
    # Include route sub-modules
    # ------------------------------------------------------------------
    from team_memory.daemon.runtime.routes import (
        debug_router,
        drafts_router,
        hooks_router,
        recall_router,
        status_router,
        wiki_router,
    )

    app.include_router(hooks_router)
    app.include_router(drafts_router)
    app.include_router(wiki_router)
    app.include_router(status_router)
    app.include_router(recall_router)
    app.include_router(debug_router)

    return app

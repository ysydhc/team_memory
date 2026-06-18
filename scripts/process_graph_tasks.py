"""Standalone worker for graphiti_ingest background tasks.

Runs as a long-lived process managed by taskctl.sh. Key design:
- Atomic task claiming (pending → running via FOR UPDATE SKIP LOCKED)
- Idle-timeout watchdog: kills only when no HTTP activity for IDLE_TIMEOUT_SECONDS
  (not a hard wall-clock limit — tasks making progress are never killed)
- State file (.task_worker.state) for rich monitoring
- Retry with back-off (up to max_retries from DB)
- Auto-restart via os.execv every N tasks to clear Kùzu/C-level state
- Polls continuously (sleeps when idle, never exits on its own)
"""

import asyncio
import gc
import json
import logging
import os
import signal
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

from sqlalchemy import text

from team_memory.config import load_settings
from team_memory.services.infra.task_runner import _handlers
from team_memory.storage.database import get_session

ROOT_DIR = Path(__file__).resolve().parent.parent
STATE_FILE = ROOT_DIR / ".task_worker.state"

IDLE_TIMEOUT_SECONDS = int(os.environ.get("IDLE_TIMEOUT", "180"))
IDLE_CHECK_INTERVAL = int(os.environ.get("IDLE_CHECK_INTERVAL", "30"))
RESTART_EVERY_N = int(os.environ.get("RESTART_EVERY", "20"))
IDLE_POLL_INTERVAL = 30

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-5s %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("graphiti_worker")

_task_start_time: float = 0.0
_last_activity: float = 0.0
_current_task_id: str | None = None
_current_retry_count: int = 0
_current_max_retries: int = 3


# ---------------------------------------------------------------------------
# Activity tracking — monkey-patch httpx to update _last_activity on responses
# ---------------------------------------------------------------------------


def _install_activity_hook():
    """Patch httpx.AsyncClient and httpx.Client to update _last_activity on every response."""
    import httpx

    _orig_async_send = httpx.AsyncClient.send
    _orig_sync_send = httpx.Client.send

    async def _patched_async_send(self, request, **kwargs):
        global _last_activity
        response = await _orig_async_send(self, request, **kwargs)
        _last_activity = time.monotonic()
        return response

    def _patched_sync_send(self, request, **kwargs):
        global _last_activity
        response = _orig_sync_send(self, request, **kwargs)
        _last_activity = time.monotonic()
        return response

    httpx.AsyncClient.send = _patched_async_send
    httpx.Client.send = _patched_sync_send


# ---------------------------------------------------------------------------
# State file — read by taskctl.sh status
# ---------------------------------------------------------------------------


def _write_state(**kwargs):
    state = {}
    if STATE_FILE.exists():
        try:
            state = json.loads(STATE_FILE.read_text())
        except Exception:
            pass
    state.update(kwargs)
    state["heartbeat"] = datetime.now(timezone.utc).isoformat()
    try:
        STATE_FILE.write_text(json.dumps(state, indent=2))
    except Exception:
        pass


# ---------------------------------------------------------------------------
# SIGALRM — periodic idle check (not a hard timeout)
# ---------------------------------------------------------------------------


def _mark_task_failed_sync(error_msg: str):
    """Use sync psycopg2 to mark the current task — safe in signal handler context."""
    if not _current_task_id:
        return
    new_retry = _current_retry_count + 1
    new_status = "failed" if new_retry >= _current_max_retries else "pending"
    try:
        import psycopg2

        settings = load_settings()
        db_url = str(settings.database.url).replace("+asyncpg", "")
        conn = psycopg2.connect(db_url)
        conn.autocommit = True
        with conn.cursor() as cur:
            cur.execute(
                "UPDATE background_tasks SET status = %s, retry_count = %s, error_message = %s WHERE id = %s",
                (new_status, new_retry, error_msg, _current_task_id),
            )
        conn.close()
    except Exception:
        pass


def _alarm_handler(signum, frame):
    global _last_activity
    now = time.monotonic()
    idle_seconds = now - _last_activity if _last_activity else now - _task_start_time
    total_elapsed = now - _task_start_time if _task_start_time else 0

    if idle_seconds < IDLE_TIMEOUT_SECONDS:
        signal.alarm(IDLE_CHECK_INTERVAL)
        return

    logger.error(
        "IDLE TIMEOUT: no HTTP activity for %.0fs (total elapsed %.0fs), force-exiting",
        idle_seconds,
        total_elapsed,
    )

    _mark_task_failed_sync(f"Idle timeout: no activity for {idle_seconds:.0f}s (total {total_elapsed:.0f}s)")

    _write_state(
        status="idle_timeout_exit",
        exit_reason=f"No activity for {idle_seconds:.0f}s",
    )
    os._exit(2)


# ---------------------------------------------------------------------------
# Startup recovery
# ---------------------------------------------------------------------------


async def _recover_orphaned_tasks(db_url: str) -> int:
    """Reset tasks stuck in 'running' from a previous crash back to 'pending'."""
    async with get_session(db_url) as session:
        result = await session.execute(
            text(
                "UPDATE background_tasks "
                "SET status = 'pending', retry_count = retry_count + 1, "
                "    error_message = 'Recovered: worker crashed while running' "
                "WHERE task_type = 'graphiti_ingest' AND status = 'running' "
                "RETURNING id"
            )
        )
        rows = result.fetchall()
        await session.commit()
        if rows:
            logger.warning("Recovered %d orphaned 'running' tasks → pending", len(rows))
        return len(rows)


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------


async def process_tasks():
    _install_activity_hook()

    settings = load_settings()
    db_url = str(settings.database.url)

    handler = _handlers.get("graphiti_ingest")
    if not handler:
        logger.error("graphiti_ingest handler not registered — exiting")
        return

    signal.signal(signal.SIGALRM, _alarm_handler)

    await _recover_orphaned_tasks(db_url)

    processed = 0
    failed = 0
    session_start = datetime.now(timezone.utc)

    _write_state(
        pid=os.getpid(),
        session_started_at=session_start.isoformat(),
        status="running",
        tasks_completed=0,
        tasks_failed=0,
        current_task=None,
    )

    logger.info(
        "Worker started (pid=%d, idle_timeout=%ds, check_interval=%ds, restart_every=%d)",
        os.getpid(),
        IDLE_TIMEOUT_SECONDS,
        IDLE_CHECK_INTERVAL,
        RESTART_EVERY_N,
    )

    idle_count = 0

    while True:
        # ---- Atomic claim: pending → running ----
        async with get_session(db_url) as session:
            result = await session.execute(
                text(
                    "UPDATE background_tasks "
                    "SET status = 'running', started_at = now() "
                    "WHERE id = ("
                    "  SELECT id FROM background_tasks "
                    "  WHERE task_type = 'graphiti_ingest' AND status = 'pending' "
                    "  ORDER BY created_at ASC LIMIT 1 "
                    "  FOR UPDATE SKIP LOCKED"
                    ") "
                    "RETURNING id, payload, retry_count, max_retries"
                )
            )
            row = result.first()
            await session.commit()

        if not row:
            if idle_count == 0:
                logger.info("No pending tasks, polling every %ds...", IDLE_POLL_INTERVAL)
            idle_count += 1
            _write_state(status="idle", current_task=None)
            await asyncio.sleep(IDLE_POLL_INTERVAL)
            continue

        idle_count = 0
        task_id = row[0]
        payload = row[1]
        retry_count = row[2]
        max_retries = row[3]
        record_id = payload.get("record_id", "?")
        source_type = payload.get("source_type", "experience")
        task_id_short = str(task_id)[:8]

        global _current_task_id, _current_retry_count, _current_max_retries
        _current_task_id = str(task_id)
        _current_retry_count = retry_count
        _current_max_retries = max_retries

        logger.info(
            "[START] %s record=%s type=%s retry=%d/%d",
            task_id_short,
            record_id[:12],
            source_type,
            retry_count,
            max_retries,
        )
        _write_state(
            status="processing",
            current_task={
                "task_id": str(task_id),
                "record_id": record_id,
                "source_type": source_type,
                "started_at": datetime.now(timezone.utc).isoformat(),
                "retry": retry_count,
            },
        )

        # ---- Execute with idle-timeout watchdog ----
        global _task_start_time, _last_activity
        _task_start_time = time.monotonic()
        _last_activity = _task_start_time
        signal.alarm(IDLE_CHECK_INTERVAL)

        try:
            await handler(payload)
            elapsed = time.monotonic() - _task_start_time
            signal.alarm(0)
            _current_task_id = None

            async with get_session(db_url) as session:
                await session.execute(
                    text("UPDATE background_tasks SET status = 'completed', completed_at = now() WHERE id = :id"),
                    {"id": task_id},
                )
                await session.commit()

            processed += 1
            logger.info(
                "[DONE]  %s %.1fs (session: %d done, %d fail)",
                task_id_short,
                elapsed,
                processed,
                failed,
            )
            _write_state(
                tasks_completed=processed,
                tasks_failed=failed,
                current_task=None,
                last_completed={
                    "task_id": str(task_id),
                    "record_id": record_id,
                    "elapsed_s": round(elapsed, 1),
                    "at": datetime.now(timezone.utc).isoformat(),
                },
            )

        except Exception as e:
            elapsed = time.monotonic() - _task_start_time
            signal.alarm(0)
            _current_task_id = None
            failed += 1

            new_retry = retry_count + 1
            if new_retry < max_retries:
                new_status = "pending"
                logger.warning(
                    "[RETRY] %s failed in %.1fs (%d/%d): %s",
                    task_id_short,
                    elapsed,
                    new_retry,
                    max_retries,
                    str(e)[:200],
                )
            else:
                new_status = "failed"
                logger.error(
                    "[FAIL]  %s failed permanently in %.1fs (%d/%d): %s",
                    task_id_short,
                    elapsed,
                    new_retry,
                    max_retries,
                    str(e)[:200],
                )

            async with get_session(db_url) as session:
                await session.execute(
                    text(
                        "UPDATE background_tasks "
                        "SET status = :status, retry_count = :retry, "
                        "    error_message = :error "
                        "WHERE id = :id"
                    ),
                    {
                        "id": task_id,
                        "status": new_status,
                        "retry": new_retry,
                        "error": str(e)[:500],
                    },
                )
                await session.commit()

            _write_state(
                tasks_completed=processed,
                tasks_failed=failed,
                current_task=None,
            )

        gc.collect()

        if processed >= RESTART_EVERY_N:
            logger.info(
                "Processed %d tasks, restarting to clear Kùzu state...",
                processed,
            )
            _write_state(status="restarting")
            os.execv(sys.executable, [sys.executable] + sys.argv)

        await asyncio.sleep(0.5)


if __name__ == "__main__":
    if sys.platform != "win32":
        asyncio.set_event_loop_policy(asyncio.DefaultEventLoopPolicy())
    asyncio.run(process_tasks())

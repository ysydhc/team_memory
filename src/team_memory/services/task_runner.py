"""Background task runner -- poll and execute persistent tasks."""

from __future__ import annotations

import logging
import time
from datetime import datetime, timezone
from typing import Any, Callable, Coroutine

from sqlalchemy import select

from team_memory.storage.database import get_session
from team_memory.storage.models import BackgroundTask

logger = logging.getLogger("team_memory.task_runner")

TaskHandler = Callable[[dict[str, Any]], Coroutine[Any, Any, None]]

_handlers: dict[str, TaskHandler] = {}


def register_handler(task_type: str, handler: TaskHandler) -> None:
    """Register an async handler for a task type."""
    _handlers[task_type] = handler
    logger.debug("Registered task handler for '%s'", task_type)


def get_handlers() -> dict[str, TaskHandler]:
    """Return a copy of the registered handlers (for testing)."""
    return dict(_handlers)


def clear_handlers() -> None:
    """Remove all registered handlers (for testing)."""
    _handlers.clear()


async def enqueue(task_type: str, payload: dict[str, Any], *, db_url: str) -> None:
    """Insert a pending task into the background_tasks table."""
    async with get_session(db_url) as session:
        task = BackgroundTask(task_type=task_type, payload=payload)
        session.add(task)


async def poll_and_execute(db_url: str, *, batch_size: int = 5) -> int:
    """Claim pending tasks and execute them. Returns count processed."""
    processed = 0
    async with get_session(db_url) as session:
        stmt = (
            select(BackgroundTask)
            .where(BackgroundTask.status == "pending")
            .order_by(BackgroundTask.created_at)
            .limit(batch_size)
        )
        result = await session.execute(stmt)
        tasks = result.scalars().all()

        for task in tasks:
            handler = _handlers.get(task.task_type)
            if handler is None:
                logger.warning("No handler for task type '%s', skipping", task.task_type)
                continue

            task.status = "running"
            task.started_at = datetime.now(timezone.utc)
            await session.flush()

            try:
                start = time.monotonic()
                await handler(task.payload)
                duration_ms = int((time.monotonic() - start) * 1000)
                task.status = "completed"
                task.completed_at = datetime.now(timezone.utc)
                processed += 1
                logger.info(
                    "Task completed: %s",
                    task.task_type,
                    extra={
                        "task_id": str(task.id),
                        "task_type": task.task_type,
                        "duration_ms": duration_ms,
                    },
                )
            except Exception as e:
                duration_ms = int((time.monotonic() - start) * 1000)
                task.retry_count += 1
                if task.retry_count >= task.max_retries:
                    task.status = "failed"
                    task.error_message = str(e)[:500]
                    logger.error(
                        "Task failed: %s (%d/%d retries)",
                        task.task_type,
                        task.retry_count,
                        task.max_retries,
                        extra={
                            "task_id": str(task.id),
                            "duration_ms": duration_ms,
                            "error": str(e)[:200],
                        },
                    )
                else:
                    task.status = "pending"  # retry later
                    task.error_message = str(e)[:500]
                    logger.info(
                        "Task %s will retry (%d/%d): %s",
                        task.id,
                        task.retry_count,
                        task.max_retries,
                        e,
                        extra={
                            "task_id": str(task.id),
                            "duration_ms": duration_ms,
                        },
                    )
            await session.flush()

    return processed

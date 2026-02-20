"""Async embedding generation queue (D2).

Decouples embedding generation from the request path. New experiences
are saved immediately with embedding_status='pending', and a background
worker picks up the task and generates the embedding asynchronously.

Usage:
    queue = EmbeddingQueue(embedding_provider, db_url)
    await queue.start()
    await queue.enqueue(experience_id, text_to_embed)
    ...
    await queue.stop()

Architecture:
    Request → save(status=pending) → queue.enqueue() → return immediately
                                          ↓
                                    Background Worker(s)
                                          ↓
                                    encode_single(text)
                                          ↓
                                    UPDATE experiences SET embedding=?, status=ready
"""

from __future__ import annotations

import asyncio
import logging
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone

from team_memory.embedding.base import EmbeddingProvider
from team_memory.services.event_bus import EventBus, Events

logger = logging.getLogger("team_memory.embedding_queue")


@dataclass
class EmbeddingTask:
    """A single embedding task to be processed by the background worker."""

    experience_id: uuid.UUID
    text: str
    retry_count: int = 0
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


class EmbeddingQueue:
    """Async embedding generation queue with background workers.

    Uses asyncio.Queue internally. Workers are started as asyncio Tasks
    during application lifespan.
    """

    def __init__(
        self,
        embedding_provider: EmbeddingProvider,
        db_url: str,
        max_workers: int = 3,
        max_retries: int = 3,
        max_queue_size: int = 1000,
        event_bus: EventBus | None = None,
    ):
        self._embedding = embedding_provider
        self._db_url = db_url
        self._max_workers = max_workers
        self._max_retries = max_retries
        self._event_bus = event_bus or EventBus()
        self._queue: asyncio.Queue[EmbeddingTask] = asyncio.Queue(
            maxsize=max_queue_size
        )
        self._workers: list[asyncio.Task] = []
        self._running = False
        self._processed = 0
        self._failed = 0

    async def enqueue(self, experience_id: uuid.UUID, text: str) -> None:
        """Add an embedding task to the queue.

        Non-blocking; raises asyncio.QueueFull if the queue is at capacity.
        """
        task = EmbeddingTask(experience_id=experience_id, text=text)
        try:
            self._queue.put_nowait(task)
            logger.debug(
                "Enqueued embedding task for experience %s (queue size: %d)",
                experience_id,
                self._queue.qsize(),
            )
        except asyncio.QueueFull:
            logger.error(
                "Embedding queue full, cannot enqueue task for %s", experience_id
            )
            raise

    async def start(self) -> None:
        """Start background worker tasks."""
        if self._running:
            return
        self._running = True
        for i in range(self._max_workers):
            worker = asyncio.create_task(
                self._worker(worker_id=i), name=f"embedding-worker-{i}"
            )
            self._workers.append(worker)
        logger.info(
            "Embedding queue started with %d workers", self._max_workers
        )

    async def stop(self) -> None:
        """Gracefully stop all workers.

        Waits for current tasks to complete, then cancels workers.
        """
        self._running = False
        for worker in self._workers:
            worker.cancel()
        if self._workers:
            await asyncio.gather(*self._workers, return_exceptions=True)
        self._workers.clear()
        logger.info("Embedding queue stopped")

    async def _worker(self, worker_id: int) -> None:
        """Background worker coroutine: dequeue tasks and process them."""
        logger.debug("Embedding worker %d started", worker_id)
        while self._running:
            try:
                task = await asyncio.wait_for(
                    self._queue.get(), timeout=5.0
                )
            except asyncio.TimeoutError:
                continue
            except asyncio.CancelledError:
                break

            try:
                await self._process_task(task)
                self._processed += 1
            except Exception:
                logger.warning(
                    "Worker %d failed to process task for experience %s",
                    worker_id,
                    task.experience_id,
                    exc_info=True,
                )
                self._failed += 1
                # Retry if within limits
                if task.retry_count < self._max_retries:
                    task.retry_count += 1
                    logger.info(
                        "Retrying embedding for %s (attempt %d/%d)",
                        task.experience_id,
                        task.retry_count,
                        self._max_retries,
                    )
                    await asyncio.sleep(2 ** task.retry_count)  # Exponential backoff
                    try:
                        self._queue.put_nowait(task)
                    except asyncio.QueueFull:
                        await self._mark_failed(task.experience_id)
                else:
                    await self._mark_failed(task.experience_id)
            finally:
                self._queue.task_done()

    async def _process_task(self, task: EmbeddingTask) -> None:
        """Generate embedding and update the database."""
        from team_memory.storage.database import get_session
        from team_memory.storage.models import Experience

        # Generate embedding
        embedding = await self._embedding.encode_single(task.text)

        # Update database
        from sqlalchemy import update

        async with get_session(self._db_url) as session:
            await session.execute(
                update(Experience)
                .where(Experience.id == task.experience_id)
                .values(embedding=embedding, embedding_status="ready")
            )
            await session.commit()

        logger.info("Embedding generated for experience %s", task.experience_id)

        await self._event_bus.emit(
            Events.EMBEDDING_COMPLETED,
            {
                "experience_id": str(task.experience_id),
                "retry_count": task.retry_count,
            },
        )

    async def _mark_failed(self, experience_id: uuid.UUID) -> None:
        """Mark an experience's embedding as failed in the database."""
        from sqlalchemy import update

        from team_memory.storage.database import get_session
        from team_memory.storage.models import Experience

        try:
            async with get_session(self._db_url) as session:
                await session.execute(
                    update(Experience)
                    .where(Experience.id == experience_id)
                    .values(embedding_status="failed")
                )
                await session.commit()
        except Exception:
            logger.error(
                "Failed to mark experience %s as failed", experience_id, exc_info=True
            )

        await self._event_bus.emit(
            Events.EMBEDDING_FAILED,
            {"experience_id": str(experience_id)},
        )

    @property
    def status(self) -> dict:
        """Return queue status information."""
        return {
            "running": self._running,
            "workers": len(self._workers),
            "pending": self._queue.qsize(),
            "processed": self._processed,
            "failed": self._failed,
        }

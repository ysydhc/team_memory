"""Tests for the background task runner (services/task_runner.py).

Validates enqueueing, polling, handler dispatch, retry logic,
and the skip-on-missing-handler path.
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from team_memory.services import task_runner
from team_memory.storage.models import BackgroundTask

# ============================================================
# Fixtures
# ============================================================


@pytest.fixture(autouse=True)
def _clean_handlers():
    """Ensure handler registry is clean before and after each test."""
    task_runner.clear_handlers()
    yield
    task_runner.clear_handlers()


def _make_task(
    task_type: str = "test_task",
    payload: dict | None = None,
    status: str = "pending",
    retry_count: int = 0,
    max_retries: int = 3,
) -> BackgroundTask:
    """Create a BackgroundTask instance for testing."""
    t = BackgroundTask(
        id=uuid.uuid4(),
        task_type=task_type,
        payload=payload or {},
        status=status,
        created_at=datetime.now(timezone.utc),
        retry_count=retry_count,
        max_retries=max_retries,
    )
    return t


# ============================================================
# register_handler / clear_handlers
# ============================================================


class TestHandlerRegistration:
    def test_register_handler(self) -> None:
        handler = AsyncMock()
        task_runner.register_handler("my_type", handler)
        assert "my_type" in task_runner.get_handlers()
        assert task_runner.get_handlers()["my_type"] is handler

    def test_clear_handlers(self) -> None:
        task_runner.register_handler("a", AsyncMock())
        task_runner.register_handler("b", AsyncMock())
        task_runner.clear_handlers()
        assert task_runner.get_handlers() == {}


# ============================================================
# enqueue
# ============================================================


class TestEnqueue:
    @pytest.mark.asyncio
    async def test_enqueue_creates_pending_task(self) -> None:
        """enqueue() should add a BackgroundTask to the session."""
        mock_session = AsyncMock()
        added_objects: list[BackgroundTask] = []
        mock_session.add = MagicMock(side_effect=lambda obj: added_objects.append(obj))

        with patch("team_memory.services.task_runner.get_session") as mock_get_session:
            mock_get_session.return_value.__aenter__ = AsyncMock(return_value=mock_session)
            mock_get_session.return_value.__aexit__ = AsyncMock(return_value=False)

            await task_runner.enqueue(
                "pattern_extraction",
                {"user_id": "alice", "raw_conversation": "hello"},
                db_url="sqlite+aiosqlite://",
            )

        assert len(added_objects) == 1
        task = added_objects[0]
        assert task.task_type == "pattern_extraction"
        assert task.payload == {"user_id": "alice", "raw_conversation": "hello"}


# ============================================================
# poll_and_execute
# ============================================================


class TestPollAndExecute:
    @pytest.mark.asyncio
    async def test_poll_and_execute_calls_handler(self) -> None:
        """A pending task should be claimed and dispatched to its handler."""
        handler = AsyncMock()
        task_runner.register_handler("test_task", handler)

        task = _make_task(payload={"key": "value"})
        mock_result = MagicMock()
        mock_result.scalars.return_value.all.return_value = [task]

        mock_session = AsyncMock()
        mock_session.execute = AsyncMock(return_value=mock_result)
        mock_session.flush = AsyncMock()

        with patch("team_memory.services.task_runner.get_session") as mock_get_session:
            mock_get_session.return_value.__aenter__ = AsyncMock(return_value=mock_session)
            mock_get_session.return_value.__aexit__ = AsyncMock(return_value=False)

            processed = await task_runner.poll_and_execute("sqlite+aiosqlite://")

        assert processed == 1
        handler.assert_awaited_once_with({"key": "value"})
        assert task.status == "completed"
        assert task.completed_at is not None

    @pytest.mark.asyncio
    async def test_failed_task_retries(self) -> None:
        """When a handler raises, the task should be marked for retry."""
        handler = AsyncMock(side_effect=RuntimeError("boom"))
        task_runner.register_handler("test_task", handler)

        task = _make_task(retry_count=0, max_retries=3)
        mock_result = MagicMock()
        mock_result.scalars.return_value.all.return_value = [task]

        mock_session = AsyncMock()
        mock_session.execute = AsyncMock(return_value=mock_result)
        mock_session.flush = AsyncMock()

        with patch("team_memory.services.task_runner.get_session") as mock_get_session:
            mock_get_session.return_value.__aenter__ = AsyncMock(return_value=mock_session)
            mock_get_session.return_value.__aexit__ = AsyncMock(return_value=False)

            processed = await task_runner.poll_and_execute("sqlite+aiosqlite://")

        assert processed == 0
        assert task.retry_count == 1
        assert task.status == "pending"  # will be retried
        assert task.error_message == "boom"

    @pytest.mark.asyncio
    async def test_failed_task_permanent_failure(self) -> None:
        """When retries exhausted, the task should be marked as failed."""
        handler = AsyncMock(side_effect=RuntimeError("permanent"))
        task_runner.register_handler("test_task", handler)

        task = _make_task(retry_count=2, max_retries=3)  # one more failure => failed
        mock_result = MagicMock()
        mock_result.scalars.return_value.all.return_value = [task]

        mock_session = AsyncMock()
        mock_session.execute = AsyncMock(return_value=mock_result)
        mock_session.flush = AsyncMock()

        with patch("team_memory.services.task_runner.get_session") as mock_get_session:
            mock_get_session.return_value.__aenter__ = AsyncMock(return_value=mock_session)
            mock_get_session.return_value.__aexit__ = AsyncMock(return_value=False)

            processed = await task_runner.poll_and_execute("sqlite+aiosqlite://")

        assert processed == 0
        assert task.retry_count == 3
        assert task.status == "failed"
        assert task.error_message == "permanent"

    @pytest.mark.asyncio
    async def test_no_handler_skips_task(self) -> None:
        """Tasks with no registered handler should be skipped (not crash)."""
        task = _make_task(task_type="unknown_type")
        mock_result = MagicMock()
        mock_result.scalars.return_value.all.return_value = [task]

        mock_session = AsyncMock()
        mock_session.execute = AsyncMock(return_value=mock_result)
        mock_session.flush = AsyncMock()

        with patch("team_memory.services.task_runner.get_session") as mock_get_session:
            mock_get_session.return_value.__aenter__ = AsyncMock(return_value=mock_session)
            mock_get_session.return_value.__aexit__ = AsyncMock(return_value=False)

            processed = await task_runner.poll_and_execute("sqlite+aiosqlite://")

        assert processed == 0
        # Task status should remain pending since it was skipped
        assert task.status == "pending"

    @pytest.mark.asyncio
    async def test_empty_queue_returns_zero(self) -> None:
        """When no pending tasks exist, poll_and_execute returns 0."""
        mock_result = MagicMock()
        mock_result.scalars.return_value.all.return_value = []

        mock_session = AsyncMock()
        mock_session.execute = AsyncMock(return_value=mock_result)

        with patch("team_memory.services.task_runner.get_session") as mock_get_session:
            mock_get_session.return_value.__aenter__ = AsyncMock(return_value=mock_session)
            mock_get_session.return_value.__aexit__ = AsyncMock(return_value=False)

            processed = await task_runner.poll_and_execute("sqlite+aiosqlite://")

        assert processed == 0

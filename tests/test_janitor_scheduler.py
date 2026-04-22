"""Tests for JanitorScheduler service."""

import asyncio
from unittest.mock import AsyncMock, MagicMock

import pytest

from team_memory.services.janitor_scheduler import JanitorScheduler


class TestJanitorScheduler:
    """Test cases for JanitorScheduler."""

    @pytest.fixture
    def mock_janitor(self):
        """Create a mock MemoryJanitor instance."""
        janitor = AsyncMock()
        janitor.run_all = AsyncMock(
            return_value={
                "project": None,
                "completed_at": "2024-01-01T00:00:00Z",
                "operations": {
                    "score_decay": {"updated_count": 5},
                    "outdated_sweep": {"found_count": 2, "deleted_count": 1},
                    "soft_deleted_purge": {"purged_count": 0},
                    "draft_expiration": {"expired_count": 1},
                    "personal_memory_pruning": {"pruned_count": 3},
                },
            }
        )
        return janitor

    @pytest.fixture
    def mock_config(self):
        """Create a mock configuration object."""
        config = MagicMock()
        config.interval_hours = 1  # Short interval for testing
        return config

    def test_init(self, mock_janitor):
        """Test scheduler initialization."""
        scheduler = JanitorScheduler(mock_janitor)

        assert scheduler._janitor is mock_janitor
        assert scheduler._config is None
        assert scheduler._task is None
        assert scheduler._running is False

    def test_init_with_config(self, mock_janitor, mock_config):
        """Test scheduler initialization with config."""
        scheduler = JanitorScheduler(mock_janitor, mock_config)

        assert scheduler._janitor is mock_janitor
        assert scheduler._config is mock_config
        assert scheduler._task is None
        assert scheduler._running is False

    def test_get_config_with_config(self, mock_janitor, mock_config):
        """Test _get_config with configuration object."""
        scheduler = JanitorScheduler(mock_janitor, mock_config)

        assert scheduler._get_config("interval_hours") == 1

        # Configure mock to raise AttributeError for nonexistent attribute
        del mock_config.nonexistent  # Ensure it doesn't exist
        mock_config.configure_mock(**{})  # Reset mock

        # Test that nonexistent attribute returns default
        assert scheduler._get_config("nonexistent", "default") == "default"

    def test_get_config_without_config(self, mock_janitor):
        """Test _get_config without configuration object."""
        scheduler = JanitorScheduler(mock_janitor)

        assert scheduler._get_config("interval_hours", 24) == 24
        assert scheduler._get_config("nonexistent") is None

    def test_is_running_initially_false(self, mock_janitor):
        """Test that scheduler is not running initially."""
        scheduler = JanitorScheduler(mock_janitor)

        assert scheduler.is_running() is False

    @pytest.mark.asyncio
    async def test_start_sets_running_flag(self, mock_janitor, mock_config):
        """Test that start() sets the running flag and creates task."""
        scheduler = JanitorScheduler(mock_janitor, mock_config)

        await scheduler.start()

        assert scheduler._running is True
        assert scheduler._task is not None
        assert scheduler.is_running() is True

        # Clean up
        await scheduler.stop()

    @pytest.mark.asyncio
    async def test_start_multiple_times_safe(self, mock_janitor, mock_config):
        """Test that calling start() multiple times is safe."""
        scheduler = JanitorScheduler(mock_janitor, mock_config)

        await scheduler.start()
        first_task = scheduler._task

        await scheduler.start()  # Should not create new task

        assert scheduler._task is first_task
        assert scheduler.is_running() is True

        # Clean up
        await scheduler.stop()

    @pytest.mark.asyncio
    async def test_stop_sets_running_flag(self, mock_janitor, mock_config):
        """Test that stop() sets running flag to False and cancels task."""
        scheduler = JanitorScheduler(mock_janitor, mock_config)

        await scheduler.start()
        assert scheduler.is_running() is True

        await scheduler.stop()

        assert scheduler._running is False
        assert scheduler._task is None
        assert scheduler.is_running() is False

    @pytest.mark.asyncio
    async def test_stop_when_not_running(self, mock_janitor):
        """Test that stop() is safe when scheduler is not running."""
        scheduler = JanitorScheduler(mock_janitor)

        # Should not raise an exception
        await scheduler.stop()

        assert scheduler._running is False
        assert scheduler._task is None

    @pytest.mark.asyncio
    async def test_run_loop_calls_janitor(self, mock_janitor, mock_config):
        """Test that the run loop calls janitor operations."""
        # Use very short interval for testing (0.0001 hours = 0.36 seconds)
        mock_config.interval_hours = 0.0001
        scheduler = JanitorScheduler(mock_janitor, mock_config)

        await scheduler.start()

        # Wait for initial sleep + a bit more for the janitor to run
        await asyncio.sleep(0.5)  # Wait longer to ensure janitor runs

        await scheduler.stop()

        # Verify janitor was called
        mock_janitor.run_all.assert_called_with(project=None)

    @pytest.mark.asyncio
    async def test_run_loop_handles_janitor_errors(self, mock_janitor, mock_config):
        """Test that the run loop handles janitor errors gracefully."""
        # Make janitor raise an exception
        mock_janitor.run_all = AsyncMock(side_effect=Exception("Test error"))

        # Use very short interval for testing
        mock_config.interval_hours = 0.01
        scheduler = JanitorScheduler(mock_janitor, mock_config)

        await scheduler.start()

        # Wait for initial sleep + error handling
        await asyncio.sleep(0.5)

        # Scheduler should still be running despite the error
        assert scheduler.is_running() is True

        await scheduler.stop()

    @pytest.mark.asyncio
    async def test_scheduler_lifecycle(self, mock_janitor, mock_config):
        """Test complete scheduler lifecycle."""
        scheduler = JanitorScheduler(mock_janitor, mock_config)

        # Initial state
        assert not scheduler.is_running()

        # Start scheduler
        await scheduler.start()
        assert scheduler.is_running()

        # Stop scheduler
        await scheduler.stop()
        assert not scheduler.is_running()

        # Can restart
        await scheduler.start()
        assert scheduler.is_running()

        # Clean up
        await scheduler.stop()

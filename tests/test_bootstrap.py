"""Tests for bootstrap module."""

from __future__ import annotations

import logging
from pathlib import Path

import pytest

from team_memory.config import load_settings, reset_settings


@pytest.fixture(autouse=True)
def _reset_settings() -> None:
    """Reset global settings before/after each test."""
    reset_settings()
    yield
    reset_settings()


def test_trim_log_files_to_max_bytes_deletes_oldest(tmp_path: Path) -> None:
    """_trim_log_files_to_max_bytes deletes oldest files until total <= max_bytes."""
    from team_memory.bootstrap import _trim_log_files_to_max_bytes

    (tmp_path / "a.log").write_bytes(b"x" * 100)
    (tmp_path / "b.log").write_bytes(b"x" * 100)
    (tmp_path / "c.log").write_bytes(b"x" * 100)
    _trim_log_files_to_max_bytes(tmp_path, max_bytes=150)
    remaining = sorted(tmp_path.glob("*.log"))
    assert len(remaining) == 1
    assert remaining[0].name == "c.log"


def test_trim_log_files_skips_when_debug(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """_trim_log_files_to_max_bytes skips when TEAM_MEMORY_DEBUG=1."""
    from team_memory.bootstrap import _trim_log_files_to_max_bytes

    monkeypatch.setenv("TEAM_MEMORY_DEBUG", "1")
    (tmp_path / "a.log").write_bytes(b"x" * 100)
    (tmp_path / "b.log").write_bytes(b"x" * 100)
    _trim_log_files_to_max_bytes(tmp_path, max_bytes=50)
    assert len(list(tmp_path.glob("*.log"))) == 2


@pytest.mark.asyncio
async def test_stop_background_tasks_stops_log_listener() -> None:
    """stop_background_tasks stops _log_listener when present."""
    import queue
    from logging.handlers import QueueListener
    from unittest.mock import MagicMock

    from team_memory.bootstrap import stop_background_tasks

    log_queue: queue.Queue = queue.Queue()
    log_listener = QueueListener(log_queue, logging.NullHandler())
    log_listener.start()
    ctx = MagicMock()
    ctx.embedding_queue = None
    ctx._stale_scanner_task = None
    ctx._log_listener = log_listener
    await stop_background_tasks(ctx)
    assert log_listener._thread is None or not log_listener._thread.is_alive()


def test_configure_logging_adds_queue_handler_when_file_enabled(tmp_path: str) -> None:
    """When log_file_enabled=True, _configure_logging adds QueueHandler to team_memory logger."""
    from logging.handlers import QueueHandler

    from team_memory.bootstrap import _configure_logging

    settings = load_settings()
    settings.logging.log_file_enabled = True
    settings.logging.log_file_path = str(tmp_path / "app.log")
    _configure_logging(settings)
    root = logging.getLogger("team_memory")
    assert any(isinstance(h, QueueHandler) for h in root.handlers)

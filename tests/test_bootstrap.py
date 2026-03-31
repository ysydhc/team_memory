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


def test_is_debug_mode_env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """_is_debug_mode returns True when TEAM_MEMORY_DEBUG=1."""
    from team_memory.bootstrap import _is_debug_mode

    monkeypatch.setenv("TEAM_MEMORY_DEBUG", "1")
    assert _is_debug_mode() is True
    monkeypatch.setenv("TEAM_MEMORY_DEBUG", "0")
    assert _is_debug_mode() is False


def test_is_debug_mode_logger_level(tmp_path: Path) -> None:
    """_is_debug_mode returns True when team_memory logger effective level is DEBUG."""
    from team_memory.bootstrap import _is_debug_mode

    root = logging.getLogger("team_memory")
    root.setLevel(logging.DEBUG)
    try:
        assert _is_debug_mode() is True
    finally:
        root.setLevel(logging.INFO)


def test_configure_logging_uses_rotating_handler_when_not_debug(tmp_path: Path) -> None:
    """When not DEBUG, _configure_logging uses RotatingFileHandler."""
    from logging.handlers import RotatingFileHandler

    from team_memory.bootstrap import _configure_logging

    settings = load_settings()
    settings.logging.log_file_enabled = True
    settings.logging.log_file_path = str(tmp_path / "app.log")
    listener = _configure_logging(settings)
    assert listener is not None
    assert len(listener.handlers) == 1
    assert isinstance(listener.handlers[0], RotatingFileHandler)


def test_configure_logging_uses_file_handler_when_debug(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """When TEAM_MEMORY_DEBUG=1, _configure_logging uses FileHandler (no size limit)."""
    from logging.handlers import RotatingFileHandler

    from team_memory.bootstrap import _configure_logging

    monkeypatch.setenv("TEAM_MEMORY_DEBUG", "1")
    settings = load_settings()
    settings.logging.log_file_enabled = True
    settings.logging.log_file_path = str(tmp_path / "app.log")
    listener = _configure_logging(settings)
    assert listener is not None
    assert len(listener.handlers) == 1
    assert isinstance(listener.handlers[0], logging.FileHandler)
    assert not isinstance(listener.handlers[0], RotatingFileHandler)


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

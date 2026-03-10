"""Tests for bootstrap module."""

from __future__ import annotations

import logging

import pytest

from team_memory.config import load_settings, reset_settings


@pytest.fixture(autouse=True)
def _reset_settings() -> None:
    """Reset global settings before/after each test."""
    reset_settings()
    yield
    reset_settings()


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

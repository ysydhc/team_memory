"""E2E tests for logging system: I/O logs and hot reload.

When LOG_IO_ENABLED=1, MCP tool calls produce [io] logs.
Optional: PUT logging config hot reload verification.
"""

from __future__ import annotations

import logging
import uuid
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from team_memory.server import mcp


class _CollectingHandler(logging.Handler):
    """Handler that collects log records for assertion."""

    def __init__(self):
        super().__init__()
        self.records: list[logging.LogRecord] = []

    def emit(self, record: logging.LogRecord) -> None:
        self.records.append(record)


# ============================================================
# test_log_io_enabled_produces_output
# ============================================================


class TestLogIoEnabledProducesOutput:
    """When LOG_IO_ENABLED=1, MCP or /health request produces I/O logs."""

    @pytest.mark.asyncio
    async def test_log_io_enabled_produces_output(self):
        """LOG_IO_ENABLED=1: MCP tool call produces [io] logs with node_id/tool_name."""
        # Enable I/O logging via settings provider (same effect as env/config)
        mock_settings = MagicMock()
        mock_settings.logging.log_io_enabled = True
        mock_settings.logging.log_io_detail = "mcp"

        mock_session = AsyncMock()
        mock_get_session = MagicMock()
        mock_get_session.return_value.__aenter__ = AsyncMock(return_value=mock_session)
        mock_get_session.return_value.__aexit__ = AsyncMock(return_value=False)

        io_logger = logging.getLogger("team_memory.io")
        collector = _CollectingHandler()
        collector.setLevel(logging.INFO)
        io_logger.addHandler(collector)
        try:
            with (
                patch("team_memory.server.io_logger._settings_provider", lambda: mock_settings),
                patch("team_memory.server._get_service") as mock_get_service,
                patch("team_memory.server.get_session", mock_get_session),
                patch("team_memory.server._get_current_user", return_value="alice"),
                patch("team_memory.server._get_db_url", return_value="sqlite:///:memory:"),
            ):
                mock_service = MagicMock()
                mock_service.feedback = AsyncMock(return_value={"message": "recorded"})
                mock_get_service.return_value = mock_service

                tools = await mcp.get_tools()
                feedback_fn = tools["tm_feedback"].fn
                await feedback_fn(experience_id=str(uuid.uuid4()), rating=1)
        finally:
            io_logger.removeHandler(collector)

        # Assert [io] format appears (e.g. "[io] tm_feedback in: {...}" or "[io] tm_feedback out: ...")
        io_records = [r for r in collector.records if "[io]" in (r.getMessage() or "")]
        assert len(io_records) >= 1, (
            f"Expected [io] logs when LOG_IO_ENABLED=1, got: {[r.getMessage() for r in collector.records]}"
        )
        combined = " ".join(r.getMessage() or "" for r in io_records)
        assert "tm_feedback" in combined or "tm_" in combined, (
            f"Expected node_id/tool_name in [io] output, got: {combined}"
        )

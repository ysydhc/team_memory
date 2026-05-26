"""Tests for team_memory hooks session_timeout — updated for HTTP-based implementation."""

from __future__ import annotations

from unittest.mock import AsyncMock, patch

import pytest

from team_memory.hooks.session_timeout import SessionTimeoutManager


class TestSessionTimeoutManagerInit:
    """SessionTimeoutManager should no longer depend on DraftBuffer/DraftRefiner."""

    def test_init_no_args(self):
        mgr = SessionTimeoutManager()
        assert mgr._timeout_minutes == 30

    def test_init_custom_timeout(self):
        mgr = SessionTimeoutManager(timeout_minutes=5)
        assert mgr._timeout_minutes == 5

    def test_init_custom_daemon_url(self):
        mgr = SessionTimeoutManager(daemon_url="http://localhost:9999")
        assert mgr._daemon_url == "http://localhost:9999"

    def test_no_daemon_imports(self):
        """Verify session_timeout does not import daemon internals."""
        import ast
        import pathlib

        src = pathlib.Path(
            "src/team_memory/hooks/session_timeout.py"
        ).read_text()
        tree = ast.parse(src)
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and node.module:
                assert not node.module.startswith(
                    "team_memory.daemon"
                ), f"session_timeout imports daemon: {node.module}"


class TestSessionTimeoutManagerHTTP:
    """SessionTimeoutManager uses HTTP to communicate with daemon."""

    @pytest.mark.asyncio
    async def test_check_timeouts_calls_daemon(self):
        mgr = SessionTimeoutManager(timeout_minutes=30)
        with patch(
            "team_memory.hooks.session_timeout.async_post_hook",
            new_callable=AsyncMock,
        ) as mock_post:
            # First call returns timed-out sessions
            mock_post.side_effect = [
                {"timed_out_sessions": ["sess-1", "sess-2"]},
                {"status": "ok"},
                {"status": "ok"},
            ]
            await mgr._check_timeouts()
            # session_timeout_check + session_end for each session
            assert mock_post.call_count == 3

    @pytest.mark.asyncio
    async def test_check_timeouts_no_timed_out(self):
        mgr = SessionTimeoutManager(timeout_minutes=30)
        with patch(
            "team_memory.hooks.session_timeout.async_post_hook",
            new_callable=AsyncMock,
            return_value={"timed_out_sessions": []},
        ) as mock_post:
            await mgr._check_timeouts()
            # Only the check call, no session_end calls
            assert mock_post.call_count == 1

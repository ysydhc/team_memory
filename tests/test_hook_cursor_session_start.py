"""Tests for scripts/hooks/cursor_session_start.py — thin forwarder to TM Daemon."""
from __future__ import annotations

import io
import json
from unittest.mock import patch

import httpx
import pytest


class TestCursorSessionStartThin:
    """Tests for the thin cursor_session_start hook."""

    def test_forwards_input_to_daemon(self) -> None:
        from hooks.cursor_session_start import main

        mock_response = httpx.Response(
            200,
            text=json.dumps({"additionalContext": "some context"}),
            request=httpx.Request("POST", "http://127.0.0.1:3901/hooks/session_start"),
        )

        input_data = {"workspace_roots": ["/tmp"], "conversation_id": "sess-1"}
        with patch("sys.stdin", io.StringIO(json.dumps(input_data))):
            with patch("hooks.cursor_session_start.httpx.post", return_value=mock_response) as mock_post:
                with patch("builtins.print") as mock_print:
                    main()

        mock_post.assert_called_once()
        mock_print.assert_called_once_with(
            json.dumps({"additionalContext": "some context"})
        )

    def test_daemon_not_running_returns_ok(self) -> None:
        from hooks.cursor_session_start import main

        input_data = {"workspace_roots": ["/tmp"]}
        with patch("sys.stdin", io.StringIO(json.dumps(input_data))):
            with patch("hooks.cursor_session_start.httpx.post", side_effect=httpx.ConnectError("refused")):
                with patch("builtins.print") as mock_print:
                    main()

        mock_print.assert_called_once_with(
            json.dumps({"action": "ok", "message": "daemon not running"})
        )

    def test_invalid_stdin_returns_error(self) -> None:
        from hooks.cursor_session_start import main

        with patch("sys.stdin", io.StringIO("invalid!")):
            with patch("builtins.print") as mock_print:
                main()

        mock_print.assert_called_once_with(
            json.dumps({"action": "error", "message": "invalid input"})
        )

    def test_timeout_returns_error(self) -> None:
        from hooks.cursor_session_start import main

        input_data = {"workspace_roots": ["/tmp"]}
        with patch("sys.stdin", io.StringIO(json.dumps(input_data))):
            with patch("hooks.cursor_session_start.httpx.post", side_effect=httpx.TimeoutException("timed out")):
                with patch("builtins.print") as mock_print:
                    main()

        mock_print.assert_called_once_with(
            json.dumps({"action": "error", "message": "timed out"})
        )

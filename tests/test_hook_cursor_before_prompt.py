"""Tests for scripts/hooks/cursor_before_prompt.py — thin forwarder to TM Daemon."""
from __future__ import annotations

import io
import json
from unittest.mock import patch

import httpx
import pytest


class TestCursorBeforePromptThin:
    """Tests for the thin cursor_before_prompt hook."""

    def test_forwards_input_to_daemon(self) -> None:
        from hooks.cursor_before_prompt import main

        mock_response = httpx.Response(
            200,
            text=json.dumps({"additionalContext": "retrieved context"}),
            request=httpx.Request("POST", "http://127.0.0.1:3901/hooks/before_prompt"),
        )

        input_data = {"prompt": "之前的问题", "workspace_roots": ["/tmp"]}
        with patch("sys.stdin", io.StringIO(json.dumps(input_data))):
            with patch("hooks.cursor_before_prompt.httpx.post", return_value=mock_response) as mock_post:
                with patch("builtins.print") as mock_print:
                    main()

        mock_post.assert_called_once()
        mock_print.assert_called_once_with(
            json.dumps({"additionalContext": "retrieved context"})
        )

    def test_daemon_not_running_returns_ok(self) -> None:
        from hooks.cursor_before_prompt import main

        input_data = {"prompt": "test", "workspace_roots": []}
        with patch("sys.stdin", io.StringIO(json.dumps(input_data))):
            with patch("hooks.cursor_before_prompt.httpx.post", side_effect=httpx.ConnectError("refused")):
                with patch("builtins.print") as mock_print:
                    main()

        mock_print.assert_called_once_with(
            json.dumps({"action": "ok", "message": "daemon not running"})
        )

    def test_invalid_stdin_returns_error(self) -> None:
        from hooks.cursor_before_prompt import main

        with patch("sys.stdin", io.StringIO("not-json")):
            with patch("builtins.print") as mock_print:
                main()

        mock_print.assert_called_once_with(
            json.dumps({"action": "error", "message": "invalid input"})
        )

    def test_timeout_returns_error(self) -> None:
        from hooks.cursor_before_prompt import main

        input_data = {"prompt": "test"}
        with patch("sys.stdin", io.StringIO(json.dumps(input_data))):
            with patch("hooks.cursor_before_prompt.httpx.post", side_effect=httpx.TimeoutException("timed out")):
                with patch("builtins.print") as mock_print:
                    main()

        mock_print.assert_called_once_with(
            json.dumps({"action": "error", "message": "timed out"})
        )

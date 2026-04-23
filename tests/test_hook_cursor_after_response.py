"""Tests for scripts/hooks/cursor_after_response.py — thin forwarder to TM Daemon.

The hook now simply reads stdin JSON and POSTs to the daemon HTTP API.
Heavy logic (DraftBuffer, ConvergenceDetector, etc.) lives in the daemon.
"""
from __future__ import annotations

import io
import json
from unittest.mock import patch

import httpx
import pytest

sys_path_needs_hooks = True


class TestCursorAfterResponseThin:
    """Tests for the thin cursor_after_response hook."""

    def test_forwards_input_to_daemon(self) -> None:
        from hooks.cursor_after_response import main

        mock_response = httpx.Response(
            200,
            text=json.dumps({"action": "draft_saved", "convergence": False, "draft_id": "d1"}),
            request=httpx.Request("POST", "http://127.0.0.1:3901/hooks/after_response"),
        )

        input_data = {"conversation_id": "sess-1", "prompt": "hello", "workspace_roots": ["/tmp"]}
        with patch("sys.stdin", io.StringIO(json.dumps(input_data))):
            with patch("hooks.cursor_after_response.httpx.post", return_value=mock_response) as mock_post:
                with patch("builtins.print") as mock_print:
                    main()

        mock_post.assert_called_once()
        call_kwargs = mock_post.call_args
        assert call_kwargs[1]["json"] == input_data
        assert "after_response" in call_kwargs[0][0] or "after_response" in call_kwargs[1].get("url", "")
        mock_print.assert_called_once_with(
            json.dumps({"action": "draft_saved", "convergence": False, "draft_id": "d1"})
        )

    def test_daemon_not_running_returns_ok(self) -> None:
        from hooks.cursor_after_response import main

        input_data = {"conversation_id": "sess-1", "prompt": "hello"}
        with patch("sys.stdin", io.StringIO(json.dumps(input_data))):
            with patch("hooks.cursor_after_response.httpx.post", side_effect=httpx.ConnectError("refused")):
                with patch("builtins.print") as mock_print:
                    main()

        mock_print.assert_called_once_with(
            json.dumps({"action": "ok", "message": "daemon not running"})
        )

    def test_invalid_stdin_returns_error(self) -> None:
        from hooks.cursor_after_response import main

        with patch("sys.stdin", io.StringIO("not json!!!")):
            with patch("builtins.print") as mock_print:
                main()

        mock_print.assert_called_once_with(
            json.dumps({"action": "error", "message": "invalid input"})
        )

    def test_timeout_returns_error(self) -> None:
        from hooks.cursor_after_response import main

        input_data = {"conversation_id": "sess-1", "prompt": "hello"}
        with patch("sys.stdin", io.StringIO(json.dumps(input_data))):
            with patch("hooks.cursor_after_response.httpx.post", side_effect=httpx.TimeoutException("timed out")):
                with patch("builtins.print") as mock_print:
                    main()

        mock_print.assert_called_once_with(
            json.dumps({"action": "error", "message": "timed out"})
        )

    def test_generic_exception_returns_error(self) -> None:
        from hooks.cursor_after_response import main

        input_data = {"conversation_id": "sess-1", "prompt": "hello"}
        with patch("sys.stdin", io.StringIO(json.dumps(input_data))):
            with patch("hooks.cursor_after_response.httpx.post", side_effect=RuntimeError("fail")):
                with patch("builtins.print") as mock_print:
                    main()

        mock_print.assert_called_once_with(
            json.dumps({"action": "error", "message": "fail"})
        )

    def test_daemon_url_is_correct(self) -> None:
        from hooks.cursor_after_response import DAEMON_URL

        assert DAEMON_URL == "http://127.0.0.1:3901"

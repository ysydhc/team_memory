"""Tests for thin hook scripts — forward to TM Daemon HTTP API.

Tests each thin hook's main() function with mocked httpx,
stdin parsing, daemon-not-running fallback, and timeout handling.
"""
from __future__ import annotations

import io
import json
from unittest.mock import MagicMock, patch

import httpx
import pytest


def _mock_post_side_effect(exc: Exception) -> MagicMock:
    """Create a mock httpx.post that raises the given exception."""
    mock = MagicMock(side_effect=exc)
    return mock


# -----------------------------------------------------------------------
# cursor_after_response
# -----------------------------------------------------------------------


class TestCursorAfterResponse:
    """Tests for scripts/hooks/cursor_after_response.py."""

    def test_forwards_to_daemon(self) -> None:
        from hooks.cursor_after_response import main

        mock_response = httpx.Response(
            200,
            text=json.dumps({"action": "draft_saved", "convergence": False}),
            request=httpx.Request("POST", "http://127.0.0.1:3901/hooks/after_response"),
        )

        input_data = {"conversation_id": "sess-1", "prompt": "hello"}
        with patch("sys.stdin", io.StringIO(json.dumps(input_data))):
            with patch("hooks.cursor_after_response.httpx.post", return_value=mock_response):
                with patch("builtins.print") as mock_print:
                    main()

        mock_print.assert_called_once_with(
            json.dumps({"action": "draft_saved", "convergence": False})
        )

    def test_daemon_not_running(self) -> None:
        from hooks.cursor_after_response import main

        input_data = {"conversation_id": "sess-1", "prompt": "hello"}
        with patch("sys.stdin", io.StringIO(json.dumps(input_data))):
            with patch("hooks.cursor_after_response.httpx.post", side_effect=httpx.ConnectError("Connection refused")):
                with patch("builtins.print") as mock_print:
                    main()

        mock_print.assert_called_once_with(
            json.dumps({"action": "ok", "message": "daemon not running"})
        )

    def test_invalid_stdin(self) -> None:
        from hooks.cursor_after_response import main

        with patch("sys.stdin", io.StringIO("not json!!!")):
            with patch("builtins.print") as mock_print:
                main()

        mock_print.assert_called_once_with(
            json.dumps({"action": "error", "message": "invalid input"})
        )

    def test_generic_exception(self) -> None:
        from hooks.cursor_after_response import main

        input_data = {"conversation_id": "sess-1", "prompt": "hello"}
        with patch("sys.stdin", io.StringIO(json.dumps(input_data))):
            with patch("hooks.cursor_after_response.httpx.post", side_effect=RuntimeError("something broke")):
                with patch("builtins.print") as mock_print:
                    main()

        mock_print.assert_called_once_with(
            json.dumps({"action": "error", "message": "something broke"})
        )

    def test_timeout_exception(self) -> None:
        from hooks.cursor_after_response import main

        input_data = {"conversation_id": "sess-1", "prompt": "hello"}
        with patch("sys.stdin", io.StringIO(json.dumps(input_data))):
            with patch("hooks.cursor_after_response.httpx.post", side_effect=httpx.TimeoutException("timed out")):
                with patch("builtins.print") as mock_print:
                    main()

        mock_print.assert_called_once_with(
            json.dumps({"action": "error", "message": "timed out"})
        )


# -----------------------------------------------------------------------
# cursor_session_start
# -----------------------------------------------------------------------


class TestCursorSessionStart:
    """Tests for scripts/hooks/cursor_session_start.py."""

    def test_forwards_to_daemon(self) -> None:
        from hooks.cursor_session_start import main

        mock_response = httpx.Response(
            200,
            text=json.dumps({"additionalContext": "some context"}),
            request=httpx.Request("POST", "http://127.0.0.1:3901/hooks/session_start"),
        )

        input_data = {"workspace_roots": ["/tmp"], "conversation_id": "sess-1"}
        with patch("sys.stdin", io.StringIO(json.dumps(input_data))):
            with patch("hooks.cursor_session_start.httpx.post", return_value=mock_response):
                with patch("builtins.print") as mock_print:
                    main()

        mock_print.assert_called_once_with(
            json.dumps({"additionalContext": "some context"})
        )

    def test_daemon_not_running(self) -> None:
        from hooks.cursor_session_start import main

        input_data = {"workspace_roots": ["/tmp"]}
        with patch("sys.stdin", io.StringIO(json.dumps(input_data))):
            with patch("hooks.cursor_session_start.httpx.post", side_effect=httpx.ConnectError("Connection refused")):
                with patch("builtins.print") as mock_print:
                    main()

        mock_print.assert_called_once_with(
            json.dumps({"action": "ok", "message": "daemon not running"})
        )

    def test_invalid_stdin(self) -> None:
        from hooks.cursor_session_start import main

        with patch("sys.stdin", io.StringIO("invalid!")):
            with patch("builtins.print") as mock_print:
                main()

        mock_print.assert_called_once_with(
            json.dumps({"action": "error", "message": "invalid input"})
        )


# -----------------------------------------------------------------------
# cursor_before_prompt
# -----------------------------------------------------------------------


class TestCursorBeforePrompt:
    """Tests for scripts/hooks/cursor_before_prompt.py."""

    def test_forwards_to_daemon(self) -> None:
        from hooks.cursor_before_prompt import main

        mock_response = httpx.Response(
            200,
            text=json.dumps({"additionalContext": "retrieved context"}),
            request=httpx.Request("POST", "http://127.0.0.1:3901/hooks/before_prompt"),
        )

        input_data = {"prompt": "之前的问题", "workspace_roots": ["/tmp"]}
        with patch("sys.stdin", io.StringIO(json.dumps(input_data))):
            with patch("hooks.cursor_before_prompt.httpx.post", return_value=mock_response):
                with patch("builtins.print") as mock_print:
                    main()

        mock_print.assert_called_once_with(
            json.dumps({"additionalContext": "retrieved context"})
        )

    def test_daemon_not_running(self) -> None:
        from hooks.cursor_before_prompt import main

        input_data = {"prompt": "test", "workspace_roots": []}
        with patch("sys.stdin", io.StringIO(json.dumps(input_data))):
            with patch("hooks.cursor_before_prompt.httpx.post", side_effect=httpx.ConnectError("Connection refused")):
                with patch("builtins.print") as mock_print:
                    main()

        mock_print.assert_called_once_with(
            json.dumps({"action": "ok", "message": "daemon not running"})
        )

    def test_invalid_stdin(self) -> None:
        from hooks.cursor_before_prompt import main

        with patch("sys.stdin", io.StringIO("not-json")):
            with patch("builtins.print") as mock_print:
                main()

        mock_print.assert_called_once_with(
            json.dumps({"action": "error", "message": "invalid input"})
        )

    def test_timeout_exception(self) -> None:
        from hooks.cursor_before_prompt import main

        input_data = {"prompt": "test"}
        with patch("sys.stdin", io.StringIO(json.dumps(input_data))):
            with patch("hooks.cursor_before_prompt.httpx.post", side_effect=httpx.TimeoutException("timed out")):
                with patch("builtins.print") as mock_print:
                    main()

        mock_print.assert_called_once_with(
            json.dumps({"action": "error", "message": "timed out"})
        )


# -----------------------------------------------------------------------
# claude_session_start
# -----------------------------------------------------------------------


class TestClaudeSessionStart:
    """Tests for scripts/hooks/claude_session_start.py."""

    def test_forwards_to_daemon(self) -> None:
        from hooks.claude_session_start import main

        mock_response = httpx.Response(
            200,
            text=json.dumps({"additionalContext": "claude context"}),
            request=httpx.Request("POST", "http://127.0.0.1:3901/hooks/session_start"),
        )

        input_data = {"workspace_roots": ["/tmp"], "conversation_id": "sess-1"}
        with patch("sys.stdin", io.StringIO(json.dumps(input_data))):
            with patch("hooks.claude_session_start.httpx.post", return_value=mock_response):
                with patch("builtins.print") as mock_print:
                    main()

        mock_print.assert_called_once_with(
            json.dumps({"additionalContext": "claude context"})
        )

    def test_daemon_not_running(self) -> None:
        from hooks.claude_session_start import main

        input_data = {"workspace_roots": ["/tmp"]}
        with patch("sys.stdin", io.StringIO(json.dumps(input_data))):
            with patch("hooks.claude_session_start.httpx.post", side_effect=httpx.ConnectError("Connection refused")):
                with patch("builtins.print") as mock_print:
                    main()

        mock_print.assert_called_once_with(
            json.dumps({"action": "ok", "message": "daemon not running"})
        )

    def test_invalid_stdin(self) -> None:
        from hooks.claude_session_start import main

        with patch("sys.stdin", io.StringIO("not-json")):
            with patch("builtins.print") as mock_print:
                main()

        mock_print.assert_called_once_with(
            json.dumps({"action": "error", "message": "invalid input"})
        )


# -----------------------------------------------------------------------
# HermesPipeline (thin)
# -----------------------------------------------------------------------


class TestHermesPipelineThin:
    """Tests for thin HermesPipeline that forwards to daemon HTTP API."""

    @pytest.mark.asyncio
    async def test_on_turn_start_success(self) -> None:
        from hooks.hermes_pipeline import HermesPipeline

        pipeline = HermesPipeline()

        mock_response = httpx.Response(
            200,
            json={"action": "project_context", "context": "some context"},
            request=httpx.Request("POST", "http://127.0.0.1:3901/hooks/before_prompt"),
        )

        with patch.object(pipeline._client, "post", return_value=mock_response):
            result = await pipeline.on_turn_start("之前的问题", project="team_doc")

        assert result == {"action": "project_context", "context": "some context"}
        await pipeline.close()

    @pytest.mark.asyncio
    async def test_on_turn_start_daemon_not_running(self) -> None:
        from hooks.hermes_pipeline import HermesPipeline

        pipeline = HermesPipeline()

        with patch.object(
            pipeline._client, "post", side_effect=httpx.ConnectError("refused")
        ):
            result = await pipeline.on_turn_start("hello", project="team_doc")

        assert result == {"action": "ok", "message": "daemon not running"}
        await pipeline.close()

    @pytest.mark.asyncio
    async def test_on_turn_end_success(self) -> None:
        from hooks.hermes_pipeline import HermesPipeline

        pipeline = HermesPipeline()

        mock_response = httpx.Response(
            200,
            json={"action": "draft_saved", "draft_id": "d1"},
            request=httpx.Request("POST", "http://127.0.0.1:3901/hooks/after_response"),
        )

        with patch.object(pipeline._client, "post", return_value=mock_response):
            result = await pipeline.on_turn_end("sess-1", "response text", project="team_doc")

        assert result == {"action": "draft_saved", "draft_id": "d1"}
        await pipeline.close()

    @pytest.mark.asyncio
    async def test_on_turn_end_daemon_not_running(self) -> None:
        from hooks.hermes_pipeline import HermesPipeline

        pipeline = HermesPipeline()

        with patch.object(
            pipeline._client, "post", side_effect=httpx.ConnectError("refused")
        ):
            result = await pipeline.on_turn_end("sess-1", "response text")

        assert result == {"action": "ok", "message": "daemon not running"}
        await pipeline.close()

    @pytest.mark.asyncio
    async def test_on_session_end_success(self) -> None:
        from hooks.hermes_pipeline import HermesPipeline

        pipeline = HermesPipeline()

        mock_response = httpx.Response(
            200,
            json={"action": "published", "draft_id": "d1"},
            request=httpx.Request("POST", "http://127.0.0.1:3901/hooks/session_end"),
        )

        with patch.object(pipeline._client, "post", return_value=mock_response):
            result = await pipeline.on_session_end("sess-1")

        assert result == {"action": "published", "draft_id": "d1"}
        await pipeline.close()

    @pytest.mark.asyncio
    async def test_on_session_end_daemon_not_running(self) -> None:
        from hooks.hermes_pipeline import HermesPipeline

        pipeline = HermesPipeline()

        with patch.object(
            pipeline._client, "post", side_effect=httpx.ConnectError("refused")
        ):
            result = await pipeline.on_session_end("sess-1")

        assert result is None
        await pipeline.close()

    @pytest.mark.asyncio
    async def test_on_session_end_generic_error(self) -> None:
        from hooks.hermes_pipeline import HermesPipeline

        pipeline = HermesPipeline()

        with patch.object(
            pipeline._client, "post", side_effect=RuntimeError("something broke")
        ):
            result = await pipeline.on_session_end("sess-1")

        assert result == {"action": "error", "message": "something broke"}
        await pipeline.close()

    @pytest.mark.asyncio
    async def test_context_manager(self) -> None:
        from hooks.hermes_pipeline import HermesPipeline

        async with HermesPipeline() as pipeline:
            mock_response = httpx.Response(
                200,
                json={"action": "ok"},
                request=httpx.Request("POST", "http://127.0.0.1:3901/hooks/before_prompt"),
            )
            with patch.object(pipeline._client, "post", return_value=mock_response):
                result = await pipeline.on_turn_start("test")
                assert result == {"action": "ok"}

    @pytest.mark.asyncio
    async def test_on_turn_start_generic_error(self) -> None:
        from hooks.hermes_pipeline import HermesPipeline

        pipeline = HermesPipeline()

        with patch.object(
            pipeline._client, "post", side_effect=RuntimeError("fail")
        ):
            result = await pipeline.on_turn_start("hello")

        assert result == {"action": "error", "message": "fail"}
        await pipeline.close()

    @pytest.mark.asyncio
    async def test_on_turn_end_generic_error(self) -> None:
        from hooks.hermes_pipeline import HermesPipeline

        pipeline = HermesPipeline()

        with patch.object(
            pipeline._client, "post", side_effect=RuntimeError("fail")
        ):
            result = await pipeline.on_turn_end("sess-1", "hello")

        assert result == {"action": "error", "message": "fail"}
        await pipeline.close()

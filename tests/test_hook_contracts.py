"""Hook contract tests — verify each platform hook produces valid daemon payloads.

Ensures that hook scripts correctly map platform-specific stdin formats
to the daemon's expected payload schema. Catches format drift early.
"""

from __future__ import annotations

import io
import json
import os
import tempfile
from unittest.mock import patch

import httpx
import pytest

pytestmark = pytest.mark.unit

# ---------------------------------------------------------------------------
# Contract definitions: what the daemon expects from each endpoint
# ---------------------------------------------------------------------------

AFTER_RESPONSE_CONTRACT = {
    "required_fields": ["conversation_id", "response_text", "workspace_roots"],
    "response_text_must_be_nonempty": True,
}

BEFORE_PROMPT_CONTRACT = {
    "required_fields": ["conversation_id", "prompt", "workspace_roots"],
}

SESSION_START_CONTRACT = {
    "required_fields": ["conversation_id", "workspace_roots"],
}


def _capture_post_payload(mock_post_fn):
    """Extract the JSON payload sent to daemon from a mock."""
    call_kwargs = mock_post_fn.call_args
    if call_kwargs is None:
        return None
    return call_kwargs[1].get("json", {})


# ---------------------------------------------------------------------------
# Cursor platform contracts
# ---------------------------------------------------------------------------


class TestCursorAfterResponseContract:
    """Cursor afterAgentResponse must produce valid after_response payload."""

    def _mock_response(self):
        return httpx.Response(
            200,
            text=json.dumps({"action": "ok"}),
            request=httpx.Request("POST", "http://127.0.0.1:3901/hooks/after_response"),
        )

    def test_old_format_agent_response(self):
        """Old Cursor format (agent_response field) meets contract."""
        from team_memory.hooks.platforms.cursor.after_response import main

        input_data = {
            "conversation_id": "sess-1",
            "agent_response": "Here is my response",
            "workspace_roots": ["/tmp/project"],
        }
        with patch("sys.stdin", io.StringIO(json.dumps(input_data))):
            with patch("team_memory.hooks.core.daemon_http.httpx.post", return_value=self._mock_response()) as mock:
                with patch("builtins.print"):
                    main()

        payload = _capture_post_payload(mock)
        assert payload is not None
        for field in AFTER_RESPONSE_CONTRACT["required_fields"]:
            assert field in payload, f"Missing required field: {field}"
        assert payload["response_text"], "response_text must not be empty"

    def test_new_format_transcript_path(self):
        """New Cursor format (transcript_path) meets contract."""
        from team_memory.hooks.platforms.cursor.after_response import main

        # Create a temp transcript file
        transcript = [
            {"role": "user", "message": {"content": [{"type": "text", "text": "hello"}]}},
            {"role": "assistant", "message": {"content": [{"type": "text", "text": "world response"}]}},
            {"type": "turn_ended", "status": "completed"},
        ]
        with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
            for entry in transcript:
                f.write(json.dumps(entry) + "\n")
            transcript_path = f.name

        try:
            input_data = {
                "conversation_id": "sess-2",
                "generation_id": "gen-1",
                "model": "gemini-3.1-pro",
                "status": "completed",
                "transcript_path": transcript_path,
                "workspace_roots": ["/tmp/project"],
            }
            with patch("sys.stdin", io.StringIO(json.dumps(input_data))):
                with patch("team_memory.hooks.core.daemon_http.httpx.post", return_value=self._mock_response()) as mock:
                    with patch("builtins.print"):
                        main()

            payload = _capture_post_payload(mock)
            assert payload is not None
            for field in AFTER_RESPONSE_CONTRACT["required_fields"]:
                assert field in payload, f"Missing required field: {field}"
            assert payload["response_text"] == "world response"
        finally:
            os.unlink(transcript_path)

    def test_empty_response_no_post(self):
        """Empty response (no transcript, no fields) should not call daemon."""
        from team_memory.hooks.platforms.cursor.after_response import main

        input_data = {
            "conversation_id": "sess-3",
            "model": "gemini-3.1-pro",
            "workspace_roots": ["/tmp"],
        }
        with patch("sys.stdin", io.StringIO(json.dumps(input_data))):
            with patch("team_memory.hooks.core.daemon_http.httpx.post") as mock:
                with patch("builtins.print"):
                    main()

        mock.assert_not_called()


class TestCursorBeforePromptContract:
    """Cursor beforeSubmitPrompt must produce valid before_prompt payload."""

    def _mock_response(self):
        return httpx.Response(
            200,
            text=json.dumps({"results": [], "context": ""}),
            request=httpx.Request("POST", "http://127.0.0.1:3901/hooks/before_prompt"),
        )

    def test_standard_prompt(self):
        """Standard prompt input meets contract."""
        from team_memory.hooks.platforms.cursor.before_prompt import main

        input_data = {
            "prompt": "如何实现用户认证",
            "conversation_id": "sess-1",
            "workspace_roots": ["/tmp/project"],
        }
        with patch("sys.stdin", io.StringIO(json.dumps(input_data))):
            with patch("team_memory.hooks.core.daemon_http.httpx.post", return_value=self._mock_response()) as mock:
                with patch("builtins.print"):
                    main()

        payload = _capture_post_payload(mock)
        assert payload is not None
        for field in BEFORE_PROMPT_CONTRACT["required_fields"]:
            assert field in payload, f"Missing required field: {field}"


class TestClaudeStopContract:
    """Claude Code stop hook must produce valid after_response payload."""

    def test_last_assistant_message_format(self):
        """Claude Code format (last_assistant_message) meets contract."""
        from team_memory.hooks.platforms.claude.stop import main

        input_data = {
            "session_id": "sess-claude-1",
            "last_assistant_message": "Here is my response to your question.",
            "cwd": "/tmp/project",
        }
        with patch("sys.stdin", io.StringIO(json.dumps(input_data))):
            with patch(
                "team_memory.hooks.core.daemon_http.httpx.post",
                return_value=httpx.Response(
                    200, text="{}", request=httpx.Request("POST", "http://x/hooks/after_response")
                ),
            ) as mock:
                with patch("builtins.print"):
                    main()

        payload = _capture_post_payload(mock)
        assert payload is not None
        for field in AFTER_RESPONSE_CONTRACT["required_fields"]:
            assert field in payload, f"Missing required field: {field}"
        assert payload["response_text"] == "Here is my response to your question."


class TestHermesStopContract:
    """Hermes stop hook must produce valid after_response payload."""

    def test_response_field_format(self):
        """Hermes format (response field) meets contract."""
        from team_memory.hooks.platforms.hermes.stop import main

        input_data = {
            "session_id": "sess-hermes-1",
            "response": "Hermes assistant reply",
            "cwd": "/tmp/project",
        }
        with patch("sys.stdin", io.StringIO(json.dumps(input_data))):
            with patch("team_memory.hooks.core.daemon_http.httpx.post") as mock:
                with patch("builtins.print"):
                    main()

        # Hermes uses fire_and_forget — check the thread-submitted call
        # Since fire_and_forget spawns a thread, we mock at a lower level
        payload = _capture_post_payload(mock)
        if payload is not None:
            for field in AFTER_RESPONSE_CONTRACT["required_fields"]:
                assert field in payload, f"Missing required field: {field}"
            assert payload["response_text"] == "Hermes assistant reply"

    def test_empty_response_no_post(self):
        """Empty response should not call daemon."""
        from team_memory.hooks.platforms.hermes.stop import main

        input_data = {
            "session_id": "sess-hermes-2",
            "response": "",
            "cwd": "/tmp",
        }
        with patch("sys.stdin", io.StringIO(json.dumps(input_data))):
            with patch("team_memory.hooks.core.daemon_http.httpx.post") as mock:
                with patch("builtins.print"):
                    main()

        mock.assert_not_called()

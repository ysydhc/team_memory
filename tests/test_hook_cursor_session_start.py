"""Tests for scripts/hooks/cursor_session_start.py — sessionStart hook."""
from __future__ import annotations

import json
import sys
from pathlib import Path
from unittest.mock import patch

# Ensure scripts/hooks/ is importable
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts" / "hooks"))

import cursor_session_start  # noqa: I001,E402


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _hook_input(
    workspace_roots: list | None = None,
    conversation_id: str = "conv-session-1",
) -> dict:
    return {
        "workspace_roots": workspace_roots or ["/Users/yeshouyou/Work/agent/team_doc"],
        "conversation_id": conversation_id,
    }


# ---------------------------------------------------------------------------
# Test: successful session start
# ---------------------------------------------------------------------------

class TestSessionStartSuccess:
    """sessionStart hook outputs additionalContext on success."""

    def test_outputs_additional_context_on_success(self):
        mock_mcp_result = {
            "results": [
                {"title": "Docker部署经验", "content": "使用docker-compose时需要注意..."}
            ]
        }
        with patch("cursor_session_start.parse_hook_input", return_value=_hook_input()):
            with patch("cursor_session_start.get_project_from_path", return_value="team_doc"):
                with patch(
                    "cursor_session_start.call_mcp_tool",
                    return_value=mock_mcp_result,
                ):
                    output = cursor_session_start.main()

        result = json.loads(output)
        assert "additionalContext" in result
        assert result["additionalContext"]  # non-empty

    def test_calls_memory_context_with_project(self):
        mock_mcp_result = {"results": []}
        with patch("cursor_session_start.parse_hook_input", return_value=_hook_input()):
            with patch("cursor_session_start.get_project_from_path", return_value="team_doc"):
                with patch(
                    "cursor_session_start.call_mcp_tool",
                    return_value=mock_mcp_result,
                ) as mock_call:
                    cursor_session_start.main()

        mock_call.assert_called_once_with("memory_context", {"project": "team_doc"})

    def test_no_project_returns_empty_additional_context(self):
        with patch("cursor_session_start.parse_hook_input", return_value=_hook_input()):
            with patch("cursor_session_start.get_project_from_path", return_value=None):
                output = cursor_session_start.main()

        result = json.loads(output)
        # No project resolved — should still output valid JSON
        assert "additionalContext" in result
        assert result["additionalContext"] == ""


# ---------------------------------------------------------------------------
# Test: error handling
# ---------------------------------------------------------------------------

class TestSessionStartErrorHandling:
    """sessionStart hook should never crash."""

    def test_mcp_failure_returns_error_output(self):
        with patch("cursor_session_start.parse_hook_input", return_value=_hook_input()):
            with patch("cursor_session_start.get_project_from_path", return_value="team_doc"):
                with patch(
                    "cursor_session_start.call_mcp_tool",
                    side_effect=Exception("MCP down"),
                ):
                    output = cursor_session_start.main()

        result = json.loads(output)
        assert result.get("status") == "error"
        assert "message" in result

    def test_empty_input_returns_error(self):
        with patch("cursor_session_start.parse_hook_input", return_value={}):
            output = cursor_session_start.main()

        result = json.loads(output)
        # Should not crash — either error or empty additionalContext
        assert isinstance(result, dict)

    def test_no_workspace_roots_returns_empty_context(self):
        inp = {"conversation_id": "conv-1"}
        with patch("cursor_session_start.parse_hook_input", return_value=inp):
            with patch("cursor_session_start.get_project_from_path", return_value=None):
                output = cursor_session_start.main()

        result = json.loads(output)
        assert "additionalContext" in result
        assert result["additionalContext"] == ""

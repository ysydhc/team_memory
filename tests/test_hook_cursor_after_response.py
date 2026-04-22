"""Tests for scripts/hooks/cursor_after_response.py — afterAgentResponse hook."""
from __future__ import annotations

import asyncio
import io
import json
import os
import sys
import tempfile
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

# Ensure scripts/hooks/ is importable
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts" / "hooks"))

import cursor_after_response
import draft_buffer
from convergence_detector import ConvergenceDetector


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_db_path() -> str:
    fd, path = tempfile.mkstemp(suffix=".db")
    os.close(fd)
    return path


def _hook_input(
    conversation_id: str = "conv-123",
    prompt: str = "hello world",
    workspace_roots: list | None = None,
    model: str = "claude-3.5",
) -> str:
    """Build a JSON stdin string for the hook."""
    return json.dumps({
        "conversation_id": conversation_id,
        "prompt": prompt,
        "workspace_roots": workspace_roots or ["/Users/yeshouyou/Work/agent/team_doc"],
        "model": model,
    })


# ---------------------------------------------------------------------------
# Test: main — basic flow
# ---------------------------------------------------------------------------

class TestMainBasicFlow:
    """cursor_after_response.main handles the basic afterAgentResponse flow."""

    def test_outputs_valid_json(self):
        """Main should output valid JSON with status, convergence, draft_id."""
        with patch("sys.stdin", new_callable=lambda: io.StringIO(_hook_input(prompt="工作还在进行中"))):
            with patch("cursor_after_response.DraftBuffer") as MockBuf:
                mock_instance = AsyncMock()
                mock_instance.create_draft = AsyncMock(return_value="draft-uuid-1")
                mock_instance.find_pending_by_conversation = AsyncMock(return_value=[])
                mock_instance.__aenter__ = AsyncMock(return_value=mock_instance)
                mock_instance.__aexit__ = AsyncMock(return_value=False)
                MockBuf.return_value = mock_instance

                with patch("cursor_after_response.get_project_from_path", return_value="team_doc"):
                    with patch("cursor_after_response.parse_hook_input") as mock_parse:
                        mock_parse.return_value = {
                            "conversation_id": "conv-123",
                            "prompt": "工作还在进行中",
                            "workspace_roots": ["/Users/yeshouyou/Work/agent/team_doc"],
                            "model": "claude-3.5",
                        }
                        output = cursor_after_response.main()

        result = json.loads(output)
        assert result["status"] == "ok"
        assert "convergence" in result
        assert "draft_id" in result

    def test_no_convergence_when_no_signal(self):
        """Should report convergence=False when no signal present."""
        with patch("cursor_after_response.parse_hook_input") as mock_parse:
            mock_parse.return_value = {
                "conversation_id": "conv-123",
                "prompt": "还在修改代码中",
                "workspace_roots": ["/Users/yeshouyou/Work/agent/team_doc"],
                "model": "claude-3.5",
            }
            with patch("cursor_after_response.DraftBuffer") as MockBuf:
                mock_instance = AsyncMock()
                mock_instance.create_draft = AsyncMock(return_value="draft-uuid-1")
                mock_instance.find_pending_by_conversation = AsyncMock(return_value=[])
                mock_instance.__aenter__ = AsyncMock(return_value=mock_instance)
                mock_instance.__aexit__ = AsyncMock(return_value=False)
                MockBuf.return_value = mock_instance

                with patch("cursor_after_response.get_project_from_path", return_value="team_doc"):
                    output = cursor_after_response.main()

        result = json.loads(output)
        assert result["status"] == "ok"
        assert result["convergence"] is False

    def test_convergence_detected(self):
        """Should report convergence=True when signal found."""
        with patch("cursor_after_response.parse_hook_input") as mock_parse:
            mock_parse.return_value = {
                "conversation_id": "conv-123",
                "prompt": "问题已经解决了，不用再改了",
                "workspace_roots": ["/Users/yeshouyou/Work/agent/team_doc"],
                "model": "claude-3.5",
            }
            with patch("cursor_after_response.DraftBuffer") as MockBuf:
                mock_instance = AsyncMock()
                mock_instance.create_draft = AsyncMock(return_value="draft-uuid-1")
                mock_instance.find_pending_by_conversation = AsyncMock(return_value=[])
                mock_instance.__aenter__ = AsyncMock(return_value=mock_instance)
                mock_instance.__aexit__ = AsyncMock(return_value=False)
                MockBuf.return_value = mock_instance

                with patch("cursor_after_response.get_project_from_path", return_value="team_doc"):
                    output = cursor_after_response.main()

        result = json.loads(output)
        assert result["status"] == "ok"
        assert result["convergence"] is True
        assert result["draft_id"] == "draft-uuid-1"


# ---------------------------------------------------------------------------
# Test: draft creation vs update
# ---------------------------------------------------------------------------

class TestDraftManagement:
    """Verify draft creation and update logic."""

    def test_creates_new_draft_when_no_pending(self):
        """If no pending draft for conversation, create a new one."""
        with patch("cursor_after_response.parse_hook_input") as mock_parse:
            mock_parse.return_value = {
                "conversation_id": "conv-new",
                "prompt": "some response",
                "workspace_roots": ["/Users/yeshouyou/Work/agent/team_doc"],
            }
            with patch("cursor_after_response.DraftBuffer") as MockBuf:
                mock_instance = AsyncMock()
                mock_instance.create_draft = AsyncMock(return_value="new-draft-id")
                mock_instance.find_pending_by_conversation = AsyncMock(return_value=[])
                mock_instance.__aenter__ = AsyncMock(return_value=mock_instance)
                mock_instance.__aexit__ = AsyncMock(return_value=False)
                MockBuf.return_value = mock_instance

                with patch("cursor_after_response.get_project_from_path", return_value="team_doc"):
                    output = cursor_after_response.main()

        result = json.loads(output)
        assert result["draft_id"] == "new-draft-id"
        mock_instance.create_draft.assert_called_once()

    def test_updates_existing_draft_when_pending(self):
        """If a pending draft exists for this conversation, update it."""
        existing_draft = {
            "id": "existing-draft-id",
            "project": "team_doc",
            "conversation_id": "conv-exist",
            "content": "old content",
        }
        with patch("cursor_after_response.parse_hook_input") as mock_parse:
            mock_parse.return_value = {
                "conversation_id": "conv-exist",
                "prompt": "more info added",
                "workspace_roots": ["/Users/yeshouyou/Work/agent/team_doc"],
            }
            with patch("cursor_after_response.DraftBuffer") as MockBuf:
                mock_instance = AsyncMock()
                mock_instance.update_draft = AsyncMock(return_value=None)
                mock_instance.find_pending_by_conversation = AsyncMock(
                    return_value=[existing_draft]
                )
                mock_instance.__aenter__ = AsyncMock(return_value=mock_instance)
                mock_instance.__aexit__ = AsyncMock(return_value=False)
                MockBuf.return_value = mock_instance

                with patch("cursor_after_response.get_project_from_path", return_value="team_doc"):
                    output = cursor_after_response.main()

        result = json.loads(output)
        assert result["draft_id"] == "existing-draft-id"
        mock_instance.update_draft.assert_called_once()
        mock_instance.create_draft.assert_not_called()


# ---------------------------------------------------------------------------
# Test: convergence marking
# ---------------------------------------------------------------------------

class TestConvergenceMarking:
    """When convergence detected, draft should be marked for publishing."""

    def test_convergence_marks_draft_for_publishing(self):
        """mark_for_publishing should be called when convergence detected."""
        with patch("cursor_after_response.parse_hook_input") as mock_parse:
            mock_parse.return_value = {
                "conversation_id": "conv-mark",
                "prompt": "搞定了",
                "workspace_roots": ["/Users/yeshouyou/Work/agent/team_doc"],
            }
            with patch("cursor_after_response.DraftBuffer") as MockBuf:
                mock_instance = AsyncMock()
                mock_instance.create_draft = AsyncMock(return_value="draft-mark-id")
                mock_instance.find_pending_by_conversation = AsyncMock(return_value=[])
                mock_instance.mark_for_publishing = AsyncMock(return_value=None)
                mock_instance.__aenter__ = AsyncMock(return_value=mock_instance)
                mock_instance.__aexit__ = AsyncMock(return_value=False)
                MockBuf.return_value = mock_instance

                with patch("cursor_after_response.get_project_from_path", return_value="team_doc"):
                    output = cursor_after_response.main()

        result = json.loads(output)
        assert result["convergence"] is True
        mock_instance.mark_for_publishing.assert_called_once_with("draft-mark-id")

    def test_no_marking_without_convergence(self):
        """mark_for_publishing should NOT be called without convergence."""
        with patch("cursor_after_response.parse_hook_input") as mock_parse:
            mock_parse.return_value = {
                "conversation_id": "conv-nomark",
                "prompt": "still working on it",
                "workspace_roots": ["/Users/yeshouyou/Work/agent/team_doc"],
            }
            with patch("cursor_after_response.DraftBuffer") as MockBuf:
                mock_instance = AsyncMock()
                mock_instance.create_draft = AsyncMock(return_value="draft-nomark-id")
                mock_instance.find_pending_by_conversation = AsyncMock(return_value=[])
                mock_instance.mark_for_publishing = AsyncMock(return_value=None)
                mock_instance.__aenter__ = AsyncMock(return_value=mock_instance)
                mock_instance.__aexit__ = AsyncMock(return_value=False)
                MockBuf.return_value = mock_instance

                with patch("cursor_after_response.get_project_from_path", return_value="team_doc"):
                    output = cursor_after_response.main()

        result = json.loads(output)
        assert result["convergence"] is False
        mock_instance.mark_for_publishing.assert_not_called()


# ---------------------------------------------------------------------------
# Test: error handling
# ---------------------------------------------------------------------------

class TestErrorHandling:
    """Hook should never crash — all errors should be caught."""

    def test_empty_input_returns_ok(self):
        """Empty stdin should not crash the hook."""
        with patch("cursor_after_response.parse_hook_input", return_value={}):
            with patch("cursor_after_response.DraftBuffer") as MockBuf:
                mock_instance = AsyncMock()
                mock_instance.create_draft = AsyncMock(return_value="draft-id")
                mock_instance.find_pending_by_conversation = AsyncMock(return_value=[])
                mock_instance.__aenter__ = AsyncMock(return_value=mock_instance)
                mock_instance.__aexit__ = AsyncMock(return_value=False)
                MockBuf.return_value = mock_instance

                with patch("cursor_after_response.get_project_from_path", return_value=None):
                    output = cursor_after_response.main()

        result = json.loads(output)
        assert result["status"] == "ok"

    def test_draft_buffer_exception_returns_error_status(self):
        """If DraftBuffer raises, the hook should still output JSON, not crash."""
        with patch("cursor_after_response.parse_hook_input") as mock_parse:
            mock_parse.return_value = {
                "conversation_id": "conv-err",
                "prompt": "hello",
                "workspace_roots": ["/path"],
            }
            with patch("cursor_after_response.DraftBuffer") as MockBuf:
                mock_instance = AsyncMock()
                mock_instance.find_pending_by_conversation = AsyncMock(
                    side_effect=Exception("DB error")
                )
                mock_instance.__aenter__ = AsyncMock(return_value=mock_instance)
                mock_instance.__aexit__ = AsyncMock(return_value=False)
                MockBuf.return_value = mock_instance

                with patch("cursor_after_response.get_project_from_path", return_value="team_doc"):
                    output = cursor_after_response.main()

        result = json.loads(output)
        assert result["status"] == "error"

    def test_no_project_returns_ok(self):
        """If project cannot be resolved, still returns ok with no-op."""
        with patch("cursor_after_response.parse_hook_input") as mock_parse:
            mock_parse.return_value = {
                "conversation_id": "conv-noproj",
                "prompt": "hello",
                "workspace_roots": ["/unknown/path"],
            }
            with patch("cursor_after_response.get_project_from_path", return_value=None):
                output = cursor_after_response.main()

        result = json.loads(output)
        assert result["status"] == "ok"
        assert result["draft_id"] == ""

"""Tests for scripts/hooks/cursor_before_prompt.py — beforeSubmitPrompt hook."""
from __future__ import annotations

import json
import sys
from pathlib import Path
from unittest.mock import patch

# Ensure scripts/hooks/ is importable
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts" / "hooks"))

import cursor_before_prompt  # noqa: I001,E402


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _hook_input(
    prompt: str = "帮我写个函数",
    workspace_roots: list | None = None,
    conversation_id: str = "conv-prompt-1",
) -> dict:
    return {
        "prompt": prompt,
        "workspace_roots": workspace_roots or ["/Users/yeshouyou/Work/agent/team_doc"],
        "conversation_id": conversation_id,
    }


# ---------------------------------------------------------------------------
# Test: keyword-triggered retrieval
# ---------------------------------------------------------------------------

class TestBeforePromptRetrieval:
    """beforeSubmitPrompt triggers retrieval for keyword prompts."""

    def test_retrieval_triggered_for_keyword_prompt(self):
        mock_mcp_result = {
            "results": [
                {"title": "Docker经验", "content": "之前遇到Docker问题时..."}
            ]
        }
        inp = _hook_input(prompt="之前遇到的Docker问题")
        with patch("cursor_before_prompt.parse_hook_input", return_value=inp):
            with patch("cursor_before_prompt.get_project_from_path", return_value="team_doc"):
                with patch(
                    "cursor_before_prompt.call_mcp_tool",
                    return_value=mock_mcp_result,
                ) as mock_call:
                    output = cursor_before_prompt.main()

        result = json.loads(output)
        assert "additionalContext" in result
        mock_call.assert_called_once()
        call_args = mock_call.call_args
        assert call_args[0][0] == "memory_recall"
        assert call_args[0][1]["query"] == "之前遇到的Docker问题"
        assert call_args[0][1]["max_results"] == 3
        assert call_args[0][1]["project"] == "team_doc"

    def test_retrieval_triggered_for_shangci(self):
        inp = _hook_input(prompt="上次说的方案怎么样")
        with patch("cursor_before_prompt.parse_hook_input", return_value=inp):
            with patch("cursor_before_prompt.get_project_from_path", return_value="team_doc"):
                with patch(
                    "cursor_before_prompt.call_mcp_tool",
                    return_value={"results": []},
                ) as mock_call:
                    output = cursor_before_prompt.main()

        result = json.loads(output)
        assert "additionalContext" in result
        mock_call.assert_called_once()

    def test_retrieval_skipped_for_normal_prompt(self):
        inp = _hook_input(prompt="帮我写个函数")
        with patch("cursor_before_prompt.parse_hook_input", return_value=inp):
            with patch("cursor_before_prompt.get_project_from_path", return_value="team_doc"):
                with patch("cursor_before_prompt.call_mcp_tool", return_value={}) as mock_call:
                    output = cursor_before_prompt.main()

        result = json.loads(output)
        assert result == {}
        mock_call.assert_not_called()

    def test_retrieval_skipped_for_english_prompt(self):
        inp = _hook_input(prompt="help me debug this")
        with patch("cursor_before_prompt.parse_hook_input", return_value=inp):
            with patch("cursor_before_prompt.get_project_from_path", return_value="team_doc"):
                with patch("cursor_before_prompt.call_mcp_tool", return_value={}) as mock_call:
                    output = cursor_before_prompt.main()

        result = json.loads(output)
        assert result == {}
        mock_call.assert_not_called()


# ---------------------------------------------------------------------------
# Test: no project resolved
# ---------------------------------------------------------------------------

class TestBeforePromptNoProject:
    """When project can't be resolved, skip retrieval."""

    def test_no_project_keyword_prompt_returns_empty(self):
        inp = _hook_input(prompt="之前遇到的问题")
        with patch("cursor_before_prompt.parse_hook_input", return_value=inp):
            with patch("cursor_before_prompt.get_project_from_path", return_value=None):
                with patch("cursor_before_prompt.call_mcp_tool", return_value={}) as mock_call:
                    output = cursor_before_prompt.main()

        result = json.loads(output)
        assert result == {}
        mock_call.assert_not_called()


# ---------------------------------------------------------------------------
# Test: error handling
# ---------------------------------------------------------------------------

class TestBeforePromptErrorHandling:
    """beforeSubmitPrompt should never crash — errors produce empty output."""

    def test_mcp_failure_returns_empty(self):
        inp = _hook_input(prompt="之前遇到的问题")
        with patch("cursor_before_prompt.parse_hook_input", return_value=inp):
            with patch("cursor_before_prompt.get_project_from_path", return_value="team_doc"):
                with patch(
                    "cursor_before_prompt.call_mcp_tool",
                    side_effect=Exception("MCP down"),
                ):
                    output = cursor_before_prompt.main()

        result = json.loads(output)
        assert result == {}

    def test_empty_input_returns_empty(self):
        with patch("cursor_before_prompt.parse_hook_input", return_value={}):
            output = cursor_before_prompt.main()

        result = json.loads(output)
        assert result == {}

    def test_malformed_input_returns_empty(self):
        with patch(
            "cursor_before_prompt.parse_hook_input",
            return_value={"conversation_id": "conv-1"},
        ):
            output = cursor_before_prompt.main()

        result = json.loads(output)
        assert result == {}

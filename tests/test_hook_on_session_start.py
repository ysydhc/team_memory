"""Tests for scripts/hooks/on_session_start.py — sessionStart hook (async TMClient version)."""
from __future__ import annotations

import json
import sys
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

# Ensure scripts/hooks/ is importable
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts" / "hooks"))

from on_session_start import process_session_start  # noqa: I001,E402
from shared import HookInput, PipelineConfig  # noqa: I001,E402


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _hook_input(
    workspace_roots: list | None = None,
    conversation_id: str = "conv-session-1",
) -> HookInput:
    return HookInput(
        workspace_roots=workspace_roots or ["/Users/yeshouyou/Work/agent/team_doc"],
        conversation_id=conversation_id,
    )


def _config() -> PipelineConfig:
    return PipelineConfig(tm_url="http://localhost:3900")


# ---------------------------------------------------------------------------
# Test: with project → calls get_context → outputs additional_context
# ---------------------------------------------------------------------------

class TestSessionStartWithContext:
    """sessionStart hook fetches and outputs context when project is resolved."""

    @pytest.mark.asyncio
    async def test_outputs_additional_context_on_success(self, capsys):
        mock_result = {
            "results": [
                {"title": "Docker部署经验", "content": "使用docker-compose时需要注意..."}
            ]
        }
        with patch("on_session_start.TMClient") as MockTM:
            instance = MockTM.return_value
            instance.get_context = AsyncMock(return_value=mock_result)
            await process_session_start(_hook_input(), _config())

        output = json.loads(capsys.readouterr().out)
        assert output["action"] == "context_injected"
        assert output["project"] == "team_doc"
        assert "additional_context" in output
        assert output["additional_context"]  # non-empty

    @pytest.mark.asyncio
    async def test_calls_get_context_with_project(self, capsys):
        mock_result = {"results": []}
        with patch("on_session_start.TMClient") as MockTM:
            instance = MockTM.return_value
            instance.get_context = AsyncMock(return_value=mock_result)
            await process_session_start(_hook_input(), _config())

        instance.get_context.assert_awaited_once_with(project="team_doc")

    @pytest.mark.asyncio
    async def test_context_length_capped(self, capsys):
        long_content = "x" * 5000
        mock_result = {"results": [{"title": "T", "content": long_content}]}
        with patch("on_session_start.TMClient") as MockTM:
            instance = MockTM.return_value
            instance.get_context = AsyncMock(return_value=mock_result)
            await process_session_start(_hook_input(), _config())

        output = json.loads(capsys.readouterr().out)
        assert len(output["additional_context"]) <= 2000


# ---------------------------------------------------------------------------
# Test: without project → calls get_context without project
# ---------------------------------------------------------------------------

class TestSessionStartNoProject:
    """When project can't be resolved, still calls get_context (without project)."""

    @pytest.mark.asyncio
    async def test_no_workspace_roots_calls_get_context_without_project(self, capsys):
        inp = HookInput(workspace_roots=[], conversation_id="conv-1")
        mock_result = {"results": []}
        with patch("on_session_start.TMClient") as MockTM:
            instance = MockTM.return_value
            instance.get_context = AsyncMock(return_value=mock_result)
            await process_session_start(inp, _config())

        output = json.loads(capsys.readouterr().out)
        assert output["action"] == "no_context"
        assert output["project"] is None
        instance.get_context.assert_awaited_once_with(project=None)

    @pytest.mark.asyncio
    async def test_empty_context_returns_no_context_action(self, capsys):
        mock_result = {"results": []}
        with patch("on_session_start.TMClient") as MockTM:
            instance = MockTM.return_value
            instance.get_context = AsyncMock(return_value=mock_result)
            await process_session_start(_hook_input(), _config())

        output = json.loads(capsys.readouterr().out)
        assert output["action"] == "no_context"


# ---------------------------------------------------------------------------
# Test: TM error → graceful fallback
# ---------------------------------------------------------------------------

class TestSessionStartErrorHandling:
    """sessionStart hook should never crash — errors produce graceful output."""

    @pytest.mark.asyncio
    async def test_tm_error_graceful_fallback(self, capsys):
        with patch("on_session_start.TMClient") as MockTM:
            instance = MockTM.return_value
            instance.get_context = AsyncMock(side_effect=Exception("MCP down"))
            await process_session_start(_hook_input(), _config())

        output = json.loads(capsys.readouterr().out)
        assert output["action"] == "error"
        assert "message" in output

    @pytest.mark.asyncio
    async def test_get_context_returns_none_graceful_fallback(self, capsys):
        with patch("on_session_start.TMClient") as MockTM:
            instance = MockTM.return_value
            instance.get_context = AsyncMock(return_value=None)
            await process_session_start(_hook_input(), _config())

        output = json.loads(capsys.readouterr().out)
        assert output["action"] == "no_context"

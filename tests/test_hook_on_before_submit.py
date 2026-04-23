"""Tests for scripts/hooks/on_before_submit.py — beforeSubmitPrompt hook (async TMClient version)."""
from __future__ import annotations

import json
import sys
from pathlib import Path
from unittest.mock import AsyncMock, patch

import pytest

# Ensure scripts/hooks/ is importable
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts" / "hooks"))

from on_before_submit import RETRIEVAL_KEYWORDS, process_before_submit  # noqa: I001,E402
from shared import HookInput, PipelineConfig  # noqa: I001,E402


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _hook_input(
    prompt: str = "帮我写个函数",
    workspace_roots: list | None = None,
    conversation_id: str = "conv-prompt-1",
) -> HookInput:
    return HookInput(
        workspace_roots=workspace_roots or ["/Users/yeshouyou/Work/agent/team_doc"],
        prompt=prompt,
        conversation_id=conversation_id,
    )


def _config() -> PipelineConfig:
    return PipelineConfig(tm_url="http://localhost:3900")


# ---------------------------------------------------------------------------
# Test: message with trigger keyword → calls recall → outputs results
# ---------------------------------------------------------------------------

class TestBeforeSubmitRetrieval:
    """beforeSubmitPrompt triggers retrieval for keyword prompts."""

    @pytest.mark.asyncio
    async def test_retrieval_triggered_for_zhijian(self, capsys):
        mock_result = {
            "results": [
                {"title": "Docker经验", "content": "之前遇到Docker问题时..."}
            ]
        }
        with patch("on_before_submit.TMClient") as MockTM:
            instance = MockTM.return_value
            instance.recall = AsyncMock(return_value=mock_result)
            await process_before_submit(_hook_input(prompt="之前遇到的Docker问题"), _config())

        output = json.loads(capsys.readouterr().out)
        assert output["action"] == "retrieved"
        assert "additional_context" in output
        instance.recall.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_retrieval_triggered_for_jingyan(self, capsys):
        with patch("on_before_submit.TMClient") as MockTM:
            instance = MockTM.return_value
            instance.recall = AsyncMock(return_value={"results": []})
            await process_before_submit(_hook_input(prompt="分享一些经验"), _config())

        output = json.loads(capsys.readouterr().out)
        instance.recall.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_retrieval_triggered_for_remember(self, capsys):
        mock_result = {"results": [{"title": "T", "content": "C"}]}
        with patch("on_before_submit.TMClient") as MockTM:
            instance = MockTM.return_value
            instance.recall = AsyncMock(return_value=mock_result)
            await process_before_submit(_hook_input(prompt="remember what we did"), _config())

        output = json.loads(capsys.readouterr().out)
        assert output["action"] == "retrieved"
        instance.recall.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_retrieval_triggered_for_previously(self, capsys):
        mock_result = {"results": [{"title": "T", "content": "C"}]}
        with patch("on_before_submit.TMClient") as MockTM:
            instance = MockTM.return_value
            instance.recall = AsyncMock(return_value=mock_result)
            await process_before_submit(_hook_input(prompt="previously we discussed"), _config())

        output = json.loads(capsys.readouterr().out)
        assert output["action"] == "retrieved"
        instance.recall.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_recall_called_with_correct_args(self, capsys):
        mock_result = {"results": []}
        with patch("on_before_submit.TMClient") as MockTM:
            instance = MockTM.return_value
            instance.recall = AsyncMock(return_value=mock_result)
            await process_before_submit(_hook_input(prompt="之前的方案"), _config())

        instance.recall.assert_awaited_once_with(
            query="之前的方案", project="team_doc"
        )

    @pytest.mark.asyncio
    async def test_retrieved_context_capped_at_max_chars(self, capsys):
        long_content = "y" * 5000
        mock_result = {"results": [{"title": "T", "content": long_content}]}
        with patch("on_before_submit.TMClient") as MockTM:
            instance = MockTM.return_value
            instance.recall = AsyncMock(return_value=mock_result)
            await process_before_submit(_hook_input(prompt="之前的方案"), _config())

        output = json.loads(capsys.readouterr().out)
        assert len(output["additional_context"]) <= 2000


# ---------------------------------------------------------------------------
# Test: message without trigger keyword → skip
# ---------------------------------------------------------------------------

class TestBeforeSubmitNoTrigger:
    """beforeSubmitPrompt skips retrieval when no trigger keywords present."""

    @pytest.mark.asyncio
    async def test_normal_prompt_skip(self, capsys):
        with patch("on_before_submit.TMClient") as MockTM:
            instance = MockTM.return_value
            instance.recall = AsyncMock(return_value={})
            await process_before_submit(_hook_input(prompt="帮我写个函数"), _config())

        output = json.loads(capsys.readouterr().out)
        assert output["action"] == "skip"
        assert output["reason"] == "no_trigger_keywords"
        instance.recall.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_english_prompt_no_trigger_skip(self, capsys):
        with patch("on_before_submit.TMClient") as MockTM:
            instance = MockTM.return_value
            instance.recall = AsyncMock(return_value={})
            await process_before_submit(_hook_input(prompt="help me debug this"), _config())

        output = json.loads(capsys.readouterr().out)
        assert output["action"] == "skip"
        instance.recall.assert_not_awaited()


# ---------------------------------------------------------------------------
# Test: empty message → skip
# ---------------------------------------------------------------------------

class TestBeforeSubmitEmptyMessage:
    """Empty prompt should skip retrieval."""

    @pytest.mark.asyncio
    async def test_empty_message_skip(self, capsys):
        with patch("on_before_submit.TMClient") as MockTM:
            instance = MockTM.return_value
            instance.recall = AsyncMock(return_value={})
            await process_before_submit(_hook_input(prompt=""), _config())

        output = json.loads(capsys.readouterr().out)
        assert output["action"] == "skip"
        instance.recall.assert_not_awaited()


# ---------------------------------------------------------------------------
# Test: multiple trigger keywords → only one recall call
# ---------------------------------------------------------------------------

class TestBeforeSubmitMultipleKeywords:
    """Multiple keywords in one message should trigger only one recall call."""

    @pytest.mark.asyncio
    async def test_multiple_keywords_single_recall(self, capsys):
        mock_result = {"results": [{"title": "T", "content": "C"}]}
        with patch("on_before_submit.TMClient") as MockTM:
            instance = MockTM.return_value
            instance.recall = AsyncMock(return_value=mock_result)
            await process_before_submit(
                _hook_input(prompt="之前上次经验都涉及的问题"), _config()
            )

        output = json.loads(capsys.readouterr().out)
        assert output["action"] == "retrieved"
        instance.recall.assert_awaited_once()


# ---------------------------------------------------------------------------
# Test: TM error → graceful fallback
# ---------------------------------------------------------------------------

class TestBeforeSubmitErrorHandling:
    """beforeSubmitPrompt should never crash — errors produce graceful output."""

    @pytest.mark.asyncio
    async def test_tm_error_graceful_fallback(self, capsys):
        with patch("on_before_submit.TMClient") as MockTM:
            instance = MockTM.return_value
            instance.recall = AsyncMock(side_effect=Exception("MCP down"))
            await process_before_submit(
                _hook_input(prompt="之前遇到的问题"), _config()
            )

        output = json.loads(capsys.readouterr().out)
        assert output["action"] == "error"
        assert "message" in output

    @pytest.mark.asyncio
    async def test_recall_returns_none_graceful_fallback(self, capsys):
        with patch("on_before_submit.TMClient") as MockTM:
            instance = MockTM.return_value
            instance.recall = AsyncMock(return_value=None)
            await process_before_submit(
                _hook_input(prompt="之前遇到的问题"), _config()
            )

        output = json.loads(capsys.readouterr().out)
        assert output["action"] == "no_results"

    @pytest.mark.asyncio
    async def test_no_project_still_retrieves(self, capsys):
        """Even without project, recall should still be called (with project=None)."""
        mock_result = {"results": [{"title": "T", "content": "C"}]}
        inp = HookInput(
            workspace_roots=[],
            prompt="之前遇到的问题",
            conversation_id="conv-1",
        )
        with patch("on_before_submit.TMClient") as MockTM:
            instance = MockTM.return_value
            instance.recall = AsyncMock(return_value=mock_result)
            await process_before_submit(inp, _config())

        instance.recall.assert_awaited_once_with(
            query="之前遇到的问题", project=None
        )
        output = json.loads(capsys.readouterr().out)
        assert output["action"] == "retrieved"


# ---------------------------------------------------------------------------
# Test: RETRIEVAL_KEYWORDS constant
# ---------------------------------------------------------------------------

class TestRetrievalKeywords:
    """Verify the expanded keyword list includes Chinese and English triggers."""

    def test_contains_chinese_keywords(self):
        assert "之前" in RETRIEVAL_KEYWORDS
        assert "上次" in RETRIEVAL_KEYWORDS
        assert "经验" in RETRIEVAL_KEYWORDS
        assert "踩坑" in RETRIEVAL_KEYWORDS
        assert "以前" in RETRIEVAL_KEYWORDS
        assert "历史" in RETRIEVAL_KEYWORDS

    def test_contains_english_keywords(self):
        assert "remember" in RETRIEVAL_KEYWORDS
        assert "before" in RETRIEVAL_KEYWORDS
        assert "previously" in RETRIEVAL_KEYWORDS
        assert "earlier" in RETRIEVAL_KEYWORDS

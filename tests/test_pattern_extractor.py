"""Tests for PatternExtractor — user behavior pattern extraction from conversations."""

from __future__ import annotations

import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from team_memory.services.pattern_extractor import PatternExtractor


class TestPatternExtractor:
    """Tests for PatternExtractor.extract_patterns and extract_and_save."""

    @pytest.mark.asyncio
    async def test_extract_patterns_returns_list(self):
        """LLM returns valid JSON with patterns."""
        llm_response = json.dumps(
            {
                "patterns": [
                    {
                        "pattern": "请一步步思考",
                        "category": "instruction_style",
                        "frequency_hint": "high",
                        "suggested_rule": "用户偏好逐步推理",
                    },
                    {
                        "pattern": "先搜索再执行",
                        "category": "workflow_preference",
                        "frequency_hint": "medium",
                        "suggested_rule": "执行前先检索团队知识库",
                    },
                ]
            }
        )

        mock_llm = AsyncMock()
        mock_llm.chat = AsyncMock(return_value=llm_response)

        with patch(
            "team_memory.services.llm_provider.create_llm_provider",
            return_value=mock_llm,
        ):
            extractor = PatternExtractor()
            patterns = await extractor.extract_patterns("some conversation", MagicMock())

        assert len(patterns) == 2
        assert patterns[0]["pattern"] == "请一步步思考"
        assert patterns[1]["category"] == "workflow_preference"

    @pytest.mark.asyncio
    async def test_extract_patterns_handles_invalid_json(self):
        """LLM returns invalid JSON — should return empty list."""
        mock_llm = AsyncMock()
        mock_llm.chat = AsyncMock(return_value="no json here")

        with patch(
            "team_memory.services.llm_provider.create_llm_provider",
            return_value=mock_llm,
        ):
            extractor = PatternExtractor()
            patterns = await extractor.extract_patterns("conversation", MagicMock())

        assert patterns == []

    @pytest.mark.asyncio
    async def test_extract_patterns_handles_llm_error(self):
        """LLM call fails — should return empty list."""
        mock_llm = AsyncMock()
        mock_llm.chat = AsyncMock(side_effect=RuntimeError("LLM down"))

        with patch(
            "team_memory.services.llm_provider.create_llm_provider",
            return_value=mock_llm,
        ):
            extractor = PatternExtractor()
            patterns = await extractor.extract_patterns("conversation", MagicMock())

        assert patterns == []

    @pytest.mark.asyncio
    async def test_extract_and_save_writes_to_personal_memory(self):
        """extract_and_save should call pm_service.write for each pattern."""
        llm_response = json.dumps(
            {
                "patterns": [
                    {
                        "pattern": "请给出对比方案",
                        "category": "output_format",
                        "frequency_hint": "high",
                        "suggested_rule": "输出时提供对比方案",
                    },
                ]
            }
        )

        mock_llm = AsyncMock()
        mock_llm.chat = AsyncMock(return_value=llm_response)
        mock_pm = AsyncMock()
        mock_pm.write = AsyncMock()

        with patch(
            "team_memory.services.llm_provider.create_llm_provider",
            return_value=mock_llm,
        ):
            extractor = PatternExtractor()
            count = await extractor.extract_and_save(
                conversation="user said things",
                user_id="alice",
                llm_config=MagicMock(),
                pm_service=mock_pm,
            )

        assert count == 1
        mock_pm.write.assert_awaited_once()
        call_kwargs = mock_pm.write.call_args.kwargs
        assert call_kwargs["user_id"] == "alice"
        assert call_kwargs["scope"] == "output_format"
        assert call_kwargs["profile_kind"] == "dynamic"  # output_format maps to dynamic

    @pytest.mark.asyncio
    async def test_extract_and_save_returns_zero_on_no_patterns(self):
        """No patterns extracted — return 0."""
        mock_llm = AsyncMock()
        mock_llm.chat = AsyncMock(return_value='{"patterns": []}')
        mock_pm = AsyncMock()

        with patch(
            "team_memory.services.llm_provider.create_llm_provider",
            return_value=mock_llm,
        ):
            extractor = PatternExtractor()
            count = await extractor.extract_and_save(
                conversation="nothing useful",
                user_id="bob",
                llm_config=MagicMock(),
                pm_service=mock_pm,
            )

        assert count == 0
        mock_pm.write.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_category_to_profile_kind_mapping(self):
        """instruction_style/workflow_preference → static, output_format → dynamic."""
        llm_response = json.dumps(
            {
                "patterns": [
                    {"pattern": "p1", "category": "instruction_style", "suggested_rule": "r1"},
                    {"pattern": "p2", "category": "workflow_preference", "suggested_rule": "r2"},
                    {"pattern": "p3", "category": "output_format", "suggested_rule": "r3"},
                ]
            }
        )

        mock_llm = AsyncMock()
        mock_llm.chat = AsyncMock(return_value=llm_response)
        mock_pm = AsyncMock()
        mock_pm.write = AsyncMock()

        with patch(
            "team_memory.services.llm_provider.create_llm_provider",
            return_value=mock_llm,
        ):
            extractor = PatternExtractor()
            count = await extractor.extract_and_save(
                conversation="conversation",
                user_id="alice",
                llm_config=MagicMock(),
                pm_service=mock_pm,
            )

        assert count == 3
        calls = mock_pm.write.call_args_list
        assert calls[0].kwargs["profile_kind"] == "static"  # instruction_style
        assert calls[1].kwargs["profile_kind"] == "static"  # workflow_preference
        assert calls[2].kwargs["profile_kind"] == "dynamic"  # output_format

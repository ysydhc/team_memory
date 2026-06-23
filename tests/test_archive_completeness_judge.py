"""Unit tests for ArchiveCompletenessJudge — parsing + fail-closed behavior.

The LLM call itself is mocked; these cover response parsing and the critical
fail-closed guarantee (any error → not complete, so a bad archive stays draft
rather than being wrongly auto-published).
"""

from __future__ import annotations

from unittest.mock import AsyncMock, patch

import pytest

from team_memory.services.archive.completeness_judge import (
    ArchiveCompletenessJudge,
    CompletenessResult,
)


def _judge() -> ArchiveCompletenessJudge:
    return ArchiveCompletenessJudge(base_url="http://x/v1", model="m", api_key="k")


@pytest.mark.unit
class TestParse:
    def test_plain_json_complete(self):
        r = ArchiveCompletenessJudge._parse('{"complete": true, "reason": "ok"}')
        assert r.complete is True
        assert r.reason == "ok"

    def test_plain_json_incomplete(self):
        r = ArchiveCompletenessJudge._parse('{"complete": false, "reason": "TODO 未完成"}')
        assert r.complete is False

    def test_code_fenced_json(self):
        raw = '```json\n{"complete": true, "reason": "done"}\n```'
        r = ArchiveCompletenessJudge._parse(raw)
        assert r.complete is True

    def test_json_with_surrounding_text(self):
        raw = '判断如下:{"complete": true, "reason": "完整"} 以上'
        r = ArchiveCompletenessJudge._parse(raw)
        assert r.complete is True

    def test_missing_complete_defaults_false(self):
        r = ArchiveCompletenessJudge._parse('{"reason": "no verdict"}')
        assert r.complete is False


@pytest.mark.unit
class TestJudgeFailClosed:
    @pytest.mark.asyncio
    async def test_llm_error_returns_not_complete(self):
        judge = _judge()
        with patch.object(judge, "_call_llm", new=AsyncMock(side_effect=RuntimeError("boom"))):
            result = await judge.judge(title="t", overview="o", solution_doc="s")
        assert result.complete is False
        assert "judge error" in result.reason

    @pytest.mark.asyncio
    async def test_unparseable_returns_not_complete(self):
        judge = _judge()
        with patch.object(judge, "_call_llm", new=AsyncMock(return_value="not json at all")):
            result = await judge.judge(title="t", overview="o", solution_doc="s")
        assert result.complete is False

    @pytest.mark.asyncio
    async def test_complete_verdict_passes_through(self):
        judge = _judge()
        with patch.object(judge, "_call_llm", new=AsyncMock(return_value='{"complete": true, "reason": "ok"}')):
            result = await judge.judge(title="t", overview="o", solution_doc="s")
        assert isinstance(result, CompletenessResult)
        assert result.complete is True

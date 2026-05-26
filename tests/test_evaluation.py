"""Tests for EvaluationService — marker injection and was_used detection."""

from __future__ import annotations

import pytest

from team_memory.services.analysis.evaluation import EvaluationService


class TestCheckWasUsedFuzzy:
    """Test EvaluationService.check_was_used_fuzzy."""

    @pytest.mark.asyncio
    async def test_high_overlap_returns_true(self):
        """When keywords overlap > threshold, fuzzy returns True."""
        svc = EvaluationService()
        results = [{"id": "exp-001", "solution": "Use async await pattern"}]
        out = await svc.check_was_used_fuzzy("Use async await pattern", results)
        assert out == {"exp-001": True}

    @pytest.mark.asyncio
    async def test_low_overlap_returns_false(self):
        """When keywords overlap < threshold, fuzzy returns False."""
        svc = EvaluationService()
        results = [{"id": "exp-001", "solution": "Use async await pattern"}]
        out = await svc.check_was_used_fuzzy("Docker compose networking", results)
        assert out == {"exp-001": False}

    @pytest.mark.asyncio
    async def test_empty_results(self):
        """Empty results list returns empty dict."""
        svc = EvaluationService()
        out = await svc.check_was_used_fuzzy("response", [])
        assert out == {}


class TestGetWeeklyStats:
    """Test EvaluationService.get_weekly_stats."""

    @pytest.mark.asyncio
    async def test_default_stats(self):
        """Without DB connection, returns zeroed stats."""
        svc = EvaluationService()
        stats = await svc.get_weekly_stats()
        assert stats == {"total": 0, "hit": 0, "used": 0, "use_rate": 0.0}

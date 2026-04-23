"""Tests for WeeklyReport hook — formatting and generation."""
from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from scripts.hooks.weekly_report import WeeklyReport


class TestFormatReport:
    """Test WeeklyReport.format_report."""

    def _make_report(self) -> WeeklyReport:
        """Create a WeeklyReport with a mock TMClient (URL doesn't matter)."""
        return WeeklyReport("http://localhost:3900")

    def test_high_use_rate_effective(self):
        """Use rate 46.8% → '系统有效 ✓'."""
        rpt = self._make_report()
        result = rpt.format_report(
            stats={"total": 100, "hit": 80, "used": 38, "use_rate": 0.468},
            new_count=5,
            promoted_count=2,
        )
        assert "系统有效 ✓" in result
        assert "46.8%" in result

    def test_medium_use_rate_needs_adjustment(self):
        """Use rate 30% → '需要调整 ⚠'."""
        rpt = self._make_report()
        result = rpt.format_report(
            stats={"total": 50, "hit": 30, "used": 9, "use_rate": 0.3},
        )
        assert "需要调整 ⚠" in result
        assert "30.0%" in result

    def test_low_use_rate_invalid(self):
        """Use rate 15% → '系统无效 ✗'."""
        rpt = self._make_report()
        result = rpt.format_report(
            stats={"total": 20, "hit": 10, "used": 2, "use_rate": 0.15},
        )
        assert "系统无效 ✗" in result
        assert "15.0%" in result

    def test_zero_data_shows_zeros(self):
        """Zero data → displays 0 for all counters."""
        rpt = self._make_report()
        result = rpt.format_report(
            stats={"total": 0, "hit": 0, "used": 0, "use_rate": 0.0},
            new_count=0,
            promoted_count=0,
        )
        assert "检索：0 次" in result
        assert "命中：0 次" in result
        assert "使用：0 次" in result
        assert "0.0%" in result
        assert "系统无效 ✗" in result

    def test_report_contains_key_sections(self):
        """Report has all required sections."""
        rpt = self._make_report()
        result = rpt.format_report(
            stats={"total": 10, "hit": 8, "used": 4, "use_rate": 0.5},
            new_count=3,
            promoted_count=1,
        )
        assert "记忆系统周报" in result
        assert "检索" in result
        assert "命中" in result
        assert "使用" in result
        assert "使用率" in result
        assert "本周新增" in result
        assert "本周提升" in result
        assert "评估" in result


class TestGenerate:
    """Test WeeklyReport.generate — calls TM API for stats."""

    @pytest.mark.asyncio
    async def test_generate_calls_tm_api(self):
        """generate() calls TMClient to fetch stats."""
        rpt = WeeklyReport("http://localhost:3900")
        # Mock the TMClient._call to return stats
        mock_stats = {"total": 50, "hit": 30, "used": 15, "use_rate": 0.5}
        rpt._tm._call = AsyncMock(return_value=mock_stats)

        result = await rpt.generate(days=7)

        # Should have called _call with the search_log stats endpoint
        rpt._tm._call.assert_awaited_once()
        call_args = rpt._tm._call.call_args
        assert "search_log" in call_args[0][0] or "stats" in call_args[0][0]
        # Result should contain formatted report
        assert "记忆系统周报" in result
        assert "50.0%" in result

    @pytest.mark.asyncio
    async def test_generate_handles_api_error(self):
        """generate() returns error message when API call fails."""
        rpt = WeeklyReport("http://localhost:3900")
        rpt._tm._call = AsyncMock(side_effect=Exception("API error"))

        result = await rpt.generate(days=7)
        # Should handle gracefully — either return error or zeroed report
        assert "记忆系统周报" in result or "error" in result.lower() or "Error" in result

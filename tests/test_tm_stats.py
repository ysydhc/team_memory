"""Tests for tm_stats — search quality report CLI / 搜索质量报告 CLI 测试

Unit tests cover:
- _bar() bar chart generation / 柱状图生成
- _resolve_dsn() DSN resolution / DSN 解析

Integration tests (marked @pytest.mark.integration) cover:
- Full stats report with real DB
"""

from __future__ import annotations

import pytest

from team_memory.daemon.observe.tm_stats import _bar, _resolve_dsn

pytestmark = pytest.mark.unit


class TestBar:
    """Tests for _bar() bar chart generation / _bar() 柱状图生成测试"""

    def test_full_bar(self):
        """Value equals max should fill all bars / 值等于最大值应填满"""
        result = _bar(10, 10, width=5)
        assert result == "█████"

    def test_zero_value(self):
        """Zero value should produce empty bars / 零值应产生空柱"""
        result = _bar(0, 10, width=5)
        assert result == "░░░░░"

    def test_half_bar(self):
        """Half value should fill half / 一半值应填充一半"""
        result = _bar(5, 10, width=10)
        assert "█" in result
        assert "░" in result
        assert len(result) == 10

    def test_zero_max(self):
        """Zero max_value should produce empty bars / 零最大值应产生空柱"""
        result = _bar(5, 0, width=5)
        assert result == "░░░░░"

    def test_negative_max(self):
        """Negative max_value should produce empty bars / 负最大值应产生空柱"""
        result = _bar(5, -1, width=5)
        assert result == "░░░░░"

    def test_custom_width(self):
        """Custom width should be respected / 自定义宽度应被遵守"""
        result = _bar(5, 10, width=20)
        assert len(result) == 20

    def test_small_value_large_max(self):
        """Small value with large max should produce mostly empty / 小值大最大值应几乎全空"""
        result = _bar(1, 100, width=20)
        assert result.count("█") == 0 or result.count("█") <= 1

    def test_bar_default_width(self):
        """Default width should be 20 / 默认宽度应为 20"""
        result = _bar(5, 10)
        assert len(result) == 20


class TestResolveDsn:
    """Tests for _resolve_dsn() DSN resolution / _resolve_dsn() DSN 解析测试"""

    def test_returns_string(self):
        """Should return a non-empty string / 应返回非空字符串"""
        dsn = _resolve_dsn()
        assert isinstance(dsn, str)
        assert len(dsn) > 0

    def test_contains_postgresql(self):
        """DSN should start with postgresql / DSN 应以 postgresql 开头"""
        dsn = _resolve_dsn()
        assert dsn.startswith("postgresql")

    def test_contains_team_memory(self):
        """DSN should reference team_memory database / DSN 应引用 team_memory 数据库"""
        dsn = _resolve_dsn()
        assert "team_memory" in dsn

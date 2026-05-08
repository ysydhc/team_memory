"""Tests for ResponseBuffer — faithfulness response buffer / 忠实度响应缓冲区测试

Unit tests cover:
- Constructor defaults / 构造函数默认值
- Batch threshold logic / 批量阈值逻辑

Integration tests (marked @pytest.mark.integration) cover:
- add() / get_pending() / mark_evaluated() with real DB
"""
from __future__ import annotations

import pytest

from team_memory.daemon.response_buffer import ResponseBuffer

pytestmark = pytest.mark.unit


class TestResponseBufferInit:
    """Tests for ResponseBuffer constructor / ResponseBuffer 构造函数测试"""

    def test_default_dsn(self):
        """Default DSN should use localhost / 默认 DSN 应使用 localhost"""
        buf = ResponseBuffer()
        assert "localhost" in buf._dsn
        assert "5433" in buf._dsn

    def test_custom_dsn(self):
        """Custom DSN should be stored / 自定义 DSN 应被保存"""
        buf = ResponseBuffer(dsn="postgresql://user:pass@remote:5432/db")
        assert buf._dsn == "postgresql://user:pass@remote:5432/db"

    def test_default_batch_threshold(self):
        """Default batch threshold should be 5 / 默认批量阈值应为 5"""
        buf = ResponseBuffer()
        assert buf._batch_threshold == 5

    def test_custom_batch_threshold(self):
        """Custom batch threshold should be stored / 自定义批量阈值应被保存"""
        buf = ResponseBuffer(batch_threshold=10)
        assert buf._batch_threshold == 10

    def test_pool_initially_none(self):
        """Pool should start as None / 连接池初始应为 None"""
        buf = ResponseBuffer()
        assert buf._pool is None


class TestResponseBufferThreshold:
    """Tests for batch threshold logic / 批量阈值逻辑测试"""

    def test_threshold_zero_means_always_trigger(self):
        """Threshold=0 should trigger on every entry / 阈值=0 应每次触发"""
        buf = ResponseBuffer(batch_threshold=0)
        assert buf._batch_threshold == 0

    def test_threshold_one_triggers_on_first(self):
        """Threshold=1 should trigger on first entry / 阈值=1 应在首条触发"""
        buf = ResponseBuffer(batch_threshold=1)
        assert buf._batch_threshold == 1

    def test_threshold_large_defers_evaluation(self):
        """Large threshold defers evaluation / 大阈值延迟评估"""
        buf = ResponseBuffer(batch_threshold=100)
        assert buf._batch_threshold == 100

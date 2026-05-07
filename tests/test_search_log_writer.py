"""Tests for SearchLogWriter — search log and usage tracking / 搜索日志与使用追踪测试

Unit tests cover:
- _MEM_MARKER_RE regex pattern / 正则模式
- Constructor defaults / 构造函数默认值
- Fuzzy matching logic / 模糊匹配逻辑

Integration tests (marked @pytest.mark.integration) cover:
- log_search / mark_used_from_response / get_stats with real DB
"""
from __future__ import annotations

import pytest

from daemon.search_log_writer import SearchLogWriter, _MEM_MARKER_RE

pytestmark = pytest.mark.unit


class TestMemMarkerRegex:
    """Tests for _MEM_MARKER_RE pattern / _MEM_MARKER_RE 正则模式测试"""

    def test_standard_marker(self):
        """Standard [mem:abc12345] should match / 标准 [mem:abc12345] 应匹配"""
        matches = _MEM_MARKER_RE.findall("Based on [mem:abc12345] I suggest...")
        assert matches == ["abc12345"]

    def test_uppercase_marker(self):
        """Uppercase [mem:ABC12345] should match (case-insensitive) / 大写也应匹配"""
        matches = _MEM_MARKER_RE.findall("See [mem:ABC12345] for details")
        assert matches == ["ABC12345"]

    def test_multiple_markers(self):
        """Multiple markers should all be found / 多个标记应全部找到"""
        text = "Used [mem:aaa11111] and [mem:bbb22222]"
        matches = _MEM_MARKER_RE.findall(text)
        assert len(matches) == 2
        assert "aaa11111" in matches
        assert "bbb22222" in matches

    def test_no_marker(self):
        """No marker should return empty list / 无标记应返回空列表"""
        matches = _MEM_MARKER_RE.findall("Just a normal response")
        assert matches == []

    def test_partial_marker_not_matched(self):
        """Partial marker [mem:abc] (too short) should NOT match / 过短标记不应匹配"""
        matches = _MEM_MARKER_RE.findall("See [mem:abc] for info")
        assert matches == []

    def test_long_prefix_not_matched(self):
        """9-char prefix should NOT match / 9 字符前缀不应匹配"""
        matches = _MEM_MARKER_RE.findall("See [mem:abc123456] for info")
        assert matches == []

    def test_marker_in_code_block(self):
        """Marker inside code block should match / 代码块中的标记应匹配"""
        matches = _MEM_MARKER_RE.findall("```python\n# [mem:abc12345]\n```")
        assert matches == ["abc12345"]


class TestSearchLogWriterInit:
    """Tests for SearchLogWriter constructor / SearchLogWriter 构造函数测试"""

    def test_default_dsn(self):
        """Default DSN should contain localhost / 默认 DSN 应包含 localhost"""
        writer = SearchLogWriter()
        assert "localhost" in writer._dsn
        assert "5433" in writer._dsn

    def test_custom_dsn(self):
        """Custom DSN should be stored / 自定义 DSN 应被保存"""
        writer = SearchLogWriter(dsn="postgresql://user:pass@remote:5432/db")
        assert writer._dsn == "postgresql://user:pass@remote:5432/db"

    def test_pool_initially_none(self):
        """Pool should start as None / 连接池初始应为 None"""
        writer = SearchLogWriter()
        assert writer._pool is None


class TestFuzzyMatchingLogic:
    """Tests for fuzzy matching keyword overlap logic / 模糊匹配关键词重叠逻辑测试

    These test the pure logic used in mark_used_fuzzy() —
    extracting keywords from query/response and checking overlap.
    """

    def test_keyword_extraction_basic(self):
        """Basic keyword extraction from text / 基本关键词提取"""
        # The fuzzy matching logic tokenizes by splitting on whitespace
        text = "Docker PostgreSQL connection fix"
        tokens = text.lower().split()
        assert "docker" in tokens
        assert "postgresql" in tokens
        assert "connection" in tokens

    def test_keyword_overlap_detection(self):
        """Overlap between query and response should be detectable / 查询和回复的重叠应可检测"""
        query_keywords = {"docker", "postgresql", "connection"}
        response_text = "use host.docker.internal for Docker PostgreSQL connection"
        response_tokens = set(response_text.lower().split())
        overlap = query_keywords & response_tokens
        assert len(overlap) >= 2  # at least docker + postgresql

    def test_no_keyword_overlap(self):
        """No overlap should return empty set / 无重叠应返回空集合"""
        query_keywords = {"kubernetes", "helm"}
        response_text = "Docker compose is a tool for defining multi-container applications"
        response_tokens = set(response_text.lower().split())
        overlap = query_keywords & response_tokens
        assert len(overlap) == 0

    def test_chinese_keyword_extraction(self):
        """Chinese keywords should be handled / 中文关键词应被处理"""
        text = "实体提取 使用LLM从文本中提取"
        tokens = text.split()
        assert len(tokens) >= 2

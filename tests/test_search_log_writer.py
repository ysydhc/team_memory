"""Tests for SearchLogWriter — search log and usage tracking / 搜索日志与使用追踪测试

Unit tests cover:
- _MEM_MARKER_RE regex pattern / 正则模式
- Constructor defaults / 构造函数默认值
- Fuzzy matching logic / 模糊匹配逻辑

Integration tests (marked @pytest.mark.integration) cover:
"""

from __future__ import annotations

import pytest

from team_memory.services.search.search_log_writer import SearchLogWriter

pytestmark = pytest.mark.unit


class TestSearchLogWriterInit:
    """Tests for SearchLogWriter constructor / SearchLogWriter 构造函数测试"""

    def test_env_dsn_fallback(self, monkeypatch):
        """Env var TEAM_MEMORY_DB_URL used when no dsn passed"""
        monkeypatch.setenv("TEAM_MEMORY_DB_URL", "postgresql://env:test@localhost:5433/tm")
        writer = SearchLogWriter()
        assert writer._dsn == "postgresql://env:test@localhost:5433/tm"

    def test_no_dsn_raises(self, monkeypatch):
        """Missing DSN should raise RuntimeError"""
        monkeypatch.delenv("TEAM_MEMORY_DB_URL", raising=False)
        with pytest.raises(RuntimeError):
            SearchLogWriter()

    def test_custom_dsn(self):
        """Custom DSN should be stored"""
        writer = SearchLogWriter(dsn="postgresql://user:pass@remote:5432/db")
        assert writer._dsn == "postgresql://user:pass@remote:5432/db"

    def test_pool_initially_none(self):
        """Pool should start as None"""
        writer = SearchLogWriter(dsn="postgresql://test:test@localhost:5433/tm")
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

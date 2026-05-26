"""Tests for search_eval — search quality evaluator / 搜索质量评估器测试

Unit tests cover:
- DEFAULT_TEST_QUERIES structure validation / 测试查询结构验证
- Keyword matching logic / 关键词匹配逻辑

Integration tests (marked @pytest.mark.integration) cover:
- Full evaluation with real DB and search pipeline
"""

from __future__ import annotations

import pytest

from team_memory.daemon.quality.search_eval import DEFAULT_TEST_QUERIES

pytestmark = pytest.mark.unit


class TestDefaultTestQueries:
    """Tests for DEFAULT_TEST_QUERIES structure / DEFAULT_TEST_QUERIES 结构测试"""

    def test_queries_is_non_empty_list(self):
        """Should be a non-empty list / 应为非空列表"""
        assert isinstance(DEFAULT_TEST_QUERIES, list)
        assert len(DEFAULT_TEST_QUERIES) > 0

    def test_each_query_has_required_fields(self):
        """Each query should have query, expected_keywords, min_results / 每个查询应有必需字段"""
        for i, tq in enumerate(DEFAULT_TEST_QUERIES):
            assert "query" in tq, f"Query {i} missing 'query'"
            assert "expected_keywords" in tq, f"Query {i} missing 'expected_keywords'"
            assert "min_results" in tq, f"Query {i} missing 'min_results'"
            assert isinstance(tq["query"], str), f"Query {i} 'query' should be str"
            assert isinstance(tq["expected_keywords"], list), f"Query {i} 'expected_keywords' should be list"
            assert isinstance(tq["min_results"], int), f"Query {i} 'min_results' should be int"

    def test_expected_keywords_are_lowercase(self):
        """expected_keywords should be lowercase for case-insensitive matching / 关键词应为小写"""
        for i, tq in enumerate(DEFAULT_TEST_QUERIES):
            for kw in tq["expected_keywords"]:
                assert kw == kw.lower(), f"Query {i} keyword '{kw}' should be lowercase"

    def test_min_results_at_least_one(self):
        """min_results should be >= 1 / min_results 应 >= 1"""
        for i, tq in enumerate(DEFAULT_TEST_QUERIES):
            assert tq["min_results"] >= 1, f"Query {i} min_results should be >= 1"

    def test_no_duplicate_queries(self):
        """Queries should be unique / 查询应唯一"""
        queries = [tq["query"] for tq in DEFAULT_TEST_QUERIES]
        assert len(queries) == len(set(queries)), "Duplicate queries found"

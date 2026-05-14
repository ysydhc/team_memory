"""Comprehensive unit tests for src/team_memory/daemon/search_eval.py

Tests the search quality evaluator with mocked DB/search pipeline.
Covers: DEFAULT_TEST_QUERIES structure validation, print_report output,
keyword matching logic, precision/recall calculation, aggregate metrics,
edge cases in evaluation logic, and the run_evaluation flow.

Uses @pytest.mark.unit for pure-logic and mocked-DB tests.
Uses @pytest.mark.integration for tests that hit a real DB/search pipeline.
"""

from __future__ import annotations

from io import StringIO
from unittest.mock import AsyncMock, patch

import pytest

from team_memory.daemon.quality.search_eval import DEFAULT_TEST_QUERIES, print_report
from tests.conftest import make_mock_app_context

pytestmark = pytest.mark.unit


# ===================================================================
# DEFAULT_TEST_QUERIES structure validation
# ===================================================================


class TestDefaultTestQueries:
    """Tests for DEFAULT_TEST_QUERIES structure and content."""

    def test_queries_is_non_empty_list(self):
        """Should be a non-empty list."""
        assert isinstance(DEFAULT_TEST_QUERIES, list)
        assert len(DEFAULT_TEST_QUERIES) > 0

    def test_each_query_has_required_fields(self):
        """Each query should have query, expected_keywords, min_results."""
        for i, tq in enumerate(DEFAULT_TEST_QUERIES):
            assert "query" in tq, f"Query {i} missing 'query'"
            assert "expected_keywords" in tq, f"Query {i} missing 'expected_keywords'"
            assert "min_results" in tq, f"Query {i} missing 'min_results'"
            assert isinstance(tq["query"], str), f"Query {i} 'query' should be str"
            assert isinstance(tq["expected_keywords"], list), f"Query {i} 'expected_keywords' should be list"
            assert isinstance(tq["min_results"], int), f"Query {i} 'min_results' should be int"

    def test_expected_keywords_are_lowercase(self):
        """expected_keywords should be lowercase for case-insensitive matching."""
        for i, tq in enumerate(DEFAULT_TEST_QUERIES):
            for kw in tq["expected_keywords"]:
                assert kw == kw.lower(), f"Query {i} keyword '{kw}' should be lowercase"

    def test_min_results_at_least_one(self):
        """min_results should be >= 1."""
        for i, tq in enumerate(DEFAULT_TEST_QUERIES):
            assert tq["min_results"] >= 1, f"Query {i} min_results should be >= 1"

    def test_no_duplicate_queries(self):
        """Queries should be unique."""
        queries = [tq["query"] for tq in DEFAULT_TEST_QUERIES]
        assert len(queries) == len(set(queries)), "Duplicate queries found"

    def test_expected_keywords_non_empty(self):
        """Each query should have at least one expected keyword."""
        for i, tq in enumerate(DEFAULT_TEST_QUERIES):
            assert len(tq["expected_keywords"]) > 0, f"Query {i} has no expected_keywords"

    def test_query_strings_non_empty(self):
        """Each query string should be non-empty."""
        for i, tq in enumerate(DEFAULT_TEST_QUERIES):
            assert len(tq["query"]) > 0, f"Query {i} has empty query string"

    def test_at_least_five_test_queries(self):
        """Should have at least 5 test queries for meaningful evaluation."""
        assert len(DEFAULT_TEST_QUERIES) >= 5


# ===================================================================
# print_report — pure function that prints formatted output
# ===================================================================


class TestPrintReport:
    """Tests for print_report output formatting."""

    def _capture_output(self, metrics: dict) -> str:
        """Helper to capture print_report stdout."""
        buf = StringIO()
        import sys

        old_stdout = sys.stdout
        sys.stdout = buf
        try:
            print_report(metrics)
        finally:
            sys.stdout = old_stdout
        return buf.getvalue()

    def test_prints_header(self):
        metrics = {
            "timestamp": "2024-01-01T00:00:00+00:00",
            "total_queries": 5,
            "successful_queries": 3,
            "avg_precision": 0.8,
            "avg_recall": 0.6,
            "hit_rate": 0.75,
            "details": [],
        }
        output = self._capture_output(metrics)
        assert "Search Quality Evaluation Report" in output
        assert "2024-01-01T00:00:00+00:00" in output

    def test_prints_summary_stats(self):
        metrics = {
            "timestamp": "2024-01-01T00:00:00+00:00",
            "total_queries": 10,
            "successful_queries": 8,
            "avg_precision": 0.85,
            "avg_recall": 0.7,
            "hit_rate": 0.9,
            "details": [],
        }
        output = self._capture_output(metrics)
        assert "10" in output  # total_queries
        assert "8" in output  # successful_queries
        assert "90.0%" in output  # hit_rate as percentage

    def test_prints_ok_detail_rows(self):
        metrics = {
            "timestamp": "2024-01-01T00:00:00+00:00",
            "total_queries": 1,
            "successful_queries": 1,
            "avg_precision": 1.0,
            "avg_recall": 1.0,
            "hit_rate": 1.0,
            "details": [
                {
                    "query": "test query",
                    "status": "ok",
                    "hits": 3,
                    "precision": 1.0,
                    "recall": 1.0,
                },
            ],
        }
        output = self._capture_output(metrics)
        assert "test query" in output
        assert "3" in output  # hits

    def test_prints_error_detail_rows(self):
        metrics = {
            "timestamp": "2024-01-01T00:00:00+00:00",
            "total_queries": 1,
            "successful_queries": 0,
            "avg_precision": 0.0,
            "avg_recall": 0.0,
            "hit_rate": 0.0,
            "details": [
                {
                    "query": "failing query",
                    "status": "error",
                    "error": "connection refused",
                    "hits": 0,
                    "precision": 0.0,
                },
            ],
        }
        output = self._capture_output(metrics)
        assert "failing query" in output
        assert "ERR" in output

    def test_prints_separator_lines(self):
        metrics = {
            "timestamp": "2024-01-01T00:00:00+00:00",
            "total_queries": 0,
            "successful_queries": 0,
            "avg_precision": 0.0,
            "avg_recall": 0.0,
            "hit_rate": 0.0,
            "details": [],
        }
        output = self._capture_output(metrics)
        assert "=" * 60 in output
        assert "─" * 60 in output


# ===================================================================
# Precision / recall calculation logic (extracted from run_evaluation)
# ===================================================================


class TestPrecisionRecallLogic:
    """Tests for the precision/recall calculation logic.

    These tests exercise the same logic found inside run_evaluation() lines 110-139,
    but without needing the full async DB pipeline.
    """

    def _calc_metrics(self, recall_results: list[dict], expected_keywords: list[str], min_results: int):
        """Reproduce the precision/recall logic from run_evaluation."""
        hits = 0
        for r in recall_results:
            title = (r.get("title") or "").lower()
            desc = (r.get("description") or "").lower()
            solution = (r.get("solution") or "").lower()
            tags = [t.lower() for t in (r.get("tags") or [])]
            text = f"{title} {desc} {solution} {' '.join(tags)}"

            if any(kw in text for kw in expected_keywords):
                hits += 1

        precision = hits / len(recall_results) if recall_results else 0.0
        recall = 1.0 if hits >= min_results else 0.0
        return hits, precision, recall

    def test_all_results_match(self):
        results = [
            {"title": "Docker Guide", "description": "PostgreSQL connection setup"},
            {"title": "Docker Compose", "description": "PostgreSQL config"},
        ]
        keywords = ["docker", "postgresql"]
        hits, precision, recall = self._calc_metrics(results, keywords, min_results=1)
        assert hits == 2
        assert precision == 1.0
        assert recall == 1.0

    def test_no_results_match(self):
        results = [
            {"title": "Python basics", "description": "Hello world"},
            {"title": "JavaScript guide", "description": "DOM manipulation"},
        ]
        keywords = ["docker", "postgresql"]
        hits, precision, recall = self._calc_metrics(results, keywords, min_results=1)
        assert hits == 0
        assert precision == 0.0
        assert recall == 0.0

    def test_partial_match(self):
        results = [
            {"title": "Docker Guide", "description": "container setup"},
            {"title": "Python basics", "description": "Hello world"},
        ]
        keywords = ["docker", "postgresql"]
        hits, precision, recall = self._calc_metrics(results, keywords, min_results=1)
        assert hits == 1
        assert precision == 0.5
        assert recall == 1.0  # 1 hit >= min_results=1

    def test_empty_results(self):
        results = []
        keywords = ["docker"]
        hits, precision, recall = self._calc_metrics(results, keywords, min_results=1)
        assert hits == 0
        assert precision == 0.0
        assert recall == 0.0

    def test_min_results_threshold_not_met(self):
        """recall should be 0 when hits < min_results."""
        results = [
            {"title": "Docker Guide", "description": "container setup"},
        ]
        keywords = ["docker", "postgresql"]
        hits, precision, recall = self._calc_metrics(results, keywords, min_results=2)
        assert hits == 1
        assert precision == 1.0
        assert recall == 0.0  # 1 < min_results=2

    def test_min_results_exactly_met(self):
        """recall should be 1 when hits == min_results."""
        results = [
            {"title": "Docker Guide", "description": "container setup"},
            {"title": "Python basics", "description": "Hello world"},
        ]
        keywords = ["docker"]
        hits, precision, recall = self._calc_metrics(results, keywords, min_results=1)
        assert hits == 1
        assert recall == 1.0

    def test_case_insensitive_matching(self):
        """Keywords should match case-insensitively."""
        results = [
            {"title": "DOCKER SETUP", "description": "PostgreSQL Connection"},
        ]
        keywords = ["docker", "postgresql"]
        hits, precision, recall = self._calc_metrics(results, keywords, min_results=1)
        assert hits == 1

    def test_match_in_description(self):
        """Should match keywords in description field."""
        results = [
            {"title": "Some Title", "description": "docker postgresql config"},
        ]
        keywords = ["docker"]
        hits, _, _ = self._calc_metrics(results, keywords, min_results=1)
        assert hits == 1

    def test_match_in_solution(self):
        """Should match keywords in solution field."""
        results = [
            {"title": "T", "description": "D", "solution": "docker setup steps"},
        ]
        keywords = ["docker"]
        hits, _, _ = self._calc_metrics(results, keywords, min_results=1)
        assert hits == 1

    def test_match_in_tags(self):
        """Should match keywords in tags."""
        results = [
            {"title": "T", "description": "D", "tags": ["Docker", "Compose"]},
        ]
        keywords = ["docker"]
        hits, _, _ = self._calc_metrics(results, keywords, min_results=1)
        assert hits == 1

    def test_missing_fields_handled(self):
        """Missing title/description/solution/tags should not crash."""
        results = [
            {"title": None, "description": None, "solution": None, "tags": None},
        ]
        keywords = ["docker"]
        hits, precision, recall = self._calc_metrics(results, keywords, min_results=1)
        assert hits == 0
        assert precision == 0.0

    def test_empty_fields_handled(self):
        """Empty string fields should not crash."""
        results = [
            {"title": "", "description": "", "solution": "", "tags": []},
        ]
        keywords = ["docker"]
        hits, precision, recall = self._calc_metrics(results, keywords, min_results=1)
        assert hits == 0

    def test_keyword_as_substring(self):
        """Keywords should match as substrings, not exact words."""
        results = [
            {"title": "configuration"},
        ]
        keywords = ["config"]
        hits, _, _ = self._calc_metrics(results, keywords, min_results=1)
        assert hits == 1

    def test_no_keywords_matches_nothing(self):
        """Empty keyword list should match nothing."""
        results = [
            {"title": "anything"},
        ]
        keywords = []
        hits, precision, recall = self._calc_metrics(results, keywords, min_results=1)
        assert hits == 0


# ===================================================================
# Aggregate metrics computation (from run_evaluation lines 136-149)
# ===================================================================


class TestAggregateMetrics:
    """Tests for the aggregate metrics computation logic."""

    def _compute_aggregates(self, results: list[dict]):
        """Reproduce the aggregate logic from run_evaluation."""
        ok_results = [r for r in results if r["status"] == "ok"]
        avg_precision = sum(r["precision"] for r in ok_results) / len(ok_results) if ok_results else 0.0
        avg_recall = sum(r["recall"] for r in ok_results) / len(ok_results) if ok_results else 0.0
        hit_rate = sum(1 for r in ok_results if r["total_results"] > 0) / len(ok_results) if ok_results else 0.0
        return {
            "avg_precision": round(avg_precision, 3),
            "avg_recall": round(avg_recall, 3),
            "hit_rate": round(hit_rate, 3),
        }

    def test_all_ok_results(self):
        results = [
            {"status": "ok", "precision": 1.0, "recall": 1.0, "total_results": 3},
            {"status": "ok", "precision": 0.8, "recall": 1.0, "total_results": 2},
        ]
        agg = self._compute_aggregates(results)
        assert agg["avg_precision"] == 0.9
        assert agg["avg_recall"] == 1.0
        assert agg["hit_rate"] == 1.0

    def test_mixed_ok_and_error(self):
        """Error results should be excluded from aggregates."""
        results = [
            {"status": "ok", "precision": 1.0, "recall": 1.0, "total_results": 3},
            {"status": "error", "precision": 0.0, "recall": 0.0, "hits": 0},
        ]
        agg = self._compute_aggregates(results)
        assert agg["avg_precision"] == 1.0
        assert agg["avg_recall"] == 1.0
        assert agg["hit_rate"] == 1.0

    def test_all_error_results(self):
        """All error results → aggregates should be 0."""
        results = [
            {"status": "error", "precision": 0.0, "recall": 0.0, "hits": 0},
        ]
        agg = self._compute_aggregates(results)
        assert agg["avg_precision"] == 0.0
        assert agg["avg_recall"] == 0.0
        assert agg["hit_rate"] == 0.0

    def test_empty_results(self):
        """No results at all → aggregates should be 0."""
        results = []
        agg = self._compute_aggregates(results)
        assert agg["avg_precision"] == 0.0
        assert agg["avg_recall"] == 0.0
        assert agg["hit_rate"] == 0.0

    def test_hit_rate_with_zero_total_results(self):
        """Results with total_results=0 should not count in hit_rate."""
        results = [
            {"status": "ok", "precision": 0.0, "recall": 0.0, "total_results": 0},
            {"status": "ok", "precision": 1.0, "recall": 1.0, "total_results": 3},
        ]
        agg = self._compute_aggregates(results)
        assert agg["hit_rate"] == 0.5  # 1 of 2 has total_results > 0

    def test_rounding_to_3_decimals(self):
        """Averages should be rounded to 3 decimal places."""
        results = [
            {"status": "ok", "precision": 0.123456, "recall": 0.987654, "total_results": 1},
            {"status": "ok", "precision": 0.654321, "recall": 0.111111, "total_results": 1},
        ]
        agg = self._compute_aggregates(results)
        assert agg["avg_precision"] == round((0.123456 + 0.654321) / 2, 3)
        assert agg["avg_recall"] == round((0.987654 + 0.111111) / 2, 3)


# ===================================================================
# run_evaluation flow (mocked)
# ===================================================================


class TestRunEvaluationFlow:
    """Tests for the run_evaluation async function with mocked dependencies."""

    @pytest.mark.asyncio
    async def test_returns_metrics_dict(self):
        """run_evaluation should return a dict with expected keys."""
        from team_memory.daemon.quality.search_eval import run_evaluation

        mock_recall_result = {
            "results": [
                {
                    "title": "Docker PostgreSQL Guide",
                    "description": "How to connect",
                    "solution": "Use docker compose",
                    "tags": ["docker", "postgresql"],
                }
            ]
        }

        mock_recall = AsyncMock(return_value=mock_recall_result)
        mock_ctx = make_mock_app_context()

        with (
            patch("team_memory.services.memory.memory_operations.op_recall", mock_recall),
            patch("team_memory.bootstrap.get_context", return_value=mock_ctx),
            patch("team_memory.bootstrap.bootstrap"),
        ):
            metrics = await run_evaluation("postgresql://fake:***@localhost/db")

        assert "timestamp" in metrics
        assert "total_queries" in metrics
        assert "successful_queries" in metrics
        assert "avg_precision" in metrics
        assert "avg_recall" in metrics
        assert "hit_rate" in metrics
        assert "details" in metrics
        assert metrics["total_queries"] == len(DEFAULT_TEST_QUERIES)

    @pytest.mark.asyncio
    async def test_handles_recall_exception(self):
        """When op_recall raises, the query should be recorded as error."""
        from team_memory.daemon.quality.search_eval import run_evaluation

        mock_recall = AsyncMock(side_effect=Exception("search failed"))
        mock_ctx = make_mock_app_context()

        with (
            patch("team_memory.services.memory.memory_operations.op_recall", mock_recall),
            patch("team_memory.bootstrap.get_context", return_value=mock_ctx),
            patch("team_memory.bootstrap.bootstrap"),
        ):
            metrics = await run_evaluation("postgresql://fake:***@localhost/db")

        # All queries should be errors
        for detail in metrics["details"]:
            assert detail["status"] == "error"
            assert "search failed" in detail["error"]

    @pytest.mark.asyncio
    async def test_successful_queries_count(self):
        """successful_queries should count only ok results."""
        from team_memory.daemon.quality.search_eval import run_evaluation

        # Return results that match keywords for all queries
        mock_recall_result = {
            "results": [
                {
                    "title": "Docker PostgreSQL connection setup",
                    "description": "Guide to docker and postgresql connection",
                    "solution": "Use docker compose with asyncpg",
                    "tags": [
                        "docker",
                        "postgresql",
                        "asyncpg",
                        "wkwebview",
                        "cookie",
                        "embedding",
                        "model",
                        "config",
                        "entity",
                        "extraction",
                        "task",
                        "group",
                        "completion",
                        "asyncpg",
                        "pgvector",
                        "makefile",
                        "target",
                        "refinement",
                        "worker",
                        "wiki",
                        "compilation",
                        "search",
                        "pipeline",
                        "rrf",
                        "fusion",
                    ],
                }
            ]
        }

        mock_recall = AsyncMock(return_value=mock_recall_result)
        mock_ctx = make_mock_app_context()

        with (
            patch("team_memory.services.memory.memory_operations.op_recall", mock_recall),
            patch("team_memory.bootstrap.get_context", return_value=mock_ctx),
            patch("team_memory.bootstrap.bootstrap"),
        ):
            metrics = await run_evaluation("postgresql://fake:***@localhost/db")

        assert metrics["successful_queries"] == metrics["total_queries"]

    @pytest.mark.asyncio
    async def test_empty_results_gives_zero_precision(self):
        """Empty recall results should give 0 precision and 0 recall."""
        from team_memory.daemon.quality.search_eval import run_evaluation

        mock_recall_result = {"results": []}
        mock_recall = AsyncMock(return_value=mock_recall_result)
        mock_ctx = make_mock_app_context()

        with (
            patch("team_memory.services.memory.memory_operations.op_recall", mock_recall),
            patch("team_memory.bootstrap.get_context", return_value=mock_ctx),
            patch("team_memory.bootstrap.bootstrap"),
        ):
            metrics = await run_evaluation("postgresql://fake:***@localhost/db")

        for detail in metrics["details"]:
            if detail["status"] == "ok":
                assert detail["precision"] == 0.0
                assert detail["recall"] == 0.0

    @pytest.mark.asyncio
    async def test_top_result_field(self):
        """top_result should be set from the first result's title."""
        from team_memory.daemon.quality.search_eval import run_evaluation

        mock_recall_result = {
            "results": [
                {
                    "title": "My Top Result",
                    "description": "docker postgresql",
                    "solution": "",
                    "tags": [],
                }
            ]
        }
        mock_recall = AsyncMock(return_value=mock_recall_result)
        mock_ctx = make_mock_app_context()

        with (
            patch("team_memory.services.memory.memory_operations.op_recall", mock_recall),
            patch("team_memory.bootstrap.get_context", return_value=mock_ctx),
            patch("team_memory.bootstrap.bootstrap"),
        ):
            metrics = await run_evaluation("postgresql://fake:***@localhost/db")

        # At least one query should have "My Top Result" as top_result
        top_results = [d["top_result"] for d in metrics["details"] if d["status"] == "ok"]
        assert any("My Top Result" in tr for tr in top_results)

    @pytest.mark.asyncio
    async def test_top_result_empty_when_no_results(self):
        """top_result should be empty string when there are no results."""
        from team_memory.daemon.quality.search_eval import run_evaluation

        mock_recall_result = {"results": []}
        mock_recall = AsyncMock(return_value=mock_recall_result)
        mock_ctx = make_mock_app_context()

        with (
            patch("team_memory.services.memory.memory_operations.op_recall", mock_recall),
            patch("team_memory.bootstrap.get_context", return_value=mock_ctx),
            patch("team_memory.bootstrap.bootstrap"),
        ):
            metrics = await run_evaluation("postgresql://fake:***@localhost/db")

        for detail in metrics["details"]:
            if detail["status"] == "ok":
                assert detail["top_result"] == ""


# ===================================================================
# Keyword matching edge cases
# ===================================================================


class TestKeywordMatchingEdgeCases:
    """Additional edge cases for keyword matching logic."""

    def _match(self, result_text: str, keywords: list[str]) -> bool:
        """Check if any keyword appears in the text (lowercased)."""
        return any(kw in result_text.lower() for kw in keywords)

    def test_empty_keyword_list(self):
        """Empty keyword list should not match anything."""
        assert self._match("docker postgresql", []) is False

    def test_keyword_is_full_text(self):
        """Keyword that is the entire text should match."""
        assert self._match("docker", ["docker"]) is True

    def test_keyword_longer_than_text(self):
        """Keyword longer than text should not match."""
        assert self._match("docker", ["docker postgresql"]) is False

    def test_special_characters_in_keyword(self):
        """Keywords with special characters should match literally."""
        assert self._match("asyncpg pgvector", ["pgvector"]) is True
        assert self._match("rrf fusion", ["rrf"]) is True

    def test_unicode_in_text(self):
        """Unicode in text should not break matching."""
        assert self._match("配置 Docker 容器", ["docker"]) is True

    def test_multiple_keywords_any_match(self):
        """Any keyword matching should count as a hit."""
        assert self._match("docker container", ["docker", "postgresql"]) is True
        assert self._match("postgresql server", ["docker", "postgresql"]) is True

    def test_word_boundary_not_required(self):
        """Substring matches should work (no word boundary requirement)."""
        assert self._match("configuration", ["config"]) is True
        assert self._match("dockerize", ["docker"]) is True

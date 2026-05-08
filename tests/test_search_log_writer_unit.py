"""Comprehensive unit tests for scripts/daemon/search_log_writer.py

Tests the SearchLogWriter class with mocked asyncpg to avoid DB dependency.
Covers: marker parsing, fuzzy matching logic, pool management, all async methods,
error handling paths, granularity selection, result_ids parsing, and more.

Uses @pytest.mark.unit for pure-logic and mocked-DB tests.
Uses @pytest.mark.integration for tests that hit a real PostgreSQL.
"""
from __future__ import annotations

import json
import re
import uuid
from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from daemon.search_log_writer import SearchLogWriter

pytestmark = pytest.mark.unit


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_writer(dsn: str = "postgresql://test:***@localhost:5433/team_memory") -> SearchLogWriter:
    return SearchLogWriter(dsn=dsn)


def _mock_pool(rows_fetch=None, row_fetchrow=None, execute_result=None):
    """Build a fully mocked asyncpg pool that the SearchLogWriter methods use."""
    pool = AsyncMock()
    pool._closed = False

    conn = AsyncMock()

    # fetch → returns a list of rows
    if rows_fetch is not None:
        conn.fetch = AsyncMock(return_value=rows_fetch)
    else:
        conn.fetch = AsyncMock(return_value=[])

    # fetchrow → returns a single row or None
    if row_fetchrow is not None:
        conn.fetchrow = AsyncMock(return_value=row_fetchrow)
    else:
        conn.fetchrow = AsyncMock(return_value=None)

    # execute → returns some result
    conn.execute = AsyncMock(return_value=execute_result or "INSERT 0 1")

    # acquire is an async context manager
    pool.acquire = MagicMock()
    pool.acquire.return_value.__aenter__ = AsyncMock(return_value=conn)
    pool.acquire.return_value.__aexit__ = AsyncMock(return_value=False)

    # create_pool
    pool.create_pool = AsyncMock(return_value=pool)

    return pool


# ===================================================================
# _MEM_MARKER_RE — additional edge-case tests
# ===================================================================


class TestSearchLogWriterConstructor:
    """Tests for constructor defaults and attribute storage."""

    def test_default_dsn_format(self):
        w = SearchLogWriter()
        assert w._dsn.startswith("postgresql://")
        assert "5433" in w._dsn

    def test_custom_dsn_stored(self):
        dsn = "postgresql://x:y@other:9999/testdb"
        w = SearchLogWriter(dsn=dsn)
        assert w._dsn == dsn

    def test_pool_initially_none(self):
        w = SearchLogWriter()
        assert w._pool is None

    def test_default_project_in_log_search(self):
        """Check that default arg is 'default' for project in log_search."""
        import inspect
        sig = inspect.signature(SearchLogWriter.log_search)
        assert sig.parameters["project"].default == "default"

    def test_default_source_in_log_search(self):
        import inspect
        sig = inspect.signature(SearchLogWriter.log_search)
        assert sig.parameters["source"].default == "daemon"

    def test_default_threshold_fuzzy(self):
        import inspect
        sig = inspect.signature(SearchLogWriter.mark_used_fuzzy)
        assert sig.parameters["threshold"].default == 0.3


# ===================================================================
# _ensure_pool
# ===================================================================

class TestEnsurePool:
    """Tests for pool creation and reuse."""

    @pytest.mark.asyncio
    async def test_reuses_existing_open_pool(self):
        w = _make_writer()
        existing = MagicMock()
        existing._closed = False
        w._pool = existing
        pool = await w._ensure_pool()
        assert pool is existing


class TestClose:
    """Tests for the close() method."""

    @pytest.mark.asyncio
    async def test_close_when_pool_is_none(self):
        """close() should not raise when pool is None."""
        w = _make_writer()
        await w.close()  # should not raise

    @pytest.mark.asyncio
    async def test_close_when_pool_closed(self):
        """close() should not try to close an already-closed pool."""
        w = _make_writer()
        pool = MagicMock()
        pool._closed = True
        w._pool = pool
        await w.close()  # should not call close on pool

    @pytest.mark.asyncio
    async def test_close_when_pool_open(self):
        """close() should call pool.close() on an open pool."""
        w = _make_writer()
        pool = AsyncMock()
        pool._closed = False
        w._pool = pool
        await w.close()
        pool.close.assert_awaited_once()


# ===================================================================
# log_search
# ===================================================================

class TestLogSearch:
    """Tests for log_search method with mocked DB."""

    @pytest.mark.asyncio
    async def test_success_returns_uuid_string(self):
        w = _make_writer()
        mock_pool = _mock_pool()
        w._pool = mock_pool

        result = await w.log_search(query="test query")
        assert result is not None
        # result should be a uuid string
        uuid.UUID(result)  # raises if invalid

    @pytest.mark.asyncio
    async def test_success_with_result_ids(self):
        w = _make_writer()
        mock_pool = _mock_pool()
        w._pool = mock_pool

        rids = [{"id": "abc123"}, {"id": "def456"}]
        result = await w.log_search(query="q", result_ids=rids)
        assert result is not None
        uuid.UUID(result)

    @pytest.mark.asyncio
    async def test_success_with_none_result_ids(self):
        w = _make_writer()
        mock_pool = _mock_pool()
        w._pool = mock_pool

        result = await w.log_search(query="q", result_ids=None)
        assert result is not None

    @pytest.mark.asyncio
    async def test_returns_none_on_exception(self):
        w = _make_writer()
        pool = AsyncMock()
        pool._closed = False
        conn = AsyncMock()
        conn.execute = AsyncMock(side_effect=Exception("DB down"))
        pool.acquire = MagicMock()
        pool.acquire.return_value.__aenter__ = AsyncMock(return_value=conn)
        pool.acquire.return_value.__aexit__ = AsyncMock(return_value=False)
        w._pool = pool

        result = await w.log_search(query="q")
        assert result is None

    @pytest.mark.asyncio
    async def test_query_truncated_to_500(self):
        """Query should be truncated to 500 chars in the INSERT."""
        w = _make_writer()
        mock_pool = _mock_pool()
        w._pool = mock_pool

        long_query = "a" * 1000
        await w.log_search(query=long_query)

        conn = mock_pool.acquire.return_value.__aenter__.return_value
        call_args = conn.execute.call_args
        # $2 is the query arg (second positional)
        assert len(call_args[0][2]) <= 500

    @pytest.mark.asyncio
    async def test_project_and_source_passed_through(self):
        w = _make_writer()
        mock_pool = _mock_pool()
        w._pool = mock_pool

        await w.log_search(query="q", project="myproj", source="test_src")
        conn = mock_pool.acquire.return_value.__aenter__.return_value
        call_args = conn.execute.call_args
        # project is $3, source is $4
        assert call_args[0][3] == "myproj"
        assert call_args[0][4] == "test_src"


# ===================================================================
# ===================================================================


class TestMarkUsedFuzzy:
    """Tests for fuzzy matching logic with mocked DB."""

    @pytest.mark.asyncio
    async def test_returns_0_when_no_rows(self):
        w = _make_writer()
        mock_pool = _mock_pool(rows_fetch=[])
        w._pool = mock_pool

        result = await w.mark_used_fuzzy("some response")
        assert result == 0

    @pytest.mark.asyncio
    async def test_returns_0_on_exception(self):
        w = _make_writer()
        pool = AsyncMock()
        pool._closed = False
        pool.acquire = MagicMock(side_effect=Exception("fail"))
        w._pool = pool

        result = await w.mark_used_fuzzy("response")
        assert result == 0

    @pytest.mark.asyncio
    async def test_marks_when_keyword_overlap_exceeds_threshold(self):
        """When most keywords from experience appear in response."""
        exp_id = str(uuid.uuid4())
        # Experience text: all these words should appear in response
        exp_text = "docker postgresql connection setup guide"

        row = {"id": uuid.uuid4(), "result_ids": [{"id": exp_id}]}
        exp_row = {"description": exp_text, "solution": None}

        mock_pool = _mock_pool(rows_fetch=[row], row_fetchrow=exp_row)
        w = _make_writer()
        w._pool = mock_pool

        # Response contains all keywords → overlap ratio = 1.0 > 0.8
        result = await w.mark_used_fuzzy(
            "Docker PostgreSQL connection setup is important for the guide"
        )
        assert result == 1

    @pytest.mark.asyncio
    async def test_does_not_mark_when_overlap_below_threshold(self):
        """When few keywords match, overlap < threshold → no mark."""
        exp_id = str(uuid.uuid4())
        exp_text = "docker postgresql connection setup guide configuration"

        row = {"id": uuid.uuid4(), "result_ids": [{"id": exp_id}]}
        exp_row = {"description": exp_text, "solution": None}

        mock_pool = _mock_pool(rows_fetch=[row], row_fetchrow=exp_row)
        w = _make_writer()
        w._pool = mock_pool

        # Response only matches 1 out of 6 keywords → ~0.17 < 0.8
        result = await w.mark_used_fuzzy("I like docker very much today")
        assert result == 0

    @pytest.mark.asyncio
    async def test_uses_solution_over_description(self):
        """When solution is non-null, it should be used instead of description."""
        exp_id = str(uuid.uuid4())
        row = {"id": uuid.uuid4(), "result_ids": [{"id": exp_id}]}
        # solution takes priority
        exp_row = {"description": "unrelated", "solution": "docker postgresql setup"}

        mock_pool = _mock_pool(rows_fetch=[row], row_fetchrow=exp_row)
        w = _make_writer()
        w._pool = mock_pool

        result = await w.mark_used_fuzzy(
            "Here is the docker postgresql setup for you"
        )
        assert result == 1

    @pytest.mark.asyncio
    async def test_skips_row_when_result_ids_empty(self):
        row = {"id": uuid.uuid4(), "result_ids": []}
        mock_pool = _mock_pool(rows_fetch=[row])
        w = _make_writer()
        w._pool = mock_pool

        result = await w.mark_used_fuzzy("response text")
        assert result == 0

    @pytest.mark.asyncio
    async def test_skips_invalid_uuid(self):
        row = {"id": uuid.uuid4(), "result_ids": [{"id": "not-a-uuid"}]}
        mock_pool = _mock_pool(rows_fetch=[row])
        w = _make_writer()
        w._pool = mock_pool

        result = await w.mark_used_fuzzy("response text")
        assert result == 0

    @pytest.mark.asyncio
    async def test_skips_empty_id_in_result(self):
        row = {"id": uuid.uuid4(), "result_ids": [{"id": ""}]}
        mock_pool = _mock_pool(rows_fetch=[row])
        w = _make_writer()
        w._pool = mock_pool

        result = await w.mark_used_fuzzy("response text")
        assert result == 0

    @pytest.mark.asyncio
    async def test_skips_missing_id_key(self):
        row = {"id": uuid.uuid4(), "result_ids": [{"score": 0.9}]}
        mock_pool = _mock_pool(rows_fetch=[row])
        w = _make_writer()
        w._pool = mock_pool

        result = await w.mark_used_fuzzy("response text")
        assert result == 0

    @pytest.mark.asyncio
    async def test_skips_when_experience_not_found(self):
        exp_id = str(uuid.uuid4())
        row = {"id": uuid.uuid4(), "result_ids": [{"id": exp_id}]}
        # exp_row is None → experience not in DB
        mock_pool = _mock_pool(rows_fetch=[row], row_fetchrow=None)
        w = _make_writer()
        w._pool = mock_pool

        result = await w.mark_used_fuzzy("response text")
        assert result == 0

    @pytest.mark.asyncio
    async def test_skips_when_text_empty(self):
        exp_id = str(uuid.uuid4())
        row = {"id": uuid.uuid4(), "result_ids": [{"id": exp_id}]}
        exp_row = {"description": "", "solution": ""}
        mock_pool = _mock_pool(rows_fetch=[row], row_fetchrow=exp_row)
        w = _make_writer()
        w._pool = mock_pool

        result = await w.mark_used_fuzzy("response text")
        assert result == 0

    @pytest.mark.asyncio
    async def test_skips_when_keywords_all_short(self):
        """If all tokens in experience text are <= 2 chars, no keywords."""
        exp_id = str(uuid.uuid4())
        row = {"id": uuid.uuid4(), "result_ids": [{"id": exp_id}]}
        exp_row = {"description": "a b c d", "solution": None}
        mock_pool = _mock_pool(rows_fetch=[row], row_fetchrow=exp_row)
        w = _make_writer()
        w._pool = mock_pool

        result = await w.mark_used_fuzzy("response with a b c d words")
        assert result == 0

    @pytest.mark.asyncio
    async def test_parsing_string_result_ids(self):
        exp_id = str(uuid.uuid4())
        row = {"id": uuid.uuid4(), "result_ids": json.dumps([{"id": exp_id}])}
        exp_row = {"description": "docker postgresql setup", "solution": None}
        mock_pool = _mock_pool(rows_fetch=[row], row_fetchrow=exp_row)
        w = _make_writer()
        w._pool = mock_pool

        result = await w.mark_used_fuzzy("docker postgresql setup here")
        assert result == 1

    @pytest.mark.asyncio
    async def test_only_one_fuzzy_hit_per_search_log(self):
        """Even if multiple experiences match, only 1 mark per search_log (break)."""
        exp1 = str(uuid.uuid4())
        exp2 = str(uuid.uuid4())
        row = {
            "id": uuid.uuid4(),
            "result_ids": [{"id": exp1}, {"id": exp2}],
        }

        def fetchrow_side_effect(query, *args):
            # Both experiences return high-overlap text
            return {"description": "docker postgresql setup", "solution": None}

        conn = AsyncMock()
        conn.fetch = AsyncMock(return_value=[row])
        conn.fetchrow = AsyncMock(side_effect=fetchrow_side_effect)
        conn.execute = AsyncMock()

        pool = AsyncMock()
        pool._closed = False
        pool.acquire = MagicMock()
        pool.acquire.return_value.__aenter__ = AsyncMock(return_value=conn)
        pool.acquire.return_value.__aexit__ = AsyncMock(return_value=False)

        w = _make_writer()
        w._pool = pool

        result = await w.mark_used_fuzzy(
            "docker postgresql setup is great"
        )
        # Only 1 mark even though both experiences match
        assert result == 1

    @pytest.mark.asyncio
    async def test_passes_custom_threshold(self):
        """Verify threshold parameter is used."""
        exp_id = str(uuid.uuid4())
        row = {"id": uuid.uuid4(), "result_ids": [{"id": exp_id}]}
        # 6 keywords: docker postgresql connection setup guide config
        exp_row = {"description": "docker postgresql connection setup guide config", "solution": None}
        mock_pool = _mock_pool(rows_fetch=[row], row_fetchrow=exp_row)
        w = _make_writer()
        w._pool = mock_pool

        # Response matches 4/6 = 0.667, threshold=0.5 → should match
        result = await w.mark_used_fuzzy(
            "docker postgresql connection setup works",
            threshold=0.5,
        )
        assert result == 1


# ===================================================================
# get_recent_with_results
# ===================================================================

class TestGetRecentWithResults:

    @pytest.mark.asyncio
    async def test_returns_none_when_no_rows(self):
        w = _make_writer()
        mock_pool = _mock_pool(row_fetchrow=None)
        w._pool = mock_pool

        result = await w.get_recent_with_results()
        assert result is None

    @pytest.mark.asyncio
    async def test_returns_none_on_exception(self):
        w = _make_writer()
        pool = AsyncMock()
        pool._closed = False
        pool.acquire = MagicMock(side_effect=Exception("fail"))
        w._pool = pool

        result = await w.get_recent_with_results()
        assert result is None

    @pytest.mark.asyncio
    async def test_returns_dict_with_project_row(self):
        row = {
            "id": str(uuid.uuid4()),
            "query": "test query",
            "result_ids": [{"id": "abc"}],
        }
        mock_pool = _mock_pool(row_fetchrow=row)
        w = _make_writer()
        w._pool = mock_pool

        result = await w.get_recent_with_results(project="myproj")
        assert result is not None
        assert result["query"] == "test query"
        assert isinstance(result["result_ids"], list)

    @pytest.mark.asyncio
    async def test_parses_string_result_ids(self):
        row = {
            "id": str(uuid.uuid4()),
            "query": "q",
            "result_ids": json.dumps([{"id": "x"}]),
        }
        mock_pool = _mock_pool(row_fetchrow=row)
        w = _make_writer()
        w._pool = mock_pool

        result = await w.get_recent_with_results()
        assert isinstance(result["result_ids"], list)

    @pytest.mark.asyncio
    async def test_falls_back_to_any_project(self):
        """When project-specific query returns None, fallback runs."""
        call_count = 0

        async def fetchrow_fn(query, *args):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return None  # project-specific → None
            return {"id": "123", "query": "fallback", "result_ids": "[]"}

        conn = AsyncMock()
        conn.fetchrow = AsyncMock(side_effect=fetchrow_fn)
        pool = AsyncMock()
        pool._closed = False
        pool.acquire = MagicMock()
        pool.acquire.return_value.__aenter__ = AsyncMock(return_value=conn)
        pool.acquire.return_value.__aexit__ = AsyncMock(return_value=False)

        w = _make_writer()
        w._pool = pool

        result = await w.get_recent_with_results(project="myproj")
        assert result is not None
        assert result["query"] == "fallback"


# ===================================================================
# get_stats
# ===================================================================

class TestGetStats:

    @pytest.mark.asyncio
    async def test_returns_stats_dict(self):
        row = {
            "total": 100,
            "hit": 80,
            "used": 50,
            "unjudged": 20,
            "used_fuzzy": 20,
            "used_faithfulness": 30,
        }
        project_rows = [
            {"project": "default", "total": 60, "used": 30},
            {"project": "teamA", "total": 40, "used": 20},
        ]
        top_q = [{"query": "docker setup", "cnt": 5}]

        conn = AsyncMock()
        # First call: main stats, second: per-project, third: top queries
        conn.fetchrow = AsyncMock(return_value=row)
        conn.fetch = AsyncMock(return_value=project_rows)
        # Need to distinguish between project query and top_queries query
        call_count = 0
        original_fetch = conn.fetch

        async def smart_fetch(query, *args):
            nonlocal call_count
            call_count += 1
            if "GROUP BY project" in query:
                return project_rows
            return top_q

        conn.fetch = AsyncMock(side_effect=smart_fetch)

        pool = AsyncMock()
        pool._closed = False
        pool.acquire = MagicMock()
        pool.acquire.return_value.__aenter__ = AsyncMock(return_value=conn)
        pool.acquire.return_value.__aexit__ = AsyncMock(return_value=False)

        w = _make_writer()
        w._pool = pool

        result = await w.get_stats(days=7)
        assert result["total_searches"] == 100
        assert result["hit"] == 80
        assert result["used"] == 50
        assert result["unjudged"] == 20
        assert result["used_fuzzy"] == 20
        assert result["use_rate"] == 0.625  # 50/80
        assert "default" in result["by_project"]
        assert len(result["top_queries"]) == 1

    @pytest.mark.asyncio
    async def test_use_rate_zero_when_no_hits(self):
        row = {
            "total": 10,
            "hit": 0,
            "used": 0,
            "unjudged": 0,
            "used_fuzzy": 0,
            "used_faithfulness": 0,
        }
        conn = AsyncMock()
        conn.fetchrow = AsyncMock(return_value=row)
        conn.fetch = AsyncMock(return_value=[])

        pool = AsyncMock()
        pool._closed = False
        pool.acquire = MagicMock()
        pool.acquire.return_value.__aenter__ = AsyncMock(return_value=conn)
        pool.acquire.return_value.__aexit__ = AsyncMock(return_value=False)

        w = _make_writer()
        w._pool = pool

        result = await w.get_stats(days=7)
        assert result["use_rate"] == 0.0

    @pytest.mark.asyncio
    async def test_handles_none_counts(self):
        """When DB returns None for counts, they should default to 0."""
        row = {
            "total": None,
            "hit": None,
            "used": None,
            "unjudged": None,
            "used_fuzzy": None,
            "used_faithfulness": None,
        }
        conn = AsyncMock()
        conn.fetchrow = AsyncMock(return_value=row)
        conn.fetch = AsyncMock(return_value=[])

        pool = AsyncMock()
        pool._closed = False
        pool.acquire = MagicMock()
        pool.acquire.return_value.__aenter__ = AsyncMock(return_value=conn)
        pool.acquire.return_value.__aexit__ = AsyncMock(return_value=False)

        w = _make_writer()
        w._pool = pool

        result = await w.get_stats()
        assert result["total_searches"] == 0
        assert result["use_rate"] == 0.0

    @pytest.mark.asyncio
    async def test_returns_error_on_exception(self):
        w = _make_writer()
        pool = AsyncMock()
        pool._closed = False
        pool.acquire = MagicMock(side_effect=Exception("DB error"))
        w._pool = pool

        result = await w.get_stats()
        assert "error" in result


# ===================================================================
# get_trends
# ===================================================================

class TestGetTrends:

    @pytest.mark.asyncio
    async def test_returns_empty_list_when_no_data(self):
        w = _make_writer()
        mock_pool = _mock_pool(rows_fetch=[])
        w._pool = mock_pool

        result = await w.get_trends()
        assert result == []

    @pytest.mark.asyncio
    async def test_returns_error_on_exception(self):
        w = _make_writer()
        pool = AsyncMock()
        pool._closed = False
        pool.acquire = MagicMock(side_effect=Exception("fail"))
        w._pool = pool

        result = await w.get_trends()
        assert len(result) == 1
        assert "error" in result[0]

    @pytest.mark.asyncio
    async def test_truncation_day(self):
        """Default granularity 'day' should pass 'day' to date_trunc."""
        conn = AsyncMock()
        conn.fetch = AsyncMock(return_value=[])

        pool = AsyncMock()
        pool._closed = False
        pool.acquire = MagicMock()
        pool.acquire.return_value.__aenter__ = AsyncMock(return_value=conn)
        pool.acquire.return_value.__aexit__ = AsyncMock(return_value=False)

        w = _make_writer()
        w._pool = pool

        await w.get_trends(granularity="day")
        call_args = conn.fetch.call_args
        assert call_args[0][1] == "day"

    @pytest.mark.asyncio
    async def test_truncation_hour(self):
        conn = AsyncMock()
        conn.fetch = AsyncMock(return_value=[])

        pool = AsyncMock()
        pool._closed = False
        pool.acquire = MagicMock()
        pool.acquire.return_value.__aenter__ = AsyncMock(return_value=conn)
        pool.acquire.return_value.__aexit__ = AsyncMock(return_value=False)

        w = _make_writer()
        w._pool = pool

        await w.get_trends(granularity="hour")
        call_args = conn.fetch.call_args
        assert call_args[0][1] == "hour"

    @pytest.mark.asyncio
    async def test_truncation_week(self):
        conn = AsyncMock()
        conn.fetch = AsyncMock(return_value=[])

        pool = AsyncMock()
        pool._closed = False
        pool.acquire = MagicMock()
        pool.acquire.return_value.__aenter__ = AsyncMock(return_value=conn)
        pool.acquire.return_value.__aexit__ = AsyncMock(return_value=False)

        w = _make_writer()
        w._pool = pool

        await w.get_trends(granularity="week")
        call_args = conn.fetch.call_args
        assert call_args[0][1] == "week"

    @pytest.mark.asyncio
    async def test_truncation_unknown_defaults_to_day(self):
        conn = AsyncMock()
        conn.fetch = AsyncMock(return_value=[])

        pool = AsyncMock()
        pool._closed = False
        pool.acquire = MagicMock()
        pool.acquire.return_value.__aenter__ = AsyncMock(return_value=conn)
        pool.acquire.return_value.__aexit__ = AsyncMock(return_value=False)

        w = _make_writer()
        w._pool = pool

        await w.get_trends(granularity="month")
        call_args = conn.fetch.call_args
        assert call_args[0][1] == "day"

    @pytest.mark.asyncio
    async def test_returns_formatted_row(self):
        now = datetime.now(timezone.utc)
        row = {
            "period": now,
            "total": 10,
            "hit": 8,
            "used": 5,
            "used_fuzzy": 2,
            "used_faithfulness": 3,
        }
        conn = AsyncMock()
        conn.fetch = AsyncMock(return_value=[row])

        pool = AsyncMock()
        pool._closed = False
        pool.acquire = MagicMock()
        pool.acquire.return_value.__aenter__ = AsyncMock(return_value=conn)
        pool.acquire.return_value.__aexit__ = AsyncMock(return_value=False)

        w = _make_writer()
        w._pool = pool

        result = await w.get_trends()
        assert len(result) == 1
        assert result[0]["total"] == 10
        assert result[0]["hit"] == 8
        assert result[0]["used"] == 5
        assert result[0]["use_rate"] == 0.625  # 5/8
        assert result[0]["period"] is not None

    @pytest.mark.asyncio
    async def test_handles_none_period(self):
        row = {
            "period": None,
            "total": 5,
            "hit": 3,
            "used": 1,
            "used_fuzzy": 0,
            "used_faithfulness": 0,
        }
        conn = AsyncMock()
        conn.fetch = AsyncMock(return_value=[row])

        pool = AsyncMock()
        pool._closed = False
        pool.acquire = MagicMock()
        pool.acquire.return_value.__aenter__ = AsyncMock(return_value=conn)
        pool.acquire.return_value.__aexit__ = AsyncMock(return_value=False)

        w = _make_writer()
        w._pool = pool

        result = await w.get_trends()
        assert result[0]["period"] is None


# ===================================================================
# get_miss_queries
# ===================================================================

class TestGetMissQueries:

    @pytest.mark.asyncio
    async def test_returns_empty_when_no_misses(self):
        w = _make_writer()
        mock_pool = _mock_pool(rows_fetch=[])
        w._pool = mock_pool

        result = await w.get_miss_queries()
        assert result == []

    @pytest.mark.asyncio
    async def test_returns_error_on_exception(self):
        w = _make_writer()
        pool = AsyncMock()
        pool._closed = False
        pool.acquire = MagicMock(side_effect=Exception("fail"))
        w._pool = pool

        result = await w.get_miss_queries()
        assert len(result) == 1
        assert "error" in result[0]

    @pytest.mark.asyncio
    async def test_returns_formatted_results(self):
        now = datetime.now(timezone.utc)
        row = {"query": "how to deploy", "cnt": 3, "last_seen": now}
        w = _make_writer()
        mock_pool = _mock_pool(rows_fetch=[row])
        w._pool = mock_pool

        result = await w.get_miss_queries()
        assert len(result) == 1
        assert result[0]["query"] == "how to deploy"
        assert result[0]["count"] == 3
        assert result[0]["last_seen"] is not None

    @pytest.mark.asyncio
    async def test_truncates_long_query_to_80(self):
        long_q = "a" * 200
        row = {"query": long_q, "cnt": 1, "last_seen": None}
        w = _make_writer()
        mock_pool = _mock_pool(rows_fetch=[row])
        w._pool = mock_pool

        result = await w.get_miss_queries()
        assert len(result[0]["query"]) == 80

    @pytest.mark.asyncio
    async def test_handles_null_last_seen(self):
        row = {"query": "q", "cnt": 1, "last_seen": None}
        w = _make_writer()
        mock_pool = _mock_pool(rows_fetch=[row])
        w._pool = mock_pool

        result = await w.get_miss_queries()
        assert result[0]["last_seen"] is None


# ===================================================================
# get_low_use_queries
# ===================================================================

class TestGetLowUseQueries:

    @pytest.mark.asyncio
    async def test_returns_empty_when_none(self):
        w = _make_writer()
        mock_pool = _mock_pool(rows_fetch=[])
        w._pool = mock_pool

        result = await w.get_low_use_queries()
        assert result == []

    @pytest.mark.asyncio
    async def test_returns_error_on_exception(self):
        w = _make_writer()
        pool = AsyncMock()
        pool._closed = False
        pool.acquire = MagicMock(side_effect=Exception("fail"))
        w._pool = pool

        result = await w.get_low_use_queries()
        assert len(result) == 1
        assert "error" in result[0]

    @pytest.mark.asyncio
    async def test_returns_formatted_results(self):
        row = {"query": "setup k8s", "total": 10, "used": 0}
        w = _make_writer()
        mock_pool = _mock_pool(rows_fetch=[row])
        w._pool = mock_pool

        result = await w.get_low_use_queries()
        assert len(result) == 1
        assert result[0]["query"] == "setup k8s"
        assert result[0]["count"] == 10
        assert result[0]["use_rate"] == 0.0

    @pytest.mark.asyncio
    async def test_truncates_long_query_to_80(self):
        long_q = "b" * 200
        row = {"query": long_q, "total": 5, "used": 0}
        w = _make_writer()
        mock_pool = _mock_pool(rows_fetch=[row])
        w._pool = mock_pool

        result = await w.get_low_use_queries()
        assert len(result[0]["query"]) == 80


# ===================================================================
# Keyword extraction logic (testing the actual regex used in mark_used_fuzzy)
# ===================================================================

class TestKeywordExtraction:
    """Test the tokenization logic used in mark_used_fuzzy.

    In the source code:
        keywords = [
            tok.lower()
            for tok in re.split(r"[^\\w]+", text)
            if len(tok) > 2
        ]
    """

    def test_basic_extraction(self):
        text = "Docker PostgreSQL connection"
        keywords = [tok.lower() for tok in re.split(r"[^\w]+", text) if len(tok) > 2]
        assert keywords == ["docker", "postgresql", "connection"]

    def test_filters_short_tokens(self):
        text = "a bb ccc dd eee"
        keywords = [tok.lower() for tok in re.split(r"[^\w]+", text) if len(tok) > 2]
        assert "a" not in keywords
        assert "bb" not in keywords
        assert "ccc" in keywords
        assert "eee" in keywords

    def test_handles_punctuation(self):
        text = "hello, world! how-are you?"
        keywords = [tok.lower() for tok in re.split(r"[^\w]+", text) if len(tok) > 2]
        assert "hello" in keywords
        assert "world" in keywords
        assert "how" in keywords  # len("how") = 3 > 2
        assert "are" in keywords

    def test_empty_text(self):
        text = ""
        keywords = [tok.lower() for tok in re.split(r"[^\w]+", text) if len(tok) > 2]
        assert keywords == []

    def test_only_short_tokens(self):
        text = "a b c --"
        keywords = [tok.lower() for tok in re.split(r"[^\w]+", text) if len(tok) > 2]
        assert keywords == []

    def test_numbers_included(self):
        text = "port 8080 or 443"
        keywords = [tok.lower() for tok in re.split(r"[^\w]+", text) if len(tok) > 2]
        assert "8080" in keywords

    def test_unicode_split(self):
        text = "python 代码 测试 工具"
        keywords = [tok.lower() for tok in re.split(r"[^\w]+", text) if len(tok) > 2]
        # Chinese characters may or may not be word chars depending on regex engine
        assert isinstance(keywords, list)


# ===================================================================
# Overlap ratio calculation
# ===================================================================

class TestOverlapRatioCalculation:

    def test_full_overlap(self):
        keywords = ["docker", "postgresql", "setup"]
        response = "docker postgresql setup guide".lower()
        matched = sum(1 for kw in keywords if kw in response)
        ratio = matched / len(keywords)
        assert ratio == 1.0

    def test_no_overlap(self):
        keywords = ["docker", "postgresql"]
        response = "kubernetes helm chart deployment".lower()
        matched = sum(1 for kw in keywords if kw in response)
        ratio = matched / len(keywords)
        assert ratio == 0.0

    def test_partial_overlap(self):
        keywords = ["docker", "postgresql", "redis", "nginx"]
        response = "docker and postgresql are great".lower()
        matched = sum(1 for kw in keywords if kw in response)
        ratio = matched / len(keywords)
        assert ratio == 0.5

    def test_threshold_boundary_above(self):
        keywords = ["a", "b", "c", "d", "e"]  # all > 2 chars? No, need len > 2
        keywords = ["aaa", "bbb", "ccc", "ddd", "eee"]
        response = "aaa bbb ccc ddd eee".lower()
        matched = sum(1 for kw in keywords if kw in response)
        ratio = matched / len(keywords)
        assert ratio == 1.0
        assert ratio > 0.8

    def test_threshold_boundary_below(self):
        keywords = ["aaa", "bbb", "ccc", "ddd", "eee"]
        response = "aaa bbb ccc".lower()
        matched = sum(1 for kw in keywords if kw in response)
        ratio = matched / len(keywords)
        assert ratio == 0.6
        assert ratio < 0.8

    def test_exact_threshold_not_above(self):
        """Threshold comparison uses > not >=. At exactly 0.8 should not match."""
        keywords = ["aaa", "bbb", "ccc", "ddd", "eee"]
        response = "aaa bbb ccc ddd".lower()  # 4/5 = 0.8
        matched = sum(1 for kw in keywords if kw in response)
        ratio = matched / len(keywords)
        assert ratio == 0.8
        assert not (ratio > 0.8)  # > not >=


# ===================================================================
# Integration marker check (no actual DB needed here, just marker)
# ===================================================================

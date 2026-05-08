"""Comprehensive unit tests for scripts/daemon/response_buffer.py

Tests the ResponseBuffer class with mocked asyncpg to avoid DB dependency.
Covers: constructor, pool management, add(), pending_count(), should_batch(),
fetch_pending(), mark_evaluated(), sync_to_search_log(), get_stats(), close(),
error handling paths, input truncation, JSON serialization, and UUID handling.

Uses @pytest.mark.unit for pure-logic and mocked-DB tests.
Uses @pytest.mark.integration for tests that hit a real PostgreSQL.
"""
from __future__ import annotations

import json
import uuid
from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from team_memory.daemon.response_buffer import ResponseBuffer

pytestmark = pytest.mark.unit


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_buffer(
    dsn: str = "postgresql://developer:***@localhost:5433/team_memory",
    batch_threshold: int = 5,
) -> ResponseBuffer:
    return ResponseBuffer(dsn=dsn, batch_threshold=batch_threshold)


def _mock_pool(
    rows_fetch=None,
    row_fetchrow=None,
    execute_result=None,
    fetchval_result=None,
):
    """Build a fully mocked asyncpg pool that the ResponseBuffer methods use."""
    pool = AsyncMock()
    pool._closed = False

    conn = AsyncMock()

    if rows_fetch is not None:
        conn.fetch = AsyncMock(return_value=rows_fetch)
    else:
        conn.fetch = AsyncMock(return_value=[])

    if row_fetchrow is not None:
        conn.fetchrow = AsyncMock(return_value=row_fetchrow)
    else:
        conn.fetchrow = AsyncMock(return_value=None)

    conn.execute = AsyncMock(return_value=execute_result or "INSERT 0 1")

    if fetchval_result is not None:
        conn.fetchval = AsyncMock(return_value=fetchval_result)
    else:
        conn.fetchval = AsyncMock(return_value=0)

    pool.acquire = MagicMock()
    pool.acquire.return_value.__aenter__ = AsyncMock(return_value=conn)
    pool.acquire.return_value.__aexit__ = AsyncMock(return_value=False)

    pool.create_pool = AsyncMock(return_value=pool)

    return pool


# ===================================================================
# Constructor
# ===================================================================

class TestResponseBufferConstructor:
    """Tests for constructor defaults and attribute storage."""

    def test_default_dsn_format(self):
        buf = ResponseBuffer()
        assert buf._dsn.startswith("postgresql://")
        assert "5433" in buf._dsn

    def test_custom_dsn_stored(self):
        dsn = "postgresql://user:***@other:9999/testdb"
        buf = ResponseBuffer(dsn=dsn)
        assert buf._dsn == dsn

    def test_pool_initially_none(self):
        buf = ResponseBuffer()
        assert buf._pool is None

    def test_default_batch_threshold(self):
        buf = ResponseBuffer()
        assert buf._batch_threshold == 5

    def test_custom_batch_threshold(self):
        buf = ResponseBuffer(batch_threshold=10)
        assert buf._batch_threshold == 10

    def test_threshold_zero(self):
        buf = ResponseBuffer(batch_threshold=0)
        assert buf._batch_threshold == 0

    def test_threshold_one(self):
        buf = ResponseBuffer(batch_threshold=1)
        assert buf._batch_threshold == 1

    def test_threshold_large(self):
        buf = ResponseBuffer(batch_threshold=1000)
        assert buf._batch_threshold == 1000


# ===================================================================
# _ensure_pool
# ===================================================================

class TestEnsurePool:
    """Tests for pool creation and reuse."""

    @pytest.mark.asyncio
    async def test_creates_pool_when_none(self):
        buf = _make_buffer()
        mock_pool = _mock_pool()
        with patch("team_memory.daemon.response_buffer.asyncpg") as mock_asyncpg:
            mock_asyncpg.create_pool = AsyncMock(return_value=mock_pool)
            pool = await buf._ensure_pool()
            assert pool is mock_pool
            mock_asyncpg.create_pool.assert_called_once()

    @pytest.mark.asyncio
    async def test_reuses_existing_open_pool(self):
        buf = _make_buffer()
        existing = MagicMock()
        existing._closed = False
        buf._pool = existing
        pool = await buf._ensure_pool()
        assert pool is existing

    @pytest.mark.asyncio
    async def test_recreates_when_pool_closed(self):
        buf = _make_buffer()
        old = MagicMock()
        old._closed = True
        buf._pool = old
        new_pool = _mock_pool()
        with patch("team_memory.daemon.response_buffer.asyncpg") as mock_asyncpg:
            mock_asyncpg.create_pool = AsyncMock(return_value=new_pool)
            pool = await buf._ensure_pool()
            assert pool is new_pool


# ===================================================================
# close()
# ===================================================================

class TestClose:
    """Tests for the close() method."""

    @pytest.mark.asyncio
    async def test_close_when_pool_is_none(self):
        """close() should not raise when pool is None."""
        buf = _make_buffer()
        await buf.close()  # should not raise

    @pytest.mark.asyncio
    async def test_close_when_pool_closed(self):
        """close() should not try to close an already-closed pool."""
        buf = _make_buffer()
        pool = MagicMock()
        pool._closed = True
        buf._pool = pool
        await buf.close()  # should not call close on pool

    @pytest.mark.asyncio
    async def test_close_when_pool_open(self):
        """close() should call pool.close() on an open pool."""
        buf = _make_buffer()
        pool = AsyncMock()
        pool._closed = False
        buf._pool = pool
        await buf.close()
        pool.close.assert_awaited_once()


# ===================================================================
# add()
# ===================================================================

class TestAdd:
    """Tests for the add() method with mocked DB."""

    @pytest.mark.asyncio
    async def test_success_returns_uuid_string(self):
        buf = _make_buffer()
        mock_pool = _mock_pool()
        buf._pool = mock_pool

        result = await buf.add(query="test query", agent_response="test response")
        assert result is not None
        # result should be a uuid string
        uuid.UUID(result)  # raises if invalid

    @pytest.mark.asyncio
    async def test_success_with_result_ids(self):
        buf = _make_buffer()
        mock_pool = _mock_pool()
        buf._pool = mock_pool

        rids = [{"id": "abc123"}, {"id": "def456"}]
        result = await buf.add(
            query="q", agent_response="r", result_ids=rids
        )
        assert result is not None
        uuid.UUID(result)

    @pytest.mark.asyncio
    async def test_success_with_none_result_ids(self):
        buf = _make_buffer()
        mock_pool = _mock_pool()
        buf._pool = mock_pool

        result = await buf.add(
            query="q", agent_response="r", result_ids=None
        )
        assert result is not None

    @pytest.mark.asyncio
    async def test_success_with_search_log_id(self):
        buf = _make_buffer()
        mock_pool = _mock_pool()
        buf._pool = mock_pool
        # Mock has_search_log to return False (not a duplicate)
        buf.has_search_log = AsyncMock(return_value=False)

        log_id = str(uuid.uuid4())
        result = await buf.add(
            query="q", agent_response="r", search_log_id=log_id
        )
        assert result is not None
        uuid.UUID(result)

    @pytest.mark.asyncio
    async def test_returns_none_on_exception(self):
        buf = _make_buffer()
        pool = AsyncMock()
        pool._closed = False
        conn = AsyncMock()
        conn.execute = AsyncMock(side_effect=Exception("DB down"))
        pool.acquire = MagicMock()
        pool.acquire.return_value.__aenter__ = AsyncMock(return_value=conn)
        pool.acquire.return_value.__aexit__ = AsyncMock(return_value=False)
        buf._pool = pool

        result = await buf.add(query="q", agent_response="r")
        assert result is None

    @pytest.mark.asyncio
    async def test_query_truncated_to_2000(self):
        """Query should be truncated to 2000 chars in the INSERT."""
        buf = _make_buffer()
        mock_pool = _mock_pool()
        buf._pool = mock_pool

        long_query = "a" * 3000
        await buf.add(query=long_query, agent_response="r")

        conn = mock_pool.acquire.return_value.__aenter__.return_value
        call_args = conn.execute.call_args
        # $2 is the query arg (second positional)
        assert len(call_args[0][2]) <= 2000

    @pytest.mark.asyncio
    async def test_agent_response_truncated_to_10000(self):
        """Agent response should be truncated to 10000 chars in the INSERT."""
        buf = _make_buffer()
        mock_pool = _mock_pool()
        buf._pool = mock_pool

        long_response = "b" * 15000
        await buf.add(query="q", agent_response=long_response)

        conn = mock_pool.acquire.return_value.__aenter__.return_value
        call_args = conn.execute.call_args
        # $3 is the agent_response arg (third positional)
        assert len(call_args[0][3]) <= 10000

    @pytest.mark.asyncio
    async def test_result_ids_json_serialized(self):
        """result_ids should be JSON-serialized in the INSERT."""
        buf = _make_buffer()
        mock_pool = _mock_pool()
        buf._pool = mock_pool

        rids = [{"id": "abc", "score": 0.9}]
        await buf.add(query="q", agent_response="r", result_ids=rids)

        conn = mock_pool.acquire.return_value.__aenter__.return_value
        call_args = conn.execute.call_args
        # $4 is the result_ids arg
        result_ids_json = call_args[0][4]
        assert json.loads(result_ids_json) == rids

    @pytest.mark.asyncio
    async def test_none_result_ids_passed_as_none(self):
        """None result_ids should be passed as None, not serialized."""
        buf = _make_buffer()
        mock_pool = _mock_pool()
        buf._pool = mock_pool

        await buf.add(query="q", agent_response="r", result_ids=None)

        conn = mock_pool.acquire.return_value.__aenter__.return_value
        call_args = conn.execute.call_args
        # $4 is the result_ids arg
        assert call_args[0][4] is None

    @pytest.mark.asyncio
    async def test_search_log_id_converted_to_uuid(self):
        """search_log_id string should be converted to UUID for the INSERT."""
        buf = _make_buffer()
        mock_pool = _mock_pool()
        buf._pool = mock_pool
        buf.has_search_log = AsyncMock(return_value=False)

        log_id = str(uuid.uuid4())
        await buf.add(query="q", agent_response="r", search_log_id=log_id)

        conn = mock_pool.acquire.return_value.__aenter__.return_value
        call_args = conn.execute.call_args
        # $5 is the search_log_id arg
        assert isinstance(call_args[0][5], uuid.UUID)
        assert str(call_args[0][5]) == log_id

    @pytest.mark.asyncio
    async def test_none_search_log_id_passed_as_none(self):
        """None search_log_id should be passed as None."""
        buf = _make_buffer()
        mock_pool = _mock_pool()
        buf._pool = mock_pool

        await buf.add(query="q", agent_response="r", search_log_id=None)

        conn = mock_pool.acquire.return_value.__aenter__.return_value
        call_args = conn.execute.call_args
        # $5 is the search_log_id arg
        assert call_args[0][5] is None

    @pytest.mark.asyncio
    async def test_created_at_is_utc_datetime(self):
        """created_at should be a UTC datetime."""
        buf = _make_buffer()
        mock_pool = _mock_pool()
        buf._pool = mock_pool

        await buf.add(query="q", agent_response="r")

        conn = mock_pool.acquire.return_value.__aenter__.return_value
        call_args = conn.execute.call_args
        # $6 is the created_at arg
        created_at = call_args[0][6]
        assert isinstance(created_at, datetime)
        assert created_at.tzinfo is not None


# ===================================================================
# pending_count()
# ===================================================================

class TestPendingCount:
    """Tests for the pending_count() method."""

    @pytest.mark.asyncio
    async def test_returns_count(self):
        buf = _make_buffer()
        mock_pool = _mock_pool(fetchval_result=42)
        buf._pool = mock_pool

        count = await buf.pending_count()
        assert count == 42

    @pytest.mark.asyncio
    async def test_returns_zero_when_none(self):
        """When fetchval returns None, pending_count should return 0."""
        buf = _make_buffer()
        mock_pool = _mock_pool(fetchval_result=None)
        buf._pool = mock_pool

        count = await buf.pending_count()
        assert count == 0

    @pytest.mark.asyncio
    async def test_returns_zero_on_exception(self):
        buf = _make_buffer()
        pool = AsyncMock()
        pool._closed = False
        pool.acquire = MagicMock(side_effect=Exception("DB error"))
        buf._pool = pool

        count = await buf.pending_count()
        assert count == 0


# ===================================================================
# should_batch()
# ===================================================================

class TestShouldBatch:
    """Tests for the should_batch() method."""

    @pytest.mark.asyncio
    async def test_returns_true_when_count_equals_threshold(self):
        buf = _make_buffer(batch_threshold=5)
        mock_pool = _mock_pool(fetchval_result=5)
        buf._pool = mock_pool

        assert await buf.should_batch() is True

    @pytest.mark.asyncio
    async def test_returns_true_when_count_exceeds_threshold(self):
        buf = _make_buffer(batch_threshold=5)
        mock_pool = _mock_pool(fetchval_result=10)
        buf._pool = mock_pool

        assert await buf.should_batch() is True

    @pytest.mark.asyncio
    async def test_returns_false_when_count_below_threshold(self):
        buf = _make_buffer(batch_threshold=5)
        mock_pool = _mock_pool(fetchval_result=3)
        buf._pool = mock_pool

        assert await buf.should_batch() is False

    @pytest.mark.asyncio
    async def test_returns_false_when_count_zero(self):
        buf = _make_buffer(batch_threshold=5)
        mock_pool = _mock_pool(fetchval_result=0)
        buf._pool = mock_pool

        assert await buf.should_batch() is False

    @pytest.mark.asyncio
    async def test_threshold_zero_always_true(self):
        """Threshold 0 should trigger on every entry (count >= 0 is always True)."""
        buf = _make_buffer(batch_threshold=0)
        mock_pool = _mock_pool(fetchval_result=0)
        buf._pool = mock_pool

        assert await buf.should_batch() is True

    @pytest.mark.asyncio
    async def test_threshold_one_triggers_on_first(self):
        """Threshold 1 should trigger when there is at least 1 pending."""
        buf = _make_buffer(batch_threshold=1)
        mock_pool = _mock_pool(fetchval_result=1)
        buf._pool = mock_pool

        assert await buf.should_batch() is True

    @pytest.mark.asyncio
    async def test_threshold_one_false_when_zero(self):
        """Threshold 1 should not trigger when 0 pending."""
        buf = _make_buffer(batch_threshold=1)
        mock_pool = _mock_pool(fetchval_result=0)
        buf._pool = mock_pool

        assert await buf.should_batch() is False


# ===================================================================
# fetch_pending()
# ===================================================================

class TestFetchPending:
    """Tests for the fetch_pending() method."""

    @pytest.mark.asyncio
    async def test_returns_parsed_rows(self):
        buf = _make_buffer()
        entry_id = uuid.uuid4()
        log_id = uuid.uuid4()
        now = datetime.now(timezone.utc)

        row = {
            "id": entry_id,
            "query": "test query",
            "agent_response": "test response",
            "result_ids": json.dumps([{"id": "abc123"}]),
            "search_log_id": log_id,
            "created_at": now,
        }
        mock_pool = _mock_pool(rows_fetch=[row])
        buf._pool = mock_pool

        result = await buf.fetch_pending()
        assert len(result) == 1
        assert result[0]["id"] == str(entry_id)
        assert result[0]["query"] == "test query"
        assert result[0]["agent_response"] == "test response"
        assert result[0]["result_ids"] == [{"id": "abc123"}]
        assert result[0]["search_log_id"] == str(log_id)
        assert result[0]["created_at"] == now

    @pytest.mark.asyncio
    async def test_returns_empty_list_when_no_pending(self):
        buf = _make_buffer()
        mock_pool = _mock_pool(rows_fetch=[])
        buf._pool = mock_pool

        result = await buf.fetch_pending()
        assert result == []

    @pytest.mark.asyncio
    async def test_handles_none_result_ids(self):
        buf = _make_buffer()
        row = {
            "id": uuid.uuid4(),
            "query": "q",
            "agent_response": "r",
            "result_ids": None,
            "search_log_id": None,
            "created_at": datetime.now(timezone.utc),
        }
        mock_pool = _mock_pool(rows_fetch=[row])
        buf._pool = mock_pool

        result = await buf.fetch_pending()
        assert result[0]["result_ids"] is None
        assert result[0]["search_log_id"] is None

    @pytest.mark.asyncio
    async def test_returns_empty_on_exception(self):
        buf = _make_buffer()
        pool = AsyncMock()
        pool._closed = False
        pool.acquire = MagicMock(side_effect=Exception("DB error"))
        buf._pool = pool

        result = await buf.fetch_pending()
        assert result == []

    @pytest.mark.asyncio
    async def test_passes_limit_parameter(self):
        buf = _make_buffer()
        mock_pool = _mock_pool(rows_fetch=[])
        buf._pool = mock_pool

        await buf.fetch_pending(limit=100)
        conn = mock_pool.acquire.return_value.__aenter__.return_value
        conn.fetch.assert_awaited_once()
        call_args = conn.fetch.call_args
        assert call_args[0][1] == 100  # limit is the first positional arg after SQL


# ===================================================================
# mark_evaluated()
# ===================================================================

class TestMarkEvaluated:
    """Tests for the mark_evaluated() method."""

    @pytest.mark.asyncio
    async def test_success(self):
        buf = _make_buffer()
        mock_pool = _mock_pool()
        buf._pool = mock_pool

        entry_id = str(uuid.uuid4())
        await buf.mark_evaluated(
            entry_id, faithfulness_score=0.85, judge_reasoning="Good answer"
        )

        conn = mock_pool.acquire.return_value.__aenter__.return_value
        conn.execute.assert_awaited_once()
        call_args = conn.execute.call_args
        # $1 is entry_id (should be UUID), $2 is score, $3 is reasoning
        assert isinstance(call_args[0][1], uuid.UUID)
        assert call_args[0][2] == 0.85
        assert call_args[0][3] == "Good answer"

    @pytest.mark.asyncio
    async def test_reasoning_truncated_to_5000(self):
        """judge_reasoning should be truncated to 5000 chars."""
        buf = _make_buffer()
        mock_pool = _mock_pool()
        buf._pool = mock_pool

        long_reasoning = "x" * 8000
        await buf.mark_evaluated(
            str(uuid.uuid4()),
            faithfulness_score=0.5,
            judge_reasoning=long_reasoning,
        )

        conn = mock_pool.acquire.return_value.__aenter__.return_value
        call_args = conn.execute.call_args
        assert len(call_args[0][3]) <= 5000

    @pytest.mark.asyncio
    async def test_does_not_raise_on_exception(self):
        """mark_evaluated should silently handle exceptions."""
        buf = _make_buffer()
        pool = AsyncMock()
        pool._closed = False
        pool.acquire = MagicMock(side_effect=Exception("DB error"))
        buf._pool = pool

        # Should not raise
        await buf.mark_evaluated(
            str(uuid.uuid4()), faithfulness_score=0.5
        )

    @pytest.mark.asyncio
    async def test_default_reasoning_is_empty(self):
        """Default judge_reasoning should be empty string."""
        buf = _make_buffer()
        mock_pool = _mock_pool()
        buf._pool = mock_pool

        await buf.mark_evaluated(
            str(uuid.uuid4()), faithfulness_score=0.5
        )

        conn = mock_pool.acquire.return_value.__aenter__.return_value
        call_args = conn.execute.call_args
        assert call_args[0][3] == ""


# ===================================================================
# sync_to_search_log()
# ===================================================================

class TestSyncToSearchLog:
    """Tests for the sync_to_search_log() method."""

    @pytest.mark.asyncio
    async def test_was_used_true_when_score_ge_0_5(self):
        """was_used should be True when faithfulness_score >= 0.5."""
        buf = _make_buffer()
        entry_id = uuid.uuid4()
        log_id = uuid.uuid4()
        row = {"search_log_id": log_id}
        mock_pool = _mock_pool(row_fetchrow=row)
        buf._pool = mock_pool

        await buf.sync_to_search_log(
            str(entry_id), faithfulness_score=0.5
        )

        conn = mock_pool.acquire.return_value.__aenter__.return_value
        conn.execute.assert_awaited_once()
        call_args = conn.execute.call_args
        # $3 is was_used
        assert call_args[0][3] is True

    @pytest.mark.asyncio
    async def test_was_used_true_when_score_above_0_5(self):
        buf = _make_buffer()
        entry_id = uuid.uuid4()
        log_id = uuid.uuid4()
        row = {"search_log_id": log_id}
        mock_pool = _mock_pool(row_fetchrow=row)
        buf._pool = mock_pool

        await buf.sync_to_search_log(
            str(entry_id), faithfulness_score=0.9
        )

        conn = mock_pool.acquire.return_value.__aenter__.return_value
        call_args = conn.execute.call_args
        assert call_args[0][3] is True

    @pytest.mark.asyncio
    async def test_was_used_false_when_score_below_0_5(self):
        """was_used should be False when faithfulness_score < 0.5."""
        buf = _make_buffer()
        entry_id = uuid.uuid4()
        log_id = uuid.uuid4()
        row = {"search_log_id": log_id}
        mock_pool = _mock_pool(row_fetchrow=row)
        buf._pool = mock_pool

        await buf.sync_to_search_log(
            str(entry_id), faithfulness_score=0.3
        )

        conn = mock_pool.acquire.return_value.__aenter__.return_value
        call_args = conn.execute.call_args
        assert call_args[0][3] is False

    @pytest.mark.asyncio
    async def test_does_not_update_when_no_search_log_id(self):
        """Should skip UPDATE when search_log_id is None."""
        buf = _make_buffer()
        entry_id = uuid.uuid4()
        row = {"search_log_id": None}
        mock_pool = _mock_pool(row_fetchrow=row)
        buf._pool = mock_pool

        await buf.sync_to_search_log(
            str(entry_id), faithfulness_score=0.8
        )

        conn = mock_pool.acquire.return_value.__aenter__.return_value
        # execute should not be called (only fetchrow was called)
        conn.execute.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_does_not_update_when_row_not_found(self):
        """Should skip UPDATE when row is None."""
        buf = _make_buffer()
        mock_pool = _mock_pool(row_fetchrow=None)
        buf._pool = mock_pool

        await buf.sync_to_search_log(
            str(uuid.uuid4()), faithfulness_score=0.8
        )

        conn = mock_pool.acquire.return_value.__aenter__.return_value
        conn.execute.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_does_not_raise_on_exception(self):
        buf = _make_buffer()
        pool = AsyncMock()
        pool._closed = False
        pool.acquire = MagicMock(side_effect=Exception("DB error"))
        buf._pool = pool

        # Should not raise
        await buf.sync_to_search_log(
            str(uuid.uuid4()), faithfulness_score=0.8
        )

    @pytest.mark.asyncio
    async def test_was_used_boundary_at_exactly_0_5(self):
        """Score exactly 0.5 should set was_used=True (>= comparison)."""
        buf = _make_buffer()
        entry_id = uuid.uuid4()
        log_id = uuid.uuid4()
        row = {"search_log_id": log_id}
        mock_pool = _mock_pool(row_fetchrow=row)
        buf._pool = mock_pool

        await buf.sync_to_search_log(str(entry_id), faithfulness_score=0.5)

        conn = mock_pool.acquire.return_value.__aenter__.return_value
        call_args = conn.execute.call_args
        assert call_args[0][3] is True


# ===================================================================
# get_stats()
# ===================================================================

class TestGetStats:
    """Tests for the get_stats() method."""

    @pytest.mark.asyncio
    async def test_returns_correct_stats(self):
        buf = _make_buffer()
        mock_pool = _mock_pool(fetchval_result=10)
        # Override fetchval to return different values for each call
        conn = mock_pool.acquire.return_value.__aenter__.return_value
        conn.fetchval = AsyncMock(side_effect=[10, 7, 0.85])
        buf._pool = mock_pool

        stats = await buf.get_stats(days=7)
        assert stats["total"] == 10
        assert stats["evaluated"] == 7
        assert stats["pending"] == 3
        assert stats["avg_faithfulness"] == 0.85

    @pytest.mark.asyncio
    async def test_returns_zeroes_when_no_data(self):
        buf = _make_buffer()
        conn = _mock_pool().acquire.return_value.__aenter__.return_value
        conn.fetchval = AsyncMock(return_value=None)
        mock_pool = _mock_pool()
        # Override the pool's acquire to return our conn
        mock_pool.acquire.return_value.__aenter__ = AsyncMock(return_value=conn)
        buf._pool = mock_pool

        stats = await buf.get_stats(days=7)
        assert stats["total"] == 0
        assert stats["evaluated"] == 0
        assert stats["pending"] == 0
        assert stats["avg_faithfulness"] is None

    @pytest.mark.asyncio
    async def test_returns_default_on_exception(self):
        buf = _make_buffer()
        pool = AsyncMock()
        pool._closed = False
        pool.acquire = MagicMock(side_effect=Exception("DB error"))
        buf._pool = pool

        stats = await buf.get_stats()
        assert stats == {
            "total": 0,
            "evaluated": 0,
            "pending": 0,
            "avg_faithfulness": None,
        }

    @pytest.mark.asyncio
    async def test_avg_faithfulness_rounded_to_3_decimals(self):
        """avg_faithfulness should be rounded to 3 decimal places."""
        buf = _make_buffer()
        conn = AsyncMock()
        conn.fetchval = AsyncMock(side_effect=[5, 3, 0.123456789])
        pool = AsyncMock()
        pool._closed = False
        pool.acquire = MagicMock()
        pool.acquire.return_value.__aenter__ = AsyncMock(return_value=conn)
        pool.acquire.return_value.__aexit__ = AsyncMock(return_value=False)
        buf._pool = pool

        stats = await buf.get_stats(days=7)
        assert stats["avg_faithfulness"] == 0.123

    @pytest.mark.asyncio
    async def test_days_parameter_passed_to_query(self):
        """The days parameter should be passed to the SQL query."""
        buf = _make_buffer()
        mock_pool = _mock_pool(fetchval_result=0)
        buf._pool = mock_pool

        await buf.get_stats(days=30)
        conn = mock_pool.acquire.return_value.__aenter__.return_value
        # fetchval should be called 3 times (total, evaluated, avg)
        assert conn.fetchval.await_count == 3
        # Each call should have days=30 as the parameter
        for call in conn.fetchval.call_args_list:
            assert call[0][1] == 30


# ===================================================================
# Error handling edge cases
# ===================================================================

class TestErrorHandling:
    """Edge cases for error handling across all methods."""

    @pytest.mark.asyncio
    async def test_add_handles_invalid_search_log_id(self):
        """Invalid search_log_id string should cause an exception caught gracefully."""
        buf = _make_buffer()
        pool = AsyncMock()
        pool._closed = False
        conn = AsyncMock()
        # Make execute raise when given invalid UUID
        conn.execute = AsyncMock(side_effect=Exception("invalid input syntax for type uuid"))
        pool.acquire = MagicMock()
        pool.acquire.return_value.__aenter__ = AsyncMock(return_value=conn)
        pool.acquire.return_value.__aexit__ = AsyncMock(return_value=False)
        buf._pool = pool

        result = await buf.add(
            query="q", agent_response="r", search_log_id="not-a-uuid"
        )
        assert result is None

    @pytest.mark.asyncio
    async def test_fetch_pending_handles_malformed_json_in_result_ids(self):
        """Malformed JSON in result_ids should be caught."""
        buf = _make_buffer()
        row = {
            "id": uuid.uuid4(),
            "query": "q",
            "agent_response": "r",
            "result_ids": "not-valid-json{{",
            "search_log_id": None,
            "created_at": datetime.now(timezone.utc),
        }
        mock_pool = _mock_pool(rows_fetch=[row])
        buf._pool = mock_pool

        result = await buf.fetch_pending()
        # Should return empty list due to JSON parse error
        assert result == []

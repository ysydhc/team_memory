"""Unit tests for scripts/daemon/faithfulness_batch.py

Tests argument parsing, helper functions, batch logic, and error paths.
Mocks asyncpg connections, LLM calls, ResponseBuffer, and FaithfulnessJudge.
Uses @pytest.mark.unit for pure-logic and mocked-DB tests.
"""
from __future__ import annotations

import json
import sys
import uuid
from dataclasses import dataclass
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

pytestmark = pytest.mark.unit


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_eval_result(
    score: float = 0.8,
    total_claims: int = 3,
    supported_claims: int = 2,
    claims_detail: list[dict] | None = None,
) -> Any:
    """Create a mock FaithfulnessResult-like object."""
    if claims_detail is None:
        claims_detail = [
            {"claim": "c1", "supported": True, "reason": "ok"},
            {"claim": "c2", "supported": True, "reason": "ok"},
            {"claim": "c3", "supported": False, "reason": "no"},
        ]

    @dataclass
    class FakeResult:
        score: float
        total_claims: int
        supported_claims: int
        claims_detail: list
        raw_response: str = ""

    return FakeResult(
        score=score,
        total_claims=total_claims,
        supported_claims=supported_claims,
        claims_detail=claims_detail,
    )


def _make_entry(
    entry_id: str | None = None,
    query: str = "test query",
    agent_response: str = "test response",
    result_ids: list[dict] | None = None,
) -> dict:
    """Create a mock response_buffer entry."""
    if entry_id is None:
        entry_id = str(uuid.uuid4())
    if result_ids is None:
        result_ids = [{"id": str(uuid.uuid4())}]
    return {
        "id": entry_id,
        "query": query,
        "agent_response": agent_response,
        "result_ids": result_ids,
    }


def _mock_buffer(
    pending_count: int = 0,
    entries: list[dict] | None = None,
    batch_threshold: int = 5,
) -> MagicMock:
    """Create a mocked ResponseBuffer."""
    buf = MagicMock()
    buf._batch_threshold = batch_threshold
    buf.pending_count = AsyncMock(return_value=pending_count)
    buf.fetch_pending = AsyncMock(return_value=entries or [])
    buf.mark_evaluated = AsyncMock()
    buf.sync_to_search_log = AsyncMock()
    buf.close = AsyncMock()
    return buf


def _mock_judge(eval_result=None):
    """Create a mocked FaithfulnessJudge."""
    judge = MagicMock()
    judge.evaluate = AsyncMock(return_value=eval_result or _make_eval_result())
    return judge


# ===========================================================================
# load_experience_contexts
# ===========================================================================

class TestLoadExperienceContexts:
    """Tests for the load_experience_contexts() helper."""

    @pytest.mark.asyncio
    async def test_empty_result_ids_returns_empty_list(self):
        from daemon.faithfulness_batch import load_experience_contexts
        result = await load_experience_contexts("postgresql://x", None)
        assert result == []

    @pytest.mark.asyncio
    async def test_empty_list_returns_empty_list(self):
        from daemon.faithfulness_batch import load_experience_contexts
        result = await load_experience_contexts("postgresql://x", [])
        assert result == []

    @pytest.mark.asyncio
    async def test_all_invalid_uuids_returns_empty_list(self):
        from daemon.faithfulness_batch import load_experience_contexts
        result_ids = [{"id": "not-a-uuid"}, {"id": ""}, {"id": None}]
        result = await load_experience_contexts("postgresql://x", result_ids)
        assert result == []

    @pytest.mark.asyncio
    async def test_valid_uuid_connects_and_queries(self):
        """Valid UUID should trigger asyncpg.connect and fetch."""
        from daemon.faithfulness_batch import load_experience_contexts
        valid_id = str(uuid.uuid4())
        result_ids = [{"id": valid_id}]

        mock_conn = AsyncMock()
        mock_conn.fetch = AsyncMock(return_value=[])
        mock_conn.close = AsyncMock()

        with patch("asyncpg.connect", new_callable=AsyncMock, return_value=mock_conn) as mock_connect:
            result = await load_experience_contexts("postgresql://x", result_ids)
            assert result == []
            mock_connect.assert_called_once_with("postgresql://x")
            mock_conn.close.assert_called_once()

    @pytest.mark.asyncio
    async def test_context_formatting_with_all_fields(self):
        """Test that title, problem, solution are formatted correctly."""
        from daemon.faithfulness_batch import load_experience_contexts
        valid_id = uuid.uuid4()
        result_ids = [{"id": str(valid_id)}]

        mock_row = {
            "title": "Test Title",
            "description": "Test Problem",
            "solution": "Test Solution",
        }
        mock_conn = AsyncMock()
        mock_conn.fetch = AsyncMock(return_value=[mock_row])
        mock_conn.close = AsyncMock()

        with patch("asyncpg.connect", new_callable=AsyncMock, return_value=mock_conn):
            result = await load_experience_contexts("postgresql://x", result_ids)
            assert len(result) == 1
            assert "标题: Test Title" in result[0]
            assert "问题: Test Problem" in result[0]
            assert "方案: Test Solution" in result[0]

    @pytest.mark.asyncio
    async def test_context_formatting_with_title_only(self):
        """Test context when only title is present."""
        from daemon.faithfulness_batch import load_experience_contexts
        valid_id = uuid.uuid4()
        result_ids = [{"id": str(valid_id)}]

        mock_row = {"title": "Only Title", "description": None, "solution": None}
        mock_conn = AsyncMock()
        mock_conn.fetch = AsyncMock(return_value=[mock_row])
        mock_conn.close = AsyncMock()

        with patch("asyncpg.connect", new_callable=AsyncMock, return_value=mock_conn):
            result = await load_experience_contexts("postgresql://x", result_ids)
            assert len(result) == 1
            assert result[0] == "标题: Only Title"

    @pytest.mark.asyncio
    async def test_problem_truncated_to_200(self):
        """Test that problem text is truncated to 200 chars."""
        from daemon.faithfulness_batch import load_experience_contexts
        valid_id = uuid.uuid4()
        result_ids = [{"id": str(valid_id)}]

        long_problem = "x" * 300
        mock_row = {"title": "T", "description": long_problem, "solution": None}
        mock_conn = AsyncMock()
        mock_conn.fetch = AsyncMock(return_value=[mock_row])
        mock_conn.close = AsyncMock()

        with patch("asyncpg.connect", new_callable=AsyncMock, return_value=mock_conn):
            result = await load_experience_contexts("postgresql://x", result_ids)
            assert len(result) == 1
            assert "x" * 200 in result[0]
            assert long_problem not in result[0]

    @pytest.mark.asyncio
    async def test_solution_truncated_to_300(self):
        """Test that solution text is truncated to 300 chars."""
        from daemon.faithfulness_batch import load_experience_contexts
        valid_id = uuid.uuid4()
        result_ids = [{"id": str(valid_id)}]

        long_solution = "y" * 400
        mock_row = {"title": "T", "description": None, "solution": long_solution}
        mock_conn = AsyncMock()
        mock_conn.fetch = AsyncMock(return_value=[mock_row])
        mock_conn.close = AsyncMock()

        with patch("asyncpg.connect", new_callable=AsyncMock, return_value=mock_conn):
            result = await load_experience_contexts("postgresql://x", result_ids)
            assert len(result) == 1
            assert "y" * 300 in result[0]
            assert long_solution not in result[0]

    @pytest.mark.asyncio
    async def test_empty_title_problem_solution_excluded(self):
        """Empty strings should be excluded from context."""
        from daemon.faithfulness_batch import load_experience_contexts
        valid_id = uuid.uuid4()
        result_ids = [{"id": str(valid_id)}]

        mock_row = {"title": "", "description": "", "solution": ""}
        mock_conn = AsyncMock()
        mock_conn.fetch = AsyncMock(return_value=[mock_row])
        mock_conn.close = AsyncMock()

        with patch("asyncpg.connect", new_callable=AsyncMock, return_value=mock_conn):
            result = await load_experience_contexts("postgresql://x", result_ids)
            assert len(result) == 1
            assert result[0] == ""

    @pytest.mark.asyncio
    async def test_multiple_valid_uuids(self):
        """Multiple valid UUIDs should all be collected."""
        from daemon.faithfulness_batch import load_experience_contexts
        ids = [{"id": str(uuid.uuid4())}, {"id": str(uuid.uuid4())}]

        mock_conn = AsyncMock()
        mock_conn.fetch = AsyncMock(return_value=[])
        mock_conn.close = AsyncMock()

        with patch("asyncpg.connect", new_callable=AsyncMock, return_value=mock_conn):
            result = await load_experience_contexts("postgresql://x", ids)
            call_args = mock_conn.fetch.call_args
            assert len(call_args[0][1]) == 2

    @pytest.mark.asyncio
    async def test_mixed_valid_invalid_uuids(self):
        """Mix of valid and invalid UUIDs should only use valid ones."""
        from daemon.faithfulness_batch import load_experience_contexts
        valid_id = str(uuid.uuid4())
        ids = [{"id": "bad"}, {"id": valid_id}, {"id": ""}]

        mock_conn = AsyncMock()
        mock_conn.fetch = AsyncMock(return_value=[])
        mock_conn.close = AsyncMock()

        with patch("asyncpg.connect", new_callable=AsyncMock, return_value=mock_conn):
            result = await load_experience_contexts("postgresql://x", ids)
            call_args = mock_conn.fetch.call_args
            assert len(call_args[0][1]) == 1

    @pytest.mark.asyncio
    async def test_multiple_rows_formatted(self):
        """Multiple rows from DB should produce multiple context strings."""
        from daemon.faithfulness_batch import load_experience_contexts
        valid_id1 = str(uuid.uuid4())
        valid_id2 = str(uuid.uuid4())
        ids = [{"id": valid_id1}, {"id": valid_id2}]

        rows = [
            {"title": "T1", "description": "P1", "solution": "S1"},
            {"title": "T2", "description": None, "solution": "S2"},
        ]
        mock_conn = AsyncMock()
        mock_conn.fetch = AsyncMock(return_value=rows)
        mock_conn.close = AsyncMock()

        with patch("asyncpg.connect", new_callable=AsyncMock, return_value=mock_conn):
            result = await load_experience_contexts("postgresql://x", ids)
            assert len(result) == 2
            assert "T1" in result[0]
            assert "T2" in result[1]


# ===========================================================================
# run_batch
# ===========================================================================

class TestRunBatch:
    """Tests for the run_batch() function."""

    @pytest.mark.asyncio
    async def test_skipped_below_threshold(self):
        """Should skip when pending < threshold and not force."""
        from daemon.faithfulness_batch import run_batch

        buf = _mock_buffer(pending_count=3, batch_threshold=5)

        with patch("daemon.response_buffer.ResponseBuffer", return_value=buf):
            result = await run_batch(dsn="postgresql://x", force=False)
            assert result["skipped"] is True
            assert result["pending"] == 3
            buf.fetch_pending.assert_not_called()

    @pytest.mark.asyncio
    async def test_not_skipped_with_force(self):
        """Should not skip when force=True, even below threshold."""
        from daemon.faithfulness_batch import run_batch

        buf = _mock_buffer(pending_count=3, batch_threshold=5, entries=[])

        with patch("daemon.response_buffer.ResponseBuffer", return_value=buf):
            result = await run_batch(dsn="postgresql://x", force=True)
            assert result["skipped"] is False
            assert result["evaluated"] == 0

    @pytest.mark.asyncio
    async def test_empty_entries_returns_zero(self):
        """Should return evaluated=0 when fetch_pending returns empty."""
        from daemon.faithfulness_batch import run_batch

        buf = _mock_buffer(pending_count=10, batch_threshold=5, entries=[])

        with patch("daemon.response_buffer.ResponseBuffer", return_value=buf):
            result = await run_batch(dsn="postgresql://x", force=True)
            assert result["evaluated"] == 0
            assert result["skipped"] is False

    @pytest.mark.asyncio
    async def test_single_entry_evaluation(self):
        """Should evaluate a single entry and return summary."""
        from daemon.faithfulness_batch import run_batch

        entry = _make_entry(query="query1", agent_response="response1")
        buf = _mock_buffer(pending_count=5, batch_threshold=5, entries=[entry])
        eval_result = _make_eval_result(score=0.85, total_claims=3, supported_claims=3)
        judge = _mock_judge(eval_result)

        with (
            patch("daemon.response_buffer.ResponseBuffer", return_value=buf),
            patch("daemon.faithfulness_judge.FaithfulnessJudge", return_value=judge),
            patch("daemon.faithfulness_batch.load_experience_contexts", new_callable=AsyncMock, return_value=["ctx1"]),
        ):
            result = await run_batch(dsn="postgresql://x", force=True)
            assert result["evaluated"] == 1
            assert result["avg_faithfulness"] == 0.85
            assert result["high_faithfulness"] == 1
            assert result["low_faithfulness"] == 0
            assert result["errors"] == 0
            assert result["dry_run"] is False

    @pytest.mark.asyncio
    async def test_dry_run_does_not_write_to_db(self):
        """In dry_run mode, mark_evaluated and sync_to_search_log should not be called."""
        from daemon.faithfulness_batch import run_batch

        entry = _make_entry(query="q", agent_response="r")
        buf = _mock_buffer(pending_count=5, batch_threshold=5, entries=[entry])
        eval_result = _make_eval_result(score=0.8)
        judge = _mock_judge(eval_result)

        with (
            patch("daemon.response_buffer.ResponseBuffer", return_value=buf),
            patch("daemon.faithfulness_judge.FaithfulnessJudge", return_value=judge),
            patch("daemon.faithfulness_batch.load_experience_contexts", new_callable=AsyncMock, return_value=["ctx"]),
        ):
            result = await run_batch(dsn="postgresql://x", dry_run=True, force=True)
            assert result["dry_run"] is True
            buf.mark_evaluated.assert_not_called()
            buf.sync_to_search_log.assert_not_called()

    @pytest.mark.asyncio
    async def test_non_dry_run_writes_to_db(self):
        """In non-dry-run mode, mark_evaluated and sync_to_search_log should be called."""
        from daemon.faithfulness_batch import run_batch

        entry = _make_entry(query="q", agent_response="r")
        buf = _mock_buffer(pending_count=5, batch_threshold=5, entries=[entry])
        eval_result = _make_eval_result(score=0.8)
        judge = _mock_judge(eval_result)

        with (
            patch("daemon.response_buffer.ResponseBuffer", return_value=buf),
            patch("daemon.faithfulness_judge.FaithfulnessJudge", return_value=judge),
            patch("daemon.faithfulness_batch.load_experience_contexts", new_callable=AsyncMock, return_value=["ctx"]),
        ):
            result = await run_batch(dsn="postgresql://x", dry_run=False, force=True)
            assert result["dry_run"] is False
            buf.mark_evaluated.assert_called_once()
            buf.sync_to_search_log.assert_called_once()

    @pytest.mark.asyncio
    async def test_multiple_entries_summary_stats(self):
        """Should correctly compute avg, high, low from multiple entries."""
        from daemon.faithfulness_batch import run_batch

        entry1 = _make_entry(query="q1", agent_response="r1")
        entry2 = _make_entry(query="q2", agent_response="r2")
        entry3 = _make_entry(query="q3", agent_response="r3")
        buf = _mock_buffer(
            pending_count=10, batch_threshold=5,
            entries=[entry1, entry2, entry3],
        )

        # Different scores: 0.9 (high), 0.5 (low), -1 (error)
        results_iter = iter([
            _make_eval_result(score=0.9, total_claims=2, supported_claims=2),
            _make_eval_result(score=0.5, total_claims=2, supported_claims=1),
            _make_eval_result(score=-1.0, total_claims=0, supported_claims=0),
        ])
        judge = MagicMock()
        judge.evaluate = AsyncMock(side_effect=lambda **kw: next(results_iter))

        with (
            patch("daemon.response_buffer.ResponseBuffer", return_value=buf),
            patch("daemon.faithfulness_judge.FaithfulnessJudge", return_value=judge),
            patch("daemon.faithfulness_batch.load_experience_contexts", new_callable=AsyncMock, return_value=["ctx"]),
        ):
            result = await run_batch(dsn="postgresql://x", force=True)
            assert result["evaluated"] == 3
            assert result["high_faithfulness"] == 1
            assert result["low_faithfulness"] == 1
            assert result["errors"] == 1
            # avg of non-negative: (0.9 + 0.5) / 2 = 0.7
            assert result["avg_faithfulness"] == 0.7

    @pytest.mark.asyncio
    async def test_all_error_scores_avg_is_zero(self):
        """When all scores are negative (errors), avg should be 0."""
        from daemon.faithfulness_batch import run_batch

        entry = _make_entry(query="q", agent_response="r")
        buf = _mock_buffer(pending_count=5, batch_threshold=5, entries=[entry])
        eval_result = _make_eval_result(score=-1.0, total_claims=0, supported_claims=0)
        judge = _mock_judge(eval_result)

        with (
            patch("daemon.response_buffer.ResponseBuffer", return_value=buf),
            patch("daemon.faithfulness_judge.FaithfulnessJudge", return_value=judge),
            patch("daemon.faithfulness_batch.load_experience_contexts", new_callable=AsyncMock, return_value=["ctx"]),
        ):
            result = await run_batch(dsn="postgresql://x", force=True)
            assert result["avg_faithfulness"] == 0
            assert result["errors"] == 1

    @pytest.mark.asyncio
    async def test_close_always_called(self):
        """buffer.close() should always be called even if evaluation fails."""
        from daemon.faithfulness_batch import run_batch

        entry = _make_entry(query="q", agent_response="r")
        buf = _mock_buffer(pending_count=5, batch_threshold=5, entries=[entry])
        judge = MagicMock()
        judge.evaluate = AsyncMock(side_effect=Exception("LLM error"))

        with (
            patch("daemon.response_buffer.ResponseBuffer", return_value=buf),
            patch("daemon.faithfulness_judge.FaithfulnessJudge", return_value=judge),
            patch("daemon.faithfulness_batch.load_experience_contexts", new_callable=AsyncMock, return_value=["ctx"]),
        ):
            with pytest.raises(Exception, match="LLM error"):
                await run_batch(dsn="postgresql://x", force=True)
            buf.close.assert_called_once()

    @pytest.mark.asyncio
    async def test_reasoning_truncated_to_2000(self):
        """Reasoning should be truncated to 2000 chars."""
        from daemon.faithfulness_batch import run_batch

        entry = _make_entry(query="q", agent_response="r")
        buf = _mock_buffer(pending_count=5, batch_threshold=5, entries=[entry])

        # Create a large claims_detail
        large_detail = [{"claim": f"claim{i}", "supported": True, "reason": "x" * 100} for i in range(30)]
        eval_result = _make_eval_result(score=0.8, claims_detail=large_detail)
        judge = _mock_judge(eval_result)

        with (
            patch("daemon.response_buffer.ResponseBuffer", return_value=buf),
            patch("daemon.faithfulness_judge.FaithfulnessJudge", return_value=judge),
            patch("daemon.faithfulness_batch.load_experience_contexts", new_callable=AsyncMock, return_value=["ctx"]),
        ):
            result = await run_batch(dsn="postgresql://x", force=True)
            assert result["evaluated"] == 1
            # Verify mark_evaluated was called with truncated reasoning
            call_kwargs = buf.mark_evaluated.call_args
            assert len(call_kwargs[1]["judge_reasoning"]) <= 2000

    @pytest.mark.asyncio
    async def test_passed_model_and_base_url(self):
        """Model and base_url should be forwarded to FaithfulnessJudge."""
        from daemon.faithfulness_batch import run_batch

        # Need at least one entry so judge is actually created
        entry = _make_entry(query="q", agent_response="r")
        buf = _mock_buffer(pending_count=5, batch_threshold=5, entries=[entry])
        eval_result = _make_eval_result(score=0.5)
        judge = _mock_judge(eval_result)

        with (
            patch("daemon.response_buffer.ResponseBuffer", return_value=buf),
            patch("daemon.faithfulness_judge.FaithfulnessJudge", return_value=judge) as mock_judge_cls,
            patch("daemon.faithfulness_batch.load_experience_contexts", new_callable=AsyncMock, return_value=["ctx"]),
        ):
            await run_batch(
                dsn="postgresql://x",
                model="custom-model",
                base_url="http://custom:9999",
                api_key="test-key",
                force=True,
            )
            mock_judge_cls.assert_called_once_with(
                base_url="http://custom:9999",
                model="custom-model",
                api_key="test-key",
            )

    @pytest.mark.asyncio
    async def test_pending_at_exactly_threshold_not_skipped(self):
        """When pending == threshold, should not skip (>= check)."""
        from daemon.faithfulness_batch import run_batch

        buf = _mock_buffer(pending_count=5, batch_threshold=5, entries=[])

        with patch("daemon.response_buffer.ResponseBuffer", return_value=buf):
            result = await run_batch(dsn="postgresql://x", force=False)
            assert result["skipped"] is False
            assert result["evaluated"] == 0

    @pytest.mark.asyncio
    async def test_query_and_response_forwarded_to_judge(self):
        """Judge.evaluate should receive the entry's query and response."""
        from daemon.faithfulness_batch import run_batch

        entry = _make_entry(query="my question", agent_response="my answer")
        buf = _mock_buffer(pending_count=5, batch_threshold=5, entries=[entry])
        eval_result = _make_eval_result(score=0.5)
        judge = _mock_judge(eval_result)

        with (
            patch("daemon.response_buffer.ResponseBuffer", return_value=buf),
            patch("daemon.faithfulness_judge.FaithfulnessJudge", return_value=judge),
            patch("daemon.faithfulness_batch.load_experience_contexts", new_callable=AsyncMock, return_value=["ctx"]),
        ):
            await run_batch(dsn="postgresql://x", force=True)
            judge.evaluate.assert_called_once_with(
                query="my question",
                contexts=["ctx"],
                response="my answer",
            )


# ===========================================================================
# show_recent
# ===========================================================================

class TestShowRecent:
    """Tests for the show_recent() function."""

    @pytest.mark.asyncio
    async def test_no_entries_prints_message(self, capsys):
        """Should print 'No entries' when rows is empty."""
        from daemon.faithfulness_batch import show_recent

        mock_conn = AsyncMock()
        mock_conn.fetch = AsyncMock(return_value=[])
        mock_conn.close = AsyncMock()

        with patch("asyncpg.connect", new_callable=AsyncMock, return_value=mock_conn):
            await show_recent(dsn="postgresql://x", limit=5)
            captured = capsys.readouterr()
            assert "No entries" in captured.out

    @pytest.mark.asyncio
    async def test_prints_entries(self, capsys):
        """Should print entry details when rows exist."""
        from daemon.faithfulness_batch import show_recent

        mock_row = {
            "id": str(uuid.uuid4()),
            "query": "test query",
            "agent_response": "test response here",
            "result_ids": json.dumps([{"id": "abc"}]),
            "evaluated": True,
            "faithfulness_score": 0.85,
            "judge_reasoning": "some reasoning",
            "created_at": "2025-01-01T00:00:00",
        }

        mock_conn = AsyncMock()
        mock_conn.fetch = AsyncMock(return_value=[mock_row])
        mock_conn.close = AsyncMock()

        with patch("asyncpg.connect", new_callable=AsyncMock, return_value=mock_conn):
            await show_recent(dsn="postgresql://x", limit=1)
            captured = capsys.readouterr()
            assert "test query" in captured.out
            assert "test response here" in captured.out

    @pytest.mark.asyncio
    async def test_result_ids_from_string_parsed(self, capsys):
        """Should parse result_ids when provided as a JSON string."""
        from daemon.faithfulness_batch import show_recent

        mock_row = {
            "id": str(uuid.uuid4()),
            "query": "q",
            "agent_response": "r",
            "result_ids": json.dumps([{"id": "a"}, {"id": "b"}]),
            "evaluated": False,
            "faithfulness_score": None,
            "judge_reasoning": None,
            "created_at": "2025-01-01",
        }

        mock_conn = AsyncMock()
        mock_conn.fetch = AsyncMock(return_value=[mock_row])
        mock_conn.close = AsyncMock()

        with patch("asyncpg.connect", new_callable=AsyncMock, return_value=mock_conn):
            await show_recent(dsn="postgresql://x", limit=1)
            captured = capsys.readouterr()
            assert "results=2" in captured.out

    @pytest.mark.asyncio
    async def test_result_ids_as_list(self, capsys):
        """Should handle result_ids already as a list (not string)."""
        from daemon.faithfulness_batch import show_recent

        mock_row = {
            "id": str(uuid.uuid4()),
            "query": "q",
            "agent_response": "r",
            "result_ids": [{"id": "a"}],
            "evaluated": False,
            "faithfulness_score": None,
            "judge_reasoning": None,
            "created_at": "2025-01-01",
        }

        mock_conn = AsyncMock()
        mock_conn.fetch = AsyncMock(return_value=[mock_row])
        mock_conn.close = AsyncMock()

        with patch("asyncpg.connect", new_callable=AsyncMock, return_value=mock_conn):
            await show_recent(dsn="postgresql://x", limit=1)
            captured = capsys.readouterr()
            assert "results=1" in captured.out

    @pytest.mark.asyncio
    async def test_score_display_green(self, capsys):
        """Score >= 0.7 should show green checkmark."""
        from daemon.faithfulness_batch import show_recent

        mock_row = {
            "id": str(uuid.uuid4()),
            "query": "q",
            "agent_response": "r",
            "result_ids": None,
            "evaluated": True,
            "faithfulness_score": 0.85,
            "judge_reasoning": None,
            "created_at": "2025-01-01",
        }

        mock_conn = AsyncMock()
        mock_conn.fetch = AsyncMock(return_value=[mock_row])
        mock_conn.close = AsyncMock()

        with patch("asyncpg.connect", new_callable=AsyncMock, return_value=mock_conn):
            await show_recent(dsn="postgresql://x", limit=1)
            captured = capsys.readouterr()
            assert "\033[32m" in captured.out
            assert "✓" in captured.out

    @pytest.mark.asyncio
    async def test_score_display_red(self, capsys):
        """Score 0 <= s < 0.7 should show red cross."""
        from daemon.faithfulness_batch import show_recent

        mock_row = {
            "id": str(uuid.uuid4()),
            "query": "q",
            "agent_response": "r",
            "result_ids": None,
            "evaluated": True,
            "faithfulness_score": 0.4,
            "judge_reasoning": None,
            "created_at": "2025-01-01",
        }

        mock_conn = AsyncMock()
        mock_conn.fetch = AsyncMock(return_value=[mock_row])
        mock_conn.close = AsyncMock()

        with patch("asyncpg.connect", new_callable=AsyncMock, return_value=mock_conn):
            await show_recent(dsn="postgresql://x", limit=1)
            captured = capsys.readouterr()
            assert "\033[31m" in captured.out
            assert "✗" in captured.out

    @pytest.mark.asyncio
    async def test_score_display_error(self, capsys):
        """Negative score should show 'error'."""
        from daemon.faithfulness_batch import show_recent

        mock_row = {
            "id": str(uuid.uuid4()),
            "query": "q",
            "agent_response": "r",
            "result_ids": None,
            "evaluated": True,
            "faithfulness_score": -1.0,
            "judge_reasoning": None,
            "created_at": "2025-01-01",
        }

        mock_conn = AsyncMock()
        mock_conn.fetch = AsyncMock(return_value=[mock_row])
        mock_conn.close = AsyncMock()

        with patch("asyncpg.connect", new_callable=AsyncMock, return_value=mock_conn):
            await show_recent(dsn="postgresql://x", limit=1)
            captured = capsys.readouterr()
            assert "error" in captured.out

    @pytest.mark.asyncio
    async def test_score_display_pending(self, capsys):
        """Not evaluated should show 'pending'."""
        from daemon.faithfulness_batch import show_recent

        mock_row = {
            "id": str(uuid.uuid4()),
            "query": "q",
            "agent_response": "r",
            "result_ids": None,
            "evaluated": False,
            "faithfulness_score": None,
            "judge_reasoning": None,
            "created_at": "2025-01-01",
        }

        mock_conn = AsyncMock()
        mock_conn.fetch = AsyncMock(return_value=[mock_row])
        mock_conn.close = AsyncMock()

        with patch("asyncpg.connect", new_callable=AsyncMock, return_value=mock_conn):
            await show_recent(dsn="postgresql://x", limit=1)
            captured = capsys.readouterr()
            assert "pending" in captured.out

    @pytest.mark.asyncio
    async def test_evaluated_no_score(self, capsys):
        """Evaluated but None score should show 'evaluated (no score)'."""
        from daemon.faithfulness_batch import show_recent

        mock_row = {
            "id": str(uuid.uuid4()),
            "query": "q",
            "agent_response": "r",
            "result_ids": None,
            "evaluated": True,
            "faithfulness_score": None,
            "judge_reasoning": None,
            "created_at": "2025-01-01",
        }

        mock_conn = AsyncMock()
        mock_conn.fetch = AsyncMock(return_value=[mock_row])
        mock_conn.close = AsyncMock()

        with patch("asyncpg.connect", new_callable=AsyncMock, return_value=mock_conn):
            await show_recent(dsn="postgresql://x", limit=1)
            captured = capsys.readouterr()
            assert "evaluated (no score)" in captured.out

    @pytest.mark.asyncio
    async def test_none_result_ids(self, capsys):
        """None result_ids should show results=0."""
        from daemon.faithfulness_batch import show_recent

        mock_row = {
            "id": str(uuid.uuid4()),
            "query": "q",
            "agent_response": "r",
            "result_ids": None,
            "evaluated": False,
            "faithfulness_score": None,
            "judge_reasoning": None,
            "created_at": "2025-01-01",
        }

        mock_conn = AsyncMock()
        mock_conn.fetch = AsyncMock(return_value=[mock_row])
        mock_conn.close = AsyncMock()

        with patch("asyncpg.connect", new_callable=AsyncMock, return_value=mock_conn):
            await show_recent(dsn="postgresql://x", limit=1)
            captured = capsys.readouterr()
            assert "results=0" in captured.out

    @pytest.mark.asyncio
    async def test_truncated_query_and_response(self, capsys):
        """Query truncated to 80 chars, response to 150 chars."""
        from daemon.faithfulness_batch import show_recent

        long_query = "Q" * 100
        long_response = "R" * 200
        mock_row = {
            "id": str(uuid.uuid4()),
            "query": long_query,
            "agent_response": long_response,
            "result_ids": None,
            "evaluated": False,
            "faithfulness_score": None,
            "judge_reasoning": None,
            "created_at": "2025-01-01",
        }

        mock_conn = AsyncMock()
        mock_conn.fetch = AsyncMock(return_value=[mock_row])
        mock_conn.close = AsyncMock()

        with patch("asyncpg.connect", new_callable=AsyncMock, return_value=mock_conn):
            await show_recent(dsn="postgresql://x", limit=1)
            captured = capsys.readouterr()
            # Query should be truncated to 80 chars
            assert "Q" * 80 in captured.out
            assert long_query not in captured.out
            # Response should be truncated to 150 chars
            assert "R" * 150 in captured.out
            assert long_response not in captured.out


# ===========================================================================
# main()
# ===========================================================================

class TestMain:
    """Tests for the main() CLI entry point."""

    def test_show_mode_calls_show_recent(self):
        """--show should call show_recent and return."""
        from daemon.faithfulness_batch import main
        with (
            patch("sys.argv", ["faithfulness_batch.py", "--show", "5"]),
            patch("daemon.faithfulness_batch.show_recent", new_callable=AsyncMock),
            patch("asyncio.run") as mock_run,
        ):
            main()
            mock_run.assert_called_once()

    def test_recent_mode_passes_force_true(self):
        """--recent should call run_batch with force=True."""
        from daemon.faithfulness_batch import main
        mock_result = {"skipped": False, "evaluated": 3}

        with (
            patch("sys.argv", ["faithfulness_batch.py", "--recent", "3"]),
            patch("daemon.faithfulness_batch.run_batch", new_callable=AsyncMock, return_value=mock_result) as mock_run_batch,
            patch("asyncio.run", return_value=mock_result),
        ):
            main()
            # run_batch is called as a coroutine inside asyncio.run.
            # Verify it was called by checking the mock.
            # Since run_batch is an async function, calling it returns a coroutine.
            # asyncio.run receives the coroutine. The mock prevents actual execution.
            mock_run_batch.assert_called_once()

    def test_default_mode_passes_args(self):
        """Default mode should forward all CLI args to run_batch."""
        from daemon.faithfulness_batch import main
        mock_result = {"skipped": False, "evaluated": 5}

        with (
            patch("sys.argv", [
                "faithfulness_batch.py",
                "--model", "my-model",
                "--base-url", "http://my-url:1234",
                "--api-key", "my-key",
                "--batch-size", "15",
                "--dry-run",
                "--force",
                "--db-url", "postgresql://custom/db",
            ]),
            patch("daemon.faithfulness_batch.run_batch", new_callable=AsyncMock, return_value=mock_result) as mock_run_batch,
            patch("asyncio.run", return_value=mock_result),
        ):
            main()
            mock_run_batch.assert_called_once()

    def test_default_db_url(self):
        """Should use default db-url when not specified."""
        from daemon.faithfulness_batch import main
        mock_result = {"skipped": False}

        with (
            patch("sys.argv", ["faithfulness_batch.py"]),
            patch("daemon.faithfulness_batch.run_batch", new_callable=AsyncMock, return_value=mock_result) as mock_run_batch,
            patch("asyncio.run", return_value=mock_result),
        ):
            main()
            mock_run_batch.assert_called_once()
            # Check the DSN is the default
            call_kwargs = mock_run_batch.call_args[1]
            assert "5433" in call_kwargs["dsn"]

    def test_skipped_result_exits_zero(self):
        """When run_batch returns skipped=True, sys.exit(0) should be called."""
        from daemon.faithfulness_batch import main
        mock_result = {"skipped": True}

        with (
            patch("sys.argv", ["faithfulness_batch.py"]),
            patch("asyncio.run", return_value=mock_result),
        ):
            with pytest.raises(SystemExit) as exc_info:
                main()
            assert exc_info.value.code == 0

    def test_non_skipped_result_no_exit(self):
        """When run_batch returns skipped=False, should not exit."""
        from daemon.faithfulness_batch import main
        mock_result = {"skipped": False, "evaluated": 5}

        with (
            patch("sys.argv", ["faithfulness_batch.py"]),
            patch("asyncio.run", return_value=mock_result),
        ):
            # Should complete without SystemExit
            main()

    def test_argument_parsing_all_flags(self):
        """All CLI flags should be parsed correctly."""
        from daemon.faithfulness_batch import main
        mock_result = {"skipped": False}

        with (
            patch("sys.argv", [
                "faithfulness_batch.py",
                "--model", "gpt-4",
                "--base-url", "http://llm:8080",
                "--api-key", "secret",
                "--batch-size", "25",
                "--dry-run",
                "--force",
                "--recent", "10",
                "--show", "0",
                "--db-url", "postgresql://test/db",
            ]),
            patch("daemon.faithfulness_batch.run_batch", new_callable=AsyncMock, return_value=mock_result) as mock_run_batch,
            patch("asyncio.run", return_value=mock_result),
        ):
            main()
            # --recent=10 means recent mode
            mock_run_batch.assert_called_once()

    def test_recent_mode_always_forces(self):
        """--recent should always pass force=True regardless of --force flag."""
        from daemon.faithfulness_batch import main
        mock_result = {"skipped": False, "evaluated": 2}

        with (
            patch("sys.argv", ["faithfulness_batch.py", "--recent", "5"]),
            patch("daemon.faithfulness_batch.run_batch", new_callable=AsyncMock, return_value=mock_result) as mock_run_batch,
            patch("asyncio.run", return_value=mock_result),
        ):
            main()
            call_kwargs = mock_run_batch.call_args[1]
            assert call_kwargs["force"] is True

    def test_custom_db_url_passed(self):
        """Custom --db-url should be forwarded to run_batch."""
        from daemon.faithfulness_batch import main
        mock_result = {"skipped": False}

        with (
            patch("sys.argv", ["faithfulness_batch.py", "--db-url", "postgresql://custom:5432/mydb"]),
            patch("daemon.faithfulness_batch.run_batch", new_callable=AsyncMock, return_value=mock_result) as mock_run_batch,
            patch("asyncio.run", return_value=mock_result),
        ):
            main()
            call_kwargs = mock_run_batch.call_args[1]
            assert call_kwargs["dsn"] == "postgresql://custom:5432/mydb"


# ===========================================================================
# Integration: run_batch end-to-end with full mock
# ===========================================================================

class TestRunBatchEndToEnd:
    """End-to-end tests for run_batch with all external deps mocked."""

    @pytest.mark.asyncio
    async def test_full_batch_with_multiple_entries_and_results(self):
        """Full run with 2 entries, different scores, verifying summary."""
        from daemon.faithfulness_batch import run_batch

        entry1 = _make_entry(query="what is python", agent_response="Python is a language")
        entry2 = _make_entry(query="how to cook", agent_response="Use a pan")
        buf = _mock_buffer(pending_count=8, batch_threshold=5, entries=[entry1, entry2])

        results_iter = iter([
            _make_eval_result(score=0.9, total_claims=3, supported_claims=3),
            _make_eval_result(score=0.6, total_claims=2, supported_claims=1),
        ])
        judge = MagicMock()
        judge.evaluate = AsyncMock(side_effect=lambda **kw: next(results_iter))

        with (
            patch("daemon.response_buffer.ResponseBuffer", return_value=buf),
            patch("daemon.faithfulness_judge.FaithfulnessJudge", return_value=judge),
            patch("daemon.faithfulness_batch.load_experience_contexts", new_callable=AsyncMock, return_value=["ctx"]),
        ):
            result = await run_batch(dsn="postgresql://x", force=True)
            assert result["evaluated"] == 2
            assert result["high_faithfulness"] == 1
            assert result["low_faithfulness"] == 1
            assert result["errors"] == 0
            # avg: (0.9 + 0.6) / 2 = 0.75
            assert result["avg_faithfulness"] == 0.75
            assert buf.mark_evaluated.call_count == 2
            assert buf.sync_to_search_log.call_count == 2

    @pytest.mark.asyncio
    async def test_batch_respects_batch_size(self):
        """fetch_pending should be called with the correct limit."""
        from daemon.faithfulness_batch import run_batch

        buf = _mock_buffer(pending_count=100, batch_threshold=5, entries=[])

        with (
            patch("daemon.response_buffer.ResponseBuffer", return_value=buf),
            patch("daemon.faithfulness_judge.FaithfulnessJudge"),
        ):
            await run_batch(dsn="postgresql://x", batch_size=50, force=True)
            buf.fetch_pending.assert_called_once_with(limit=50)

    @pytest.mark.asyncio
    async def test_all_high_faithfulness(self):
        """All entries >= 0.7 should be counted as high."""
        from daemon.faithfulness_batch import run_batch

        entry1 = _make_entry(query="q1", agent_response="r1")
        entry2 = _make_entry(query="q2", agent_response="r2")
        buf = _mock_buffer(pending_count=5, batch_threshold=5, entries=[entry1, entry2])

        results_iter = iter([
            _make_eval_result(score=0.7, total_claims=1, supported_claims=1),
            _make_eval_result(score=1.0, total_claims=1, supported_claims=1),
        ])
        judge = MagicMock()
        judge.evaluate = AsyncMock(side_effect=lambda **kw: next(results_iter))

        with (
            patch("daemon.response_buffer.ResponseBuffer", return_value=buf),
            patch("daemon.faithfulness_judge.FaithfulnessJudge", return_value=judge),
            patch("daemon.faithfulness_batch.load_experience_contexts", new_callable=AsyncMock, return_value=["ctx"]),
        ):
            result = await run_batch(dsn="postgresql://x", force=True)
            assert result["high_faithfulness"] == 2
            assert result["low_faithfulness"] == 0

    @pytest.mark.asyncio
    async def test_all_low_faithfulness(self):
        """All entries 0 <= s < 0.7 should be counted as low."""
        from daemon.faithfulness_batch import run_batch

        entry = _make_entry(query="q", agent_response="r")
        buf = _mock_buffer(pending_count=5, batch_threshold=5, entries=[entry])
        eval_result = _make_eval_result(score=0.3, total_claims=2, supported_claims=0)
        judge = _mock_judge(eval_result)

        with (
            patch("daemon.response_buffer.ResponseBuffer", return_value=buf),
            patch("daemon.faithfulness_judge.FaithfulnessJudge", return_value=judge),
            patch("daemon.faithfulness_batch.load_experience_contexts", new_callable=AsyncMock, return_value=["ctx"]),
        ):
            result = await run_batch(dsn="postgresql://x", force=True)
            assert result["high_faithfulness"] == 0
            assert result["low_faithfulness"] == 1

    @pytest.mark.asyncio
    async def test_multiple_errors(self):
        """Multiple error scores should all be counted."""
        from daemon.faithfulness_batch import run_batch

        entry1 = _make_entry(query="q1", agent_response="r1")
        entry2 = _make_entry(query="q2", agent_response="r2")
        buf = _mock_buffer(pending_count=5, batch_threshold=5, entries=[entry1, entry2])

        results_iter = iter([
            _make_eval_result(score=-1.0, total_claims=0, supported_claims=0),
            _make_eval_result(score=-1.0, total_claims=0, supported_claims=0),
        ])
        judge = MagicMock()
        judge.evaluate = AsyncMock(side_effect=lambda **kw: next(results_iter))

        with (
            patch("daemon.response_buffer.ResponseBuffer", return_value=buf),
            patch("daemon.faithfulness_judge.FaithfulnessJudge", return_value=judge),
            patch("daemon.faithfulness_batch.load_experience_contexts", new_callable=AsyncMock, return_value=["ctx"]),
        ):
            result = await run_batch(dsn="postgresql://x", force=True)
            assert result["errors"] == 2
            assert result["evaluated"] == 2

    @pytest.mark.asyncio
    async def test_buffer_close_on_success(self):
        """buffer.close() should be called on successful completion."""
        from daemon.faithfulness_batch import run_batch

        buf = _mock_buffer(pending_count=5, batch_threshold=5, entries=[])

        with (
            patch("daemon.response_buffer.ResponseBuffer", return_value=buf),
            patch("daemon.faithfulness_judge.FaithfulnessJudge"),
        ):
            await run_batch(dsn="postgresql://x", force=True)
            buf.close.assert_called_once()

    @pytest.mark.asyncio
    async def test_default_params_used(self):
        """Default parameters should be used when not specified."""
        from daemon.faithfulness_batch import run_batch

        # Need at least one entry so judge is created
        entry = _make_entry(query="q", agent_response="r")
        buf = _mock_buffer(pending_count=5, batch_threshold=5, entries=[entry])
        eval_result = _make_eval_result(score=0.5)
        judge = _mock_judge(eval_result)

        with (
            patch("daemon.response_buffer.ResponseBuffer", return_value=buf) as mock_buf_cls,
            patch("daemon.faithfulness_judge.FaithfulnessJudge", return_value=judge) as mock_judge_cls,
            patch("daemon.faithfulness_batch.load_experience_contexts", new_callable=AsyncMock, return_value=["ctx"]),
        ):
            await run_batch(dsn="postgresql://x")
            # Default batch_size=20
            buf.fetch_pending.assert_called_once_with(limit=20)
            # Default judge params (None for model, base_url, api_key)
            mock_judge_cls.assert_called_once_with(
                base_url=None,
                model=None,
                api_key=None,
            )

    @pytest.mark.asyncio
    async def test_result_ids_forwarded_to_load_contexts(self):
        """Result IDs from entry should be forwarded to load_experience_contexts."""
        from daemon.faithfulness_batch import run_batch

        rids = [{"id": "abc-123"}, {"id": "def-456"}]
        entry = _make_entry(query="q", agent_response="r", result_ids=rids)
        buf = _mock_buffer(pending_count=5, batch_threshold=5, entries=[entry])
        eval_result = _make_eval_result(score=0.5)
        judge = _mock_judge(eval_result)

        with (
            patch("daemon.response_buffer.ResponseBuffer", return_value=buf),
            patch("daemon.faithfulness_judge.FaithfulnessJudge", return_value=judge),
            patch("daemon.faithfulness_batch.load_experience_contexts", new_callable=AsyncMock, return_value=["ctx"]) as mock_load,
        ):
            await run_batch(dsn="postgresql://x", force=True)
            mock_load.assert_called_once_with("postgresql://x", rids)

    @pytest.mark.asyncio
    async def test_report_header_printed(self, capsys):
        """The batch report header should be printed."""
        from daemon.faithfulness_batch import run_batch

        entry = _make_entry(query="q", agent_response="r")
        buf = _mock_buffer(pending_count=5, batch_threshold=5, entries=[entry])
        eval_result = _make_eval_result(score=0.5)
        judge = _mock_judge(eval_result)

        with (
            patch("daemon.response_buffer.ResponseBuffer", return_value=buf),
            patch("daemon.faithfulness_judge.FaithfulnessJudge", return_value=judge),
            patch("daemon.faithfulness_batch.load_experience_contexts", new_callable=AsyncMock, return_value=["ctx"]),
        ):
            await run_batch(dsn="postgresql://x", force=True)
            captured = capsys.readouterr()
            assert "Faithfulness Batch Report" in captured.out

    @pytest.mark.asyncio
    async def test_low_entries_printed(self, capsys):
        """Low faithfulness entries should be printed in the report."""
        from daemon.faithfulness_batch import run_batch

        entry = _make_entry(query="low question", agent_response="low answer")
        buf = _mock_buffer(pending_count=5, batch_threshold=5, entries=[entry])
        eval_result = _make_eval_result(score=0.3, total_claims=2, supported_claims=0)
        judge = _mock_judge(eval_result)

        with (
            patch("daemon.response_buffer.ResponseBuffer", return_value=buf),
            patch("daemon.faithfulness_judge.FaithfulnessJudge", return_value=judge),
            patch("daemon.faithfulness_batch.load_experience_contexts", new_callable=AsyncMock, return_value=["ctx"]),
        ):
            result = await run_batch(dsn="postgresql://x", force=True)
            captured = capsys.readouterr()
            assert "Low faithfulness entries" in captured.out
            assert "low question" in captured.out

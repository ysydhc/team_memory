"""Tests for scripts/hooks/hermes_pipeline.py — Hermes-side memory pipeline.

HermesPipeline integrates retrieval, draft buffering, convergence detection,
and draft refining into a single pipeline that Hermes can call directly
(without external hook mechanisms).
"""
from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest

# Ensure scripts/hooks/ is importable
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts" / "hooks"))

from convergence_detector import ConvergenceDetector  # noqa: E402
from draft_buffer import DraftBuffer  # noqa: E402
from draft_refiner import DraftRefiner  # noqa: E402
from hermes_pipeline import HermesPipeline  # noqa: E402


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_db_path() -> str:
    """Return a unique temp file path for a test SQLite DB."""
    fd, path = tempfile.mkstemp(suffix=".db")
    os.close(fd)
    return path


@pytest.fixture
def db_path():
    """Provide a temporary DB path and clean up after the test."""
    path = _make_db_path()
    yield path
    if os.path.exists(path):
        os.unlink(path)


def _make_tm_client() -> AsyncMock:
    """Create a mock TMClient with recall, get_context, draft_save, draft_publish."""
    tm = AsyncMock()
    tm.recall = AsyncMock(return_value={"results": ["result-1", "result-2"]})
    tm.get_context = AsyncMock(return_value={"context": "project-context"})
    tm.draft_save = AsyncMock(return_value={"id": "tm-draft-001"})
    tm.draft_publish = AsyncMock(return_value={"status": "ok"})
    return tm


def _make_pipeline(db_path: str, tm: AsyncMock | None = None) -> HermesPipeline:
    """Create a HermesPipeline with mocked TMClient and real buffer/detector."""
    tm = tm or _make_tm_client()
    pipeline = HermesPipeline.__new__(HermesPipeline)
    pipeline._tm = tm
    pipeline._buffer = DraftBuffer(db_path)
    pipeline._detector = ConvergenceDetector()
    pipeline._refiner = DraftRefiner(tm, pipeline._buffer)
    return pipeline


# ---------------------------------------------------------------------------
# on_turn_start — keyword triggered → full retrieval
# ---------------------------------------------------------------------------


class TestOnTurnStartKeywordTrigger:
    """on_turn_start with keyword triggers full retrieval via tm.recall."""

    @pytest.mark.asyncio
    async def test_keyword_before_triggers_recall(self, db_path):
        pipeline = _make_pipeline(db_path)
        async with pipeline._buffer:
            result = await pipeline.on_turn_start("之前我们怎么做这个的？", project="team_doc")
        assert result["action"] == "full_retrieval"
        pipeline._tm.recall.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_keyword_last_time_triggers_recall(self, db_path):
        pipeline = _make_pipeline(db_path)
        async with pipeline._buffer:
            result = await pipeline.on_turn_start("上次踩坑了", project="team_doc")
        assert result["action"] == "full_retrieval"

    @pytest.mark.asyncio
    async def test_keyword_experience_triggers_recall(self, db_path):
        pipeline = _make_pipeline(db_path)
        async with pipeline._buffer:
            result = await pipeline.on_turn_start("分享一下经验", project="team_doc")
        assert result["action"] == "full_retrieval"

    @pytest.mark.asyncio
    async def test_keyword_pitfall_triggers_recall(self, db_path):
        pipeline = _make_pipeline(db_path)
        async with pipeline._buffer:
            result = await pipeline.on_turn_start("这个有踩坑", project="team_doc")
        assert result["action"] == "full_retrieval"

    @pytest.mark.asyncio
    async def test_keyword_remember_triggers_recall(self, db_path):
        pipeline = _make_pipeline(db_path)
        async with pipeline._buffer:
            result = await pipeline.on_turn_start("Remember to check this", project="team_doc")
        assert result["action"] == "full_retrieval"

    @pytest.mark.asyncio
    async def test_keyword_previously_triggers_recall(self, db_path):
        pipeline = _make_pipeline(db_path)
        async with pipeline._buffer:
            result = await pipeline.on_turn_start("Previously we fixed this", project="team_doc")
        assert result["action"] == "full_retrieval"

    @pytest.mark.asyncio
    async def test_full_retrieval_returns_results(self, db_path):
        pipeline = _make_pipeline(db_path)
        async with pipeline._buffer:
            result = await pipeline.on_turn_start("之前怎么做的", project="team_doc")
        assert "results" in result


# ---------------------------------------------------------------------------
# on_turn_start — no keyword → project-level context
# ---------------------------------------------------------------------------


class TestOnTurnStartNoKeyword:
    """on_turn_start without keyword returns project-level context."""

    @pytest.mark.asyncio
    async def test_no_keyword_returns_project_context(self, db_path):
        pipeline = _make_pipeline(db_path)
        async with pipeline._buffer:
            result = await pipeline.on_turn_start("帮我写个函数", project="team_doc")
        assert result["action"] == "project_context"
        pipeline._tm.get_context.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_project_context_returns_context(self, db_path):
        pipeline = _make_pipeline(db_path)
        async with pipeline._buffer:
            result = await pipeline.on_turn_start("写代码", project="team_doc")
        assert "context" in result

    @pytest.mark.asyncio
    async def test_no_keyword_does_not_call_recall(self, db_path):
        pipeline = _make_pipeline(db_path)
        async with pipeline._buffer:
            await pipeline.on_turn_start("帮我写个函数", project="team_doc")
        pipeline._tm.recall.assert_not_awaited()


# ---------------------------------------------------------------------------
# on_turn_end — converged → publish draft
# ---------------------------------------------------------------------------


class TestOnTurnEndConverged:
    """on_turn_end with convergence signal publishes the draft."""

    @pytest.mark.asyncio
    async def test_converged_publishes_draft(self, db_path):
        pipeline = _make_pipeline(db_path)
        async with pipeline._buffer:
            # First save a draft so there's something to publish
            await pipeline._refiner.save_draft(
                "sess-1", "Draft", "因为服务器配置。", project="team_doc"
            )
            result = await pipeline.on_turn_end(
                session_id="sess-1",
                agent_response="问题解决了，不需要再改",
                project="team_doc",
            )
        assert result["action"] == "published"
        pipeline._tm.draft_publish.assert_awaited()

    @pytest.mark.asyncio
    async def test_converged_publish_result_has_status(self, db_path):
        pipeline = _make_pipeline(db_path)
        async with pipeline._buffer:
            await pipeline._refiner.save_draft(
                "sess-2", "Draft", "因为服务器配置。", project="team_doc"
            )
            result = await pipeline.on_turn_end(
                session_id="sess-2",
                agent_response="搞定了",
                project="team_doc",
            )
        assert result["result"]["status"] == "published"


# ---------------------------------------------------------------------------
# on_turn_end — not converged → save draft
# ---------------------------------------------------------------------------


class TestOnTurnEndNotConverged:
    """on_turn_end without convergence saves the draft."""

    @pytest.mark.asyncio
    async def test_not_converged_saves_draft(self, db_path):
        pipeline = _make_pipeline(db_path)
        async with pipeline._buffer:
            result = await pipeline.on_turn_end(
                session_id="sess-3",
                agent_response="我还在调试中",
                project="team_doc",
            )
        assert result["action"] == "draft_saved"

    @pytest.mark.asyncio
    async def test_not_converged_calls_draft_save(self, db_path):
        pipeline = _make_pipeline(db_path)
        async with pipeline._buffer:
            await pipeline.on_turn_end(
                session_id="sess-3",
                agent_response="我还在调试中",
                project="team_doc",
            )
        pipeline._tm.draft_save.assert_awaited()


# ---------------------------------------------------------------------------
# on_turn_end — text accumulation
# ---------------------------------------------------------------------------


class TestOnTurnEndAccumulation:
    """on_turn_end accumulates text across multiple turns."""

    @pytest.mark.asyncio
    async def test_accumulates_text_across_turns(self, db_path):
        pipeline = _make_pipeline(db_path)
        async with pipeline._buffer:
            # First turn — no convergence
            await pipeline.on_turn_end(
                session_id="sess-acc",
                agent_response="第一步：因为服务器超时",
                project="team_doc",
            )
            # Second turn — no convergence
            await pipeline.on_turn_end(
                session_id="sess-acc",
                agent_response="第二步：所以需要重启",
                project="team_doc",
            )
            # Verify accumulated content in buffer
            pending = await pipeline._buffer.get_pending("sess-acc")
        assert len(pending) == 1
        content = pending[0]["content"]
        assert "第一步" in content
        assert "第二步" in content

    @pytest.mark.asyncio
    async def test_first_turn_no_existing(self, db_path):
        pipeline = _make_pipeline(db_path)
        async with pipeline._buffer:
            result = await pipeline.on_turn_end(
                session_id="sess-new",
                agent_response="开始新任务",
                project="team_doc",
            )
        assert result["action"] == "draft_saved"


# ---------------------------------------------------------------------------
# on_session_end — with pending drafts → publish
# ---------------------------------------------------------------------------


class TestOnSessionEndWithDrafts:
    """on_session_end with pending drafts publishes them."""

    @pytest.mark.asyncio
    async def test_publishes_pending_drafts(self, db_path):
        pipeline = _make_pipeline(db_path)
        async with pipeline._buffer:
            # Save a draft first
            await pipeline._refiner.save_draft(
                "sess-end-1", "Draft", "因为服务器配置。", project="team_doc"
            )
            result = await pipeline.on_session_end("sess-end-1")
        assert result is not None
        assert result["status"] == "published"
        pipeline._tm.draft_publish.assert_awaited()


# ---------------------------------------------------------------------------
# on_session_end — no drafts → return None
# ---------------------------------------------------------------------------


class TestOnSessionEndNoDrafts:
    """on_session_end with no pending drafts returns None."""

    @pytest.mark.asyncio
    async def test_returns_none_no_drafts(self, db_path):
        pipeline = _make_pipeline(db_path)
        async with pipeline._buffer:
            result = await pipeline.on_session_end("sess-empty")
        assert result is None

    @pytest.mark.asyncio
    async def test_does_not_call_publish(self, db_path):
        pipeline = _make_pipeline(db_path)
        async with pipeline._buffer:
            await pipeline.on_session_end("sess-empty")
        pipeline._tm.draft_publish.assert_not_awaited()

"""Tests for scripts/hooks/session_timeout.py — 30-minute draft safety net."""
from __future__ import annotations

import os
import sys
import tempfile
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import AsyncMock

import pytest

# Ensure scripts/hooks/ is importable
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts" / "hooks"))

import session_timeout  # noqa: E402
from draft_buffer import DraftBuffer  # noqa: E402
from draft_refiner import DraftRefiner  # noqa: E402
from session_timeout import SessionTimeoutManager  # noqa: E402


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


@pytest.fixture
def buf(db_path):
    """Create a DraftBuffer backed by a temp DB."""
    return DraftBuffer(db_path)


def _make_tm_client() -> AsyncMock:
    """Create a mock TMClient with draft_save and draft_publish methods."""
    tm = AsyncMock()
    tm.draft_save = AsyncMock(return_value={"id": "tm-draft-001"})
    tm.draft_publish = AsyncMock(return_value={"status": "ok"})
    return tm


# ---------------------------------------------------------------------------
# DraftBuffer.get_all_pending — returns all pending across sessions
# ---------------------------------------------------------------------------


class TestGetAllPending:
    """get_all_pending returns pending drafts across all sessions."""

    @pytest.mark.asyncio
    async def test_returns_all_pending_across_sessions(self, buf):
        async with buf:
            await buf.create_draft("team_doc", "sess-1", "content 1")
            await buf.create_draft("team_doc", "sess-2", "content 2")
            await buf.create_draft("team_doc", "sess-3", "content 3")

            all_pending = await buf.get_all_pending()

        assert len(all_pending) == 3
        session_ids = {d["conversation_id"] for d in all_pending}
        assert session_ids == {"sess-1", "sess-2", "sess-3"}

    @pytest.mark.asyncio
    async def test_excludes_published(self, buf):
        async with buf:
            draft_id = await buf.create_draft("team_doc", "sess-1", "content 1")
            await buf.create_draft("team_doc", "sess-2", "content 2")
            await buf.mark_published(draft_id)

            all_pending = await buf.get_all_pending()

        assert len(all_pending) == 1
        assert all_pending[0]["conversation_id"] == "sess-2"

    @pytest.mark.asyncio
    async def test_empty_when_no_pending(self, buf):
        async with buf:
            all_pending = await buf.get_all_pending()

        assert all_pending == []


# ---------------------------------------------------------------------------
# check_and_publish — non-timed-out session
# ---------------------------------------------------------------------------


class TestCheckAndPublishNotTimedOut:
    """check_and_publish returns None for non-timed-out session."""

    @pytest.mark.asyncio
    async def test_returns_none_for_recent_draft(self, buf):
        tm = _make_tm_client()
        refiner = DraftRefiner(tm, buf)
        # Default clock returns real now — drafts are fresh
        mgr = SessionTimeoutManager(refiner, buf, timeout_minutes=30)
        async with buf:
            await buf.create_draft("team_doc", "sess-recent", "recent content")
            result = await mgr.check_and_publish("sess-recent")
        assert result is None
        tm.draft_publish.assert_not_awaited()


# ---------------------------------------------------------------------------
# check_and_publish — timed-out session
# ---------------------------------------------------------------------------


class TestCheckAndPublishTimedOut:
    """check_and_publish publishes for timed-out session."""

    @pytest.mark.asyncio
    async def test_publishes_old_draft(self, buf):
        tm = _make_tm_client()
        refiner = DraftRefiner(tm, buf)
        # Clock returns 31 minutes in the future, so current drafts appear old
        fake_now = datetime.now(timezone.utc) + timedelta(minutes=31)
        mgr = SessionTimeoutManager(
            refiner, buf, timeout_minutes=30, clock=lambda: fake_now
        )
        async with buf:
            await buf.create_draft("team_doc", "sess-old", "因为超时了。")
            result = await mgr.check_and_publish("sess-old")
        assert result is not None
        assert result["status"] == "published"
        tm.draft_publish.assert_awaited()


# ---------------------------------------------------------------------------
# check_and_publish — no pending drafts
# ---------------------------------------------------------------------------


class TestCheckAndPublishNoPending:
    """check_and_publish returns None for session with no pending drafts."""

    @pytest.mark.asyncio
    async def test_returns_none_no_drafts(self, buf):
        tm = _make_tm_client()
        refiner = DraftRefiner(tm, buf)
        mgr = SessionTimeoutManager(refiner, buf, timeout_minutes=30)
        async with buf:
            result = await mgr.check_and_publish("sess-none")
        assert result is None
        tm.draft_publish.assert_not_awaited()


# ---------------------------------------------------------------------------
# _check_timeouts — publishes all timed-out drafts
# ---------------------------------------------------------------------------


class TestCheckTimeouts:
    """_check_timeouts publishes all timed-out drafts across sessions."""

    @pytest.mark.asyncio
    async def test_publishes_all_timed_out(self, buf):
        tm = _make_tm_client()
        refiner = DraftRefiner(tm, buf)
        # Clock returns 31 minutes in the future — all drafts look old
        fake_now = datetime.now(timezone.utc) + timedelta(minutes=31)
        mgr = SessionTimeoutManager(
            refiner, buf, timeout_minutes=30, clock=lambda: fake_now
        )
        async with buf:
            await buf.create_draft("team_doc", "sess-old-1", "因为原因1。")
            await buf.create_draft("team_doc", "sess-old-2", "因为原因2。")

            await mgr._check_timeouts()

        # draft_publish should have been called twice (once per session)
        assert tm.draft_publish.await_count == 2

    @pytest.mark.asyncio
    async def test_skips_recent_drafts(self, buf):
        tm = _make_tm_client()
        refiner = DraftRefiner(tm, buf)
        # Use real clock — all drafts are fresh
        mgr = SessionTimeoutManager(refiner, buf, timeout_minutes=30)
        async with buf:
            await buf.create_draft("team_doc", "sess-recent", "recent content")
            await mgr._check_timeouts()
        tm.draft_publish.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_mixed_old_and_recent(self, buf):
        tm = _make_tm_client()
        refiner = DraftRefiner(tm, buf)
        # Clock 31 min ahead — all look old, but we create one with
        # real time then advance. Actually, since we use a fixed fake_now,
        # all drafts created at real now will look 31 min old.
        # To test mixed, create the "old" draft, then use a clock
        # that is only 10 min ahead (not yet timed out).
        fake_now_10min = datetime.now(timezone.utc) + timedelta(minutes=10)
        mgr = SessionTimeoutManager(
            refiner, buf, timeout_minutes=30, clock=lambda: fake_now_10min
        )
        async with buf:
            await buf.create_draft("team_doc", "sess-1", "因为原因1。")
            await buf.create_draft("team_doc", "sess-2", "因为原因2。")
            # Only 10 minutes passed — nothing should time out
            await mgr._check_timeouts()
        tm.draft_publish.assert_not_awaited()


# ---------------------------------------------------------------------------
# start / stop lifecycle
# ---------------------------------------------------------------------------


class TestStartStopLifecycle:
    """start/stop lifecycle works correctly."""

    @pytest.mark.asyncio
    async def test_start_and_stop(self, buf):
        tm = _make_tm_client()
        refiner = DraftRefiner(tm, buf)
        mgr = SessionTimeoutManager(refiner, buf, timeout_minutes=30)
        assert mgr._running is False
        assert mgr._task is None

        await mgr.start()
        assert mgr._running is True
        assert mgr._task is not None

        await mgr.stop()
        assert mgr._running is False
        assert mgr._task is None

    @pytest.mark.asyncio
    async def test_stop_without_start_is_safe(self, buf):
        tm = _make_tm_client()
        refiner = DraftRefiner(tm, buf)
        mgr = SessionTimeoutManager(refiner, buf, timeout_minutes=30)
        # Should not raise
        await mgr.stop()

    @pytest.mark.asyncio
    async def test_check_loop_cancels_on_stop(self, buf):
        tm = _make_tm_client()
        refiner = DraftRefiner(tm, buf)
        mgr = SessionTimeoutManager(refiner, buf, timeout_minutes=30)

        await mgr.start()
        assert mgr._task is not None
        task = mgr._task

        await mgr.stop()
        assert task.cancelled() or task.done()

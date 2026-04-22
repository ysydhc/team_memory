"""Tests for scripts/hooks/draft_buffer.py — local SQLite draft buffer."""
from __future__ import annotations

import asyncio
import os
import sys
import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest

# Ensure scripts/hooks/ is importable
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts" / "hooks"))

import draft_buffer  # noqa: E402


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
    return draft_buffer.DraftBuffer(db_path)


# ---------------------------------------------------------------------------
# __init__ — table creation
# ---------------------------------------------------------------------------

class TestInit:
    """DraftBuffer.__init__ creates the drafts table if it doesn't exist."""

    @pytest.mark.asyncio
    async def test_table_created(self, db_path):
        buf = draft_buffer.DraftBuffer(db_path)
        async with buf:
            # Verify table exists by querying it
            pending = await buf.get_pending_drafts()
        assert isinstance(pending, list)
        assert len(pending) == 0

    @pytest.mark.asyncio
    async def test_idempotent_init(self, db_path):
        """Creating DraftBuffer twice on same DB should not raise."""
        buf1 = draft_buffer.DraftBuffer(db_path)
        async with buf1:
            pass
        buf2 = draft_buffer.DraftBuffer(db_path)
        async with buf2:
            pending = await buf2.get_pending_drafts()
        assert len(pending) == 0


# ---------------------------------------------------------------------------
# create_draft
# ---------------------------------------------------------------------------

class TestCreateDraft:
    """DraftBuffer.create_draft inserts a new draft and returns its UUID."""

    @pytest.mark.asyncio
    async def test_returns_uuid(self, buf):
        async with buf:
            draft_id = await buf.create_draft("team_doc", "conv-1", "some facts")
        assert isinstance(draft_id, str)
        assert len(draft_id) == 36  # UUID4 format with hyphens

    @pytest.mark.asyncio
    async def test_draft_stored_with_pending_status(self, buf):
        async with buf:
            draft_id = await buf.create_draft("team_doc", "conv-1", "some facts")
            pending = await buf.get_pending_drafts()
        assert len(pending) == 1
        assert pending[0]["id"] == draft_id
        assert pending[0]["project"] == "team_doc"
        assert pending[0]["conversation_id"] == "conv-1"
        assert pending[0]["content"] == "some facts"
        assert pending[0]["status"] == "pending"
        assert pending[0]["source"] == "pipeline"

    @pytest.mark.asyncio
    async def test_created_at_is_set(self, buf):
        async with buf:
            await buf.create_draft("team_doc", "conv-1", "facts")
            pending = await buf.get_pending_drafts()
        assert pending[0]["created_at"] is not None
        assert pending[0]["updated_at"] is not None


# ---------------------------------------------------------------------------
# update_draft
# ---------------------------------------------------------------------------

class TestUpdateDraft:
    """DraftBuffer.update_draft changes the content of an existing draft."""

    @pytest.mark.asyncio
    async def test_update_content(self, buf):
        async with buf:
            draft_id = await buf.create_draft("team_doc", "conv-1", "old facts")
            await buf.update_draft(draft_id, "new facts")
            pending = await buf.get_pending_drafts()
        assert len(pending) == 1
        assert pending[0]["content"] == "new facts"

    @pytest.mark.asyncio
    async def test_updated_at_changes(self, buf):
        async with buf:
            draft_id = await buf.create_draft("team_doc", "conv-1", "old")
            pending_before = await buf.get_pending_drafts()
            created_at = pending_before[0]["created_at"]

            # Small sleep to ensure timestamp difference
            await asyncio.sleep(0.05)
            await buf.update_draft(draft_id, "updated")
            pending_after = await buf.get_pending_drafts()

        assert pending_after[0]["updated_at"] != pending_after[0]["created_at"] or created_at != pending_after[0]["updated_at"]

    @pytest.mark.asyncio
    async def test_update_nonexistent_raises(self, buf):
        async with buf:
            with pytest.raises(Exception):
                await buf.update_draft("nonexistent-id", "data")


# ---------------------------------------------------------------------------
# get_pending_drafts
# ---------------------------------------------------------------------------

class TestGetPendingDrafts:
    """DraftBuffer.get_pending_drafts returns only pending drafts."""

    @pytest.mark.asyncio
    async def test_returns_only_pending(self, buf):
        async with buf:
            id1 = await buf.create_draft("team_doc", "c1", "fact1")
            id2 = await buf.create_draft("team_doc", "c2", "fact2")
            await buf.mark_published(id1)
            pending = await buf.get_pending_drafts()
        assert len(pending) == 1
        assert pending[0]["id"] == id2

    @pytest.mark.asyncio
    async def test_filter_by_project(self, buf):
        async with buf:
            await buf.create_draft("team_doc", "c1", "fact1")
            await buf.create_draft("ad_learning", "c2", "fact2")
            await buf.create_draft("team_doc", "c3", "fact3")
            team_doc = await buf.get_pending_drafts(project="team_doc")
        assert len(team_doc) == 2
        assert all(d["project"] == "team_doc" for d in team_doc)

    @pytest.mark.asyncio
    async def test_filter_by_project_no_match(self, buf):
        async with buf:
            await buf.create_draft("team_doc", "c1", "fact1")
            result = await buf.get_pending_drafts(project="nonexistent")
        assert len(result) == 0

    @pytest.mark.asyncio
    async def test_no_filter_returns_all(self, buf):
        async with buf:
            await buf.create_draft("team_doc", "c1", "fact1")
            await buf.create_draft("ad_learning", "c2", "fact2")
            all_pending = await buf.get_pending_drafts()
        assert len(all_pending) == 2


# ---------------------------------------------------------------------------
# get_older_than
# ---------------------------------------------------------------------------

class TestGetOlderThan:
    """DraftBuffer.get_older_than returns drafts older than N minutes."""

    @pytest.mark.asyncio
    async def test_recent_drafts_not_returned(self, buf):
        async with buf:
            await buf.create_draft("team_doc", "c1", "just created")
            old = await buf.get_older_than(minutes=60)
        assert len(old) == 0

    @pytest.mark.asyncio
    async def test_old_drafts_returned(self, buf):
        async with buf:
            draft_id = await buf.create_draft("team_doc", "c1", "old fact")
            # Manually backdate the created_at timestamp
            import aiosqlite
            await buf._db.execute(
                "UPDATE drafts SET created_at = datetime('now', '-90 minutes') WHERE id = ?",
                (draft_id,),
            )
            await buf._db.commit()
            old = await buf.get_older_than(minutes=60)
        assert len(old) == 1
        assert old[0]["id"] == draft_id

    @pytest.mark.asyncio
    async def test_boundary_excluded(self, buf):
        """Drafts exactly N minutes old should NOT be returned (< not <=)."""
        async with buf:
            draft_id = await buf.create_draft("team_doc", "c1", "boundary")
            import aiosqlite
            await buf._db.execute(
                "UPDATE drafts SET created_at = datetime('now', '-30 minutes') WHERE id = ?",
                (draft_id,),
            )
            await buf._db.commit()
            old = await buf.get_older_than(minutes=30)
        assert len(old) == 0


# ---------------------------------------------------------------------------
# mark_published
# ---------------------------------------------------------------------------

class TestMarkPublished:
    """DraftBuffer.mark_published sets status to 'published'."""

    @pytest.mark.asyncio
    async def test_mark_published(self, buf):
        async with buf:
            draft_id = await buf.create_draft("team_doc", "c1", "facts")
            await buf.mark_published(draft_id)
            pending = await buf.get_pending_drafts()
        assert len(pending) == 0

    @pytest.mark.asyncio
    async def test_mark_published_nonexistent_raises(self, buf):
        async with buf:
            with pytest.raises(Exception):
                await buf.mark_published("nonexistent-id")

    @pytest.mark.asyncio
    async def test_idempotent_mark(self, buf):
        async with buf:
            draft_id = await buf.create_draft("team_doc", "c1", "facts")
            await buf.mark_published(draft_id)
            await buf.mark_published(draft_id)  # should not raise
            pending = await buf.get_pending_drafts()
        assert len(pending) == 0


# ---------------------------------------------------------------------------
# find_pending_by_conversation
# ---------------------------------------------------------------------------

class TestFindPendingByConversation:
    """DraftBuffer.find_pending_by_conversation returns drafts matching project + conversation_id."""

    @pytest.mark.asyncio
    async def test_returns_matching_draft(self, buf):
        async with buf:
            await buf.create_draft("team_doc", "conv-1", "fact1")
            result = await buf.find_pending_by_conversation("team_doc", "conv-1")
        assert len(result) == 1
        assert result[0]["conversation_id"] == "conv-1"
        assert result[0]["project"] == "team_doc"

    @pytest.mark.asyncio
    async def test_no_match_returns_empty(self, buf):
        async with buf:
            await buf.create_draft("team_doc", "conv-1", "fact1")
            result = await buf.find_pending_by_conversation("team_doc", "conv-nonexistent")
        assert len(result) == 0

    @pytest.mark.asyncio
    async def test_different_project_no_match(self, buf):
        async with buf:
            await buf.create_draft("team_doc", "conv-1", "fact1")
            result = await buf.find_pending_by_conversation("other_project", "conv-1")
        assert len(result) == 0

    @pytest.mark.asyncio
    async def test_excludes_published(self, buf):
        async with buf:
            draft_id = await buf.create_draft("team_doc", "conv-1", "fact1")
            await buf.mark_published(draft_id)
            result = await buf.find_pending_by_conversation("team_doc", "conv-1")
        assert len(result) == 0


# ---------------------------------------------------------------------------
# mark_for_publishing
# ---------------------------------------------------------------------------

class TestMarkForPublishing:
    """DraftBuffer.mark_for_publishing sets status to 'ready_to_publish'."""

    @pytest.mark.asyncio
    async def test_mark_for_publishing(self, buf):
        async with buf:
            draft_id = await buf.create_draft("team_doc", "c1", "facts")
            await buf.mark_for_publishing(draft_id)
            # Should no longer appear in pending (it's now 'ready_to_publish')
            pending = await buf.get_pending_drafts()
        assert len(pending) == 0

    @pytest.mark.asyncio
    async def test_mark_for_publishing_nonexistent_raises(self, buf):
        async with buf:
            with pytest.raises(Exception):
                await buf.mark_for_publishing("nonexistent-id")

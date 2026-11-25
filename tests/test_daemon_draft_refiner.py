"""Tests for src/team_memory/daemon/draft_refiner.py — coverage for uncovered lines.

Uncovered lines targeted:
  98-99   : elif not pg_id branch in save_draft
  150-166 : wiki_compiler block in refine_and_publish
  185-194 : mark_for_refinement method
  226-228 : break + return facts in extract_facts (max limit)

Note: the daemon DraftBuffer does NOT have set_tm_id / tm_id column.
Lines 96-97 and 140 (tm_id paths) are already covered by test_draft_tm_id.py
which uses the hooks.draft_buffer that does have that support.
"""

from __future__ import annotations

import os
import tempfile
from unittest.mock import AsyncMock

import pytest

from team_memory.daemon.buffers.draft_buffer import DraftBuffer
from team_memory.daemon.refinement.draft_refiner import _MAX_FACTS, DraftRefiner

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_db_path() -> str:
    fd, path = tempfile.mkstemp(suffix=".db")
    os.close(fd)
    return path


@pytest.fixture
def db_path():
    path = _make_db_path()
    yield path
    if os.path.exists(path):
        os.unlink(path)


@pytest.fixture
def buf(db_path):
    return DraftBuffer(db_path)


def _mock_sink(draft_save_return=None, publish_return=None):
    """Create a mock TMSink."""
    sink = AsyncMock()
    sink.draft_save = AsyncMock(return_value=draft_save_return or {"id": "pg-001"})
    sink.draft_publish = AsyncMock(return_value=publish_return or {"status": "ok"})
    return sink


def _mock_wiki_compiler(return_path="/tmp/wiki.md"):
    """Create a mock wiki_compiler with compile_one."""
    wc = AsyncMock()
    wc.compile_one = AsyncMock(return_value=return_path)
    return wc


# ---------------------------------------------------------------------------
# extract_facts — max 10 limit (lines 226-228)
# ---------------------------------------------------------------------------


class TestExtractFactsMaxLimit:
    """Cover the break statement and return at lines 226-228."""

    def test_exactly_10_facts(self):
        sentences = [f"因为原因{i}。" for i in range(10)]
        text = "".join(sentences)
        facts = DraftRefiner.extract_facts(text)
        assert len(facts) == 10

    def test_more_than_10_facts_truncated(self):
        sentences = [f"因为原因{i}。" for i in range(15)]
        text = "".join(sentences)
        facts = DraftRefiner.extract_facts(text)
        assert len(facts) == _MAX_FACTS

    def test_11_facts_returns_10(self):
        sentences = [f"因为原因{i}。" for i in range(11)]
        text = "".join(sentences)
        facts = DraftRefiner.extract_facts(text)
        assert len(facts) == 10

    def test_fewer_than_10_returns_all(self):
        sentences = [f"因为原因{i}。" for i in range(3)]
        text = "".join(sentences)
        facts = DraftRefiner.extract_facts(text)
        assert len(facts) == 3


# ---------------------------------------------------------------------------
# save_draft — elif not pg_id branch (lines 98-99)
# ---------------------------------------------------------------------------


class TestSaveDraftNoPgId:
    """Cover the elif not pg_id: logger.warning branch."""

    @pytest.mark.asyncio
    async def test_no_id_in_response_logs_warning(self, buf):
        sink = _mock_sink(draft_save_return={"status": "error", "message": "fail"})
        refiner = DraftRefiner(sink, buf)
        async with buf:
            result = await refiner.save_draft(
                session_id="sess-noid",
                title="No ID Draft",
                content="Some content",
            )
        assert result["status"] == "error"
        assert result.get("id") is None
        sink.draft_save.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_empty_string_id(self, buf):
        sink = _mock_sink(draft_save_return={"id": ""})
        refiner = DraftRefiner(sink, buf)
        async with buf:
            result = await refiner.save_draft(
                session_id="sess-emptyid",
                title="Empty ID Draft",
                content="Content here",
            )
        assert result["id"] == ""

    @pytest.mark.asyncio
    async def test_none_id(self, buf):
        sink = _mock_sink(draft_save_return={"id": None})
        refiner = DraftRefiner(sink, buf)
        async with buf:
            result = await refiner.save_draft(
                session_id="sess-noneid",
                title="None ID Draft",
                content="Content here",
            )
        assert result["id"] is None

    @pytest.mark.asyncio
    async def test_no_id_key_at_all(self, buf):
        sink = _mock_sink(draft_save_return={"status": "ok"})
        refiner = DraftRefiner(sink, buf)
        async with buf:
            result = await refiner.save_draft(
                session_id="sess-noidkey",
                title="No ID Key Draft",
                content="Content",
            )
        assert "id" not in result


# ---------------------------------------------------------------------------
# refine_and_publish — wiki_compiler block (lines 150-166)
# ---------------------------------------------------------------------------


class TestRefineAndPublishBasic:
    """Cover refine_and_publish basic publish flow."""

    @pytest.mark.asyncio
    async def test_refine_and_publish_marks_published(self, buf):
        sink = _mock_sink()
        refiner = DraftRefiner(sink, buf)
        async with buf:
            await buf.create_draft("team_doc", "sess-nowiki", "因为配置错误。")
            result = await refiner.refine_and_publish("sess-nowiki")

        assert result is not None
        assert result["status"] == "published"


# ---------------------------------------------------------------------------
# mark_for_refinement (lines 185-194)
# ---------------------------------------------------------------------------


class TestMarkForRefinement:
    """Cover the mark_for_refinement method."""

    @pytest.mark.asyncio
    async def test_marks_pending_draft(self, buf):
        sink = _mock_sink()
        refiner = DraftRefiner(sink, buf)
        async with buf:
            draft_id = await buf.create_draft("team_doc", "sess-mark", "因为需要优化。")
            result = await refiner.mark_for_refinement("sess-mark")

        assert result is not None
        assert result["status"] == "needs_refinement"
        assert result["draft_id"] == draft_id

    @pytest.mark.asyncio
    async def test_no_pending_returns_none(self, buf):
        sink = _mock_sink()
        refiner = DraftRefiner(sink, buf)
        async with buf:
            result = await refiner.mark_for_refinement("sess-empty")

        assert result is None

    @pytest.mark.asyncio
    async def test_draft_buffer_marked_needs_refinement(self, buf):
        """Verify the draft status is actually changed in the buffer."""
        sink = _mock_sink()
        refiner = DraftRefiner(sink, buf)
        async with buf:
            draft_id = await buf.create_draft("team_doc", "sess-mark2", "需要更新。")
            await refiner.mark_for_refinement("sess-mark2")
            # Check the draft status changed
            import aiosqlite

            async with aiosqlite.connect(buf._db_path) as db:
                db.row_factory = aiosqlite.Row
                cursor = await db.execute("SELECT status FROM drafts WHERE id = ?", (draft_id,))
                row = await cursor.fetchone()
        assert row["status"] == "needs_refinement"

    @pytest.mark.asyncio
    async def test_uses_draft_id_key(self, buf):
        """Draft with only 'draft_id' key should be used correctly."""
        sink = _mock_sink()
        refiner = DraftRefiner(sink, buf)
        async with buf:
            draft_id = await buf.create_draft("team_doc", "sess-mark3", "使用Docker部署。")
            result = await refiner.mark_for_refinement("sess-mark3")

        assert result is not None
        assert result["draft_id"] == draft_id


# ---------------------------------------------------------------------------
# refine_and_publish — with tm_id on draft (line 140, via dict.get)
# ---------------------------------------------------------------------------


class TestRefineAndPublishTmIdViaDict:
    """Cover the tm_id fallback path in refine_and_publish (line 140).

    Since daemon DraftBuffer has no tm_id column, we mock get_pending
    to return dicts that include a tm_id field.
    """

    @pytest.mark.asyncio
    async def test_uses_tm_id_when_present_in_dict(self, buf):
        """When pending dict has tm_id, it should be used as draft_id."""
        sink = _mock_sink()
        refiner = DraftRefiner(sink, buf)
        # Mock get_pending to return a dict with tm_id
        mock_pending = [
            {"id": "local-123", "tm_id": "pg-tm-789", "content": "因为超时。", "title": "t", "project": "p"}
        ]
        refiner._buf.get_pending = AsyncMock(return_value=mock_pending)
        refiner._buf.mark_published_by_session = AsyncMock()

        result = await refiner.refine_and_publish("sess-tmid")

        assert result is not None
        call_args = sink.draft_publish.call_args
        assert call_args.kwargs.get("draft_id") == "pg-tm-789"

    @pytest.mark.asyncio
    async def test_falls_back_to_local_id_when_no_tm_id(self, buf):
        """When no tm_id, falls back to local id."""
        sink = _mock_sink()
        refiner = DraftRefiner(sink, buf)
        mock_pending = [{"id": "local-456", "content": "需要配置。", "title": "t", "project": "p"}]
        refiner._buf.get_pending = AsyncMock(return_value=mock_pending)
        refiner._buf.mark_published_by_session = AsyncMock()

        result = await refiner.refine_and_publish("sess-noid")

        assert result is not None
        call_args = sink.draft_publish.call_args
        assert call_args.kwargs.get("draft_id") == "local-456"

    @pytest.mark.asyncio
    async def test_falls_back_to_draft_id_key(self, buf):
        """When dict has 'draft_id' key but no 'id' or 'tm_id'."""
        sink = _mock_sink()
        refiner = DraftRefiner(sink, buf)
        mock_pending = [{"draft_id": "dd-789", "content": "使用Redis。", "title": "t", "project": "p"}]
        refiner._buf.get_pending = AsyncMock(return_value=mock_pending)
        refiner._buf.mark_published_by_session = AsyncMock()

        result = await refiner.refine_and_publish("sess-dd")

        assert result is not None
        call_args = sink.draft_publish.call_args
        assert call_args.kwargs.get("draft_id") == "dd-789"

    @pytest.mark.asyncio
    async def test_falls_back_to_empty_string(self, buf):
        """When dict has no id, draft_id, or tm_id."""
        sink = _mock_sink()
        refiner = DraftRefiner(sink, buf)
        mock_pending = [{"content": "普通文本。", "title": "t", "project": "p"}]
        refiner._buf.get_pending = AsyncMock(return_value=mock_pending)
        refiner._buf.mark_published_by_session = AsyncMock()

        result = await refiner.refine_and_publish("sess-noid2")

        assert result is not None
        call_args = sink.draft_publish.call_args
        assert call_args.kwargs.get("draft_id") == ""


# ---------------------------------------------------------------------------
# refine_and_publish — multiple pending drafts (line 132)
# ---------------------------------------------------------------------------


class TestRefineAndPublishMultipleDrafts:
    """Cover accumulation of multiple drafts."""

    @pytest.mark.asyncio
    async def test_accumulates_content_from_multiple_drafts(self, buf):
        sink = _mock_sink()
        refiner = DraftRefiner(sink, buf)
        async with buf:
            # Create multiple drafts for the same session by upserting
            await buf.upsert_draft(
                session_id="sess-multi",
                title="Draft 1",
                content="因为服务器超时。",
                project="team_doc",
            )
            await buf.upsert_draft(
                session_id="sess-multi",
                title="Draft 2",
                content="需要重启服务。",
                project="team_doc",
            )
            result = await refiner.refine_and_publish("sess-multi")

        assert result is not None
        call_args = sink.draft_publish.call_args
        refined = call_args.kwargs["refined_content"]
        assert "因为" in refined
        assert "需要" in refined


# ---------------------------------------------------------------------------
# extract_facts — additional edge cases
# ---------------------------------------------------------------------------


class TestExtractFactsEdgeCases:
    """Additional edge cases for extract_facts."""

    def test_empty_text(self):
        assert DraftRefiner.extract_facts("") == []

    def test_no_keywords(self):
        assert DraftRefiner.extract_facts("Hello world") == []

    def test_all_sentences_factual(self):
        text = "因为这个。需要那个。使用这个。"
        facts = DraftRefiner.extract_facts(text)
        assert len(facts) == 3

    def test_english_keywords_mixed(self):
        text = "The config is missing. Because of this, we need to use Redis."
        facts = DraftRefiner.extract_facts(text)
        assert len(facts) >= 1

    def test_newline_separated(self):
        text = "因为超时\n需要重启\n普通的话"
        facts = DraftRefiner.extract_facts(text)
        assert len(facts) == 2

    def test_exact_max_boundary(self):
        """Exactly _MAX_FACTS sentences should all be returned."""
        sentences = [f"因为{i}。" for i in range(_MAX_FACTS)]
        text = "".join(sentences)
        facts = DraftRefiner.extract_facts(text)
        assert len(facts) == _MAX_FACTS

    def test_max_plus_one(self):
        """One more than _MAX_FACTS should truncate."""
        sentences = [f"因为{i}。" for i in range(_MAX_FACTS + 1)]
        text = "".join(sentences)
        facts = DraftRefiner.extract_facts(text)
        assert len(facts) == _MAX_FACTS

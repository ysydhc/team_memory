"""Tests for scripts/hooks/draft_refiner.py — fact extraction and draft publishing."""

from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

# Ensure scripts/hooks/ is importable
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts" / "hooks"))

import draft_refiner  # noqa: E402
from draft_buffer import DraftBuffer  # noqa: E402
from draft_refiner import DraftRefiner  # noqa: E402


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
# extract_facts — factual sentences
# ---------------------------------------------------------------------------


class TestExtractFactsFactual:
    """extract_facts finds sentences containing factual keywords."""

    def test_chinese_keyword_shi(self):
        text = "这是一个重要的配置。普通的话。"
        facts = DraftRefiner.extract_facts(text)
        assert any("这是" in f for f in facts)

    def test_chinese_keyword_yinwei(self):
        text = "因为服务器配置错误导致超时。没有什么特别的。"
        facts = DraftRefiner.extract_facts(text)
        assert any("因为" in f for f in facts)

    def test_chinese_keyword_suoyi(self):
        text = "所以需要重启服务。随便聊聊。"
        facts = DraftRefiner.extract_facts(text)
        assert any("所以" in f for f in facts)

    def test_chinese_keyword_xuyao(self):
        text = "需要更新配置文件。今天天气不错。"
        facts = DraftRefiner.extract_facts(text)
        assert any("需要" in f for f in facts)

    def test_chinese_keyword_shiyong(self):
        text = "使用Docker部署应用。你好。"
        facts = DraftRefiner.extract_facts(text)
        assert any("使用" in f for f in facts)

    def test_chinese_keyword_peizhi(self):
        text = "配置文件在/etc/app/config.yaml。我忘了。"
        facts = DraftRefiner.extract_facts(text)
        assert any("配置" in f for f in facts)

    def test_chinese_keyword_yuanyin(self):
        text = "原因在于内存不足。哈哈。"
        facts = DraftRefiner.extract_facts(text)
        assert any("原因" in f for f in facts)

    def test_chinese_keyword_fangfa(self):
        text = "方法是通过API调用。早上好。"
        facts = DraftRefiner.extract_facts(text)
        assert any("方法" in f for f in facts)

    def test_chinese_keyword_jiejue(self):
        text = "解决方案是增加超时时间。没事。"
        facts = DraftRefiner.extract_facts(text)
        assert any("解决" in f for f in facts)

    def test_english_keyword_is(self):
        text = "The server is running. Hello there."
        facts = DraftRefiner.extract_facts(text)
        assert any("is" in f for f in facts)

    def test_english_keyword_because(self):
        text = "It failed because of timeout. Nice day."
        facts = DraftRefiner.extract_facts(text)
        assert any("because" in f for f in facts)

    def test_english_keyword_need(self):
        text = "We need to restart. Hi."
        facts = DraftRefiner.extract_facts(text)
        assert any("need" in f for f in facts)

    def test_english_keyword_use(self):
        text = "We use Redis for caching. Bye."
        facts = DraftRefiner.extract_facts(text)
        assert any("use" in f for f in facts)

    def test_english_keyword_config(self):
        text = "The config file is missing. Hello."
        facts = DraftRefiner.extract_facts(text)
        assert any("config" in f for f in facts)

    def test_multiple_facts(self):
        text = "因为服务器超时。所以需要重启。普通的话。"
        facts = DraftRefiner.extract_facts(text)
        assert len(facts) == 2


# ---------------------------------------------------------------------------
# extract_facts — non-factual text
# ---------------------------------------------------------------------------


class TestExtractFactsNonFactual:
    """extract_facts returns empty for text without factual keywords."""

    def test_plain_chinese(self):
        text = "今天天气不错。我们去吃饭吧。"
        facts = DraftRefiner.extract_facts(text)
        assert facts == []

    def test_plain_english(self):
        text = "Hello there. How are you doing today?"
        facts = DraftRefiner.extract_facts(text)
        assert facts == []

    def test_empty_string(self):
        facts = DraftRefiner.extract_facts("")
        assert facts == []

    def test_none_like_empty(self):
        facts = DraftRefiner.extract_facts("")
        assert facts == []


# ---------------------------------------------------------------------------
# extract_facts — max 10 sentences
# ---------------------------------------------------------------------------


class TestExtractFactsMax10:
    """extract_facts returns at most 10 sentences."""

    def test_limits_to_10(self):
        # Build 15 factual sentences
        sentences = [f"因为原因{i}。" for i in range(15)]
        text = "".join(sentences)
        facts = DraftRefiner.extract_facts(text)
        assert len(facts) == 10

    def test_under_10_returns_all(self):
        sentences = [f"因为原因{i}。" for i in range(5)]
        text = "".join(sentences)
        facts = DraftRefiner.extract_facts(text)
        assert len(facts) == 5


# ---------------------------------------------------------------------------
# save_draft — calls tm_client and draft_buffer
# ---------------------------------------------------------------------------


class TestSaveDraft:
    """save_draft calls tm_client.draft_save and draft_buffer.upsert_draft."""

    @pytest.mark.asyncio
    async def test_calls_tm_client(self, buf):
        tm = _make_tm_client()
        refiner = DraftRefiner(tm, buf)
        async with buf:
            result = await refiner.save_draft(
                session_id="sess-1",
                title="Test Draft",
                content="Some content",
                project="team_doc",
            )
        tm.draft_save.assert_awaited_once_with(
            title="Test Draft",
            content="Some content",
            project="team_doc",
            group_key=None,
            conversation_id="sess-1",
        )
        assert result["id"] == "tm-draft-001"

    @pytest.mark.asyncio
    async def test_calls_draft_buffer(self, buf):
        tm = _make_tm_client()
        refiner = DraftRefiner(tm, buf)
        async with buf:
            await refiner.save_draft(
                session_id="sess-1",
                title="Test Draft",
                content="Some content",
                project="team_doc",
            )
            # Verify draft was stored in local buffer
            pending = await buf.get_pending("sess-1")
        assert len(pending) == 1
        assert pending[0]["content"] == "Some content"

    @pytest.mark.asyncio
    async def test_with_group_key(self, buf):
        tm = _make_tm_client()
        refiner = DraftRefiner(tm, buf)
        async with buf:
            result = await refiner.save_draft(
                session_id="sess-1",
                title="Title",
                content="Content",
                project="team_doc",
                group_key="gk-1",
            )
        tm.draft_save.assert_awaited_once_with(
            title="Title",
            content="Content",
            project="team_doc",
            group_key="gk-1",
            conversation_id="sess-1",
        )
        assert result["id"] == "tm-draft-001"


# ---------------------------------------------------------------------------
# refine_and_publish — no pending drafts
# ---------------------------------------------------------------------------


class TestRefineAndPublishNoPending:
    """refine_and_publish returns None when no pending drafts exist."""

    @pytest.mark.asyncio
    async def test_returns_none_no_pending(self, buf):
        tm = _make_tm_client()
        refiner = DraftRefiner(tm, buf)
        async with buf:
            result = await refiner.refine_and_publish("sess-empty")
        assert result is None
        tm.draft_publish.assert_not_awaited()


# ---------------------------------------------------------------------------
# refine_and_publish — with pending drafts
# ---------------------------------------------------------------------------


class TestRefineAndPublishWithPending:
    """refine_and_publish publishes and marks pending drafts."""

    @pytest.mark.asyncio
    async def test_publishes_and_marks(self, buf):
        tm = _make_tm_client()
        refiner = DraftRefiner(tm, buf)
        async with buf:
            # Create a pending draft in the buffer
            draft_id = await buf.create_draft("team_doc", "sess-1", "因为服务器超时。")

            result = await refiner.refine_and_publish("sess-1")

        assert result is not None
        assert result["status"] == "published"
        assert result["draft_id"] == draft_id
        tm.draft_publish.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_marks_published_in_buffer(self, buf):
        tm = _make_tm_client()
        refiner = DraftRefiner(tm, buf)
        async with buf:
            await buf.create_draft("team_doc", "sess-1", "因为服务器超时。")
            await refiner.refine_and_publish("sess-1")
            # After publishing, no more pending drafts for this session
            pending = await buf.get_pending("sess-1")
        assert len(pending) == 0


# ---------------------------------------------------------------------------
# refine_and_publish — extracts facts and uses as refined_content
# ---------------------------------------------------------------------------


class TestRefineAndPublishExtractsFacts:
    """refine_and_publish extracts facts and passes them as refined_content."""

    @pytest.mark.asyncio
    async def test_uses_extracted_facts(self, buf):
        tm = _make_tm_client()
        refiner = DraftRefiner(tm, buf)
        async with buf:
            await buf.create_draft(
                "team_doc",
                "sess-1",
                "因为服务器配置错误。普通的话。所以需要重启。",
            )
            await refiner.refine_and_publish("sess-1")

        # Check that draft_publish was called with refined_content containing facts
        call_args = tm.draft_publish.call_args
        refined_content = call_args.kwargs.get("refined_content") or call_args[1].get(
            "refined_content"
        ) or call_args[0][1] if len(call_args[0]) > 1 else call_args.kwargs["refined_content"]

        assert "因为" in refined_content
        assert "所以" in refined_content
        # The non-factual sentence should not be in refined_content
        assert "普通的话" not in refined_content

    @pytest.mark.asyncio
    async def test_no_facts_uses_accumulated(self, buf):
        tm = _make_tm_client()
        refiner = DraftRefiner(tm, buf)
        async with buf:
            await buf.create_draft("team_doc", "sess-1", "今天天气不错。")
            await refiner.refine_and_publish("sess-1")

        # When no facts extracted, refined_content should be the accumulated text
        call_args = tm.draft_publish.call_args
        refined_content = call_args.kwargs.get("refined_content") or call_args.kwargs["refined_content"]
        assert "今天天气不错" in refined_content

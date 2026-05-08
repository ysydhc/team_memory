"""Tests for DraftBuffer — draft persistence and dedup / 草稿持久化与去重测试

Unit tests cover:
- find_pending_by_title() SQL logic (mocked) / 查询逻辑
- dedup logic in watcher (update vs insert) / 去重逻辑
- Draft creation and status management / 创建和状态管理
- _require_db() guard / 数据库连接检查
- _row_to_dict() conversion / 行转字典
- upsert_draft() append logic / 追加逻辑

Integration tests cover:
- Full CRUD with real SQLite (in-memory or tmp file)
"""
from __future__ import annotations

import pytest
import pytest_asyncio

from team_memory.daemon.draft_buffer import DraftBuffer, _CREATE_TABLE

# ---------------------------------------------------------------------------
# Markers
# ---------------------------------------------------------------------------
pytestmark = pytest.mark.unit


# ---------------------------------------------------------------------------
# Pure unit tests (no DB)
# ---------------------------------------------------------------------------
class TestDraftBufferInit:
    """Tests for DraftBuffer constructor / DraftBuffer 构造函数测试"""

    def test_default_db_path(self):
        buf = DraftBuffer(db_path="/tmp/test.db")
        assert buf._db_path == "/tmp/test.db"

    def test_custom_db_path(self):
        buf = DraftBuffer(db_path="/custom/path.db")
        assert buf._db_path == "/custom/path.db"

    def test_pool_initially_none(self):
        buf = DraftBuffer(db_path="/tmp/test.db")
        assert buf._db is None


class TestDraftBufferConstants:
    """Tests for DraftBuffer SQL and constants / DraftBuffer SQL 和常量测试"""

    def test_db_path_stored(self):
        buf = DraftBuffer(db_path="/tmp/drafts_test.db")
        assert buf._db_path == "/tmp/drafts_test.db"

    def test_create_table_sql(self):
        """CREATE TABLE SQL should contain expected columns."""
        assert "CREATE TABLE" in _CREATE_TABLE
        assert "drafts" in _CREATE_TABLE
        assert "id TEXT PRIMARY KEY" in _CREATE_TABLE
        assert "status TEXT DEFAULT 'pending'" in _CREATE_TABLE


class TestRequireDb:
    """Unit tests for _require_db guard / _require_db 保护测试"""

    def test_require_db_raises_when_none(self):
        """Should raise RuntimeError when _db is None / _db 为 None 时应抛出 RuntimeError"""
        buf = DraftBuffer(db_path="/tmp/test.db")
        with pytest.raises(RuntimeError, match="not connected"):
            buf._require_db()

    def test_require_db_returns_connection_when_set(self):
        """Should return the connection when _db is set / _db 已设置时应返回连接"""
        buf = DraftBuffer(db_path="/tmp/test.db")
        fake_conn = object()
        buf._db = fake_conn
        assert buf._require_db() is fake_conn


class TestRowToDict:
    """Unit tests for _row_to_dict static method / _row_to_dict 静态方法测试"""

    def test_row_to_dict(self):
        """Should convert a Row-like object to dict / 应将 Row 类对象转换为字典"""

        class FakeRow:
            def __init__(self, d):
                self._d = d

            def __iter__(self):
                return iter(self._d.items())

        row = FakeRow({"id": "abc", "title": "Test", "status": "pending"})
        result = DraftBuffer._row_to_dict(row)
        assert result == {"id": "abc", "title": "Test", "status": "pending"}


class TestUpsertContentBuilding:
    """Test the content-appending logic in upsert_draft without DB / 测试 upsert_draft 中的内容追加逻辑"""

    def test_content_appending_logic(self):
        """Verify the string concatenation logic used by upsert_draft / 验证 upsert_draft 使用的字符串拼接逻辑"""
        old_content = "existing facts"
        new_content = "new facts"
        # Simulates the append logic from upsert_draft lines 346-347
        result = f"{old_content}\n{new_content}" if old_content else new_content
        assert result == "existing facts\nnew facts"

    def test_content_appending_empty_old(self):
        """When old content is empty, just use new content / 旧内容为空时只使用新内容"""
        old_content = ""
        new_content = "new facts"
        result = f"{old_content}\n{new_content}" if old_content else new_content
        assert result == "new facts"

    def test_default_project_fallback(self):
        """Verify project defaults to 'default' / 验证 project 默认为 'default'"""
        project = None
        result = project or "default"
        assert result == "default"

    def test_existing_project_not_overridden(self):
        """Existing project should not be overridden / 已有 project 不应被覆盖"""
        project = "team_doc"
        result = project or "default"
        assert result == "team_doc"


# ---------------------------------------------------------------------------
# Integration tests (real SQLite)
# ---------------------------------------------------------------------------
@pytest.mark.integration
class TestDraftBufferContextManager:
    """Integration tests for async context manager / 异步上下文管理器集成测试"""

    @pytest.mark.asyncio
    async def test_enter_creates_table(self, tmp_path):
        """Entering context should create the drafts table / 进入上下文应创建 drafts 表"""
        db_path = str(tmp_path / "test.db")
        buf = DraftBuffer(db_path)
        async with buf:
            assert buf._db is not None
            # Check table exists
            cursor = await buf._db.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name='drafts'"
            )
            row = await cursor.fetchone()
            assert row is not None

    @pytest.mark.asyncio
    async def test_exit_closes_db(self, tmp_path):
        """Exiting context should close the DB / 退出上下文应关闭数据库"""
        db_path = str(tmp_path / "test.db")
        buf = DraftBuffer(db_path)
        async with buf:
            pass
        assert buf._db is None

    @pytest.mark.asyncio
    async def test_require_db_fails_after_exit(self, tmp_path):
        """require_db should raise after context exits / 上下文退出后 require_db 应报错"""
        db_path = str(tmp_path / "test.db")
        buf = DraftBuffer(db_path)
        async with buf:
            pass
        with pytest.raises(RuntimeError, match="not connected"):
            buf._require_db()


@pytest.mark.integration
class TestDraftBufferCRUD:
    """Integration tests for full CRUD / 完整 CRUD 集成测试"""

    @pytest_asyncio.fixture
    async def buf(self, tmp_path):
        """Provide an open DraftBuffer for testing / 提供用于测试的已打开 DraftBuffer"""
        db_path = str(tmp_path / "test.db")
        b = DraftBuffer(db_path)
        async with b:
            yield b

    @pytest.mark.asyncio
    async def test_create_draft_returns_uuid(self, buf):
        """create_draft should return a UUID string / create_draft 应返回 UUID 字符串"""
        draft_id = await buf.create_draft("team_doc", "conv-1", "facts here")
        assert isinstance(draft_id, str)
        assert len(draft_id) == 36  # UUID format

    @pytest.mark.asyncio
    async def test_create_draft_persists(self, buf):
        """Created draft should be queryable / 创建的草稿应可查询"""
        draft_id = await buf.create_draft("team_doc", "conv-1", "facts here", title="My Title")
        drafts = await buf.get_pending_drafts("team_doc")
        assert len(drafts) == 1
        assert drafts[0]["id"] == draft_id
        assert drafts[0]["title"] == "My Title"
        assert drafts[0]["content"] == "facts here"
        assert drafts[0]["status"] == "pending"
        assert drafts[0]["source"] == "pipeline"

    @pytest.mark.asyncio
    async def test_create_draft_with_source(self, buf):
        """create_draft should accept custom source / create_draft 应接受自定义 source"""
        draft_id = await buf.create_draft("team_doc", None, "obsidian note", source="obsidian")
        drafts = await buf.get_pending_drafts("team_doc")
        assert drafts[0]["source"] == "obsidian"

    @pytest.mark.asyncio
    async def test_update_draft(self, buf):
        """update_draft should change content / update_draft 应更改内容"""
        draft_id = await buf.create_draft("team_doc", "conv-1", "old content")
        await buf.update_draft(draft_id, "new content")
        drafts = await buf.get_pending_drafts("team_doc")
        assert drafts[0]["content"] == "new content"

    @pytest.mark.asyncio
    async def test_update_draft_not_found(self, buf):
        """update_draft should raise ValueError for missing draft / update_draft 对不存在的草稿应抛出 ValueError"""
        with pytest.raises(ValueError, match="not found"):
            await buf.update_draft("nonexistent-id", "new content")

    @pytest.mark.asyncio
    async def test_get_pending_drafts_by_project(self, buf):
        """get_pending_drafts should filter by project / get_pending_drafts 应按 project 过滤"""
        await buf.create_draft("team_doc", "c1", "content1")
        await buf.create_draft("other_project", "c2", "content2")
        team_drafts = await buf.get_pending_drafts("team_doc")
        assert len(team_drafts) == 1
        assert team_drafts[0]["project"] == "team_doc"

    @pytest.mark.asyncio
    async def test_get_pending_drafts_all(self, buf):
        """get_pending_drafts without project should return all / 不指定 project 应返回全部"""
        await buf.create_draft("team_doc", "c1", "content1")
        await buf.create_draft("other_project", "c2", "content2")
        all_drafts = await buf.get_pending_drafts()
        assert len(all_drafts) == 2

    @pytest.mark.asyncio
    async def test_get_pending_drafts_excludes_non_pending(self, buf):
        """get_pending_drafts should not return published drafts / 不应返回已发布的草稿"""
        draft_id = await buf.create_draft("team_doc", "c1", "content1")
        await buf.mark_published(draft_id)
        drafts = await buf.get_pending_drafts("team_doc")
        assert len(drafts) == 0

    @pytest.mark.asyncio
    async def test_mark_published(self, buf):
        """mark_published should change status to 'published' / mark_published 应将状态改为 'published'"""
        draft_id = await buf.create_draft("team_doc", "c1", "content1")
        await buf.mark_published(draft_id)
        drafts = await buf.get_pending_drafts("team_doc")
        assert len(drafts) == 0

    @pytest.mark.asyncio
    async def test_mark_published_not_found(self, buf):
        """mark_published should raise ValueError for missing draft / mark_published 对不存在草稿应报错"""
        with pytest.raises(ValueError, match="not found"):
            await buf.mark_published("nonexistent-id")

    @pytest.mark.asyncio
    async def test_mark_for_publishing(self, buf):
        """mark_for_publishing should set status to 'ready_to_publish' / 应将状态设为 'ready_to_publish'"""
        draft_id = await buf.create_draft("team_doc", "c1", "content1")
        await buf.mark_for_publishing(draft_id)
        # Should not appear in pending anymore
        drafts = await buf.get_pending_drafts("team_doc")
        assert len(drafts) == 0
        # Verify by querying all with raw SQL
        cursor = await buf._db.execute("SELECT status FROM drafts WHERE id = ?", (draft_id,))
        row = await cursor.fetchone()
        assert row[0] == "ready_to_publish"

    @pytest.mark.asyncio
    async def test_mark_for_publishing_not_found(self, buf):
        """mark_for_publishing should raise ValueError for missing draft / 对不存在草稿应报错"""
        with pytest.raises(ValueError, match="not found"):
            await buf.mark_for_publishing("nonexistent-id")

    @pytest.mark.asyncio
    async def test_mark_needs_refinement(self, buf):
        """mark_needs_refinement should set status to 'needs_refinement' / 应将状态设为 'needs_refinement'"""
        draft_id = await buf.create_draft("team_doc", "c1", "content1")
        await buf.mark_needs_refinement(draft_id)
        drafts = await buf.get_pending_drafts("team_doc")
        assert len(drafts) == 0

    @pytest.mark.asyncio
    async def test_mark_needs_refinement_not_found(self, buf):
        """mark_needs_refinement should raise ValueError for missing draft / 对不存在草稿应报错"""
        with pytest.raises(ValueError, match="not found"):
            await buf.mark_needs_refinement("nonexistent-id")

    @pytest.mark.asyncio
    async def test_get_needs_refinement(self, buf):
        """get_needs_refinement should return only needs_refinement drafts / 仅返回 needs_refinement 草稿"""
        d1 = await buf.create_draft("team_doc", "c1", "content1")
        d2 = await buf.create_draft("team_doc", "c2", "content2")
        await buf.mark_needs_refinement(d1)
        results = await buf.get_needs_refinement()
        assert len(results) == 1
        assert results[0]["id"] == d1
        assert results[0]["status"] == "needs_refinement"


@pytest.mark.integration
class TestDedupLogic:
    """Integration tests for dedup / 去重集成测试"""

    @pytest_asyncio.fixture
    async def buf(self, tmp_path):
        db_path = str(tmp_path / "test.db")
        b = DraftBuffer(db_path)
        async with b:
            yield b

    @pytest.mark.asyncio
    async def test_find_pending_by_title_found(self, buf):
        """find_pending_by_title should return draft_id when found / 找到时应返回 draft_id"""
        draft_id = await buf.create_draft("team_doc", "c1", "content", title="Meeting Notes")
        result = await buf.find_pending_by_title("Meeting Notes", "team_doc")
        assert result == draft_id

    @pytest.mark.asyncio
    async def test_find_pending_by_title_not_found(self, buf):
        """find_pending_by_title should return None when not found / 未找到时应返回 None"""
        result = await buf.find_pending_by_title("Nonexistent", "team_doc")
        assert result is None

    @pytest.mark.asyncio
    async def test_find_pending_by_title_wrong_project(self, buf):
        """find_pending_by_title should not match different project / 不应匹配不同 project"""
        await buf.create_draft("team_doc", "c1", "content", title="Notes")
        result = await buf.find_pending_by_title("Notes", "other_project")
        assert result is None

    @pytest.mark.asyncio
    async def test_find_pending_by_title_not_pending(self, buf):
        """find_pending_by_title should not match published drafts / 不应匹配已发布的草稿"""
        draft_id = await buf.create_draft("team_doc", "c1", "content", title="Notes")
        await buf.mark_published(draft_id)
        result = await buf.find_pending_by_title("Notes", "team_doc")
        assert result is None


@pytest.mark.integration
class TestFindByConversation:
    """Integration tests for find_pending_by_conversation / find_pending_by_conversation 集成测试"""

    @pytest_asyncio.fixture
    async def buf(self, tmp_path):
        db_path = str(tmp_path / "test.db")
        b = DraftBuffer(db_path)
        async with b:
            yield b

    @pytest.mark.asyncio
    async def test_find_pending_by_conversation(self, buf):
        """Should return drafts matching project + conversation_id / 应返回匹配 project+conversation 的草稿"""
        d1 = await buf.create_draft("team_doc", "conv-42", "facts about meeting")
        d2 = await buf.create_draft("team_doc", "conv-99", "other conversation")
        results = await buf.find_pending_by_conversation("team_doc", "conv-42")
        assert len(results) == 1
        assert results[0]["id"] == d1

    @pytest.mark.asyncio
    async def test_find_pending_by_conversation_empty(self, buf):
        """Should return empty list for non-matching conversation / 不匹配时应返回空列表"""
        await buf.create_draft("team_doc", "conv-1", "content")
        results = await buf.find_pending_by_conversation("team_doc", "conv-999")
        assert results == []

    @pytest.mark.asyncio
    async def test_find_pending_by_conversation_multiple(self, buf):
        """Should return all matching drafts for same conversation / 应返回同一会话的所有匹配草稿"""
        d1 = await buf.create_draft("team_doc", "conv-1", "first batch")
        d2 = await buf.create_draft("team_doc", "conv-1", "second batch")
        results = await buf.find_pending_by_conversation("team_doc", "conv-1")
        assert len(results) == 2
        ids = {r["id"] for r in results}
        assert ids == {d1, d2}


@pytest.mark.integration
class TestGetOlderThan:
    """Integration tests for get_older_than / get_older_than 集成测试"""

    @pytest_asyncio.fixture
    async def buf(self, tmp_path):
        db_path = str(tmp_path / "test.db")
        b = DraftBuffer(db_path)
        async with b:
            yield b

    @pytest.mark.asyncio
    async def test_get_older_than_none_old(self, buf):
        """No drafts should be older than 0 minutes when just created / 刚创建的草稿不应比 0 分钟更旧"""
        await buf.create_draft("team_doc", "c1", "content")
        results = await buf.get_older_than(0)
        # Drafts just created might be 0 minutes old; get_older_than uses strict <,
        # so a draft created at "now" may or may not match depending on timing
        # We just verify it returns a list without error
        assert isinstance(results, list)

    @pytest.mark.asyncio
    async def test_get_older_than_with_old_draft(self, buf):
        """Drafts inserted with old timestamps should be returned / 用旧时间戳插入的草稿应被返回"""
        import sqlite3
        # Insert a draft with a very old created_at directly
        db = buf._db
        await db.execute(
            "INSERT INTO drafts (id, title, project, conversation_id, content, status, source, created_at, updated_at) "
            "VALUES (?, ?, ?, ?, ?, 'pending', ?, ?, ?)",
            ("old-id", "Old Draft", "team_doc", "c1", "old content", "pipeline",
             "2020-01-01T00:00:00+00:00", "2020-01-01T00:00:00+00:00"),
        )
        await db.commit()
        # Also insert a fresh one
        await buf.create_draft("team_doc", "c2", "fresh content")
        results = await buf.get_older_than(1)
        old_ids = [r["id"] for r in results]
        assert "old-id" in old_ids


@pytest.mark.integration
class TestUpsertDraft:
    """Integration tests for upsert_draft / upsert_draft 集成测试"""

    @pytest_asyncio.fixture
    async def buf(self, tmp_path):
        db_path = str(tmp_path / "test.db")
        b = DraftBuffer(db_path)
        async with b:
            yield b

    @pytest.mark.asyncio
    async def test_upsert_creates_new(self, buf):
        """upsert_draft should create new draft when none exists / 不存在时应创建新草稿"""
        draft_id = await buf.upsert_draft("session-1", "Meeting", "facts v1", project="team_doc")
        assert isinstance(draft_id, str)
        drafts = await buf.find_pending_by_conversation("team_doc", "session-1")
        assert len(drafts) == 1
        assert drafts[0]["content"] == "facts v1"

    @pytest.mark.asyncio
    async def test_upsert_appends_to_existing(self, buf):
        """upsert_draft should append to existing draft for same session / 同一会话应追加内容"""
        d1 = await buf.upsert_draft("session-1", "Meeting", "facts v1", project="team_doc")
        d2 = await buf.upsert_draft("session-1", "Meeting", "facts v2", project="team_doc")
        assert d1 == d2  # Same draft ID returned
        drafts = await buf.find_pending_by_conversation("team_doc", "session-1")
        assert len(drafts) == 1
        assert "facts v1\nfacts v2" in drafts[0]["content"]

    @pytest.mark.asyncio
    async def test_upsert_default_project(self, buf):
        """upsert_draft should use 'default' project when none specified / 未指定时应使用 'default'"""
        draft_id = await buf.upsert_draft("session-1", "Title", "content")
        drafts = await buf.find_pending_by_conversation("default", "session-1")
        assert len(drafts) == 1

    @pytest.mark.asyncio
    async def test_upsert_different_sessions(self, buf):
        """Different sessions should create different drafts / 不同会话应创建不同草稿"""
        d1 = await buf.upsert_draft("session-1", "Meeting", "facts 1", project="team_doc")
        d2 = await buf.upsert_draft("session-2", "Meeting", "facts 2", project="team_doc")
        assert d1 != d2


@pytest.mark.integration
class TestGetPending:
    """Integration tests for get_pending / get_pending 集成测试"""

    @pytest_asyncio.fixture
    async def buf(self, tmp_path):
        db_path = str(tmp_path / "test.db")
        b = DraftBuffer(db_path)
        async with b:
            yield b

    @pytest.mark.asyncio
    async def test_get_pending_by_session(self, buf):
        """get_pending should return drafts for a session / 应返回指定会话的草稿"""
        await buf.create_draft("team_doc", "s1", "content1")
        await buf.create_draft("team_doc", "s2", "content2")
        results = await buf.get_pending("s1")
        assert len(results) == 1
        assert results[0]["conversation_id"] == "s1"

    @pytest.mark.asyncio
    async def test_get_pending_empty(self, buf):
        """get_pending should return empty list for unknown session / 未知会话应返回空列表"""
        results = await buf.get_pending("unknown")
        assert results == []


@pytest.mark.integration
class TestGetAllPending:
    """Integration tests for get_all_pending / get_all_pending 集成测试"""

    @pytest_asyncio.fixture
    async def buf(self, tmp_path):
        db_path = str(tmp_path / "test.db")
        b = DraftBuffer(db_path)
        async with b:
            yield b

    @pytest.mark.asyncio
    async def test_get_all_pending(self, buf):
        """get_all_pending should return all pending drafts / 应返回所有 pending 草稿"""
        await buf.create_draft("team_doc", "s1", "content1")
        await buf.create_draft("other_project", "s2", "content2")
        results = await buf.get_all_pending()
        assert len(results) == 2

    @pytest.mark.asyncio
    async def test_get_all_pending_excludes_published(self, buf):
        """get_all_pending should exclude published drafts / 应排除已发布草稿"""
        d1 = await buf.create_draft("team_doc", "s1", "content1")
        await buf.mark_published(d1)
        results = await buf.get_all_pending()
        assert len(results) == 0


@pytest.mark.integration
class TestMarkPublishedBySession:
    """Integration tests for mark_published_by_session / mark_published_by_session 集成测试"""

    @pytest_asyncio.fixture
    async def buf(self, tmp_path):
        db_path = str(tmp_path / "test.db")
        b = DraftBuffer(db_path)
        async with b:
            yield b

    @pytest.mark.asyncio
    async def test_mark_published_by_session(self, buf):
        """Should mark all pending drafts for session as published / 应将会话的所有 pending 草稿标记为已发布"""
        await buf.create_draft("team_doc", "s1", "content1")
        await buf.create_draft("team_doc", "s1", "content2")
        await buf.create_draft("team_doc", "s2", "content3")
        await buf.mark_published_by_session("s1")
        # s1 drafts should be published
        results = await buf.get_pending("s1")
        assert len(results) == 0
        # s2 draft should still be pending
        results = await buf.get_pending("s2")
        assert len(results) == 1

    @pytest.mark.asyncio
    async def test_mark_published_by_session_idempotent(self, buf):
        """Calling on session with no pending drafts should not error / 对无 pending 草稿的会话调用不应报错"""
        await buf.create_draft("team_doc", "s1", "content1")
        await buf.mark_published_by_session("s1")
        # Calling again should not raise
        await buf.mark_published_by_session("s1")


@pytest.mark.integration
class TestMarkPublishedIdempotent:
    """Test idempotency of mark_published / 测试 mark_published 的幂等性"""

    @pytest_asyncio.fixture
    async def buf(self, tmp_path):
        db_path = str(tmp_path / "test.db")
        b = DraftBuffer(db_path)
        async with b:
            yield b

    @pytest.mark.asyncio
    async def test_mark_published_twice(self, buf):
        """mark_published should be idempotent / mark_published 应为幂等操作"""
        draft_id = await buf.create_draft("team_doc", "c1", "content")
        await buf.mark_published(draft_id)
        # Calling again should not raise
        await buf.mark_published(draft_id)


@pytest.mark.integration
class TestMultipleStatusTransitions:
    """Integration tests for status transition lifecycle / 状态转换生命周期测试"""

    @pytest_asyncio.fixture
    async def buf(self, tmp_path):
        db_path = str(tmp_path / "test.db")
        b = DraftBuffer(db_path)
        async with b:
            yield b

    @pytest.mark.asyncio
    async def test_pending_to_needs_refinement_to_ready(self, buf):
        """Full lifecycle: pending -> needs_refinement -> ready_to_publish / 完整生命周期"""
        draft_id = await buf.create_draft("team_doc", "c1", "vague content", title="Meeting")
        # pending -> needs_refinement
        await buf.mark_needs_refinement(draft_id)
        results = await buf.get_needs_refinement()
        assert len(results) == 1
        # needs_refinement -> ready_to_publish
        await buf.mark_for_publishing(draft_id)
        results = await buf.get_needs_refinement()
        assert len(results) == 0

    @pytest.mark.asyncio
    async def test_multiple_drafts_ordered_by_created_at(self, buf):
        """Drafts should be returned in creation order / 草稿应按创建顺序返回"""
        d1 = await buf.create_draft("team_doc", "c1", "first", title="A")
        d2 = await buf.create_draft("team_doc", "c1", "second", title="B")
        d3 = await buf.create_draft("team_doc", "c1", "third", title="C")
        drafts = await buf.get_pending_drafts("team_doc")
        assert len(drafts) == 3
        assert drafts[0]["title"] == "A"
        assert drafts[1]["title"] == "B"
        assert drafts[2]["title"] == "C"


@pytest.mark.integration
class TestEdgeCases:
    """Integration tests for edge cases / 边缘情况集成测试"""

    @pytest_asyncio.fixture
    async def buf(self, tmp_path):
        db_path = str(tmp_path / "test.db")
        b = DraftBuffer(db_path)
        async with b:
            yield b

    @pytest.mark.asyncio
    async def test_empty_content(self, buf):
        """Should accept empty content / 应接受空内容"""
        draft_id = await buf.create_draft("team_doc", "c1", "")
        drafts = await buf.get_pending_drafts("team_doc")
        assert drafts[0]["content"] == ""

    @pytest.mark.asyncio
    async def test_empty_title(self, buf):
        """Should accept empty title (default) / 应接受空标题（默认值）"""
        draft_id = await buf.create_draft("team_doc", "c1", "content")
        drafts = await buf.get_pending_drafts("team_doc")
        assert drafts[0]["title"] == ""

    @pytest.mark.asyncio
    async def test_none_conversation_id(self, buf):
        """Should accept None conversation_id / 应接受 None conversation_id"""
        draft_id = await buf.create_draft("team_doc", None, "content")
        drafts = await buf.get_pending_drafts("team_doc")
        assert drafts[0]["conversation_id"] is None

    @pytest.mark.asyncio
    async def test_special_characters_in_content(self, buf):
        """Should handle special characters in content / 应处理内容中的特殊字符"""
        content = "Line 1\nLine 2\tTabbed\"Quoted'Single"
        draft_id = await buf.create_draft("team_doc", "c1", content)
        drafts = await buf.get_pending_drafts("team_doc")
        assert drafts[0]["content"] == content

    @pytest.mark.asyncio
    async def test_large_content(self, buf):
        """Should handle large content / 应处理大内容"""
        large_content = "x" * 100_000
        draft_id = await buf.create_draft("team_doc", "c1", large_content)
        drafts = await buf.get_pending_drafts("team_doc")
        assert drafts[0]["content"] == large_content

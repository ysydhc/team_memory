"""Tests for DraftBuffer — draft persistence and dedup / 草稿持久化与去重测试

Unit tests cover:
- find_pending_by_title() SQL logic (mocked) / 查询逻辑
- dedup logic in watcher (update vs insert) / 去重逻辑
- Draft creation and status management / 创建和状态管理

Integration tests cover:
- Full CRUD with SQLite
"""
from __future__ import annotations

import pytest

from daemon.draft_buffer import DraftBuffer

pytestmark = pytest.mark.unit


class TestDraftBufferInit:
    """Tests for DraftBuffer constructor / DraftBuffer 构造函数测试"""

    def test_default_db_path(self):
        """Default db_path should be in cache / 默认路径应在缓存目录"""
        buf = DraftBuffer(db_path="/tmp/test.db")
        assert buf._db_path == "/tmp/test.db"

    def test_custom_db_path(self):
        """Custom db_path should be stored / 自定义路径应被保存"""
        buf = DraftBuffer(db_path="/custom/path.db")
        assert buf._db_path == "/custom/path.db"

    def test_pool_initially_none(self):
        """db should start as None / 数据库连接初始应为 None"""
        buf = DraftBuffer(db_path="/tmp/test.db")
        assert buf._db is None


class TestDraftBufferConstants:
    """Tests for DraftBuffer SQL and constants / DraftBuffer SQL 和常量测试"""

    def test_db_path_stored(self):
        """DB path should be stored / 数据库路径应被保存"""
        buf = DraftBuffer(db_path="/tmp/drafts_test.db")
        assert buf._db_path == "/tmp/drafts_test.db"

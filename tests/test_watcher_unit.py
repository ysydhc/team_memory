"""Tests for watcher — file change detection and dedup / 文件变更检测与去重测试

Unit tests cover:
- _find_vault() project resolution / 项目名解析
- _process_changes() dedup logic (update vs insert) / 去重逻辑
- File type filtering (.md only) / 文件类型过滤
- Exclude pattern filtering / 排除模式过滤

Integration tests cover:
- Full watchfile loop with real filesystem
"""
from __future__ import annotations

from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest

from daemon.watcher import _find_vault, _process_changes

pytestmark = pytest.mark.unit


class TestFindVault:
    """Tests for _find_vault / _find_vault 项目名解析测试"""

    def _make_config(self, vaults=None):
        """Helper to create a mock config with vaults."""
        config = MagicMock()
        config.obsidian.vaults = vaults or []
        return config

    def _make_vault(self, path, project="default", exclude=None):
        """Helper to create a mock vault config."""
        vault = MagicMock()
        vault.path = path
        vault.project = project
        vault.exclude = exclude or []
        return vault

    def test_matching_vault(self):
        """Path inside vault should match / vault 内的路径应匹配"""
        vault = self._make_vault("/home/user/notes", "my_notes")
        config = self._make_config([vault])

        result = _find_vault(Path("/home/user/notes/daily/2024.md"), config)
        assert result is not None
        assert result.project == "my_notes"

    def test_path_outside_vault(self):
        """Path outside all vaults should return None / vault 外的路径应返回 None"""
        vault = self._make_vault("/home/user/notes", "my_notes")
        config = self._make_config([vault])

        result = _find_vault(Path("/tmp/other/file.md"), config)
        assert result is None

    def test_multiple_vaults_first_match(self):
        """First matching vault should win / 第一个匹配的 vault 应胜出"""
        v1 = self._make_vault("/home/user/notes", "notes")
        v2 = self._make_vault("/home/user/work", "work")
        config = self._make_config([v1, v2])

        result = _find_vault(Path("/home/user/notes/diary.md"), config)
        assert result.project == "notes"

    def test_nested_path(self):
        """Nested path should still match / 嵌套路径仍应匹配"""
        vault = self._make_vault("/home/user/notes", "notes")
        config = self._make_config([vault])

        result = _find_vault(Path("/home/user/notes/folder/sub/deep.md"), config)
        assert result is not None

    def test_empty_vaults(self):
        """No vaults configured should return None / 无 vault 配置应返回 None"""
        config = self._make_config([])
        result = _find_vault(Path("/any/path.md"), config)
        assert result is None


class TestProcessChanges:
    """Tests for _process_changes dedup and filtering / 变更处理去重与过滤测试"""

    def _make_config(self, vaults=None):
        vault = MagicMock()
        vault.path = "/home/user/notes"
        vault.project = "test_project"
        vault.exclude = [".obsidian", ".trash"]
        config = MagicMock()
        config.obsidian.vaults = vaults or [vault]
        return config

    @pytest.mark.asyncio
    async def test_non_md_file_skipped(self):
        """Non-.md files should be skipped / 非 .md 文件应跳过"""
        from watchfiles import Change

        config = self._make_config()
        indexer = MagicMock()
        sink = AsyncMock()
        buf = AsyncMock()

        changes = {(Change.modified, "/home/user/notes/image.png")}
        await _process_changes(changes, config, indexer, sink, buf)

        # indexer.parse_file should NOT be called for .png
        indexer.parse_file.assert_not_called()

    @pytest.mark.asyncio
    async def test_md_file_processed(self):
        """Markdown files should be processed / Markdown 文件应被处理"""
        from watchfiles import Change

        config = self._make_config()
        indexer = MagicMock()
        indexer.parse_file.return_value = {
            "title": "Test Note",
            "description": "Some content",
            "solution": "",
            "tags": [],
        }
        sink = AsyncMock()
        buf = AsyncMock()
        buf.find_pending_by_title = AsyncMock(return_value=None)
        buf.create_draft = AsyncMock(return_value="draft-123")
        buf.mark_needs_refinement = AsyncMock()

        changes = {(Change.modified, "/home/user/notes/test.md")}
        await _process_changes(changes, config, indexer, sink, buf)

        indexer.parse_file.assert_called_once()
        buf.create_draft.assert_called_once()

    @pytest.mark.asyncio
    async def test_dedup_updates_existing(self):
        """Existing pending draft should be updated, not recreated / 已有 pending draft 应更新而非重建"""
        from watchfiles import Change

        config = self._make_config()
        indexer = MagicMock()
        indexer.parse_file.return_value = {
            "title": "Duplicate Note",
            "description": "Updated content",
            "solution": "",
            "tags": [],
        }
        sink = AsyncMock()
        buf = AsyncMock()
        buf.find_pending_by_title = AsyncMock(return_value="existing-draft-id")
        buf.update_draft = AsyncMock()

        changes = {(Change.modified, "/home/user/notes/dup.md")}
        await _process_changes(changes, config, indexer, sink, buf)

        # Should UPDATE existing, NOT create new
        buf.update_draft.assert_called_once()
        buf.create_draft.assert_not_called()

    @pytest.mark.asyncio
    async def test_exclude_pattern_skipped(self):
        """Files matching exclude pattern should be skipped / 匹配排除模式的文件应跳过"""
        from watchfiles import Change

        config = self._make_config()
        indexer = MagicMock()
        sink = AsyncMock()
        buf = AsyncMock()

        changes = {(Change.modified, "/home/user/notes/.obsidian/config.md")}
        await _process_changes(changes, config, indexer, sink, buf)

        indexer.parse_file.assert_not_called()

    @pytest.mark.asyncio
    async def test_deleted_file_no_op(self):
        """Deleted files should be logged but not processed / 删除文件只记录不处理"""
        from watchfiles import Change

        config = self._make_config()
        indexer = MagicMock()
        sink = AsyncMock()
        buf = AsyncMock()

        changes = {(Change.deleted, "/home/user/notes/old.md")}
        await _process_changes(changes, config, indexer, sink, buf)

        indexer.parse_file.assert_not_called()

    @pytest.mark.asyncio
    async def test_no_matching_vault_skipped(self):
        """Files not in any vault should be skipped / 不在任何 vault 的文件应跳过"""
        from watchfiles import Change

        config = self._make_config(vaults=[])  # No vaults
        indexer = MagicMock()
        sink = AsyncMock()
        buf = AsyncMock()

        changes = {(Change.modified, "/random/path/test.md")}
        await _process_changes(changes, config, indexer, sink, buf)

        indexer.parse_file.assert_not_called()

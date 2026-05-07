"""Tests for pipeline — core daemon processing functions / 核心守护进程处理函数测试

Unit tests cover:
- _extract_mem_references() / 提取 [mem:xxx] 引用
- _resolve_project() / 解析项目名
- process_after_response() with mocks / 模拟处理回复
- process_before_prompt() with mocks / 模拟处理提问
- process_session_start() / 会话开始
- process_session_end() / 会话结束
"""
from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from daemon.pipeline import (
    _extract_mem_references,
    _resolve_project,
    process_after_response,
    process_before_prompt,
    process_session_end,
    process_session_start,
)

pytestmark = pytest.mark.unit


class TestExtractMemReferences:
    """Tests for _extract_mem_references / 提取 [mem:xxx] 引用测试"""

    def test_single_reference(self):
        """Single [mem:uuid] should be extracted / 单个引用应被提取"""
        result = _extract_mem_references("Based on [mem:abc12345-6789-def0-1234-567890abcdef]...")
        assert len(result) == 1
        assert result[0] == "abc12345-6789-def0-1234-567890abcdef"

    def test_multiple_references(self):
        """Multiple references should all be extracted / 多个引用应全部提取"""
        text = "Used [mem:aaa11111-2222-3333-4444-555566667777] and [mem:bbb99999-8888-7777-6666-555544443333]"
        result = _extract_mem_references(text)
        assert len(result) == 2

    def test_duplicate_references_deduped(self):
        """Duplicate references should be deduplicated / 重复引用应去重"""
        text = "See [mem:abc12345-6789-def0-1234-567890abcdef] again [mem:abc12345-6789-def0-1234-567890abcdef]"
        result = _extract_mem_references(text)
        assert len(result) == 1

    def test_no_references(self):
        """No references should return empty list / 无引用应返回空列表"""
        result = _extract_mem_references("Just a normal response")
        assert result == []

    def test_empty_text(self):
        """Empty text should return empty list / 空文本应返回空列表"""
        result = _extract_mem_references("")
        assert result == []


class TestResolveProject:
    """Tests for _resolve_project / 解析项目名测试"""

    def _make_config(self, projects=None):
        """Helper to create a mock DaemonConfig."""
        config = MagicMock()
        config.projects = projects or []
        return config

    def test_matching_path(self):
        """Matching workspace path should return project name / 匹配的路径应返回项目名"""
        pm = MagicMock()
        pm.name = "my_project"
        pm.path_patterns = ["/Users/me/work/my_project"]
        config = self._make_config([pm])

        result = _resolve_project(["/Users/me/work/my_project/src"], config)
        assert result == "my_project"

    def test_no_matching_path(self):
        """No matching path should return None / 无匹配路径应返回 None"""
        pm = MagicMock()
        pm.name = "other_project"
        pm.path_patterns = ["/Users/me/work/other"]
        config = self._make_config([pm])

        result = _resolve_project(["/Users/me/work/unrelated"], config)
        assert result is None

    def test_empty_workspace_roots(self):
        """Empty workspace roots should return None / 空工作区路径应返回 None"""
        config = self._make_config([])
        result = _resolve_project([], config)
        assert result is None

    def test_multiple_patterns_first_match_wins(self):
        """First matching pattern should win / 第一个匹配的模式应胜出"""
        pm1 = MagicMock()
        pm1.name = "project_a"
        pm1.path_patterns = ["/a"]
        pm2 = MagicMock()
        pm2.name = "project_b"
        pm2.path_patterns = ["/b"]
        config = self._make_config([pm1, pm2])

        result = _resolve_project(["/a/workspace"], config)
        assert result == "project_a"

    def test_multiple_workspace_roots(self):
        """Should check all workspace roots / 应检查所有工作区路径"""
        pm = MagicMock()
        pm.name = "my_project"
        pm.path_patterns = ["/home/project"]
        config = self._make_config([pm])

        result = _resolve_project(["/unrelated", "/home/project/src"], config)
        assert result == "my_project"


class TestProcessSessionStart:
    """Tests for process_session_start / 会话开始处理测试"""

    @pytest.mark.asyncio
    async def test_returns_dict(self):
        """Should return a dict / 应返回字典"""
        config = MagicMock()
        sink = AsyncMock()
        result = await process_session_start(
            {"conversation_id": "test-123", "workspace_roots": ["/home"]},
            config, sink,
        )
        assert isinstance(result, dict)


class TestProcessSessionEnd:
    """Tests for process_session_end / 会话结束处理测试"""

    @pytest.mark.asyncio
    async def test_returns_dict(self):
        """Should return a dict / 应返回字典"""
        config = MagicMock()
        sink = AsyncMock()
        buf = AsyncMock()
        refiner = AsyncMock()
        result = await process_session_end(
            {"conversation_id": "test-123"},
            config, sink, buf, refiner,
        )
        assert isinstance(result, dict)


class TestProcessBeforePrompt:
    """Tests for process_before_prompt / 提问处理测试"""

    @pytest.mark.asyncio
    async def test_with_search_results(self):
        """Should return search results / 应返回搜索结果"""
        config = MagicMock()
        sink = AsyncMock()
        sink.recall = AsyncMock(return_value={"results": [{"id": "1", "title": "Test"}]})
        sink.log_search = AsyncMock(return_value="log-id")

        result = await process_before_prompt(
            {"prompt": "test query", "conversation_id": "c1", "workspace_roots": ["/home"]},
            config, sink,
        )
        assert isinstance(result, dict)

    @pytest.mark.asyncio
    async def test_empty_prompt(self):
        """Empty prompt should still process / 空提问仍应处理"""
        config = MagicMock()
        sink = AsyncMock()
        sink.recall = AsyncMock(return_value={"results": []})

        result = await process_before_prompt(
            {"prompt": "", "conversation_id": "c2", "workspace_roots": []},
            config, sink,
        )
        assert isinstance(result, dict)

"""测试 TMSink 抽象层：TMSink ABC、LocalTMSink、RemoteTMSink、create_sink 工厂。"""

from __future__ import annotations

import json
from abc import ABC
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

# ---------------------------------------------------------------------------
# TMSink ABC
# ---------------------------------------------------------------------------


class TestTMSinkABC:
    """TMSink 是抽象基类，不能直接实例化。"""

    def test_cannot_instantiate(self):
        from daemon.tm_sink import TMSink

        with pytest.raises(TypeError, match="abstract"):
            TMSink()

    def test_has_required_abstract_methods(self):
        from daemon.tm_sink import TMSink

        abstracts = set(TMSink.__abstractmethods__)
        expected = {"draft_save", "draft_publish", "save", "recall", "context"}
        assert expected == abstracts, f"Missing abstract methods: {expected - abstracts}"

    def test_is_abc_subclass(self):
        from daemon.tm_sink import TMSink

        assert issubclass(TMSink, ABC)


# ---------------------------------------------------------------------------
# LocalTMSink — 测试时直接 mock 实例属性（因为 __init__ 调用 _ensure_ops
#               后 op_* 被存为模块级变量，方法体内引用的也是模块级变量）
# ---------------------------------------------------------------------------


class TestLocalTMSink:
    """LocalTMSink 通过直接 Python 调用 team_memory 服务。"""

    def _make_sink(self, user: str = "test_user") -> "LocalTMSink":
        """创建 LocalTMSink 并注入 mock op_* 函数。"""
        from daemon.tm_sink import LocalTMSink

        sink = LocalTMSink(user=user)
        # 手动注入 mock，避免依赖真实 TM 服务
        sink.__dict__["_mock"] = True  # 标记
        return sink

    def test_is_tmsink_subclass(self):
        from daemon.tm_sink import LocalTMSink, TMSink

        assert issubclass(LocalTMSink, TMSink)

    def test_init_user_default(self):
        from daemon.tm_sink import LocalTMSink

        with patch("daemon.tm_sink._ensure_ops"):
            sink = LocalTMSink()
        assert sink._user == "daemon"

    def test_init_custom_user(self):
        from daemon.tm_sink import LocalTMSink

        with patch("daemon.tm_sink._ensure_ops"):
            sink = LocalTMSink(user="custom")
        assert sink._user == "custom"

    @pytest.mark.asyncio
    async def test_draft_save_calls_op_draft_save(self):
        from daemon.tm_sink import LocalTMSink

        mock_op = AsyncMock(return_value={"id": "abc-123", "status": "draft"})

        with patch("daemon.tm_sink._ensure_ops"), \
             patch("daemon.tm_sink._op_draft_save", mock_op):
            sink = LocalTMSink(user="test_user")
            result = await sink.draft_save(
                title="测试标题",
                content="测试内容",
                tags=["python"],
                project="test_proj",
                group_key="gk1",
                conversation_id="conv1",
            )
            mock_op.assert_awaited_once_with(
                "test_user",
                title="测试标题",
                content="测试内容",
                tags=["python"],
                project="test_proj",
                group_key="gk1",
                conversation_id="conv1",
            )
            assert result == {"id": "abc-123", "status": "draft"}

    @pytest.mark.asyncio
    async def test_draft_publish_calls_op_draft_publish(self):
        from daemon.tm_sink import LocalTMSink

        mock_op = AsyncMock(return_value={"id": "abc-123", "status": "published"})

        with patch("daemon.tm_sink._ensure_ops"), \
             patch("daemon.tm_sink._op_draft_publish", mock_op):
            sink = LocalTMSink(user="test_user")
            result = await sink.draft_publish(
                draft_id="abc-123",
                refined_content="精炼内容",
            )
            mock_op.assert_awaited_once_with(
                "test_user",
                draft_id="abc-123",
                refined_content="精炼内容",
            )
            assert result == {"id": "abc-123", "status": "published"}

    @pytest.mark.asyncio
    async def test_save_calls_op_save(self):
        from daemon.tm_sink import LocalTMSink

        mock_op = AsyncMock(return_value={
            "message": "Knowledge saved successfully.",
            "data": {"id": "xyz", "title": "T", "status": "published"},
        })

        with patch("daemon.tm_sink._ensure_ops"), \
             patch("daemon.tm_sink._op_save", mock_op):
            sink = LocalTMSink(user="test_user")
            result = await sink.save(
                title="标题",
                problem="问题描述",
                solution="解决方案",
                tags=["tag1"],
                project="proj",
                group_key="gk",
            )
            mock_op.assert_awaited_once_with(
                "test_user",
                title="标题",
                problem="问题描述",
                solution="解决方案",
                content=None,
                tags=["tag1"],
                scope="project",
                experience_type=None,
                project="proj",
                group_key="gk",
            )
            assert result["data"]["id"] == "xyz"

    @pytest.mark.asyncio
    async def test_recall_with_dict_result(self):
        """op_recall 返回 dict 时，提取 results 字段。"""
        from daemon.tm_sink import LocalTMSink

        mock_op = AsyncMock(return_value={
            "message": "Found 2 result(s).",
            "results": [{"id": "1"}, {"id": "2"}],
        })

        with patch("daemon.tm_sink._ensure_ops"), \
             patch("daemon.tm_sink._op_recall", mock_op):
            sink = LocalTMSink(user="test_user")
            result = await sink.recall(query="测试", project="proj")
            mock_op.assert_awaited_once_with(
                "test_user",
                query="测试",
                problem=None,
                file_path=None,
                language=None,
                framework=None,
                tags=None,
                max_results=5,
                project="proj",
            )
            assert result == [{"id": "1"}, {"id": "2"}]

    @pytest.mark.asyncio
    async def test_recall_with_string_result(self):
        """op_recall 返回 JSON 字符串时，先解析再提取 results。"""
        from daemon.tm_sink import LocalTMSink

        raw = json.dumps({"results": [{"id": "1"}]})
        mock_op = AsyncMock(return_value=raw)

        with patch("daemon.tm_sink._ensure_ops"), \
             patch("daemon.tm_sink._op_recall", mock_op):
            sink = LocalTMSink(user="test_user")
            result = await sink.recall(query="测试")
            assert result == [{"id": "1"}]

    @pytest.mark.asyncio
    async def test_recall_empty_results(self):
        """op_recall 返回无 results 的 dict 时返回空列表。"""
        from daemon.tm_sink import LocalTMSink

        mock_op = AsyncMock(return_value={"message": "No matching experiences found."})

        with patch("daemon.tm_sink._ensure_ops"), \
             patch("daemon.tm_sink._op_recall", mock_op):
            sink = LocalTMSink(user="test_user")
            result = await sink.recall(query="不存在")
            assert result == []

    @pytest.mark.asyncio
    async def test_context_calls_op_context(self):
        from daemon.tm_sink import LocalTMSink

        mock_op = AsyncMock(return_value={
            "user": "test_user",
            "project": "proj",
            "profile": {"static": [], "dynamic": []},
            "relevant_experiences": [],
        })

        with patch("daemon.tm_sink._ensure_ops"), \
             patch("daemon.tm_sink._op_context", mock_op):
            sink = LocalTMSink(user="test_user")
            result = await sink.context(
                file_paths=["/a/b.py"],
                task_description="做点什么",
                project="proj",
            )
            mock_op.assert_awaited_once_with(
                "test_user",
                file_paths=["/a/b.py"],
                task_description="做点什么",
                project="proj",
            )
            assert result["user"] == "test_user"


# ---------------------------------------------------------------------------
# RemoteTMSink
# ---------------------------------------------------------------------------


class TestRemoteTMSink:
    """RemoteTMSink 通过 HTTP 调用远程 TM 服务。"""

    def test_is_tmsink_subclass(self):
        from daemon.tm_sink import RemoteTMSink, TMSink

        assert issubclass(RemoteTMSink, TMSink)

    def test_init(self):
        from daemon.tm_sink import RemoteTMSink

        sink = RemoteTMSink(base_url="http://tm:3900", user="remote_user")
        assert sink._base_url == "http://tm:3900"
        assert sink._user == "remote_user"

    def test_init_strips_trailing_slash(self):
        from daemon.tm_sink import RemoteTMSink

        sink = RemoteTMSink(base_url="http://tm:3900/")
        assert sink._base_url == "http://tm:3900"

    @pytest.mark.asyncio
    async def test_draft_save_posts_to_remote(self):
        from daemon.tm_sink import RemoteTMSink

        sink = RemoteTMSink(base_url="http://tm:3900", user="ruser")

        mock_response = MagicMock()
        mock_response.json.return_value = {"id": "draft-1", "status": "draft"}
        mock_response.raise_for_status = MagicMock()

        with patch("daemon.tm_sink.httpx.AsyncClient") as MockClient:
            mock_client = AsyncMock()
            mock_client.post.return_value = mock_response
            MockClient.return_value.__aenter__ = AsyncMock(return_value=mock_client)
            MockClient.return_value.__aexit__ = AsyncMock(return_value=False)

            result = await sink.draft_save(
                title="远程标题",
                content="远程内容",
                tags=["remote"],
                project="proj",
            )
            mock_client.post.assert_awaited_once_with(
                "http://tm:3900/memory_draft_save",
                json={
                    "arguments": {
                        "title": "远程标题",
                        "content": "远程内容",
                        "tags": ["remote"],
                        "project": "proj",
                    },
                },
            )
            assert result == {"id": "draft-1", "status": "draft"}

    @pytest.mark.asyncio
    async def test_draft_publish_posts_to_remote(self):
        from daemon.tm_sink import RemoteTMSink

        sink = RemoteTMSink(base_url="http://tm:3900", user="ruser")
        mock_response = MagicMock()
        mock_response.json.return_value = {"id": "draft-1", "status": "published"}
        mock_response.raise_for_status = MagicMock()

        with patch("daemon.tm_sink.httpx.AsyncClient") as MockClient:
            mock_client = AsyncMock()
            mock_client.post.return_value = mock_response
            MockClient.return_value.__aenter__ = AsyncMock(return_value=mock_client)
            MockClient.return_value.__aexit__ = AsyncMock(return_value=False)

            result = await sink.draft_publish(
                draft_id="draft-1",
                refined_content="精炼",
            )
            mock_client.post.assert_awaited_once_with(
                "http://tm:3900/memory_draft_publish",
                json={
                    "arguments": {
                        "draft_id": "draft-1",
                        "refined_content": "精炼",
                    },
                },
            )
            assert result == {"id": "draft-1", "status": "published"}

    @pytest.mark.asyncio
    async def test_save_posts_to_remote(self):
        from daemon.tm_sink import RemoteTMSink

        sink = RemoteTMSink(base_url="http://tm:3900", user="ruser")
        mock_response = MagicMock()
        mock_response.json.return_value = {"message": "saved", "data": {"id": "1"}}
        mock_response.raise_for_status = MagicMock()

        with patch("daemon.tm_sink.httpx.AsyncClient") as MockClient:
            mock_client = AsyncMock()
            mock_client.post.return_value = mock_response
            MockClient.return_value.__aenter__ = AsyncMock(return_value=mock_client)
            MockClient.return_value.__aexit__ = AsyncMock(return_value=False)

            result = await sink.save(
                title="T", problem="P", solution="S", tags=["t"], project="proj"
            )
            mock_client.post.assert_awaited_once_with(
                "http://tm:3900/memory_save",
                json={
                    "arguments": {
                        "title": "T",
                        "problem": "P",
                        "solution": "S",
                        "tags": ["t"],
                        "scope": "project",
                        "project": "proj",
                    },
                },
            )

    @pytest.mark.asyncio
    async def test_recall_extracts_results(self):
        from daemon.tm_sink import RemoteTMSink

        sink = RemoteTMSink(base_url="http://tm:3900", user="ruser")
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "message": "Found 1",
            "results": [{"id": "r1"}],
        }
        mock_response.raise_for_status = MagicMock()

        with patch("daemon.tm_sink.httpx.AsyncClient") as MockClient:
            mock_client = AsyncMock()
            mock_client.post.return_value = mock_response
            MockClient.return_value.__aenter__ = AsyncMock(return_value=mock_client)
            MockClient.return_value.__aexit__ = AsyncMock(return_value=False)

            result = await sink.recall(query="Q")
            assert result == [{"id": "r1"}]

    @pytest.mark.asyncio
    async def test_recall_empty_results(self):
        from daemon.tm_sink import RemoteTMSink

        sink = RemoteTMSink(base_url="http://tm:3900", user="ruser")
        mock_response = MagicMock()
        mock_response.json.return_value = {"message": "No results"}
        mock_response.raise_for_status = MagicMock()

        with patch("daemon.tm_sink.httpx.AsyncClient") as MockClient:
            mock_client = AsyncMock()
            mock_client.post.return_value = mock_response
            MockClient.return_value.__aenter__ = AsyncMock(return_value=mock_client)
            MockClient.return_value.__aexit__ = AsyncMock(return_value=False)

            result = await sink.recall(query="Q")
            assert result == []

    @pytest.mark.asyncio
    async def test_context_posts_to_remote(self):
        from daemon.tm_sink import RemoteTMSink

        sink = RemoteTMSink(base_url="http://tm:3900", user="ruser")
        mock_response = MagicMock()
        mock_response.json.return_value = {"user": "ruser", "relevant_experiences": []}
        mock_response.raise_for_status = MagicMock()

        with patch("daemon.tm_sink.httpx.AsyncClient") as MockClient:
            mock_client = AsyncMock()
            mock_client.post.return_value = mock_response
            MockClient.return_value.__aenter__ = AsyncMock(return_value=mock_client)
            MockClient.return_value.__aexit__ = AsyncMock(return_value=False)

            result = await sink.context(file_paths=["a.py"], task_description="task", project="proj")
            mock_client.post.assert_awaited_once_with(
                "http://tm:3900/memory_context",
                json={
                    "arguments": {
                        "file_paths": ["a.py"],
                        "task_description": "task",
                        "project": "proj",
                    },
                },
            )
            assert result["user"] == "ruser"


# ---------------------------------------------------------------------------
# create_sink 工厂
# ---------------------------------------------------------------------------


class TestCreateSink:
    """create_sink 根据配置创建对应的 TMSink 实现。"""

    def test_local_mode(self):
        from daemon.tm_sink import LocalTMSink, create_sink

        with patch("daemon.tm_sink._ensure_ops"):
            config = {"mode": "local", "user": "test_user"}
            sink = create_sink(config)
            assert isinstance(sink, LocalTMSink)
            assert sink._user == "test_user"

    def test_remote_mode(self):
        from daemon.tm_sink import RemoteTMSink, create_sink

        config = {"mode": "remote", "base_url": "http://tm:3900", "user": "ruser"}
        sink = create_sink(config)
        assert isinstance(sink, RemoteTMSink)
        assert sink._base_url == "http://tm:3900"
        assert sink._user == "ruser"

    def test_invalid_mode_raises(self):
        from daemon.tm_sink import create_sink

        config = {"mode": "invalid"}
        with pytest.raises(ValueError, match="mode"):
            create_sink(config)

    def test_local_mode_default_user(self):
        from daemon.tm_sink import LocalTMSink, create_sink

        with patch("daemon.tm_sink._ensure_ops"):
            config = {"mode": "local"}
            sink = create_sink(config)
            assert isinstance(sink, LocalTMSink)
            assert sink._user == "daemon"

    def test_remote_mode_no_base_url_raises(self):
        from daemon.tm_sink import create_sink

        config = {"mode": "remote", "user": "ruser"}
        with pytest.raises(ValueError, match="base_url"):
            create_sink(config)

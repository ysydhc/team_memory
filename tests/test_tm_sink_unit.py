"""Tests for TMSink — TM storage write operations / TM 存储写入操作测试

Unit tests cover:
- LocalTMSink constructor and routing / 构造函数和路由
- RemoteTMSink constructor and headers / 构造函数和请求头
- _ensure_ops() guard / 操作守卫
- LocalTMSink.recall edge cases
- LocalTMSink.update_experience
- LocalTMSink.increment_used_count
- RemoteTMSink._request with path_params and GET
- RemoteTMSink.draft_save all optional params
- RemoteTMSink.save all optional params
- RemoteTMSink.recall all params and edge cases
- RemoteTMSink.update_experience
- RemoteTMSink.increment_used_count (success + exception)

Integration tests cover:
- Full CRUD with real MCP/Web server
"""
from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from daemon.tm_sink import LocalTMSink, RemoteTMSink, TMSink

pytestmark = pytest.mark.unit


class TestLocalTMSinkInit:
    """Tests for LocalTMSink constructor / LocalTMSink 构造函数测试"""

    def test_default_user(self):
        """Default user should be 'daemon' / 默认用户应为 daemon"""
        sink = LocalTMSink()
        assert sink._user == "daemon"

    def test_custom_user(self):
        """Custom user should be stored / 自定义用户应被保存"""
        sink = LocalTMSink(user="test_user")
        assert sink._user == "test_user"


class TestRemoteTMSinkInit:
    """Tests for RemoteTMSink constructor / RemoteTMSink 构造函数测试"""

    def test_base_url_stored(self):
        """Base URL should be stored / 基础 URL 应被保存"""
        sink = RemoteTMSink(base_url="http://localhost:9111")
        assert sink._base_url == "http://localhost:9111"

    def test_default_user(self):
        """Default user should be 'daemon' / 默认用户应为 daemon"""
        sink = RemoteTMSink(base_url="http://localhost:9111")
        assert sink._user == "daemon"

    def test_custom_user(self):
        """Custom user should be stored / 自定义用户应被保存"""
        sink = RemoteTMSink(base_url="http://localhost:9111", user="remote_user")
        assert sink._user == "remote_user"

    def test_headers_contain_content_type(self):
        """Headers should contain Content-Type / 请求头应包含 Content-Type"""
        sink = RemoteTMSink(base_url="http://localhost:9111")
        headers = sink._headers()
        assert headers.get("Content-Type") == "application/json"

    def test_headers_contain_authorization(self):
        """Headers should contain Authorization / 请求头应包含 Authorization"""
        sink = RemoteTMSink(base_url="http://localhost:9111")
        headers = sink._headers()
        assert "Authorization" in headers


class TestEnsureOps:
    """Tests for _ensure_ops guard / _ensure_ops 守卫测试"""

    def test_ensure_ops_imports_successfully(self):
        """_ensure_ops should be importable / _ensure_ops 应可导入"""
        from daemon.tm_sink import _ensure_ops
        # Just verify it's callable, not that it succeeds without MCP
        assert callable(_ensure_ops)


# ---------------------------------------------------------------------------
# LocalTMSink method coverage tests
# ---------------------------------------------------------------------------

class TestLocalTMSinkRecall:
    """Tests for LocalTMSink.recall edge cases."""

    @pytest.mark.asyncio
    async def test_recall_returns_empty_list_when_raw_is_list(self):
        """When _op_recall returns a list (not dict, not str), recall returns []."""
        sink = LocalTMSink(user="test_user")
        with patch("daemon.tm_sink._op_recall", new_callable=AsyncMock) as mock_recall:
            mock_recall.return_value = [{"id": "1"}, {"id": "2"}]
            result = await sink.recall(query="test")
            assert result == []

    @pytest.mark.asyncio
    async def test_recall_returns_empty_list_when_raw_is_none(self):
        """When _op_recall returns None, recall returns []."""
        sink = LocalTMSink(user="test_user")
        with patch("daemon.tm_sink._op_recall", new_callable=AsyncMock) as mock_recall:
            mock_recall.return_value = None
            result = await sink.recall(query="test")
            assert result == []


class TestLocalTMSinkUpdateExperience:
    """Tests for LocalTMSink.update_experience."""

    @pytest.mark.asyncio
    async def test_update_experience_calls_op(self):
        """update_experience should call op_experience_update with correct args."""
        sink = LocalTMSink(user="test_user")
        mock_op = AsyncMock(return_value={"id": "exp1", "title": "Updated"})
        with patch("team_memory.services.memory_operations.op_experience_update", mock_op):
            result = await sink.update_experience(
                experience_id="exp1",
                title="Updated",
                problem="new problem",
                solution="new solution",
                tags=["tag1"],
                experience_type="lesson",
                exp_status="active",
            )
        mock_op.assert_awaited_once_with(
            "test_user",
            experience_id="exp1",
            title="Updated",
            problem="new problem",
            solution="new solution",
            tags=["tag1"],
            experience_type="lesson",
            exp_status="active",
        )
        assert result == {"id": "exp1", "title": "Updated"}


class TestLocalTMSinkIncrementUsedCount:
    """Tests for LocalTMSink.increment_used_count."""

    @pytest.mark.asyncio
    async def test_increment_used_count(self):
        """increment_used_count should call repo.increment_used_count."""
        sink = LocalTMSink(user="test_user")
        mock_session = AsyncMock()
        mock_repo = MagicMock()
        mock_repo.increment_used_count = AsyncMock()

        # Create an async generator for get_session
        async def fake_get_session():
            yield mock_session

        with (
            patch("team_memory.storage.database.get_session", fake_get_session),
            patch("team_memory.storage.repository.ExperienceRepository", return_value=mock_repo),
        ):
            await sink.increment_used_count("00000000-0000-0000-0000-000000000001")

        mock_repo.increment_used_count.assert_awaited_once()
        mock_session.commit.assert_awaited_once()


# ---------------------------------------------------------------------------
# RemoteTMSink method coverage tests
# ---------------------------------------------------------------------------

class TestRemoteTMSinkRequest:
    """Tests for RemoteTMSink._request with path_params and GET."""

    @pytest.mark.asyncio
    async def test_request_with_path_params(self):
        """_request should format path when path_params given."""
        sink = RemoteTMSink(base_url="http://localhost:9111")
        mock_resp = MagicMock()
        mock_resp.json.return_value = {"ok": True}
        mock_resp.raise_for_status = MagicMock()

        mock_client = AsyncMock()
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)
        mock_client.request = AsyncMock(return_value=mock_resp)

        with patch("httpx.AsyncClient", return_value=mock_client):
            result = await sink._request(
                "PUT", "/api/v1/experiences/{experience_id}",
                body={"title": "t"}, path_params={"experience_id": "exp42"},
            )
        # The path should have been formatted
        mock_client.request.assert_awaited_once()
        call_args = mock_client.request.call_args
        assert "exp42" in call_args[0][1]  # URL contains the formatted id
        assert result == {"ok": True}

    @pytest.mark.asyncio
    async def test_request_get_method(self):
        """_request should use client.get for GET method."""
        sink = RemoteTMSink(base_url="http://localhost:9111")
        mock_resp = MagicMock()
        mock_resp.json.return_value = {"data": "val"}
        mock_resp.raise_for_status = MagicMock()

        mock_client = AsyncMock()
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)
        mock_client.get = AsyncMock(return_value=mock_resp)

        with patch("httpx.AsyncClient", return_value=mock_client):
            result = await sink._request("GET", "/api/v1/mcp/archive/123")

        mock_client.get.assert_awaited_once()
        assert result == {"data": "val"}


class TestRemoteTMSinkDraftSave:
    """Tests for RemoteTMSink.draft_save with all optional params."""

    @pytest.mark.asyncio
    async def test_draft_save_all_optional_params(self):
        """draft_save should include group_key, conversation_id, skip_dedup."""
        sink = RemoteTMSink(base_url="http://localhost:9111")
        with patch.object(sink, "_request", new_callable=AsyncMock) as mock_req:
            mock_req.return_value = {"id": "draft1"}
            result = await sink.draft_save(
                title="t", content="c",
                tags=["a"], project="p",
                group_key="gk", conversation_id="conv1", skip_dedup=True,
            )
        call_kwargs = mock_req.call_args
        body = call_kwargs[0][2] if len(call_kwargs[0]) > 2 else call_kwargs[1].get("body")
        # Verify body was built correctly
        # The _request is called as: _request("POST", "/api/v1/mcp/draft-save", body)
        args, kwargs = mock_req.call_args
        body = args[2]
        assert body["group_key"] == "gk"
        assert body["conversation_id"] == "conv1"
        assert body["skip_dedup"] is True
        assert body["tags"] == ["a"]
        assert body["project"] == "p"


class TestRemoteTMSinkSave:
    """Tests for RemoteTMSink.save with content, experience_type, group_key."""

    @pytest.mark.asyncio
    async def test_save_all_optional_params(self):
        """save should include content, experience_type, group_key."""
        sink = RemoteTMSink(base_url="http://localhost:9111")
        with patch.object(sink, "_request", new_callable=AsyncMock) as mock_req:
            mock_req.return_value = {"id": "exp1"}
            await sink.save(
                title="t", problem="p", solution="s",
                content="full content", tags=["a"],
                scope="global", experience_type="lesson",
                project="proj", group_key="gk",
            )
        args, _ = mock_req.call_args
        body = args[2]
        assert body["content"] == "full content"
        assert body["experience_type"] == "lesson"
        assert body["group_key"] == "gk"
        assert body["title"] == "t"
        assert body["problem"] == "p"
        assert body["solution"] == "s"
        assert body["tags"] == ["a"]
        assert body["scope"] == "global"
        assert body["project"] == "proj"


class TestRemoteTMSinkRecall:
    """Tests for RemoteTMSink.recall with all params and edge cases."""

    @pytest.mark.asyncio
    async def test_recall_all_optional_params(self):
        """recall should include all optional body fields."""
        sink = RemoteTMSink(base_url="http://localhost:9111")
        with patch.object(sink, "_request", new_callable=AsyncMock) as mock_req:
            mock_req.return_value = {"results": [{"id": "1"}]}
            result = await sink.recall(
                query="q", problem="pr", file_path="/f",
                language="py", framework="django",
                tags=["t"], max_results=10, project="proj",
            )
        args, _ = mock_req.call_args
        body = args[2]
        assert body["query"] == "q"
        assert body["problem"] == "pr"
        assert body["file_path"] == "/f"
        assert body["language"] == "py"
        assert body["framework"] == "django"
        assert body["tags"] == ["t"]
        assert body["max_results"] == 10
        assert body["project"] == "proj"
        assert result == [{"id": "1"}]

    @pytest.mark.asyncio
    async def test_recall_returns_empty_when_raw_not_dict(self):
        """recall should return [] when _request returns non-dict."""
        sink = RemoteTMSink(base_url="http://localhost:9111")
        with patch.object(sink, "_request", new_callable=AsyncMock) as mock_req:
            mock_req.return_value = "some string"  # not a dict
            result = await sink.recall(query="q")
        assert result == []

    @pytest.mark.asyncio
    async def test_recall_returns_empty_when_raw_is_list(self):
        """recall should return [] when _request returns a list."""
        sink = RemoteTMSink(base_url="http://localhost:9111")
        with patch.object(sink, "_request", new_callable=AsyncMock) as mock_req:
            mock_req.return_value = [{"id": "1"}]  # list, not dict
            result = await sink.recall(query="q")
        assert result == []

    @pytest.mark.asyncio
    async def test_recall_default_max_results_not_in_body(self):
        """recall should not include max_results in body when it's the default (5)."""
        sink = RemoteTMSink(base_url="http://localhost:9111")
        with patch.object(sink, "_request", new_callable=AsyncMock) as mock_req:
            mock_req.return_value = {"results": []}
            await sink.recall(query="q", max_results=5)
        args, _ = mock_req.call_args
        body = args[2]
        assert "max_results" not in body


class TestRemoteTMSinkUpdateExperience:
    """Tests for RemoteTMSink.update_experience."""

    @pytest.mark.asyncio
    async def test_update_experience_all_params(self):
        """update_experience should include all optional body fields."""
        sink = RemoteTMSink(base_url="http://localhost:9111")
        with patch.object(sink, "_request", new_callable=AsyncMock) as mock_req:
            mock_req.return_value = {"id": "exp1"}
            await sink.update_experience(
                experience_id="exp1",
                title="t", problem="p", solution="s",
                tags=["a"], experience_type="lesson", exp_status="active",
            )
        args, kwargs = mock_req.call_args
        method, path, body = args[0], args[1], args[2]
        assert method == "PUT"
        assert "{experience_id}" in path
        assert body["title"] == "t"
        assert body["problem"] == "p"
        assert body["solution"] == "s"
        assert body["tags"] == ["a"]
        assert body["experience_type"] == "lesson"
        assert body["exp_status"] == "active"
        assert kwargs.get("path_params") == {"experience_id": "exp1"}

    @pytest.mark.asyncio
    async def test_update_experience_minimal_params(self):
        """update_experience with only experience_id should have empty body."""
        sink = RemoteTMSink(base_url="http://localhost:9111")
        with patch.object(sink, "_request", new_callable=AsyncMock) as mock_req:
            mock_req.return_value = {"id": "exp1"}
            await sink.update_experience(experience_id="exp1")
        args, _ = mock_req.call_args
        body = args[2]
        assert body == {}


class TestRemoteTMSinkIncrementUsedCount:
    """Tests for RemoteTMSink.increment_used_count."""

    @pytest.mark.asyncio
    async def test_increment_used_count_success(self):
        """increment_used_count should call feedback endpoint."""
        sink = RemoteTMSink(base_url="http://localhost:9111")
        with patch.object(sink, "_request", new_callable=AsyncMock) as mock_req:
            mock_req.return_value = {"ok": True}
            await sink.increment_used_count("exp42")
        args, _ = mock_req.call_args
        method = args[0]
        path = args[1]
        body = args[2]
        assert method == "POST"
        assert path == "/api/v1/mcp/feedback"
        assert body["experience_id"] == "exp42"
        assert body["rating"] == 5
        assert body["comment"] == "auto:referenced"

    @pytest.mark.asyncio
    async def test_increment_used_count_exception_swallowed(self):
        """increment_used_count should swallow exceptions and log debug."""
        sink = RemoteTMSink(base_url="http://localhost:9111")
        with patch.object(sink, "_request", new_callable=AsyncMock) as mock_req:
            mock_req.side_effect = Exception("network error")
            # Should not raise
            await sink.increment_used_count("exp42")
        mock_req.assert_awaited_once()


# ---------------------------------------------------------------------------
# LocalTMSink method delegation tests (cover actual method bodies)
# ---------------------------------------------------------------------------

class TestLocalTMSinkMethods:
    """Tests for LocalTMSink method delegation to op_* functions."""

    @pytest.mark.asyncio
    async def test_draft_save_delegates(self):
        """draft_save should call _op_draft_save."""
        sink = LocalTMSink(user="u1")
        mock_op = AsyncMock(return_value={"id": "d1"})
        with patch("daemon.tm_sink._op_draft_save", mock_op):
            result = await sink.draft_save(
                title="t", content="c", tags=["a"],
                project="p", group_key="gk",
                conversation_id="conv1", skip_dedup=True,
            )
        mock_op.assert_awaited_once_with(
            "u1", title="t", content="c", tags=["a"],
            project="p", group_key="gk",
            conversation_id="conv1", skip_dedup=True,
        )
        assert result == {"id": "d1"}

    @pytest.mark.asyncio
    async def test_draft_publish_delegates(self):
        """draft_publish should call _op_draft_publish."""
        sink = LocalTMSink(user="u1")
        mock_op = AsyncMock(return_value={"id": "d1"})
        with patch("daemon.tm_sink._op_draft_publish", mock_op):
            result = await sink.draft_publish(
                draft_id="d1", refined_content="rc",
            )
        mock_op.assert_awaited_once_with(
            "u1", draft_id="d1", refined_content="rc",
        )
        assert result == {"id": "d1"}

    @pytest.mark.asyncio
    async def test_save_delegates(self):
        """save should call _op_save."""
        sink = LocalTMSink(user="u1")
        mock_op = AsyncMock(return_value={"id": "e1"})
        with patch("daemon.tm_sink._op_save", mock_op):
            result = await sink.save(
                title="t", problem="p", solution="s",
                content="c", tags=["a"], scope="global",
                experience_type="lesson", project="proj",
                group_key="gk",
            )
        mock_op.assert_awaited_once_with(
            "u1", title="t", problem="p", solution="s",
            content="c", tags=["a"], scope="global",
            experience_type="lesson", project="proj",
            group_key="gk",
        )
        assert result == {"id": "e1"}

    @pytest.mark.asyncio
    async def test_recall_with_string_return(self):
        """recall should parse JSON string returns."""
        import json
        sink = LocalTMSink(user="u1")
        data = {"results": [{"id": "1"}]}
        mock_op = AsyncMock(return_value=json.dumps(data))
        with patch("daemon.tm_sink._op_recall", mock_op):
            result = await sink.recall(query="q")
        assert result == [{"id": "1"}]

    @pytest.mark.asyncio
    async def test_recall_with_dict_return(self):
        """recall should extract 'results' from dict returns."""
        sink = LocalTMSink(user="u1")
        mock_op = AsyncMock(return_value={"results": [{"id": "1"}, {"id": "2"}]})
        with patch("daemon.tm_sink._op_recall", mock_op):
            result = await sink.recall(query="q")
        assert result == [{"id": "1"}, {"id": "2"}]

    @pytest.mark.asyncio
    async def test_context_delegates(self):
        """context should call _op_context."""
        sink = LocalTMSink(user="u1")
        mock_op = AsyncMock(return_value={"ctx": "data"})
        with patch("daemon.tm_sink._op_context", mock_op):
            result = await sink.context(
                file_paths=["/f"], task_description="d", project="p",
            )
        mock_op.assert_awaited_once_with(
            "u1", file_paths=["/f"], task_description="d", project="p",
        )
        assert result == {"ctx": "data"}


class TestRemoteTMSinkDraftPublish:
    """Tests for RemoteTMSink.draft_publish."""

    @pytest.mark.asyncio
    async def test_draft_publish_with_refined_content(self):
        """draft_publish should include refined_content in body."""
        sink = RemoteTMSink(base_url="http://localhost:9111")
        with patch.object(sink, "_request", new_callable=AsyncMock) as mock_req:
            mock_req.return_value = {"id": "d1"}
            await sink.draft_publish(
                draft_id="d1", refined_content="refined text",
            )
        args, _ = mock_req.call_args
        body = args[2]
        assert body["draft_id"] == "d1"
        assert body["refined_content"] == "refined text"

    @pytest.mark.asyncio
    async def test_draft_publish_without_refined_content(self):
        """draft_publish without refined_content should omit it."""
        sink = RemoteTMSink(base_url="http://localhost:9111")
        with patch.object(sink, "_request", new_callable=AsyncMock) as mock_req:
            mock_req.return_value = {"id": "d1"}
            await sink.draft_publish(draft_id="d1")
        args, _ = mock_req.call_args
        body = args[2]
        assert body == {"draft_id": "d1"}


class TestRemoteTMSinkContext:
    """Tests for RemoteTMSink.context."""

    @pytest.mark.asyncio
    async def test_context_all_params(self):
        """context should include file_paths, task_description, project."""
        sink = RemoteTMSink(base_url="http://localhost:9111")
        with patch.object(sink, "_request", new_callable=AsyncMock) as mock_req:
            mock_req.return_value = {"user_profile": {}, "experiences": []}
            result = await sink.context(
                file_paths=["/a.py", "/b.py"],
                task_description="fix bug",
                project="proj",
            )
        args, _ = mock_req.call_args
        body = args[2]
        assert body["file_paths"] == ["/a.py", "/b.py"]
        assert body["task_description"] == "fix bug"
        assert body["project"] == "proj"

    @pytest.mark.asyncio
    async def test_context_empty_body(self):
        """context with no params should send empty body."""
        sink = RemoteTMSink(base_url="http://localhost:9111")
        with patch.object(sink, "_request", new_callable=AsyncMock) as mock_req:
            mock_req.return_value = {}
            await sink.context()
        args, _ = mock_req.call_args
        body = args[2]
        assert body == {}


class TestCreateSink:
    """Tests for create_sink factory function."""

    def test_create_local_sink(self):
        """create_sink with mode='local' returns LocalTMSink."""
        from daemon.tm_sink import create_sink
        sink = create_sink({"mode": "local", "user": "u1"})
        assert isinstance(sink, LocalTMSink)
        assert sink._user == "u1"

    def test_create_local_sink_default(self):
        """create_sink with no mode defaults to local."""
        from daemon.tm_sink import create_sink
        sink = create_sink({})
        assert isinstance(sink, LocalTMSink)

    def test_create_remote_sink(self):
        """create_sink with mode='remote' returns RemoteTMSink."""
        from daemon.tm_sink import create_sink
        sink = create_sink({"mode": "remote", "base_url": "http://x", "user": "u2"})
        assert isinstance(sink, RemoteTMSink)
        assert sink._user == "u2"
        assert sink._base_url == "http://x"

    def test_create_remote_sink_no_url_raises(self):
        """create_sink with mode='remote' and no base_url raises ValueError."""
        from daemon.tm_sink import create_sink
        with pytest.raises(ValueError, match="requires 'base_url'"):
            create_sink({"mode": "remote"})

    def test_create_unknown_mode_raises(self):
        """create_sink with unknown mode raises ValueError."""
        from daemon.tm_sink import create_sink
        with pytest.raises(ValueError, match="Unsupported tm mode"):
            create_sink({"mode": "unknown"})

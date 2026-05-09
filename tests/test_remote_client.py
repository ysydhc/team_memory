"""Tests for team_memory.remote_client — RemoteTMClient and mode detection."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest


class TestRemoteTMClient:
    """RemoteTMClient routes op_* calls to /api/v1/mcp/* endpoints."""

    def _make_client(self, base_url: str = "http://tm:9111", api_key: str = "key-x"):
        from team_memory.remote_client import RemoteTMClient
        return RemoteTMClient(base_url=base_url, api_key=api_key)

    def _mock_response(self, data: dict):
        mock = MagicMock()
        mock.json.return_value = data
        mock.raise_for_status = MagicMock()
        return mock

    @pytest.mark.asyncio
    async def test_op_save(self):
        client = self._make_client()
        resp = self._mock_response({"message": "saved", "data": {"id": "e1"}})
        client._client = AsyncMock()
        client._client.post.return_value = resp
        result = await client.op_save("user", title="t", problem="p", solution="s")
        assert result["data"]["id"] == "e1"
        call = client._client.post.call_args
        assert "/mcp/save" in call[0][0]

    @pytest.mark.asyncio
    async def test_op_recall(self):
        client = self._make_client()
        resp = self._mock_response({"results": [{"id": "r1"}]})
        client._client = AsyncMock()
        client._client.post.return_value = resp
        result = await client.op_recall("user", query="test")
        assert result["results"][0]["id"] == "r1"
        call = client._client.post.call_args
        assert "/mcp/recall" in call[0][0]

    @pytest.mark.asyncio
    async def test_op_context(self):
        client = self._make_client()
        resp = self._mock_response({"context": "..."})
        client._client = AsyncMock()
        client._client.post.return_value = resp
        result = await client.op_context("user", task_description="deploy")
        assert "context" in result

    @pytest.mark.asyncio
    async def test_op_get_archive(self):
        client = self._make_client()
        resp = self._mock_response({"id": "a1", "title": "arch"})
        client._client = AsyncMock()
        client._client.get.return_value = resp
        result = await client.op_get_archive("user", archive_id="a1")
        assert result["id"] == "a1"

    @pytest.mark.asyncio
    async def test_op_feedback(self):
        client = self._make_client()
        resp = self._mock_response({"message": "rated"})
        client._client = AsyncMock()
        client._client.post.return_value = resp
        result = await client.op_feedback("user", experience_id="e1", rating=5)
        assert result["message"] == "rated"

    @pytest.mark.asyncio
    async def test_op_draft_save(self):
        client = self._make_client()
        resp = self._mock_response({"draft_id": "d1"})
        client._client = AsyncMock()
        client._client.post.return_value = resp
        result = await client.op_draft_save("user", title="draft", content="c")
        assert result["draft_id"] == "d1"

    @pytest.mark.asyncio
    async def test_op_draft_publish(self):
        client = self._make_client()
        resp = self._mock_response({"experience_id": "e2"})
        client._client = AsyncMock()
        client._client.post.return_value = resp
        result = await client.op_draft_publish("user", draft_id="d1")
        assert result["experience_id"] == "e2"


class TestDetectRemoteMode:
    """detect_remote_mode checks env and config correctly."""

    def test_remote_forced(self):
        from team_memory.remote_client import detect_remote_mode
        with patch.dict("os.environ", {"TM_REMOTE_MODE": "1", "TM_SERVER_URL": ""}):
            is_remote, _, _ = detect_remote_mode()
            assert is_remote is True

    def test_remote_via_server_url(self):
        from team_memory.remote_client import detect_remote_mode
        with patch.dict("os.environ", {"TM_SERVER_URL": "http://remote:9111"}, clear=False):
            is_remote, url, _ = detect_remote_mode()
            assert is_remote is True
            assert url == "http://remote:9111"

    def test_local_mode_default(self):
        from team_memory.remote_client import detect_remote_mode
        with patch.dict("os.environ", {}, clear=True):
            is_remote, url, _ = detect_remote_mode()
            assert isinstance(is_remote, bool)


class TestModeInit:
    """_init_mode in server.py sets _remote_client correctly."""

    def test_init_mode_local(self):
        from team_memory import server
        # Reset
        server._remote_client = None
        with patch.dict("os.environ", {"TM_REMOTE_MODE": ""}, clear=False):
            server._init_mode()
        assert server._remote_client is None

    def test_init_mode_remote(self):
        from team_memory import server
        server._remote_client = None
        with patch.dict("os.environ", {"TM_REMOTE_MODE": "1"}, clear=False):
            server._init_mode()
        assert server._remote_client is not None
        # Cleanup
        server._remote_client = None

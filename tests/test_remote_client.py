"""Tests for team_memory.remote_client — RemoteMCPClient and setup_remote_ops."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest


class TestRemoteMCPClient:
    """RemoteMCPClient routes op_* calls to the correct /mcp/ endpoints."""

    def _make_client(self, base_url: str = "http://tm:9111", api_key: str = "key-x"):
        from team_memory.remote_client import RemoteMCPClient
        return RemoteMCPClient(base_url=base_url, api_key=api_key)

    def _mock_http(self, response_data: dict):
        """Return (MockClient ctx, mock_client) with pre-configured response."""
        mock_response = MagicMock()
        mock_response.json.return_value = response_data
        mock_response.raise_for_status = MagicMock()
        mock_client = AsyncMock()
        mock_client.post.return_value = mock_response
        mock_client.get.return_value = mock_response
        return mock_client, mock_response

    @pytest.mark.asyncio
    async def test_op_save_posts_to_mcp_save(self):
        client = self._make_client()
        mock_client, _ = self._mock_http({"id": "e1"})
        with patch("team_memory.remote_client.httpx.AsyncClient") as MC:
            MC.return_value.__aenter__ = AsyncMock(return_value=mock_client)
            MC.return_value.__aexit__ = AsyncMock(return_value=False)
            result = await client.op_save(
                "user1", title="T", problem="P", solution="S", project="proj"
            )
        mock_client.post.assert_awaited_once()
        call_args = mock_client.post.call_args
        assert call_args.args[0] == "http://tm:9111/mcp/save"
        body = call_args.kwargs["json"]
        assert body["title"] == "T"
        assert body["problem"] == "P"
        assert body["project"] == "proj"
        assert result == {"id": "e1"}

    @pytest.mark.asyncio
    async def test_op_recall_posts_to_mcp_recall(self):
        client = self._make_client()
        mock_client, _ = self._mock_http({"results": [{"id": "r1"}]})
        with patch("team_memory.remote_client.httpx.AsyncClient") as MC:
            MC.return_value.__aenter__ = AsyncMock(return_value=mock_client)
            MC.return_value.__aexit__ = AsyncMock(return_value=False)
            result = await client.op_recall("user1", query="search term")
        call_args = mock_client.post.call_args
        assert call_args.args[0] == "http://tm:9111/mcp/recall"
        assert call_args.kwargs["json"]["query"] == "search term"
        assert result == {"results": [{"id": "r1"}]}

    @pytest.mark.asyncio
    async def test_op_context_posts_to_mcp_context(self):
        client = self._make_client()
        mock_client, _ = self._mock_http({"profile": {}, "relevant_experiences": []})
        with patch("team_memory.remote_client.httpx.AsyncClient") as MC:
            MC.return_value.__aenter__ = AsyncMock(return_value=mock_client)
            MC.return_value.__aexit__ = AsyncMock(return_value=False)
            await client.op_context("user1", file_paths=["a.py"], project="proj")
        call_args = mock_client.post.call_args
        assert call_args.args[0] == "http://tm:9111/mcp/context"
        assert call_args.kwargs["json"]["file_paths"] == ["a.py"]

    @pytest.mark.asyncio
    async def test_op_get_archive_gets_mcp_archive(self):
        client = self._make_client()
        mock_client, _ = self._mock_http({"id": "arch-1", "title": "T"})
        with patch("team_memory.remote_client.httpx.AsyncClient") as MC:
            MC.return_value.__aenter__ = AsyncMock(return_value=mock_client)
            MC.return_value.__aexit__ = AsyncMock(return_value=False)
            result = await client.op_get_archive("user1", archive_id="arch-1")
        mock_client.get.assert_awaited_once()
        call_args = mock_client.get.call_args
        assert call_args.args[0] == "http://tm:9111/mcp/archive/arch-1"
        assert result == {"id": "arch-1", "title": "T"}

    @pytest.mark.asyncio
    async def test_op_feedback_posts_to_mcp_feedback(self):
        client = self._make_client()
        mock_client, _ = self._mock_http({"ok": True})
        with patch("team_memory.remote_client.httpx.AsyncClient") as MC:
            MC.return_value.__aenter__ = AsyncMock(return_value=mock_client)
            MC.return_value.__aexit__ = AsyncMock(return_value=False)
            await client.op_feedback("user1", experience_id="e1", rating=5)
        call_args = mock_client.post.call_args
        assert call_args.args[0] == "http://tm:9111/mcp/feedback"
        body = call_args.kwargs["json"]
        assert body["experience_id"] == "e1"
        assert body["rating"] == 5

    @pytest.mark.asyncio
    async def test_op_draft_save_posts_to_mcp_draft_save(self):
        client = self._make_client()
        mock_client, _ = self._mock_http({"id": "d1", "status": "draft"})
        with patch("team_memory.remote_client.httpx.AsyncClient") as MC:
            MC.return_value.__aenter__ = AsyncMock(return_value=mock_client)
            MC.return_value.__aexit__ = AsyncMock(return_value=False)
            result = await client.op_draft_save(
                "user1", title="Draft T", content="Draft C", project="proj"
            )
        call_args = mock_client.post.call_args
        assert call_args.args[0] == "http://tm:9111/mcp/draft-save"
        body = call_args.kwargs["json"]
        assert body["title"] == "Draft T"
        assert body["project"] == "proj"
        assert result == {"id": "d1", "status": "draft"}

    @pytest.mark.asyncio
    async def test_op_draft_publish_posts_to_mcp_draft_publish(self):
        client = self._make_client()
        mock_client, _ = self._mock_http({"id": "d1", "status": "published"})
        with patch("team_memory.remote_client.httpx.AsyncClient") as MC:
            MC.return_value.__aenter__ = AsyncMock(return_value=mock_client)
            MC.return_value.__aexit__ = AsyncMock(return_value=False)
            result = await client.op_draft_publish(
                "user1", draft_id="d1", refined_content="refined"
            )
        call_args = mock_client.post.call_args
        assert call_args.args[0] == "http://tm:9111/mcp/draft-publish"
        body = call_args.kwargs["json"]
        assert body["draft_id"] == "d1"
        assert body["refined_content"] == "refined"
        assert result == {"id": "d1", "status": "published"}

    def test_auth_header_included(self):
        client = self._make_client(api_key="secret-key")
        headers = client._headers()
        assert headers["Authorization"] == "Bearer secret-key"

    def test_base_url_trailing_slash_stripped(self):
        from team_memory.remote_client import RemoteMCPClient
        c = RemoteMCPClient(base_url="http://tm:9111/", api_key="k")
        assert c._base_url == "http://tm:9111"


class TestSetupRemoteOps:
    """setup_remote_ops patches memory_operations in-place."""

    def test_patches_all_op_functions(self):
        from team_memory.remote_client import RemoteMCPClient, setup_remote_ops
        from team_memory.services import memory_operations

        # Save originals
        orig_save = memory_operations.op_save

        try:
            client = setup_remote_ops(
                base_url="http://tm:9111", api_key="test-key"
            )
            assert isinstance(client, RemoteMCPClient)
            # Patched functions should be bound methods of RemoteMCPClient
            assert memory_operations.op_save == client.op_save
            assert memory_operations.op_recall == client.op_recall
            assert memory_operations.op_context == client.op_context
            assert memory_operations.op_get_archive == client.op_get_archive
            assert memory_operations.op_archive_upsert == client.op_archive_upsert
            assert memory_operations.op_feedback == client.op_feedback
            assert memory_operations.op_draft_save == client.op_draft_save
            assert memory_operations.op_draft_publish == client.op_draft_publish
        finally:
            # Restore originals so other tests are not affected
            memory_operations.op_save = orig_save  # type: ignore[method-assign]

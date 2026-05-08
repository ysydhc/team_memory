"""Tests for MCP callback hook and search_log_id correlation.

Phase A of the use_rate fix:
- write_search_log returns log_id
- _maybe_write_search_log injects search_log_id into result
- /hooks/mcp_callback registers MCP calls in recent_searches
- after_response processes mcp_calls into response_buffer
- response_buffer.has_search_log for dedup
"""
import pytest
import json
import uuid
from unittest.mock import AsyncMock, MagicMock, patch


class TestWriteSearchLogReturnsId:
    """A-1: write_search_log returns log_id."""

    @pytest.mark.asyncio
    async def test_returns_log_id_on_success(self):
        from team_memory.services.memory_operations import write_search_log

        mock_log = MagicMock()
        mock_log.id = uuid.uuid4()
        mock_repo = MagicMock()
        mock_repo.create = AsyncMock(return_value=mock_log)
        mock_session = AsyncMock()
        mock_session.__aenter__ = AsyncMock(return_value=mock_session)
        mock_session.__aexit__ = AsyncMock(return_value=False)
        mock_session.commit = AsyncMock()

        with patch(
            "team_memory.services.memory_operations.get_session",
            return_value=mock_session,
        ), patch(
            "team_memory.services.memory_operations.SearchLogRepository",
            return_value=mock_repo,
        ):
            result = await write_search_log(
                db_url="postgresql://test",
                query="test query",
                source="mcp",
            )
            assert result is not None
            assert result == str(mock_log.id)

    @pytest.mark.asyncio
    async def test_returns_none_on_failure(self):
        from team_memory.services.memory_operations import write_search_log

        with patch(
            "team_memory.services.memory_operations.get_session",
            side_effect=Exception("db error"),
        ):
            result = await write_search_log(
                db_url="postgresql://test",
                query="test query",
            )
            assert result is None


class TestMaybeWriteSearchLogInjectsId:
    """A-2: _maybe_write_search_log injects search_log_id into result."""

    @pytest.mark.asyncio
    async def test_injects_search_log_id(self):
        from team_memory.services.memory_operations import _maybe_write_search_log

        log_id = str(uuid.uuid4())
        result = {"results": [{"id": "exp1", "score": 0.9}], "intent_type": "solve"}

        with patch(
            "team_memory.services.memory_operations.write_search_log",
            new_callable=AsyncMock,
            return_value=log_id,
        ):
            await _maybe_write_search_log(result, "postgresql://test", "team_doc", query="test")

        assert "search_log_id" in result
        assert result["search_log_id"] == log_id

    @pytest.mark.asyncio
    async def test_no_inject_when_write_fails(self):
        from team_memory.services.memory_operations import _maybe_write_search_log

        result = {"results": [], "intent_type": "general"}

        with patch(
            "team_memory.services.memory_operations.write_search_log",
            new_callable=AsyncMock,
            return_value=None,
        ):
            await _maybe_write_search_log(result, "postgresql://test", "team_doc", query="test")

        assert "search_log_id" not in result


class TestMcpCallbackEndpoint:
    """A-4: /hooks/mcp_callback registers MCP calls."""

    @pytest.fixture
    def app_client(self):
        """Create a test client for the daemon app."""
        from httpx import AsyncClient, ASGITransport
        from daemon.app import create_app

        app = create_app()
        app.state.recent_searches = {}
        transport = ASGITransport(app=app)
        return AsyncClient(transport=transport, base_url="http://test")

    @pytest.mark.asyncio
    async def test_callback_appends_to_existing_session(self, app_client):
        """When before_prompt already ran, mcp_callback appends to mcp_calls."""
        async with app_client as client:
            # Simulate before_prompt already created an entry
            client._transport.app.state.recent_searches["sess-123"] = {
                "log_id": "daemon-log-id",
                "query": "test",
                "result_ids": [],
                "project": "team_doc",
                "T1": 100.0,
                "T1_iso": "2026-05-08T12:00:00",
                "mcp_calls": [],
            }

            resp = await client.post("/hooks/mcp_callback", json={
                "session_id": "sess-123",
                "tool_use_id": "tool-456",
                "search_log_id": str(uuid.uuid4()),
                "tool_name": "mcp__team_memory__recall",
                "result_ids": [{"id": "exp1", "score": 0.9}],
            })

            assert resp.status_code == 200
            data = resp.json()
            assert data["status"] == "ok"
            assert data["mcp_calls"] == 1

            # Verify mcp_calls was populated
            entry = client._transport.app.state.recent_searches["sess-123"]
            assert len(entry["mcp_calls"]) == 1
            assert entry["mcp_calls"][0]["tool_use_id"] == "tool-456"

    @pytest.mark.asyncio
    async def test_callback_creates_stub_when_no_before_prompt(self, app_client):
        """When no before_prompt entry exists, creates a stub."""
        async with app_client as client:
            resp = await client.post("/hooks/mcp_callback", json={
                "session_id": "new-session",
                "tool_use_id": "tool-789",
                "search_log_id": str(uuid.uuid4()),
                "tool_name": "mcp__team_memory__recall",
                "result_ids": [],
            })

            assert resp.status_code == 200
            data = resp.json()
            assert data["status"] == "ok"
            assert data["mcp_calls"] == 1

            entry = client._transport.app.state.recent_searches["new-session"]
            assert entry["log_id"] is None
            assert len(entry["mcp_calls"]) == 1

    @pytest.mark.asyncio
    async def test_callback_multiple_mcp_calls(self, app_client):
        """Multiple MCP calls in same session are all recorded."""
        async with app_client as client:
            client._transport.app.state.recent_searches["multi-sess"] = {
                "log_id": "daemon-log",
                "query": "test",
                "result_ids": [],
                "project": "team_doc",
                "T1": 100.0,
                "T1_iso": "2026-05-08T12:00:00",
                "mcp_calls": [],
            }

            for i in range(3):
                resp = await client.post("/hooks/mcp_callback", json={
                    "session_id": "multi-sess",
                    "tool_use_id": f"tool-{i}",
                    "search_log_id": str(uuid.uuid4()),
                    "tool_name": "mcp__team_memory__recall",
                    "result_ids": [],
                })
                assert resp.status_code == 200

            entry = client._transport.app.state.recent_searches["multi-sess"]
            assert len(entry["mcp_calls"]) == 3


class TestResponseBufferHasSearchLog:
    """A-5: response_buffer.has_search_log for dedup."""

    @pytest.mark.asyncio
    async def test_has_search_log_returns_true_for_existing(self):
        from daemon.response_buffer import ResponseBuffer

        buf = ResponseBuffer(dsn="postgresql://test")
        log_id = str(uuid.uuid4())

        mock_conn = AsyncMock()
        mock_conn.fetchval = AsyncMock(return_value=1)
        mock_pool = AsyncMock()
        mock_pool.acquire = MagicMock(return_value=mock_conn)
        mock_pool.__aenter__ = AsyncMock(return_value=mock_pool)
        mock_pool.__aexit__ = AsyncMock(return_value=False)

        with patch.object(buf, "_ensure_pool", new_callable=AsyncMock, return_value=mock_pool):
            result = await buf.has_search_log(log_id)
            assert result is True

    @pytest.mark.asyncio
    async def test_has_search_log_returns_false_for_missing(self):
        from daemon.response_buffer import ResponseBuffer

        buf = ResponseBuffer(dsn="postgresql://test")
        log_id = str(uuid.uuid4())

        mock_conn = AsyncMock()
        mock_conn.fetchval = AsyncMock(return_value=None)

        # Properly mock async context manager for pool.acquire
        mock_acquire = MagicMock()
        mock_acquire.__aenter__ = AsyncMock(return_value=mock_conn)
        mock_acquire.__aexit__ = AsyncMock(return_value=False)
        mock_pool = AsyncMock()
        mock_pool.acquire = MagicMock(return_value=mock_acquire)

        with patch.object(buf, "_ensure_pool", new_callable=AsyncMock, return_value=mock_pool):
            result = await buf.has_search_log(log_id)
            assert result is False

    @pytest.mark.asyncio
    async def test_has_search_log_returns_false_on_error(self):
        from daemon.response_buffer import ResponseBuffer

        buf = ResponseBuffer(dsn="postgresql://test")

        with patch.object(buf, "_ensure_pool", new_callable=AsyncMock, side_effect=Exception("db error")):
            result = await buf.has_search_log(str(uuid.uuid4()))
            assert result is False

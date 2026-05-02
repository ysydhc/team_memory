"""Tests for TM Daemon FastAPI application skeleton.

Tests each endpoint using httpx AsyncClient with ASGITransport.
Pipeline endpoints use mocked TMSink; draft endpoints test with real mocks.
"""

from __future__ import annotations

from contextlib import asynccontextmanager
from unittest.mock import AsyncMock, patch

import pytest
import httpx
from httpx import ASGITransport

from daemon.app import create_app
from daemon.config import DaemonConfig, DaemonSettings, DraftSettings, TMSettings
from daemon.tm_sink import TMSink


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class _StubTMSink(TMSink):
    """A concrete TMSink that records calls for assertions."""

    def __init__(self) -> None:
        self.draft_save_calls: list[dict] = []
        self.draft_publish_calls: list[dict] = []
        self.recall_calls: list[dict] = []
        self._results: list[dict] = []

    def set_recall_results(self, results: list[dict]) -> None:
        self._results = results

    async def draft_save(self, **kwargs) -> dict:
        self.draft_save_calls.append(kwargs)
        return {"id": "test-draft-id", "status": "draft"}

    async def draft_publish(self, **kwargs) -> dict:
        self.draft_publish_calls.append(kwargs)
        return {"id": kwargs.get("draft_id", ""), "status": "published"}

    async def save(self, **kwargs) -> dict:
        return {"id": "test-save-id", "status": "published"}

    async def recall(self, **kwargs) -> list[dict]:
        self.recall_calls.append(kwargs)
        return self._results

    async def update_experience(self, **kwargs) -> dict:
        return {"id": kwargs.get("experience_id", ""), "status": "updated"}

    async def context(self, **kwargs) -> dict:
        return {"user": "test", "relevant_experiences": []}

    async def increment_used_count(self, experience_id: str) -> None:
        pass


def _make_config() -> DaemonConfig:
    """Create a DaemonConfig suitable for testing (remote mode, in-memory DB).

    Use remote mode so the lifespan does NOT call bootstrap() — tests stub
    the sink anyway, so no real TM connection is needed.
    """
    return DaemonConfig(
        daemon=DaemonSettings(host="127.0.0.1", port=3901),
        tm=TMSettings(mode="remote", base_url="http://tm-stub", user="test"),
        draft=DraftSettings(db_path=""),  # will use :memory:
    )


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def stub_sink() -> _StubTMSink:
    return _StubTMSink()


@pytest.fixture
def config() -> DaemonConfig:
    return _make_config()


@pytest.fixture
async def client(stub_sink: _StubTMSink, config: DaemonConfig):
    """Create an httpx AsyncClient with the FastAPI app and injected stub sink.

    httpx ASGITransport does not send ASGI lifespan events, so we manually
    invoke the lifespan context manager to populate app.state.
    """
    with patch("daemon.app.create_sink", return_value=stub_sink):
        app = create_app(config)

        # Manually trigger the lifespan to populate app.state
        lifespan_gen = app.router.lifespan_context(app)
        await lifespan_gen.__aenter__()

        transport = ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as c:
            yield c

        # Cleanup lifespan
        await lifespan_gen.__aexit__(None, None, None)


# ---------------------------------------------------------------------------
# Tests: /status
# ---------------------------------------------------------------------------


class TestStatusEndpoint:
    @pytest.mark.asyncio
    async def test_status_running(self, client: httpx.AsyncClient):
        resp = await client.get("/status")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "running"
        assert data["tm_mode"] == "remote"

    @pytest.mark.asyncio
    async def test_status_remote_mode(self, stub_sink: _StubTMSink):
        config = DaemonConfig(
            daemon=DaemonSettings(),
            tm=TMSettings(mode="remote", base_url="http://tm:3900", user="test"),
            draft=DraftSettings(db_path=""),
        )
        with patch("daemon.app.create_sink", return_value=stub_sink):
            app = create_app(config)
            lifespan_gen = app.router.lifespan_context(app)
            await lifespan_gen.__aenter__()

            transport = ASGITransport(app=app)
            async with httpx.AsyncClient(transport=transport, base_url="http://test") as c:
                resp = await c.get("/status")
                data = resp.json()
                assert data["tm_mode"] == "remote"

            await lifespan_gen.__aexit__(None, None, None)


# ---------------------------------------------------------------------------
# Tests: Hook endpoints (pipeline stubs)
# ---------------------------------------------------------------------------


class TestHookEndpoints:
    @pytest.mark.asyncio
    async def test_after_response(self, client: httpx.AsyncClient):
        resp = await client.post(
            "/hooks/after_response",
            json={
                "conversation_id": "conv-1",
                "prompt": "test prompt",
                "workspace_roots": ["/tmp"],
                "model": "gpt-4",
            },
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["action"] == "ok"

    @pytest.mark.asyncio
    async def test_session_start(self, client: httpx.AsyncClient):
        resp = await client.post(
            "/hooks/session_start",
            json={
                "conversation_id": "conv-1",
                "prompt": "",
                "workspace_roots": ["/tmp"],
                "model": "gpt-4",
            },
        )
        assert resp.status_code == 200
        data = resp.json()
        assert "additional_context" in data

    @pytest.mark.asyncio
    async def test_before_prompt(self, client: httpx.AsyncClient):
        resp = await client.post(
            "/hooks/before_prompt",
            json={
                "conversation_id": "conv-1",
                "prompt": "how to configure X?",
                "workspace_roots": [],
                "model": "",
            },
        )
        assert resp.status_code == 200
        data = resp.json()
        assert "results" in data

    @pytest.mark.asyncio
    async def test_session_end(self, client: httpx.AsyncClient):
        resp = await client.post(
            "/hooks/session_end",
            json={
                "conversation_id": "conv-1",
                "prompt": "",
                "workspace_roots": [],
                "model": "",
            },
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["action"] == "ok"


# ---------------------------------------------------------------------------
# Tests: /draft/save
# ---------------------------------------------------------------------------


class TestDraftSaveEndpoint:
    @pytest.mark.asyncio
    async def test_draft_save_calls_sink(self, client: httpx.AsyncClient, stub_sink: _StubTMSink):
        resp = await client.post(
            "/draft/save",
            json={
                "title": "Test Draft",
                "content": "Some content",
                "project": "test_proj",
                "group_key": "gk1",
                "conversation_id": "conv-1",
            },
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["id"] == "test-draft-id"
        assert len(stub_sink.draft_save_calls) == 1
        call = stub_sink.draft_save_calls[0]
        assert call["title"] == "Test Draft"
        assert call["content"] == "Some content"
        assert call["project"] == "test_proj"

    @pytest.mark.asyncio
    async def test_draft_save_minimal(self, client: httpx.AsyncClient, stub_sink: _StubTMSink):
        resp = await client.post(
            "/draft/save",
            json={
                "title": "Minimal",
                "content": "content",
            },
        )
        assert resp.status_code == 200
        call = stub_sink.draft_save_calls[0]
        assert call["title"] == "Minimal"
        assert call["project"] is None


# ---------------------------------------------------------------------------
# Tests: /draft/publish
# ---------------------------------------------------------------------------


class TestDraftPublishEndpoint:
    @pytest.mark.asyncio
    async def test_draft_publish_calls_sink(
        self, client: httpx.AsyncClient, stub_sink: _StubTMSink
    ):
        resp = await client.post(
            "/draft/publish",
            json={
                "draft_id": "abc-123",
                "refined_content": "Refined text",
            },
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "published"
        assert len(stub_sink.draft_publish_calls) == 1
        call = stub_sink.draft_publish_calls[0]
        assert call["draft_id"] == "abc-123"
        assert call["refined_content"] == "Refined text"

    @pytest.mark.asyncio
    async def test_draft_publish_without_refined(
        self, client: httpx.AsyncClient, stub_sink: _StubTMSink
    ):
        resp = await client.post(
            "/draft/publish",
            json={
                "draft_id": "abc-123",
            },
        )
        assert resp.status_code == 200
        call = stub_sink.draft_publish_calls[0]
        assert call["draft_id"] == "abc-123"
        assert call["refined_content"] is None


# ---------------------------------------------------------------------------
# Tests: /recall
# ---------------------------------------------------------------------------


class TestRecallEndpoint:
    @pytest.mark.asyncio
    async def test_recall_with_query(self, client: httpx.AsyncClient, stub_sink: _StubTMSink):
        stub_sink.set_recall_results([{"id": "r1", "title": "Result 1"}])
        resp = await client.get("/recall", params={"query": "test query", "project": "proj"})
        assert resp.status_code == 200
        data = resp.json()
        assert len(data) == 1
        assert data[0]["id"] == "r1"
        assert len(stub_sink.recall_calls) == 1
        call = stub_sink.recall_calls[0]
        assert call["query"] == "test query"
        assert call["project"] == "proj"

    @pytest.mark.asyncio
    async def test_recall_empty(self, client: httpx.AsyncClient, stub_sink: _StubTMSink):
        stub_sink.set_recall_results([])
        resp = await client.get("/recall", params={"query": "nothing"})
        assert resp.status_code == 200
        data = resp.json()
        assert data == []


# ---------------------------------------------------------------------------
# Tests: Daemon DraftRefiner adaptation
# ---------------------------------------------------------------------------


class TestDaemonDraftRefiner:
    @pytest.mark.asyncio
    async def test_save_draft_uses_sink(self):
        sink = _StubTMSink()
        buf = AsyncMock()
        buf.upsert_draft = AsyncMock()

        from daemon.draft_refiner import DraftRefiner

        refiner = DraftRefiner(sink=sink, draft_buffer=buf)
        result = await refiner.save_draft(
            session_id="conv-1",
            title="Title",
            content="Content",
            project="proj",
        )
        assert result["id"] == "test-draft-id"
        assert len(sink.draft_save_calls) == 1
        assert sink.draft_save_calls[0]["conversation_id"] == "conv-1"
        buf.upsert_draft.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_refine_and_publish_uses_sink(self):
        sink = _StubTMSink()
        buf = AsyncMock()
        buf.get_pending = AsyncMock(
            return_value=[{"id": "d-1", "content": "使用了Python解决配置问题"}]
        )
        buf.mark_published_by_session = AsyncMock()

        from daemon.draft_refiner import DraftRefiner

        refiner = DraftRefiner(sink=sink, draft_buffer=buf)
        result = await refiner.refine_and_publish("conv-1")

        assert result is not None
        assert result["draft_id"] == "d-1"
        assert result["status"] == "published"
        assert len(sink.draft_publish_calls) == 1
        assert sink.draft_publish_calls[0]["draft_id"] == "d-1"
        buf.mark_published_by_session.assert_awaited_once_with("conv-1")

    @pytest.mark.asyncio
    async def test_refine_and_publish_no_pending(self):
        sink = _StubTMSink()
        buf = AsyncMock()
        buf.get_pending = AsyncMock(return_value=[])

        from daemon.draft_refiner import DraftRefiner

        refiner = DraftRefiner(sink=sink, draft_buffer=buf)
        result = await refiner.refine_and_publish("conv-1")

        assert result is None
        assert len(sink.draft_publish_calls) == 0

    def test_extract_facts(self):
        from daemon.draft_refiner import DraftRefiner

        facts = DraftRefiner.extract_facts("因为配置错误所以需要修复。其他无关句子")
        assert len(facts) >= 1
        assert any("因为" in f for f in facts)

    def test_extract_facts_empty(self):
        from daemon.draft_refiner import DraftRefiner

        assert DraftRefiner.extract_facts("") == []
        assert DraftRefiner.extract_facts("hello world") == []

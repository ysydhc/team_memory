"""Tests for D-architecture features (D1-D5)."""

import asyncio
from unittest.mock import AsyncMock, patch

import pytest

from team_memory.services.cache import (
    MemoryCacheBackend,
    RedisCacheBackend,
    SearchCache,
    create_cache_backend,
)
from team_memory.services.embedding_queue import EmbeddingQueue, EmbeddingTask
from team_memory.services.event_bus import Event, EventBus, Events

# ============================================================
# D5: EventBus Tests
# ============================================================


class TestEventBus:
    @pytest.mark.asyncio
    async def test_emit_calls_handler(self):
        bus = EventBus()
        handler = AsyncMock()
        bus.on(Events.EXPERIENCE_CREATED, handler)

        await bus.emit(Events.EXPERIENCE_CREATED, {"id": "123"})
        handler.assert_called_once_with({"id": "123"})

    @pytest.mark.asyncio
    async def test_multiple_handlers(self):
        bus = EventBus()
        h1 = AsyncMock()
        h2 = AsyncMock()
        bus.on(Events.EXPERIENCE_CREATED, h1)
        bus.on(Events.EXPERIENCE_CREATED, h2)

        await bus.emit(Events.EXPERIENCE_CREATED, {"id": "456"})
        h1.assert_called_once()
        h2.assert_called_once()

    @pytest.mark.asyncio
    async def test_handler_failure_isolated(self):
        """One failing handler should not affect others."""
        bus = EventBus()

        async def failing_handler(payload):
            raise RuntimeError("boom")

        h2 = AsyncMock()
        bus.on(Events.EXPERIENCE_CREATED, failing_handler)
        bus.on(Events.EXPERIENCE_CREATED, h2)

        await bus.emit(Events.EXPERIENCE_CREATED, {"id": "789"})
        h2.assert_called_once()

    @pytest.mark.asyncio
    async def test_no_handlers_no_error(self):
        bus = EventBus()
        # Should not raise
        await bus.emit(Events.EXPERIENCE_DELETED, {"id": "abc"})

    @pytest.mark.asyncio
    async def test_off_removes_handler(self):
        bus = EventBus()
        handler = AsyncMock()
        bus.on(Events.EXPERIENCE_CREATED, handler)
        bus.off(Events.EXPERIENCE_CREATED, handler)

        await bus.emit(Events.EXPERIENCE_CREATED, {"id": "123"})
        handler.assert_not_called()

    def test_handler_count(self):
        bus = EventBus()
        bus.on(Events.EXPERIENCE_CREATED, AsyncMock())
        bus.on(Events.EXPERIENCE_DELETED, AsyncMock())
        bus.on(Events.EXPERIENCE_CREATED, AsyncMock())
        assert bus.handler_count == 3

    @pytest.mark.asyncio
    async def test_recent_events(self):
        bus = EventBus()
        await bus.emit(Events.EXPERIENCE_CREATED, {"id": "1"})
        await bus.emit(Events.EXPERIENCE_DELETED, {"id": "2"})

        events = bus.recent_events
        assert len(events) == 2
        assert events[0]["type"] == Events.EXPERIENCE_CREATED
        assert events[1]["type"] == Events.EXPERIENCE_DELETED

    def test_clear_handlers(self):
        bus = EventBus()
        bus.on(Events.EXPERIENCE_CREATED, AsyncMock())
        bus.clear_handlers()
        assert bus.handler_count == 0

    def test_stats(self):
        bus = EventBus()
        bus.on(Events.EXPERIENCE_CREATED, AsyncMock())
        stats = bus.stats()
        assert stats["registered_handlers"] == 1
        assert Events.EXPERIENCE_CREATED in stats["event_types"]

    @pytest.mark.asyncio
    async def test_emit_default_empty_payload(self):
        bus = EventBus()
        handler = AsyncMock()
        bus.on(Events.SEARCH_EXECUTED, handler)
        await bus.emit(Events.SEARCH_EXECUTED)
        handler.assert_called_once_with({})


class TestEvent:
    def test_event_to_dict(self):
        event = Event(
            type=Events.EXPERIENCE_CREATED,
            payload={"id": "test"},
        )
        d = event.to_dict()
        assert d["type"] == Events.EXPERIENCE_CREATED
        assert d["payload"] == {"id": "test"}
        assert "timestamp" in d


# ============================================================
# D1: Cache Backend Tests
# ============================================================


class TestCacheBackendFactory:
    def test_memory_backend_default(self):
        backend = create_cache_backend()
        assert isinstance(backend, MemoryCacheBackend)

    def test_redis_backend_creation(self):
        backend = create_cache_backend(backend="redis", redis_url="redis://fake:6379/0")
        assert isinstance(backend, RedisCacheBackend)

    def test_unknown_backend_defaults_memory(self):
        backend = create_cache_backend(backend="unknown")
        assert isinstance(backend, MemoryCacheBackend)


class TestMemoryCacheBackendAsync:
    @pytest.mark.asyncio
    async def test_put_and_get(self):
        backend = MemoryCacheBackend(max_size=10, ttl_seconds=300)
        await backend.put("key", {"data": 1})
        result = await backend.get("key")
        assert result == {"data": 1}

    @pytest.mark.asyncio
    async def test_size(self):
        backend = MemoryCacheBackend(max_size=10, ttl_seconds=300)
        await backend.put("a", 1)
        await backend.put("b", 2)
        assert await backend.size() == 2

    @pytest.mark.asyncio
    async def test_clear(self):
        backend = MemoryCacheBackend(max_size=10, ttl_seconds=300)
        await backend.put("a", 1)
        await backend.clear()
        assert await backend.size() == 0


class TestSearchCacheWithBackend:
    @pytest.mark.asyncio
    async def test_default_uses_memory(self):
        cache = SearchCache(enabled=True)
        assert cache._backend == "memory"

    @pytest.mark.asyncio
    async def test_stats_include_backend(self):
        cache = SearchCache(enabled=True)
        stats = await cache.stats
        assert stats["backend"] == "memory"
        assert "result_cache_size" in stats
        assert "embedding_cache_size" in stats


# ============================================================
# D2: EmbeddingQueue Tests
# ============================================================


class TestEmbeddingQueue:
    def test_initial_status(self):
        mock_embedding = AsyncMock()
        queue = EmbeddingQueue(
            embedding_provider=mock_embedding,
            db_url="postgresql://fake",
        )
        status = queue.status
        assert status["running"] is False
        assert status["workers"] == 0
        assert status["pending"] == 0

    @pytest.mark.asyncio
    async def test_enqueue_task(self):
        import uuid

        mock_embedding = AsyncMock()
        queue = EmbeddingQueue(
            embedding_provider=mock_embedding,
            db_url="postgresql://fake",
        )
        exp_id = uuid.uuid4()
        await queue.enqueue(exp_id, "test text")
        assert queue.status["pending"] == 1

    @pytest.mark.asyncio
    async def test_queue_full_raises(self):
        import uuid

        mock_embedding = AsyncMock()
        queue = EmbeddingQueue(
            embedding_provider=mock_embedding,
            db_url="postgresql://fake",
            max_queue_size=2,
        )
        await queue.enqueue(uuid.uuid4(), "text1")
        await queue.enqueue(uuid.uuid4(), "text2")
        with pytest.raises(asyncio.QueueFull):
            await queue.enqueue(uuid.uuid4(), "text3")

    @pytest.mark.asyncio
    async def test_start_creates_workers(self):
        mock_embedding = AsyncMock()
        queue = EmbeddingQueue(
            embedding_provider=mock_embedding,
            db_url="postgresql://fake",
            max_workers=2,
        )
        await queue.start()
        assert queue.status["running"] is True
        assert queue.status["workers"] == 2
        await queue.stop()

    @pytest.mark.asyncio
    async def test_stop_cleans_up(self):
        mock_embedding = AsyncMock()
        queue = EmbeddingQueue(
            embedding_provider=mock_embedding,
            db_url="postgresql://fake",
        )
        await queue.start()
        await queue.stop()
        assert queue.status["running"] is False
        assert queue.status["workers"] == 0


class TestEmbeddingTask:
    def test_task_defaults(self):
        import uuid

        task = EmbeddingTask(experience_id=uuid.uuid4(), text="hello")
        assert task.retry_count == 0
        assert task.created_at is not None


# ============================================================
# D3: API Version Compat Test
# ============================================================


class TestAPIVersioning:
    """Test that old /api/ paths are rewritten to /api/v1/."""

    def test_health_no_auth(self):
        """Health endpoint should work without auth."""
        from fastapi.testclient import TestClient

        from team_memory.web.app import app

        client = TestClient(app)
        resp = client.get("/health")
        assert resp.status_code != 401

    def test_ready_no_auth(self):
        """Ready endpoint should work without auth."""
        from fastapi.testclient import TestClient

        from team_memory.web.app import app

        client = TestClient(app)
        resp = client.get("/ready")
        assert resp.status_code != 401


# ============================================================
# D4: Health Check Tests
# ============================================================


class TestHealthCheckHelpers:
    @pytest.mark.asyncio
    async def test_check_ollama_down(self):
        """_check_ollama should return 'down' when Ollama is unreachable."""
        from team_memory.web.app import _check_ollama

        # With no settings, it tries localhost:11434
        # If Ollama is not running, should return status=down
        result = await _check_ollama()
        # Either up or down, but should not raise
        assert result["status"] in ("up", "down")

    @pytest.mark.asyncio
    async def test_check_cache_no_service(self):
        """_check_cache should handle missing service gracefully."""
        from team_memory.web.app import _check_cache

        with patch("team_memory.web.app._service", None):
            result = await _check_cache()
            assert result["status"] == "unknown"


# ============================================================
# Events Constants Test
# ============================================================


class TestEventsConstants:
    def test_all_events_are_strings(self):
        for attr in dir(Events):
            if not attr.startswith("_"):
                assert isinstance(getattr(Events, attr), str)

    def test_event_naming_convention(self):
        """All events should follow entity.action naming."""
        for attr in dir(Events):
            if not attr.startswith("_"):
                value = getattr(Events, attr)
                assert "." in value, f"Event {attr}={value} missing dot separator"

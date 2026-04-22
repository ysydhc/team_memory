"""Tests for SearchLog model and SearchLogRepository.

Covers:
- Creating a SearchLog entry
- mark_used updates was_used to True
- get_stats returns correct counts
- get_stats with no data returns zeros
- get_recent returns entries within date range
"""

from __future__ import annotations

import uuid
from datetime import datetime, timedelta, timezone

import pytest
from sqlalchemy import select
from sqlalchemy.ext.asyncio import (
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)

from team_memory.storage.models import Base, SearchLog
from team_memory.storage.search_log_repository import SearchLogRepository

# ============================================================
# Fixtures
# ============================================================

# Use SQLite for testing (no pgvector needed for SearchLog)
TEST_DB_URL = "sqlite+aiosqlite://"


@pytest.fixture
async def engine():
    """Create a fresh async engine with only the search_logs table."""
    eng = create_async_engine(TEST_DB_URL, echo=False)
    async with eng.begin() as conn:
        await conn.run_sync(
            lambda sync_conn: Base.metadata.create_all(
                sync_conn, tables=[SearchLog.__table__], checkfirst=True
            )
        )
    yield eng
    async with eng.begin() as conn:
        await conn.run_sync(
            lambda sync_conn: Base.metadata.drop_all(
                sync_conn, tables=[SearchLog.__table__], checkfirst=True
            )
        )
    await eng.dispose()


@pytest.fixture
async def session(engine) -> AsyncSession:
    """Provide an async session for tests."""
    factory = async_sessionmaker(
        engine, class_=AsyncSession, expire_on_commit=False
    )
    async with factory() as sess:
        yield sess


@pytest.fixture
def repo(session: AsyncSession) -> SearchLogRepository:
    """Provide a SearchLogRepository instance."""
    return SearchLogRepository(session)


# ============================================================
# Tests
# ============================================================


class TestSearchLogCreate:
    """Test creating a SearchLog entry."""

    @pytest.mark.asyncio
    async def test_create_basic(
        self, repo: SearchLogRepository, session: AsyncSession
    ):
        """Create a minimal search log entry."""
        log = await repo.create(query="how to deploy")
        await session.commit()

        assert log.id is not None
        assert isinstance(log.id, uuid.UUID)
        assert log.query == "how to deploy"
        assert log.intent_type == "unknown"
        assert log.project == "default"
        assert log.source == "mcp"
        assert log.result_ids is None
        assert log.was_used is None
        assert log.agent_response_snippet is None
        assert log.created_at is not None
        assert log.updated_at is not None

    @pytest.mark.asyncio
    async def test_create_with_all_fields(
        self, repo: SearchLogRepository, session: AsyncSession
    ):
        """Create a search log with all fields populated."""
        results = [
            {"id": "exp-001", "score": 0.95, "source_layer": "L2"},
            {"id": "exp-002", "score": 0.82, "source_layer": "L3"},
        ]
        log = await repo.create(
            query="authentication error",
            intent_type="troubleshooting",
            project="my-project",
            source="api",
            result_ids=results,
        )
        await session.commit()

        assert log.query == "authentication error"
        assert log.intent_type == "troubleshooting"
        assert log.project == "my-project"
        assert log.source == "api"
        assert log.result_ids == results


class TestSearchLogMarkUsed:
    """Test mark_used updates was_used to True."""

    @pytest.mark.asyncio
    async def test_mark_used_sets_true(
        self, repo: SearchLogRepository, session: AsyncSession
    ):
        """Marking a search log as used sets was_used=True."""
        log = await repo.create(query="test query")
        await session.commit()

        assert log.was_used is None

        await repo.mark_used(log.id)
        await session.commit()

        # Re-fetch to verify persistence
        result = await session.execute(
            select(SearchLog).where(SearchLog.id == log.id)
        )
        refreshed = result.scalar_one()
        assert refreshed.was_used is True

    @pytest.mark.asyncio
    async def test_mark_used_with_snippet(
        self, repo: SearchLogRepository, session: AsyncSession
    ):
        """Marking as used with agent response snippet."""
        log = await repo.create(query="test query")
        await session.commit()

        await repo.mark_used(log.id, agent_snippet="I used this to fix the bug")
        await session.commit()

        result = await session.execute(
            select(SearchLog).where(SearchLog.id == log.id)
        )
        refreshed = result.scalar_one()
        assert refreshed.was_used is True
        assert refreshed.agent_response_snippet == "I used this to fix the bug"

    @pytest.mark.asyncio
    async def test_mark_used_nonexistent_id(
        self, repo: SearchLogRepository
    ):
        """Marking a non-existent ID should not raise (silent no-op)."""
        fake_id = uuid.uuid4()
        # Should not raise
        await repo.mark_used(fake_id)


class TestSearchLogGetStats:
    """Test get_stats returns correct counts."""

    @pytest.mark.asyncio
    async def test_get_stats_empty(self, repo: SearchLogRepository):
        """get_stats with no data returns zeros."""
        stats = await repo.get_stats(days=7)

        assert stats["total"] == 0
        assert stats["hit"] == 0
        assert stats["used"] == 0
        assert stats["use_rate"] == 0.0

    @pytest.mark.asyncio
    async def test_get_stats_with_data(
        self, repo: SearchLogRepository, session: AsyncSession
    ):
        """get_stats returns correct counts for mixed entries."""
        # Entry with results (hit) and used
        await repo.create(
            query="query1",
            result_ids=[{"id": "exp-1", "score": 0.9}],
        )
        # Entry with results (hit) but not used
        await repo.create(
            query="query2",
            result_ids=[{"id": "exp-2", "score": 0.8}],
        )
        # Entry with no results (not a hit)
        await repo.create(query="query3")
        # Entry with empty results list (not a hit)
        await repo.create(query="query4", result_ids=[])
        await session.commit()

        # Mark first entry as used
        result = await session.execute(
            select(SearchLog).where(SearchLog.query == "query1")
        )
        log1 = result.scalar_one()
        await repo.mark_used(log1.id)
        await session.commit()

        stats = await repo.get_stats(days=7)

        assert stats["total"] == 4
        assert stats["hit"] == 2  # query1 and query2 have non-empty result_ids
        assert stats["used"] == 1  # only query1 was marked used
        assert stats["use_rate"] == 0.5  # 1/2

    @pytest.mark.asyncio
    async def test_get_stats_zero_hit_rate(
        self, repo: SearchLogRepository, session: AsyncSession
    ):
        """use_rate is 0 when hit=0."""
        await repo.create(query="no results query")
        await session.commit()

        stats = await repo.get_stats(days=7)
        assert stats["hit"] == 0
        assert stats["use_rate"] == 0.0


class TestSearchLogGetRecent:
    """Test get_recent returns entries within date range."""

    @pytest.mark.asyncio
    async def test_get_recent_returns_entries(
        self, repo: SearchLogRepository, session: AsyncSession
    ):
        """get_recent returns recent search logs ordered by created_at desc."""
        await repo.create(query="first query")
        await repo.create(query="second query")
        await repo.create(query="third query")
        await session.commit()

        recent = await repo.get_recent(days=7, limit=100)

        assert len(recent) == 3
        # Ordered by created_at desc — most recent first
        assert recent[0].query == "third query"
        assert recent[1].query == "second query"
        assert recent[2].query == "first query"

    @pytest.mark.asyncio
    async def test_get_recent_respects_limit(
        self, repo: SearchLogRepository, session: AsyncSession
    ):
        """get_recent respects the limit parameter."""
        for i in range(5):
            await repo.create(query=f"query {i}")
        await session.commit()

        recent = await repo.get_recent(days=7, limit=2)
        assert len(recent) == 2

    @pytest.mark.asyncio
    async def test_get_recent_excludes_old_entries(
        self, repo: SearchLogRepository, session: AsyncSession
    ):
        """get_recent only returns entries within the specified day range."""
        # Insert an entry with an old created_at
        old_log = SearchLog(
            query="old query",
            created_at=datetime.now(timezone.utc) - timedelta(days=30),
            updated_at=datetime.now(timezone.utc) - timedelta(days=30),
        )
        session.add(old_log)
        await session.flush()

        # Insert a recent entry
        await repo.create(query="recent query")
        await session.commit()

        recent = await repo.get_recent(days=7, limit=100)
        assert len(recent) == 1
        assert recent[0].query == "recent query"

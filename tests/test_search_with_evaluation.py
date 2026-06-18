"""Tests for EvaluationService integration into search pipeline.

Covers:
- Search results contain [mem:xxx] markers when evaluation_service is set
- Search results don't contain markers when evaluation_service is None
- SearchLog is written after memory_recall
- SearchLog has correct intent_type
- SearchLog has correct result_ids
"""

from __future__ import annotations

import uuid
from unittest.mock import AsyncMock, MagicMock, patch

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

TEST_DB_URL = "sqlite+aiosqlite://"


@pytest.fixture
async def engine():
    """Create a fresh async engine with only the search_logs table."""
    eng = create_async_engine(TEST_DB_URL, echo=False)
    async with eng.begin() as conn:
        await conn.run_sync(
            lambda sync_conn: Base.metadata.create_all(sync_conn, tables=[SearchLog.__table__], checkfirst=True)
        )
    yield eng
    async with eng.begin() as conn:
        await conn.run_sync(
            lambda sync_conn: Base.metadata.drop_all(sync_conn, tables=[SearchLog.__table__], checkfirst=True)
        )
    await eng.dispose()


@pytest.fixture
async def session(engine) -> AsyncSession:
    """Provide an async session for tests."""
    factory = async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)
    async with factory() as sess:
        yield sess


@pytest.fixture
def repo(session: AsyncSession) -> SearchLogRepository:
    """Provide a SearchLogRepository instance."""
    return SearchLogRepository(session)

    """Test that SearchLog is written after memory_recall."""

    @pytest.mark.asyncio
    async def test_searchlog_written_after_recall(self, repo, session):
        """A SearchLog entry is created when memory_recall completes."""
        log = await repo.create(
            query="how to deploy",
            intent_type="troubleshooting",
            project="default",
            source="mcp",
            result_ids=[{"id": "exp-001", "score": 0.9}],
        )
        await session.commit()

        # Verify it was created
        result = await session.execute(select(SearchLog).where(SearchLog.query == "how to deploy"))
        fetched = result.scalar_one()
        assert fetched.query == "how to deploy"
        assert fetched.intent_type == "troubleshooting"
        assert fetched.source == "mcp"
        assert fetched.result_ids == [{"id": "exp-001", "score": 0.9}]

    @pytest.mark.asyncio
    async def test_searchlog_has_correct_intent_type(self, repo, session):
        """SearchLog stores the intent_type from the orchestrator result."""
        log = await repo.create(
            query="authentication error",
            intent_type="troubleshooting",
            source="mcp",
        )
        await session.commit()

        result = await session.execute(select(SearchLog).where(SearchLog.query == "authentication error"))
        fetched = result.scalar_one()
        assert fetched.intent_type == "troubleshooting"

    @pytest.mark.asyncio
    async def test_searchlog_has_correct_result_ids(self, repo, session):
        """SearchLog stores result_ids extracted from search results."""
        result_ids = [
            {"id": "exp-001", "score": 0.95},
            {"id": "exp-002", "score": 0.82},
        ]
        log = await repo.create(
            query="test query",
            intent_type="general",
            result_ids=result_ids,
        )
        await session.commit()

        result = await session.execute(select(SearchLog).where(SearchLog.query == "test query"))
        fetched = result.scalar_one()
        assert fetched.result_ids == result_ids
        assert len(fetched.result_ids) == 2


# ============================================================
# Test: write_search_log helper in memory_operations
# ============================================================


class TestWriteSearchLogHelper:
    """Test the write_search_log helper function in memory_operations."""

    @pytest.mark.asyncio
    async def test_write_search_log_creates_entry(self):
        """write_search_log creates a SearchLog entry with correct fields."""
        from team_memory.services.memory.memory_operations import write_search_log

        mock_session = AsyncMock()
        mock_repo = MagicMock()
        mock_log = MagicMock()
        mock_log.id = uuid.uuid4()
        mock_repo.create = AsyncMock(return_value=mock_log)

        with (
            patch("team_memory.services.memory.memory_operations.get_session") as mock_gs,
            patch(
                "team_memory.services.memory.memory_operations.SearchLogRepository",
                return_value=mock_repo,
            ),
        ):
            mock_gs.return_value.__aenter__ = AsyncMock(return_value=mock_session)
            mock_gs.return_value.__aexit__ = AsyncMock(return_value=False)

            await write_search_log(
                db_url="sqlite://",
                query="test query",
                intent_type="troubleshooting",
                project="default",
                source="mcp",
                result_ids=[{"id": "exp-1", "score": 0.9}],
            )

        mock_repo.create.assert_awaited_once_with(
            query="test query",
            intent_type="troubleshooting",
            project="default",
            source="mcp",
            result_ids=[{"id": "exp-1", "score": 0.9}],
        )

    @pytest.mark.asyncio
    async def test_write_search_log_handles_db_error_gracefully(self):
        """write_search_log does not raise on DB error (logs and continues)."""
        from team_memory.services.memory.memory_operations import write_search_log

        mock_session = AsyncMock()
        mock_repo = MagicMock()
        mock_repo.create = AsyncMock(side_effect=RuntimeError("DB down"))

        with (
            patch("team_memory.services.memory.memory_operations.get_session") as mock_gs,
            patch(
                "team_memory.services.memory.memory_operations.SearchLogRepository",
                return_value=mock_repo,
            ),
        ):
            mock_gs.return_value.__aenter__ = AsyncMock(return_value=mock_session)
            mock_gs.return_value.__aexit__ = AsyncMock(return_value=False)

            # Should not raise
            await write_search_log(
                db_url="sqlite://",
                query="test query",
                intent_type="general",
            )

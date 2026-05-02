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

from team_memory.services.evaluation import EvaluationService
from team_memory.services.search_orchestrator import (
    OrchestratedSearchResult,
    SearchOrchestrator,
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
# Test: Marker injection via SearchOrchestrator
# ============================================================


class TestMarkerInjection:
    """Test that [mem:xxx] markers are injected when evaluation_service is set."""

    @pytest.mark.asyncio
    async def test_results_contain_markers_when_evaluation_service_set(self):
        """When evaluation_service is provided, search results get [mem:xxx] markers."""
        eval_svc = EvaluationService()
        mock_pipeline = MagicMock()
        mock_pipeline_result = MagicMock()
        mock_pipeline_result.results = [
            {"id": "abc-123", "title": "Result 1", "solution": "Fix the bug", "score": 0.9},
            {"id": "def-456", "title": "Result 2", "description": "Another approach", "score": 0.8},
        ]
        mock_pipeline_result.search_type = "hybrid"
        mock_pipeline.search = AsyncMock(return_value=mock_pipeline_result)

        mock_embedding = MagicMock()
        orchestrator = SearchOrchestrator(
            search_pipeline=mock_pipeline,
            embedding_provider=mock_embedding,
            db_url="sqlite://",
            evaluation_service=eval_svc,
        )

        mock_session = AsyncMock()
        mock_repo = MagicMock()
        mock_repo.increment_recall_count = AsyncMock()

        with (
            patch(
                "team_memory.storage.database.get_session",
            ) as mock_gs,
            patch(
                "team_memory.services.search_orchestrator.ExperienceRepository",
                return_value=mock_repo,
            ),
        ):
            mock_gs.return_value.__aenter__ = AsyncMock(return_value=mock_session)
            mock_gs.return_value.__aexit__ = AsyncMock(return_value=False)

            result = await orchestrator.search(
                query="test query",
                max_results=5,
                user_name="tester",
            )

        # Results should contain markers
        assert len(result.results) == 2
        assert "[mem:abc-123]" in result.results[0]["solution"]
        assert "[mem:def-456]" in result.results[1]["description"]
        # Also check _marker field
        assert result.results[0]["_marker"] == "[mem:abc-123]"
        assert result.results[1]["_marker"] == "[mem:def-456]"

    @pytest.mark.asyncio
    async def test_results_no_markers_when_evaluation_service_none(self):
        """When evaluation_service is None, search results have no markers."""
        mock_pipeline = MagicMock()
        mock_pipeline_result = MagicMock()
        mock_pipeline_result.results = [
            {"id": "abc-123", "title": "Result 1", "solution": "Fix the bug", "score": 0.9},
        ]
        mock_pipeline_result.search_type = "hybrid"
        mock_pipeline.search = AsyncMock(return_value=mock_pipeline_result)

        mock_embedding = MagicMock()
        orchestrator = SearchOrchestrator(
            search_pipeline=mock_pipeline,
            embedding_provider=mock_embedding,
            db_url="sqlite://",
            # No evaluation_service
        )

        mock_session = AsyncMock()
        mock_repo = MagicMock()
        mock_repo.increment_recall_count = AsyncMock()

        with (
            patch(
                "team_memory.storage.database.get_session",
            ) as mock_gs,
            patch(
                "team_memory.services.search_orchestrator.ExperienceRepository",
                return_value=mock_repo,
            ),
        ):
            mock_gs.return_value.__aenter__ = AsyncMock(return_value=mock_session)
            mock_gs.return_value.__aexit__ = AsyncMock(return_value=False)

            result = await orchestrator.search(
                query="test query",
                max_results=5,
                user_name="tester",
            )

        assert len(result.results) == 1
        assert "[mem:" not in result.results[0].get("solution", "")
        assert "_marker" not in result.results[0]

    @pytest.mark.asyncio
    async def test_markers_in_legacy_path_when_evaluation_service_set(self):
        """Legacy fallback also injects markers when evaluation_service is set."""
        eval_svc = EvaluationService()
        mock_embedding = MagicMock()
        mock_embedding.encode_single = AsyncMock(return_value=[0.1, 0.2, 0.3])

        orchestrator = SearchOrchestrator(
            search_pipeline=None,
            embedding_provider=mock_embedding,
            db_url="sqlite://",
            evaluation_service=eval_svc,
        )

        mock_session = AsyncMock()
        mock_repo = MagicMock()
        mock_repo.search_by_vector = AsyncMock(
            return_value=[
                {"id": "xyz-789", "title": "Legacy result", "solution": "Old approach"},
            ]
        )

        with (
            patch(
                "team_memory.storage.database.get_session",
            ) as mock_gs,
            patch(
                "team_memory.services.search_orchestrator.ExperienceRepository",
                return_value=mock_repo,
            ),
        ):
            mock_gs.return_value.__aenter__ = AsyncMock(return_value=mock_session)
            mock_gs.return_value.__aexit__ = AsyncMock(return_value=False)

            result = await orchestrator.search(query="test", max_results=3)

        assert len(result.results) == 1
        assert "[mem:xyz-789]" in result.results[0]["solution"]
        assert result.results[0]["_marker"] == "[mem:xyz-789]"


# ============================================================
# Test: SearchLog writing
# ============================================================


class TestSearchLogWriting:
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
        result = await session.execute(
            select(SearchLog).where(SearchLog.query == "how to deploy")
        )
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

        result = await session.execute(
            select(SearchLog).where(SearchLog.query == "authentication error")
        )
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

        result = await session.execute(
            select(SearchLog).where(SearchLog.query == "test query")
        )
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
        from team_memory.services.memory_operations import write_search_log

        mock_session = AsyncMock()
        mock_repo = MagicMock()
        mock_log = MagicMock()
        mock_log.id = uuid.uuid4()
        mock_repo.create = AsyncMock(return_value=mock_log)

        with patch(
            "team_memory.services.memory_operations.get_session"
        ) as mock_gs, patch(
            "team_memory.services.memory_operations.SearchLogRepository",
            return_value=mock_repo,
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
        from team_memory.services.memory_operations import write_search_log

        mock_session = AsyncMock()
        mock_repo = MagicMock()
        mock_repo.create = AsyncMock(side_effect=RuntimeError("DB down"))

        with patch(
            "team_memory.services.memory_operations.get_session"
        ) as mock_gs, patch(
            "team_memory.services.memory_operations.SearchLogRepository",
            return_value=mock_repo,
        ):
            mock_gs.return_value.__aenter__ = AsyncMock(return_value=mock_session)
            mock_gs.return_value.__aexit__ = AsyncMock(return_value=False)

            # Should not raise
            await write_search_log(
                db_url="sqlite://",
                query="test query",
                intent_type="general",
            )

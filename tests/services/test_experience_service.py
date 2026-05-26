"""Unit tests for the 4 new ExperienceService methods.

Tests:
  - list_published: returns only published/promoted, respects filters
  - count_published: returns an int count
  - soft_delete_by_id: delegates to soft_delete
  - increment_used_count: executes an UPDATE and commits
"""

from __future__ import annotations

import uuid
from contextlib import asynccontextmanager
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from team_memory.services.memory.experience import ExperienceService

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_service():
    """Build an ExperienceService with minimal stubs (no real DB needed)."""
    mock_embedding = MagicMock()
    mock_embedding.encode_single = AsyncMock(return_value=[0.0] * 768)
    mock_embedding.encode = AsyncMock(return_value=[[0.0] * 768])
    mock_embedding.dimension = 768

    mock_auth = MagicMock()
    mock_auth.authenticate = AsyncMock(return_value=MagicMock(name="anon"))

    return ExperienceService(
        embedding_provider=mock_embedding,
        auth_provider=mock_auth,
        db_url="sqlite+aiosqlite://",
    )


def _mock_session_cm(session: AsyncMock):
    """Return an asynccontextmanager that yields *session*."""
    @asynccontextmanager
    async def _cm(_db_url):
        yield session

    return _cm


# ---------------------------------------------------------------------------
# list_published
# ---------------------------------------------------------------------------


class TestListPublished:
    @pytest.mark.asyncio
    async def test_returns_published_and_promoted(self):
        """list_published should only return experiences with allowed statuses."""
        svc = _make_service()

        mock_session = AsyncMock()

        # Simulate two results from DB
        exp1 = MagicMock()
        exp1.to_dict.return_value = {"id": str(uuid.uuid4()), "status": "published"}
        exp2 = MagicMock()
        exp2.to_dict.return_value = {"id": str(uuid.uuid4()), "status": "promoted"}

        scalars_mock = MagicMock()
        scalars_mock.all.return_value = [exp1, exp2]
        execute_result = MagicMock()
        execute_result.scalars.return_value = scalars_mock
        mock_session.execute = AsyncMock(return_value=execute_result)

        with patch("team_memory.storage.database.get_session", _mock_session_cm(mock_session)):
            results = await svc.list_published()

        assert len(results) == 2
        statuses = {r["status"] for r in results}
        assert statuses == {"published", "promoted"}

    @pytest.mark.asyncio
    async def test_returns_empty_list_when_no_results(self):
        svc = _make_service()

        mock_session = AsyncMock()
        scalars_mock = MagicMock()
        scalars_mock.all.return_value = []
        execute_result = MagicMock()
        execute_result.scalars.return_value = scalars_mock
        mock_session.execute = AsyncMock(return_value=execute_result)

        with patch("team_memory.storage.database.get_session", _mock_session_cm(mock_session)):
            results = await svc.list_published(project="nonexistent")

        assert results == []

    @pytest.mark.asyncio
    async def test_passes_filters(self):
        """list_published should execute a query (filter args accepted without error)."""
        svc = _make_service()

        mock_session = AsyncMock()
        scalars_mock = MagicMock()
        scalars_mock.all.return_value = []
        execute_result = MagicMock()
        execute_result.scalars.return_value = scalars_mock
        mock_session.execute = AsyncMock(return_value=execute_result)

        import datetime

        with patch("team_memory.storage.database.get_session", _mock_session_cm(mock_session)):
            results = await svc.list_published(
                project="alpha",
                quality_tier="gold",
                since=datetime.datetime(2026, 1, 1),
                limit=10,
            )

        # execute was called once (no error means filters were applied OK)
        mock_session.execute.assert_awaited_once()
        assert results == []


# ---------------------------------------------------------------------------
# count_published
# ---------------------------------------------------------------------------


class TestCountPublished:
    @pytest.mark.asyncio
    async def test_returns_integer(self):
        svc = _make_service()

        mock_session = AsyncMock()
        execute_result = MagicMock()
        execute_result.scalar.return_value = 7
        mock_session.execute = AsyncMock(return_value=execute_result)

        with patch("team_memory.storage.database.get_session", _mock_session_cm(mock_session)):
            count = await svc.count_published()

        assert isinstance(count, int)
        assert count == 7

    @pytest.mark.asyncio
    async def test_returns_zero_when_none(self):
        """count_published falls back to 0 when scalar() returns None."""
        svc = _make_service()

        mock_session = AsyncMock()
        execute_result = MagicMock()
        execute_result.scalar.return_value = None
        mock_session.execute = AsyncMock(return_value=execute_result)

        with patch("team_memory.storage.database.get_session", _mock_session_cm(mock_session)):
            count = await svc.count_published(project="empty")

        assert count == 0

    @pytest.mark.asyncio
    async def test_accepts_since_filter(self):
        """count_published should accept a since argument without error."""
        import datetime

        svc = _make_service()

        mock_session = AsyncMock()
        execute_result = MagicMock()
        execute_result.scalar.return_value = 3
        mock_session.execute = AsyncMock(return_value=execute_result)

        with patch("team_memory.storage.database.get_session", _mock_session_cm(mock_session)):
            count = await svc.count_published(since=datetime.datetime(2026, 1, 1))

        assert count == 3


# ---------------------------------------------------------------------------
# soft_delete_by_id
# ---------------------------------------------------------------------------


class TestSoftDeleteById:
    @pytest.mark.asyncio
    async def test_delegates_to_soft_delete(self):
        """soft_delete_by_id must call self.soft_delete with the same ID."""
        svc = _make_service()

        exp_id = str(uuid.uuid4())
        svc.soft_delete = AsyncMock(return_value=True)

        result = await svc.soft_delete_by_id(exp_id)

        svc.soft_delete.assert_awaited_once_with(exp_id)
        assert result is True

    @pytest.mark.asyncio
    async def test_returns_false_when_not_found(self):
        svc = _make_service()

        exp_id = str(uuid.uuid4())
        svc.soft_delete = AsyncMock(return_value=False)

        result = await svc.soft_delete_by_id(exp_id)

        assert result is False


# ---------------------------------------------------------------------------
# increment_used_count
# ---------------------------------------------------------------------------


class TestIncrementUsedCount:
    @pytest.mark.asyncio
    async def test_executes_update_and_commits(self):
        """increment_used_count should execute an UPDATE and commit the session."""
        svc = _make_service()

        mock_session = AsyncMock()
        mock_session.execute = AsyncMock()
        mock_session.commit = AsyncMock()

        with patch("team_memory.storage.database.get_session", _mock_session_cm(mock_session)):
            await svc.increment_used_count(str(uuid.uuid4()))

        mock_session.execute.assert_awaited_once()
        mock_session.commit.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_raises_on_invalid_uuid(self):
        """increment_used_count should raise ValueError for an invalid UUID string."""
        svc = _make_service()

        with pytest.raises((ValueError, AttributeError)):
            await svc.increment_used_count("not-a-uuid")

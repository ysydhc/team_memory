"""Unit tests for BatchOpsService.

All tests mock get_session so no real database is required.
"""

from __future__ import annotations

import uuid
from contextlib import asynccontextmanager
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from team_memory.services.maintenance.batch_ops import BatchOpsService

DB_URL = "postgresql+asyncpg://test/test"


def _svc() -> BatchOpsService:
    return BatchOpsService(db_url=DB_URL)


def _mock_session_cm(session: AsyncMock):
    """Return an asynccontextmanager that yields *session*."""

    @asynccontextmanager
    async def _cm(db_url):
        yield session

    return _cm


def _rows(*dicts):
    """Build a mock execute result whose .mappings().all() returns list of dicts."""
    mappings_mock = MagicMock()
    mappings_mock.all.return_value = [MagicMock(**{"__iter__": lambda s: iter(d.items()), "keys": lambda: d.keys(), **d}) for d in dicts]
    # Simpler: just return plain dicts wrapped so dict(r) works
    mappings_mock.all.return_value = dicts
    result = MagicMock()
    result.mappings.return_value = mappings_mock
    return result


def _scalar_rows(*tuples):
    """Build a mock execute result whose .all() returns list of tuples."""
    result = MagicMock()
    result.all.return_value = list(tuples)
    return result


# ---------------------------------------------------------------------------
# list_experiences_missing_embedding
# ---------------------------------------------------------------------------


class TestListExperiencesMissingEmbedding:
    @pytest.mark.asyncio
    async def test_returns_rows(self):
        svc = _svc()
        row = {"id": str(uuid.uuid4()), "title": "t"}
        mock_session = AsyncMock()
        mock_session.execute = AsyncMock(return_value=_rows(row))

        with patch("team_memory.services.maintenance.batch_ops.get_session", _mock_session_cm(mock_session)):
            results = await svc.list_experiences_missing_embedding(limit=10)

        mock_session.execute.assert_awaited_once()
        assert results == [row]

    @pytest.mark.asyncio
    async def test_cursor_id_included_in_query(self):
        svc = _svc()
        mock_session = AsyncMock()
        mock_session.execute = AsyncMock(return_value=_rows())

        cursor = str(uuid.uuid4())
        with patch("team_memory.services.maintenance.batch_ops.get_session", _mock_session_cm(mock_session)):
            await svc.list_experiences_missing_embedding(limit=5, cursor_id=cursor)

        call_args = mock_session.execute.call_args
        # The SQL text should contain the cursor clause
        sql_str = str(call_args[0][0])
        assert "cursor" in sql_str
        params = call_args[0][1]
        assert params["cursor"] == cursor

    @pytest.mark.asyncio
    async def test_no_cursor_no_cursor_param(self):
        svc = _svc()
        mock_session = AsyncMock()
        mock_session.execute = AsyncMock(return_value=_rows())

        with patch("team_memory.services.maintenance.batch_ops.get_session", _mock_session_cm(mock_session)):
            await svc.list_experiences_missing_embedding()

        params = mock_session.execute.call_args[0][1]
        assert "cursor" not in params


# ---------------------------------------------------------------------------
# list_archives_missing_embedding
# ---------------------------------------------------------------------------


class TestListArchivesMissingEmbedding:
    @pytest.mark.asyncio
    async def test_returns_rows(self):
        svc = _svc()
        row = {"id": str(uuid.uuid4()), "title": "arch"}
        mock_session = AsyncMock()
        mock_session.execute = AsyncMock(return_value=_rows(row))

        with patch("team_memory.services.maintenance.batch_ops.get_session", _mock_session_cm(mock_session)):
            results = await svc.list_archives_missing_embedding()

        assert results == [row]


# ---------------------------------------------------------------------------
# list_personal_memories_missing_embedding
# ---------------------------------------------------------------------------


class TestListPersonalMemoriesMissingEmbedding:
    @pytest.mark.asyncio
    async def test_returns_rows(self):
        svc = _svc()
        row = {"id": str(uuid.uuid4()), "content": "mem"}
        mock_session = AsyncMock()
        mock_session.execute = AsyncMock(return_value=_rows(row))

        with patch("team_memory.services.maintenance.batch_ops.get_session", _mock_session_cm(mock_session)):
            results = await svc.list_personal_memories_missing_embedding()

        assert results == [row]


# ---------------------------------------------------------------------------
# update_experience_embedding
# ---------------------------------------------------------------------------


class TestUpdateExperienceEmbedding:
    @pytest.mark.asyncio
    async def test_executes_update(self):
        svc = _svc()
        mock_session = AsyncMock()

        with patch("team_memory.services.maintenance.batch_ops.get_session", _mock_session_cm(mock_session)):
            await svc.update_experience_embedding(str(uuid.uuid4()), [0.1, 0.2])

        mock_session.execute.assert_awaited_once()
        params = mock_session.execute.call_args[0][1]
        assert params["e"] == [0.1, 0.2]


# ---------------------------------------------------------------------------
# update_archive_embedding
# ---------------------------------------------------------------------------


class TestUpdateArchiveEmbedding:
    @pytest.mark.asyncio
    async def test_executes_update(self):
        svc = _svc()
        mock_session = AsyncMock()

        with patch("team_memory.services.maintenance.batch_ops.get_session", _mock_session_cm(mock_session)):
            await svc.update_archive_embedding(str(uuid.uuid4()), [0.3])

        mock_session.execute.assert_awaited_once()


# ---------------------------------------------------------------------------
# list_published_experiences
# ---------------------------------------------------------------------------


class TestListPublishedExperiences:
    @pytest.mark.asyncio
    async def test_returns_rows(self):
        svc = _svc()
        row = {"id": str(uuid.uuid4()), "exp_status": "published"}
        mock_session = AsyncMock()
        mock_session.execute = AsyncMock(return_value=_rows(row))

        with patch("team_memory.services.maintenance.batch_ops.get_session", _mock_session_cm(mock_session)):
            results = await svc.list_published_experiences()

        assert results == [row]


# ---------------------------------------------------------------------------
# list_published_archives
# ---------------------------------------------------------------------------


class TestListPublishedArchives:
    @pytest.mark.asyncio
    async def test_returns_rows(self):
        svc = _svc()
        row = {"id": str(uuid.uuid4()), "title": "arch"}
        mock_session = AsyncMock()
        mock_session.execute = AsyncMock(return_value=_rows(row))

        with patch("team_memory.services.maintenance.batch_ops.get_session", _mock_session_cm(mock_session)):
            results = await svc.list_published_archives()

        assert results == [row]


# ---------------------------------------------------------------------------
# load_experience_entity_map
# ---------------------------------------------------------------------------


class TestLoadExperienceEntityMap:
    @pytest.mark.asyncio
    async def test_groups_entities_by_experience(self):
        svc = _svc()
        exp_id = str(uuid.uuid4())
        mock_session = AsyncMock()
        mock_session.execute = AsyncMock(return_value=_scalar_rows(
            (exp_id, "EntityA"),
            (exp_id, "EntityB"),
        ))

        with patch("team_memory.services.maintenance.batch_ops.get_session", _mock_session_cm(mock_session)):
            result = await svc.load_experience_entity_map()

        assert result == {exp_id: ["EntityA", "EntityB"]}

    @pytest.mark.asyncio
    async def test_returns_empty_dict_when_no_rows(self):
        svc = _svc()
        mock_session = AsyncMock()
        mock_session.execute = AsyncMock(return_value=_scalar_rows())

        with patch("team_memory.services.maintenance.batch_ops.get_session", _mock_session_cm(mock_session)):
            result = await svc.load_experience_entity_map()

        assert result == {}


# ---------------------------------------------------------------------------
# list_experiences_for_wiki
# ---------------------------------------------------------------------------


class TestListExperiencesForWiki:
    @pytest.mark.asyncio
    async def test_no_project_filter(self):
        svc = _svc()
        mock_session = AsyncMock()
        mock_session.execute = AsyncMock(return_value=_rows())

        with patch("team_memory.services.maintenance.batch_ops.get_session", _mock_session_cm(mock_session)):
            await svc.list_experiences_for_wiki()

        params = mock_session.execute.call_args[0][1]
        assert "project" not in params

    @pytest.mark.asyncio
    async def test_with_project_filter(self):
        svc = _svc()
        mock_session = AsyncMock()
        mock_session.execute = AsyncMock(return_value=_rows())

        with patch("team_memory.services.maintenance.batch_ops.get_session", _mock_session_cm(mock_session)):
            await svc.list_experiences_for_wiki(project="alpha")

        params = mock_session.execute.call_args[0][1]
        assert params["project"] == "alpha"


# ---------------------------------------------------------------------------
# list_entities_missing_embedding
# ---------------------------------------------------------------------------


class TestListEntitiesMissingEmbedding:
    @pytest.mark.asyncio
    async def test_returns_rows(self):
        svc = _svc()
        row = {"id": str(uuid.uuid4()), "name": "Alice"}
        mock_session = AsyncMock()
        mock_session.execute = AsyncMock(return_value=_rows(row))

        with patch("team_memory.services.maintenance.batch_ops.get_session", _mock_session_cm(mock_session)):
            results = await svc.list_entities_missing_embedding(limit=5)

        assert results == [row]


# ---------------------------------------------------------------------------
# update_entity_embedding
# ---------------------------------------------------------------------------


class TestUpdateEntityEmbedding:
    @pytest.mark.asyncio
    async def test_executes_update(self):
        svc = _svc()
        mock_session = AsyncMock()

        with patch("team_memory.services.maintenance.batch_ops.get_session", _mock_session_cm(mock_session)):
            await svc.update_entity_embedding(str(uuid.uuid4()), [0.5, 0.6])

        mock_session.execute.assert_awaited_once()
        params = mock_session.execute.call_args[0][1]
        assert params["e"] == [0.5, 0.6]


# ---------------------------------------------------------------------------
# list_experiences_missing_solution
# ---------------------------------------------------------------------------


class TestListExperiencesMissingSolution:
    @pytest.mark.asyncio
    async def test_returns_rows(self):
        svc = _svc()
        row = {"id": str(uuid.uuid4()), "title": "unsolved"}
        mock_session = AsyncMock()
        mock_session.execute = AsyncMock(return_value=_rows(row))

        with patch("team_memory.services.maintenance.batch_ops.get_session", _mock_session_cm(mock_session)):
            results = await svc.list_experiences_missing_solution()

        assert results == [row]


# ---------------------------------------------------------------------------
# update_experience_solution
# ---------------------------------------------------------------------------


class TestUpdateExperienceSolution:
    @pytest.mark.asyncio
    async def test_executes_update(self):
        svc = _svc()
        mock_session = AsyncMock()

        with patch("team_memory.services.maintenance.batch_ops.get_session", _mock_session_cm(mock_session)):
            await svc.update_experience_solution(str(uuid.uuid4()), "Fixed by doing X")

        mock_session.execute.assert_awaited_once()
        params = mock_session.execute.call_args[0][1]
        assert params["s"] == "Fixed by doing X"

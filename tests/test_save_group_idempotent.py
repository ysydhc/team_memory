"""Tests for save_group idempotency by group_id (one group experience per group)."""

from __future__ import annotations

import json
import uuid
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from team_memory.server import mcp
from team_memory.storage.repository import ExperienceRepository

# ---------- Unit: ExperienceRepository.get_root_by_source_context ----------


class TestGetRootBySourceContext:
    """Unit tests for get_root_by_source_context."""

    @pytest.mark.asyncio
    async def test_empty_source_context_returns_none(self):
        """Empty or whitespace source_context -> None."""
        session = AsyncMock()
        repo = ExperienceRepository(session)
        assert await repo.get_root_by_source_context("default", "") is None
        assert await repo.get_root_by_source_context("default", "   ") is None

    @pytest.mark.asyncio
    async def test_returns_none_when_no_match(self):
        """No existing root with that source_context -> None."""
        session = AsyncMock()
        result_mock = MagicMock()
        result_mock.scalar_one_or_none.return_value = None
        session.execute = AsyncMock(return_value=result_mock)
        repo = ExperienceRepository(session)
        out = await repo.get_root_by_source_context(
            "default", "task_group:11111111-1111-1111-1111-111111111111"
        )
        assert out is None
        session.execute.assert_called_once()

    @pytest.mark.asyncio
    async def test_returns_experience_when_found(self):
        """Existing root with source_context -> returns that experience."""
        session = AsyncMock()
        exp = MagicMock()
        exp.id = uuid.uuid4()
        exp.parent_id = None
        result_mock = MagicMock()
        result_mock.scalar_one_or_none.return_value = exp
        session.execute = AsyncMock(return_value=result_mock)
        repo = ExperienceRepository(session)
        out = await repo.get_root_by_source_context(
            "default", "task_group:22222222-2222-2222-2222-222222222222"
        )
        assert out is exp
        session.execute.assert_called_once()


# ---------- Integration-style: tm_save_group with group_id (mocked DB) ----------


def _mock_session_for_save_group(
    get_root_first_return=None,
    get_with_children_return=None,
    save_group_return=None,
):
    """Build mock session: repo get_root/get_with_children, service.save_group."""
    mock_session = AsyncMock()
    mock_repo = MagicMock(spec=ExperienceRepository)
    mock_repo.get_root_by_source_context = AsyncMock(return_value=get_root_first_return)
    mock_repo.get_with_children = AsyncMock(return_value=get_with_children_return)
    mock_get_session = AsyncMock()
    mock_get_session.__aenter__ = AsyncMock(return_value=mock_session)
    mock_get_session.__aexit__ = AsyncMock(return_value=False)
    return mock_session, mock_repo, mock_get_session, save_group_return


class TestTmSaveGroupIdempotent:
    """tm_save_group with group_id: idempotent second call returns already_exists."""

    @pytest.mark.asyncio
    async def test_no_group_id_calls_save_group_without_source_context(self):
        """Without group_id, save_group is called and parent has no source_context."""
        mock_session = AsyncMock()
        mock_get_session = AsyncMock()
        mock_get_session.__aenter__ = AsyncMock(return_value=mock_session)
        mock_get_session.__aexit__ = AsyncMock(return_value=False)
        save_group_return = {"id": str(uuid.uuid4()), "title": "Parent", "children": []}
        with (
            patch("team_memory.server.get_session", return_value=mock_get_session),
            patch("team_memory.server._get_db_url", return_value="sqlite+aiosqlite:///:memory:"),
            patch("team_memory.server._get_service") as mock_svc,
            patch("team_memory.server._get_current_user", return_value="test"),
            patch("team_memory.server._resolve_project", return_value="default"),
        ):
            mock_svc.return_value.save_group = AsyncMock(return_value=save_group_return)
            tools = await mcp.get_tools()
            out = await tools["tm_save_group"].fn(
                parent_title="P",
                parent_problem="Prob",
                children=[{"title": "C", "problem": "Cp", "solution": "Cs"}],
                project="default",
            )
            data = json.loads(out)
            assert "already_exists" not in data or data.get("already_exists") is False
            assert data.get("group", {}).get("id") == save_group_return["id"]
            call_kw = mock_svc.return_value.save_group.call_args.kwargs
            assert call_kw["parent"].get("source_context") is None

    @pytest.mark.asyncio
    async def test_with_group_id_existing_returns_already_exists(self):
        """With group_id, existing root -> already_exists, no duplicate save."""
        gid = str(uuid.uuid4())
        existing_root = MagicMock()
        existing_root.id = uuid.uuid4()
        full = MagicMock()
        full.to_dict.return_value = {
            "id": str(existing_root.id),
            "title": "Existing",
            "children": [],
        }
        mock_session = AsyncMock()
        mock_repo = MagicMock(spec=ExperienceRepository)
        mock_repo.get_root_by_source_context = AsyncMock(return_value=existing_root)
        mock_repo.get_with_children = AsyncMock(return_value=full)
        mock_get_session = AsyncMock()
        mock_get_session.__aenter__ = AsyncMock(return_value=mock_session)
        mock_get_session.__aexit__ = AsyncMock(return_value=False)
        with (
            patch("team_memory.server.get_session", return_value=mock_get_session),
            patch("team_memory.server._get_db_url", return_value="sqlite+aiosqlite:///:memory:"),
            patch("team_memory.server._get_service") as mock_svc,
            patch("team_memory.server._get_current_user", return_value="test"),
            patch("team_memory.server._resolve_project", return_value="default"),
            patch(
                "team_memory.storage.repository.ExperienceRepository",
                return_value=mock_repo,
            ),
        ):
            tools = await mcp.get_tools()
            out = await tools["tm_save_group"].fn(
                parent_title="P",
                parent_problem="Prob",
                children=[{"title": "C", "problem": "Cp", "solution": "Cs"}],
                project="default",
                group_id=gid,
            )
            data = json.loads(out)
            assert data.get("already_exists") is True
            assert "该任务组已有组级经验" in data.get("message", "")
            assert data.get("group", {}).get("id") == str(existing_root.id)
            mock_svc.return_value.save_group.assert_not_called()
            mock_repo.get_root_by_source_context.assert_called_once()
            (_, ctx) = mock_repo.get_root_by_source_context.call_args[0]
            assert ctx == f"task_group:{gid}"

    @pytest.mark.asyncio
    async def test_with_group_id_no_existing_calls_save_group_with_source_context(self):
        """With group_id, when no existing root -> save_group called with source_context set."""
        gid = str(uuid.uuid4())
        mock_session = AsyncMock()
        mock_repo = MagicMock(spec=ExperienceRepository)
        mock_repo.get_root_by_source_context = AsyncMock(return_value=None)
        mock_get_session = AsyncMock()
        mock_get_session.__aenter__ = AsyncMock(return_value=mock_session)
        mock_get_session.__aexit__ = AsyncMock(return_value=False)
        save_group_return = {"id": str(uuid.uuid4()), "title": "New", "children": []}
        with (
            patch("team_memory.server.get_session", return_value=mock_get_session),
            patch("team_memory.server._get_db_url", return_value="sqlite+aiosqlite:///:memory:"),
            patch("team_memory.server._get_service") as mock_svc,
            patch("team_memory.server._get_current_user", return_value="test"),
            patch("team_memory.server._resolve_project", return_value="default"),
            patch(
                "team_memory.storage.repository.ExperienceRepository",
                return_value=mock_repo,
            ),
        ):
            mock_svc.return_value.save_group = AsyncMock(return_value=save_group_return)
            tools = await mcp.get_tools()
            out = await tools["tm_save_group"].fn(
                parent_title="P",
                parent_problem="Prob",
                children=[{"title": "C", "problem": "Cp", "solution": "Cs"}],
                project="default",
                group_id=gid,
            )
            data = json.loads(out)
            assert data.get("already_exists", False) is False
            assert data.get("group", {}).get("id") == save_group_return["id"]
            call_kw = mock_svc.return_value.save_group.call_args.kwargs
            assert call_kw["parent"].get("source_context") == f"task_group:{gid}"

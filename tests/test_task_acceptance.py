"""Tests for task acceptance_criteria and acceptance_met fields (6.6)."""

from __future__ import annotations

import json
import uuid
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from team_memory.storage.models import PersonalTask
from team_memory.server import mcp


def _make_task_mock(task_id: str, **kw):
    """Create a mock task with optional acceptance fields."""
    t = MagicMock()
    t.id = uuid.UUID(task_id)
    t.title = kw.get("title", "Test")
    t.description = kw.get("description")
    t.group_id = kw.get("group_id")
    t.status = kw.get("status", "wait")
    t.project = kw.get("project", "default")
    t.user_id = kw.get("user_id", "user1")
    t.labels = kw.get("labels") or []
    t.acceptance_criteria = kw.get("acceptance_criteria")
    t.acceptance_met = kw.get("acceptance_met")

    t.to_dict.return_value = {
        "id": task_id,
        "title": t.title,
        "status": t.status,
        "acceptance_criteria": t.acceptance_criteria,
        "acceptance_met": t.acceptance_met,
    }
    return t


def test_personal_task_to_dict_includes_acceptance_fields():
    """PersonalTask.to_dict includes acceptance_criteria and acceptance_met."""
    # Create minimal task via SQLAlchemy (requires DB) - use model default
    # Instead test the model fields exist
    task = PersonalTask(
        title="Test",
        description=None,
        user_id="user1",
        project="default",
        acceptance_criteria="ruff pass, pytest pass",
        acceptance_met=True,
    )
    d = task.to_dict()
    assert "acceptance_criteria" in d
    assert d["acceptance_criteria"] == "ruff pass, pytest pass"
    assert "acceptance_met" in d
    assert d["acceptance_met"] is True


def test_personal_task_to_dict_acceptance_none():
    """PersonalTask.to_dict handles acceptance_met=False and None."""
    task = PersonalTask(
        title="Test",
        description=None,
        user_id="user1",
        project="default",
        acceptance_criteria=None,
        acceptance_met=False,
    )
    d = task.to_dict()
    assert d["acceptance_criteria"] is None
    assert d["acceptance_met"] is False


class TestTmTaskUpdateAcceptanceFields:
    """tm_task update with acceptance_criteria and acceptance_met."""

    @pytest.mark.asyncio
    async def test_update_passes_acceptance_to_repo(self):
        """tm_task update with acceptance_criteria/acceptance_met passes to update_task."""
        task_id = str(uuid.uuid4())
        task = _make_task_mock(task_id, status="in_progress")
        updated = _make_task_mock(
            task_id,
            acceptance_criteria="ruff pass",
            acceptance_met=True,
            status="completed",
        )

        mock_repo = MagicMock()
        mock_repo.get_task = AsyncMock(return_value=task)
        mock_repo.update_task = AsyncMock(return_value=updated)
        mock_repo.group_all_completed_or_cancelled = AsyncMock(return_value=False)
        mock_repo.check_wip = AsyncMock(return_value=(True, 0))

        mock_session = MagicMock()
        mock_session.commit = AsyncMock()

        with (
            patch("team_memory.server.get_session") as mock_get_session,
            patch("team_memory.server._get_db_url", return_value="sqlite+aiosqlite:///:memory:"),
            patch("team_memory.server._resolve_project", return_value="default"),
            patch("team_memory.server._get_current_user", return_value="user1"),
            patch(
                "team_memory.storage.repository.TaskRepository",
                return_value=mock_repo,
            ),
            patch("team_memory.server._get_service") as mock_svc,
            patch("team_memory.server._create_reflection", AsyncMock(return_value=None)),
            patch(
                "team_memory.services.llm_parser.compute_quality_score",
                return_value=0.8,
            ),
        ):
            mock_svc.return_value.save = AsyncMock(return_value=MagicMock(id=uuid.uuid4()))
            mock_get_session.return_value.__aenter__ = AsyncMock(return_value=mock_session)
            mock_get_session.return_value.__aexit__ = AsyncMock(return_value=False)

            tools = await mcp.get_tools()
            fn = tools["tm_task"].fn
            result = await fn(
                action="update",
                task_id=task_id,
                status="completed",
                summary="done",
                acceptance_criteria="ruff pass",
                acceptance_met=True,
            )
            out = json.loads(result)
            assert out.get("error") is not True
            mock_repo.update_task.assert_called_once()
            call_kwargs = mock_repo.update_task.call_args[1]
            assert call_kwargs.get("acceptance_criteria") == "ruff pass"
            assert call_kwargs.get("acceptance_met") is True

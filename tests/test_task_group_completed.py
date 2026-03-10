"""Tests for task group completion and list group_progress in tm_task."""

from __future__ import annotations

import json
import uuid
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from team_memory.server import mcp
from team_memory.storage.repository import TaskRepository

# ---------- Unit: TaskRepository.group_all_completed_or_cancelled ----------


class TestGroupAllCompletedOrCancelled:
    """Unit tests for TaskRepository.group_all_completed_or_cancelled."""

    @pytest.mark.asyncio
    async def test_empty_group_returns_false(self):
        """No tasks in group -> False."""
        session = AsyncMock()
        repo = TaskRepository(session)
        repo.list_tasks = AsyncMock(return_value=[])
        gid = uuid.uuid4()
        out = await repo.group_all_completed_or_cancelled("default", gid)
        assert out is False
        repo.list_tasks.assert_called_once_with(project="default", group_id=gid)

    @pytest.mark.asyncio
    async def test_all_completed_returns_true(self):
        """All tasks completed -> True."""
        session = AsyncMock()
        repo = TaskRepository(session)
        t = MagicMock()
        t.status = "completed"
        repo.list_tasks = AsyncMock(return_value=[t, t])
        gid = uuid.uuid4()
        out = await repo.group_all_completed_or_cancelled("default", gid)
        assert out is True

    @pytest.mark.asyncio
    async def test_all_cancelled_returns_true(self):
        """All tasks cancelled -> True."""
        session = AsyncMock()
        repo = TaskRepository(session)
        t = MagicMock()
        t.status = "cancelled"
        repo.list_tasks = AsyncMock(return_value=[t])
        gid = uuid.uuid4()
        out = await repo.group_all_completed_or_cancelled("default", gid)
        assert out is True

    @pytest.mark.asyncio
    async def test_mixed_completed_cancelled_returns_true(self):
        """Completed + cancelled -> True."""
        session = AsyncMock()
        repo = TaskRepository(session)
        a, b = MagicMock(), MagicMock()
        a.status, b.status = "completed", "cancelled"
        repo.list_tasks = AsyncMock(return_value=[a, b])
        gid = uuid.uuid4()
        out = await repo.group_all_completed_or_cancelled("default", gid)
        assert out is True

    @pytest.mark.asyncio
    async def test_any_wait_returns_false(self):
        """Any task wait -> False."""
        session = AsyncMock()
        repo = TaskRepository(session)
        a, b = MagicMock(), MagicMock()
        a.status, b.status = "completed", "wait"
        repo.list_tasks = AsyncMock(return_value=[a, b])
        gid = uuid.uuid4()
        out = await repo.group_all_completed_or_cancelled("default", gid)
        assert out is False

    @pytest.mark.asyncio
    async def test_any_in_progress_returns_false(self):
        """Any task in_progress -> False."""
        session = AsyncMock()
        repo = TaskRepository(session)
        t = MagicMock()
        t.status = "in_progress"
        repo.list_tasks = AsyncMock(return_value=[t])
        gid = uuid.uuid4()
        out = await repo.group_all_completed_or_cancelled("default", gid)
        assert out is False


# ---------- Integration-style: tm_task update response (mocked DB) ----------


def _make_task_mock(task_id: str, group_id: str | None, status: str = "wait"):
    t = MagicMock()
    t.id = uuid.UUID(task_id)
    t.group_id = uuid.UUID(group_id) if group_id else None
    t.project = "default"
    t.title = "Task"
    t.labels = []
    t.user_id = "alice"
    t.status = status
    t.to_dict.return_value = {
        "id": task_id,
        "group_id": group_id,
        "status": status,
        "title": "Task",
    }
    return t


class TestTmTaskUpdateGroupCompleted:
    """tm_task update: response includes group_completed when group is done."""

    @pytest.mark.asyncio
    async def test_update_completed_no_group_id_no_group_completed(self):
        """Task without group_id -> response has no group_completed true."""
        task_id = str(uuid.uuid4())
        task = _make_task_mock(task_id, group_id=None)
        task.status = "wait"
        updated = _make_task_mock(task_id, group_id=None, status="completed")

        mock_repo = MagicMock()
        mock_repo.get_task = AsyncMock(return_value=task)
        mock_repo.update_task = AsyncMock(return_value=updated)
        mock_repo.group_all_completed_or_cancelled = AsyncMock(return_value=False)

        with (
            patch("team_memory.server.get_session") as mock_get_session,
            patch("team_memory.server._get_db_url", return_value="sqlite+aiosqlite:///:memory:"),
            patch("team_memory.server._resolve_project", return_value="default"),
            patch("team_memory.server._get_current_user", return_value="alice"),
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
            mock_session = MagicMock()
            mock_session.commit = AsyncMock()
            mock_get_session.return_value.__aenter__ = AsyncMock(return_value=mock_session)
            mock_get_session.return_value.__aexit__ = AsyncMock(return_value=False)

            tools = await mcp.get_tools()
            fn = tools["tm_task"].fn
            result = await fn(
                action="update",
                task_id=task_id,
                status="completed",
                summary="Done",
            )

        data = json.loads(result)
        assert data.get("group_completed") is not True
        assert "group_id" not in data or data.get("group_completed") is not True

    @pytest.mark.asyncio
    async def test_update_completed_group_all_done_returns_group_completed(self):
        """Task with group_id and group all completed -> response has group_completed."""
        task_id = str(uuid.uuid4())
        gid = str(uuid.uuid4())
        task = _make_task_mock(task_id, group_id=gid)
        task.status = "wait"
        updated = _make_task_mock(task_id, group_id=gid, status="completed")

        mock_repo = MagicMock()
        mock_repo.get_task = AsyncMock(return_value=task)
        mock_repo.update_task = AsyncMock(return_value=updated)
        mock_repo.group_all_completed_or_cancelled = AsyncMock(return_value=True)

        with (
            patch("team_memory.server.get_session") as mock_get_session,
            patch("team_memory.server._get_db_url", return_value="sqlite+aiosqlite:///:memory:"),
            patch("team_memory.server._resolve_project", return_value="default"),
            patch("team_memory.server._get_current_user", return_value="alice"),
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
            mock_session = MagicMock()
            mock_session.commit = AsyncMock()
            mock_get_session.return_value.__aenter__ = AsyncMock(return_value=mock_session)
            mock_get_session.return_value.__aexit__ = AsyncMock(return_value=False)

            tools = await mcp.get_tools()
            fn = tools["tm_task"].fn
            result = await fn(
                action="update",
                task_id=task_id,
                status="completed",
                summary="Done",
            )

        data = json.loads(result)
        assert data.get("group_completed") is True
        assert data.get("group_id") == gid
        assert "group_completed_hint" in data
        hint = data["group_completed_hint"]
        assert "tm_save_group" in hint or "组级复盘" in hint

    @pytest.mark.asyncio
    async def test_update_completed_group_not_all_done_no_group_completed(self):
        """Task with group_id but group not all done -> no group_completed true."""
        task_id = str(uuid.uuid4())
        gid = str(uuid.uuid4())
        task = _make_task_mock(task_id, group_id=gid)
        updated = _make_task_mock(task_id, group_id=gid, status="completed")

        mock_repo = MagicMock()
        mock_repo.get_task = AsyncMock(return_value=task)
        mock_repo.update_task = AsyncMock(return_value=updated)
        mock_repo.group_all_completed_or_cancelled = AsyncMock(return_value=False)

        with (
            patch("team_memory.server.get_session") as mock_get_session,
            patch("team_memory.server._get_db_url", return_value="sqlite+aiosqlite:///:memory:"),
            patch("team_memory.server._resolve_project", return_value="default"),
            patch("team_memory.server._get_current_user", return_value="alice"),
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
            mock_session = MagicMock()
            mock_session.commit = AsyncMock()
            mock_get_session.return_value.__aenter__ = AsyncMock(return_value=mock_session)
            mock_get_session.return_value.__aexit__ = AsyncMock(return_value=False)

            tools = await mcp.get_tools()
            fn = tools["tm_task"].fn
            result = await fn(
                action="update",
                task_id=task_id,
                status="completed",
                summary="Done",
            )

        data = json.loads(result)
        assert data.get("group_completed") is not True
        assert "sediment_experience_id" in data

    @pytest.mark.asyncio
    async def test_update_completed_with_changed_files_writes_architecture_bindings(
        self,
    ):
        """tm_task update completed + changed_files -> replace_architecture_bindings called."""
        task_id = str(uuid.uuid4())
        exp_id = uuid.uuid4()
        task = _make_task_mock(task_id, group_id=None)
        task.status = "wait"
        task.project = "default"
        updated = _make_task_mock(task_id, group_id=None, status="completed")

        mock_repo = MagicMock()
        mock_repo.get_task = AsyncMock(return_value=task)
        mock_repo.update_task = AsyncMock(return_value=updated)
        mock_repo.group_all_completed_or_cancelled = AsyncMock(return_value=False)

        mock_exp_repo = MagicMock()
        mock_exp_repo.replace_architecture_bindings = AsyncMock()

        with (
            patch("team_memory.server.get_session") as mock_get_session,
            patch("team_memory.server._get_db_url", return_value="sqlite+aiosqlite:///:memory:"),
            patch("team_memory.server._resolve_project", return_value="default"),
            patch("team_memory.server._get_current_user", return_value="alice"),
            patch(
                "team_memory.storage.repository.TaskRepository",
                return_value=mock_repo,
            ),
            patch(
                "team_memory.storage.repository.ExperienceRepository",
                return_value=mock_exp_repo,
            ),
            patch("team_memory.server._get_service") as mock_svc,
            patch("team_memory.server._create_reflection", AsyncMock(return_value=None)),
            patch(
                "team_memory.services.llm_parser.compute_quality_score",
                return_value=0.8,
            ),
        ):
            mock_svc.return_value.save = AsyncMock(
                return_value=MagicMock(id=exp_id)
            )
            mock_session = MagicMock()
            mock_session.commit = AsyncMock()
            mock_get_session.return_value.__aenter__ = AsyncMock(return_value=mock_session)
            mock_get_session.return_value.__aexit__ = AsyncMock(return_value=False)

            tools = await mcp.get_tools()
            fn = tools["tm_task"].fn
            result = await fn(
                action="update",
                task_id=task_id,
                status="completed",
                summary="Done",
                changed_files=["src/foo.py", "./bar/baz.ts"],
            )

        data = json.loads(result)
        assert data.get("error") is not True
        mock_exp_repo.replace_architecture_bindings.assert_called_once()
        call_args = mock_exp_repo.replace_architecture_bindings.call_args
        assert call_args[0][0] == exp_id
        assert call_args[0][1] == ["src/foo.py", "bar/baz.ts"]  # normalized
        assert call_args[1]["project"] == "default"

    @pytest.mark.asyncio
    async def test_update_completed_empty_changed_files_git_auto_parse(
        self,
    ):
        """tm_task update completed + empty changed_files + project in paths -> Git auto-parse."""
        task_id = str(uuid.uuid4())
        exp_id = uuid.uuid4()
        task = _make_task_mock(task_id, group_id=None)
        task.status = "wait"
        task.project = "team_doc"
        updated = _make_task_mock(task_id, group_id=None, status="completed")

        mock_repo = MagicMock()
        mock_repo.get_task = AsyncMock(return_value=task)
        mock_repo.update_task = AsyncMock(return_value=updated)
        mock_repo.group_all_completed_or_cancelled = AsyncMock(return_value=False)

        mock_exp_repo = MagicMock()
        mock_exp_repo.replace_architecture_bindings = AsyncMock()

        with (
            patch("team_memory.server.get_session") as mock_get_session,
            patch("team_memory.server._get_db_url", return_value="sqlite+aiosqlite:///:memory:"),
            patch("team_memory.server._resolve_project", return_value="default"),
            patch("team_memory.server._get_current_user", return_value="alice"),
            patch(
                "team_memory.storage.repository.TaskRepository",
                return_value=mock_repo,
            ),
            patch(
                "team_memory.storage.repository.ExperienceRepository",
                return_value=mock_exp_repo,
            ),
            patch("team_memory.server._get_service") as mock_svc,
            patch("team_memory.server._create_reflection", AsyncMock(return_value=None)),
            patch(
                "team_memory.services.llm_parser.compute_quality_score",
                return_value=0.8,
            ),
            patch(
                "team_memory.web.routes.analytics._get_scan_config",
                return_value={"project_paths": {"team_doc": "/path/to/team_doc"}},
            ),
            patch(
                "team_memory.utils.git_utils.get_changed_files",
                return_value=(["src/a.py", "tests/b.py"], None),
            ),
        ):
            mock_svc.return_value.save = AsyncMock(
                return_value=MagicMock(id=exp_id)
            )
            mock_session = MagicMock()
            mock_session.commit = AsyncMock()
            mock_get_session.return_value.__aenter__ = AsyncMock(return_value=mock_session)
            mock_get_session.return_value.__aexit__ = AsyncMock(return_value=False)

            tools = await mcp.get_tools()
            fn = tools["tm_task"].fn
            result = await fn(
                action="update",
                task_id=task_id,
                status="completed",
                summary="Done",
                changed_files=None,
            )

        data = json.loads(result)
        assert data.get("error") is not True
        mock_exp_repo.replace_architecture_bindings.assert_called_once()
        call_args = mock_exp_repo.replace_architecture_bindings.call_args
        assert call_args[0][0] == exp_id
        assert call_args[0][1] == ["src/a.py", "tests/b.py"]
        assert call_args[1]["project"] == "team_doc"


# ---------- tm_task list: group_progress when group_id given ----------


class TestTmTaskListGroupProgress:
    """tm_task action=list with group_id returns group_progress."""

    @pytest.mark.asyncio
    async def test_list_without_group_id_no_group_progress(self):
        """list without group_id -> response has no group_progress."""
        with (
            patch("team_memory.server.get_session") as mock_get_session,
            patch(
                "team_memory.server._get_db_url",
                return_value="sqlite+aiosqlite:///:memory:",
            ),
            patch("team_memory.server._resolve_project", return_value="default"),
            patch("team_memory.server._get_current_user", return_value="alice"),
            patch(
                "team_memory.storage.repository.TaskRepository",
                return_value=MagicMock(list_tasks=AsyncMock(return_value=[])),
            ),
        ):
            mock_session = MagicMock()
            mock_session.commit = AsyncMock()
            mock_get_session.return_value.__aenter__ = AsyncMock(
                return_value=mock_session
            )
            mock_get_session.return_value.__aexit__ = AsyncMock(return_value=False)
            tools = await mcp.get_tools()
            fn = tools["tm_task"].fn
            result = await fn(action="list")
        data = json.loads(result)
        assert "group_progress" not in data
        assert data["total"] == 0

    @pytest.mark.asyncio
    async def test_list_with_group_id_returns_group_progress(self):
        """list with group_id -> response has group_progress total and completed."""
        gid = str(uuid.uuid4())
        t1, t2 = MagicMock(), MagicMock()
        t1.status, t2.status = "completed", "wait"
        t1.to_dict.return_value = {"id": str(uuid.uuid4()), "status": "completed"}
        t2.to_dict.return_value = {"id": str(uuid.uuid4()), "status": "wait"}
        mock_repo = MagicMock()
        mock_repo.list_tasks = AsyncMock(return_value=[t1, t2])
        with (
            patch("team_memory.server.get_session") as mock_get_session,
            patch(
                "team_memory.server._get_db_url",
                return_value="sqlite+aiosqlite:///:memory:",
            ),
            patch("team_memory.server._resolve_project", return_value="default"),
            patch("team_memory.server._get_current_user", return_value="alice"),
            patch(
                "team_memory.storage.repository.TaskRepository",
                return_value=mock_repo,
            ),
        ):
            mock_session = MagicMock()
            mock_get_session.return_value.__aenter__ = AsyncMock(
                return_value=mock_session
            )
            mock_get_session.return_value.__aexit__ = AsyncMock(return_value=False)
            tools = await mcp.get_tools()
            fn = tools["tm_task"].fn
            result = await fn(action="list", group_id=gid)
        data = json.loads(result)
        assert "group_progress" in data
        assert data["group_progress"]["total"] == 2
        assert data["group_progress"]["completed"] == 1

    @pytest.mark.asyncio
    async def test_list_with_group_id_empty_tasks_returns_zero_progress(self):
        """list with group_id and no tasks -> group_progress 0/0."""
        gid = str(uuid.uuid4())
        mock_repo = MagicMock()
        mock_repo.list_tasks = AsyncMock(return_value=[])
        with (
            patch("team_memory.server.get_session") as mock_get_session,
            patch(
                "team_memory.server._get_db_url",
                return_value="sqlite+aiosqlite:///:memory:",
            ),
            patch("team_memory.server._resolve_project", return_value="default"),
            patch("team_memory.server._get_current_user", return_value="alice"),
            patch(
                "team_memory.storage.repository.TaskRepository",
                return_value=mock_repo,
            ),
        ):
            mock_get_session.return_value.__aenter__ = AsyncMock(
                return_value=MagicMock()
            )
            mock_get_session.return_value.__aexit__ = AsyncMock(return_value=False)
            tools = await mcp.get_tools()
            fn = tools["tm_task"].fn
            result = await fn(action="list", group_id=gid)
        data = json.loads(result)
        assert data["group_progress"] == {"total": 0, "completed": 0}

"""Tests for janitor API routes."""

from __future__ import annotations

import pytest
from unittest.mock import AsyncMock, MagicMock

from fastapi import HTTPException
from team_memory.auth.provider import User
from team_memory.web.routes.janitor import (
    get_janitor_status,
    list_outdated_experiences,
    run_janitor,
)


class TestJanitorRoutes:
    """Test janitor API routes."""

    @pytest.fixture
    def admin_user(self):
        """Create an admin user for testing."""
        return User(name="admin", role="admin")

    @pytest.fixture
    def mock_context(self):
        """Mock AppContext for testing."""
        context = MagicMock()
        context.janitor = MagicMock()
        context.janitor_scheduler = MagicMock()
        return context

    @pytest.mark.asyncio
    async def test_run_janitor_success(self, admin_user, mock_context, monkeypatch):
        """Test successful janitor run."""
        # Mock get_context
        monkeypatch.setattr("team_memory.web.routes.janitor.get_context", lambda: mock_context)
        monkeypatch.setattr("team_memory.web.routes.janitor._resolve_project", lambda x: x)

        # Mock janitor run_all
        mock_result = {
            "score_decay": {"processed": 10},
            "outdated_sweep": {"found": 5},
        }
        mock_context.janitor.run_all = AsyncMock(return_value=mock_result)

        result = await run_janitor(project="test_project", user=admin_user)

        assert result["message"] == "Janitor cleanup completed successfully"
        assert result["project"] == "test_project"
        assert result["results"] == mock_result
        mock_context.janitor.run_all.assert_called_once_with("test_project")

    @pytest.mark.asyncio
    async def test_run_janitor_no_service(self, admin_user, monkeypatch):
        """Test janitor run when service is not available."""
        mock_context = MagicMock()
        mock_context.janitor = None
        monkeypatch.setattr("team_memory.web.routes.janitor.get_context", lambda: mock_context)

        with pytest.raises(HTTPException) as exc_info:
            await run_janitor(user=admin_user)

        assert exc_info.value.status_code == 503
        assert "not available" in exc_info.value.detail

    @pytest.mark.asyncio
    async def test_run_janitor_exception(self, admin_user, mock_context, monkeypatch):
        """Test janitor run when an exception occurs."""
        monkeypatch.setattr("team_memory.web.routes.janitor.get_context", lambda: mock_context)
        monkeypatch.setattr("team_memory.web.routes.janitor._resolve_project", lambda x: x)

        # Mock janitor to raise exception
        mock_context.janitor.run_all = AsyncMock(side_effect=Exception("Test error"))

        with pytest.raises(HTTPException) as exc_info:
            await run_janitor(user=admin_user)

        assert exc_info.value.status_code == 500
        assert "Test error" in exc_info.value.detail

    @pytest.mark.asyncio
    async def test_get_janitor_status_full(self, admin_user, mock_context, monkeypatch):
        """Test janitor status with all services available."""
        monkeypatch.setattr("team_memory.web.routes.janitor.get_context", lambda: mock_context)

        # Mock scheduler
        mock_context.janitor_scheduler.is_running.return_value = True
        mock_context.janitor_scheduler._config = MagicMock()
        mock_context.janitor_scheduler._config.janitor_interval_hours = 6
        mock_context.janitor_scheduler._config.janitor_enabled = True

        # Mock janitor config
        mock_context.janitor._config = MagicMock()
        mock_context.janitor._config.protection_period_days = 10
        mock_context.janitor._config.auto_soft_delete_outdated = False

        result = await get_janitor_status(user=admin_user)

        assert result["janitor_available"] is True
        assert result["scheduler_available"] is True
        assert result["scheduler_running"] is True
        assert result["config"]["interval_hours"] == 6
        assert result["janitor_config"]["protection_period_days"] == 10

    @pytest.mark.asyncio
    async def test_get_janitor_status_minimal(self, admin_user, monkeypatch):
        """Test janitor status with minimal services."""
        mock_context = MagicMock()
        mock_context.janitor = None
        mock_context.janitor_scheduler = None
        monkeypatch.setattr("team_memory.web.routes.janitor.get_context", lambda: mock_context)

        result = await get_janitor_status(user=admin_user)

        assert result["janitor_available"] is False
        assert result["scheduler_available"] is False
        assert result["scheduler_running"] is False

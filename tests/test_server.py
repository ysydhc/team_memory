"""Tests for MCP Server tool registration and basic functionality.

Tests that tools are properly registered and callable.
Full integration tests (with real DB) are in test_integration.py.
"""

from __future__ import annotations

import json
import uuid
from unittest.mock import AsyncMock, patch, MagicMock

import pytest

from team_doc.server import mcp


class TestMCPToolRegistration:
    """Verify that all expected tools are registered."""

    @pytest.mark.asyncio
    async def test_search_experiences_registered(self):
        tools = await mcp.get_tools()
        assert "search_experiences" in tools

    @pytest.mark.asyncio
    async def test_save_experience_registered(self):
        tools = await mcp.get_tools()
        assert "save_experience" in tools

    @pytest.mark.asyncio
    async def test_feedback_experience_registered(self):
        tools = await mcp.get_tools()
        assert "feedback_experience" in tools

    @pytest.mark.asyncio
    async def test_update_experience_registered(self):
        tools = await mcp.get_tools()
        assert "update_experience" in tools

    @pytest.mark.asyncio
    async def test_all_tools_have_descriptions(self):
        tools = await mcp.get_tools()
        for name, tool in tools.items():
            assert tool.description, f"Tool {name} has no description"


class TestMCPResourceRegistration:
    """Verify that all expected resources are registered."""

    @pytest.mark.asyncio
    async def test_recent_resource_registered(self):
        resources = await mcp.get_resources()
        assert "experiences://recent" in resources

    @pytest.mark.asyncio
    async def test_stats_resource_registered(self):
        resources = await mcp.get_resources()
        assert "experiences://stats" in resources


class TestSearchExperiencesTool:
    """Test search_experiences tool function."""

    @pytest.mark.asyncio
    async def test_search_returns_json(self):
        """search_experiences should return valid JSON."""
        mock_results = [
            {
                "id": str(uuid.uuid4()),
                "title": "Test experience",
                "description": "Test problem",
                "solution": "Test solution",
                "tags": ["python"],
                "similarity": 0.85,
            }
        ]

        with patch("team_doc.server._get_service") as mock_get_service, \
             patch("team_doc.server.get_session") as mock_get_session:

            mock_service = MagicMock()
            mock_service.search = AsyncMock(return_value=mock_results)
            mock_get_service.return_value = mock_service

            mock_session = AsyncMock()
            mock_get_session.return_value.__aenter__ = AsyncMock(
                return_value=mock_session
            )
            mock_get_session.return_value.__aexit__ = AsyncMock(return_value=False)

            # Get the original function from the tool
            tools = await mcp.get_tools()
            search_fn = tools["search_experiences"].fn
            result = await search_fn(query="test problem")

        data = json.loads(result)
        assert "results" in data
        assert len(data["results"]) == 1
        assert data["results"][0]["title"] == "Test experience"

    @pytest.mark.asyncio
    async def test_search_empty_results(self):
        """search_experiences should handle empty results."""
        with patch("team_doc.server._get_service") as mock_get_service, \
             patch("team_doc.server.get_session") as mock_get_session:

            mock_service = MagicMock()
            mock_service.search = AsyncMock(return_value=[])
            mock_get_service.return_value = mock_service

            mock_session = AsyncMock()
            mock_get_session.return_value.__aenter__ = AsyncMock(
                return_value=mock_session
            )
            mock_get_session.return_value.__aexit__ = AsyncMock(return_value=False)

            tools = await mcp.get_tools()
            search_fn = tools["search_experiences"].fn
            result = await search_fn(query="nonexistent")

        data = json.loads(result)
        assert data["results"] == []
        assert "No matching" in data["message"]


class TestSaveExperienceTool:
    """Test save_experience tool function."""

    @pytest.mark.asyncio
    async def test_save_returns_json(self):
        """save_experience should return valid JSON with experience ID."""
        exp_id = str(uuid.uuid4())
        mock_result = {
            "id": exp_id,
            "title": "Fix Docker issue",
            "created_at": "2026-01-01T00:00:00+00:00",
        }

        with patch("team_doc.server._get_service") as mock_get_service, \
             patch("team_doc.server.get_session") as mock_get_session, \
             patch("team_doc.server._get_current_user", return_value="alice"):

            mock_service = MagicMock()
            mock_service.save = AsyncMock(return_value=mock_result)
            mock_get_service.return_value = mock_service

            mock_session = AsyncMock()
            mock_get_session.return_value.__aenter__ = AsyncMock(
                return_value=mock_session
            )
            mock_get_session.return_value.__aexit__ = AsyncMock(return_value=False)

            tools = await mcp.get_tools()
            save_fn = tools["save_experience"].fn
            result = await save_fn(
                title="Fix Docker issue",
                problem="Container won't start",
                solution="Check port conflicts",
                tags=["docker"],
            )

        data = json.loads(result)
        assert "experience" in data
        assert data["experience"]["id"] == exp_id
        assert "saved successfully" in data["message"].lower()


class TestFeedbackExperienceTool:
    """Test feedback_experience tool function."""

    @pytest.mark.asyncio
    async def test_feedback_success(self):
        """feedback_experience should return success message."""
        with patch("team_doc.server._get_service") as mock_get_service, \
             patch("team_doc.server.get_session") as mock_get_session, \
             patch("team_doc.server._get_current_user", return_value="alice"):

            mock_service = MagicMock()
            mock_service.feedback = AsyncMock(return_value=True)
            mock_get_service.return_value = mock_service

            mock_session = AsyncMock()
            mock_get_session.return_value.__aenter__ = AsyncMock(
                return_value=mock_session
            )
            mock_get_session.return_value.__aexit__ = AsyncMock(return_value=False)

            tools = await mcp.get_tools()
            feedback_fn = tools["feedback_experience"].fn
            result = await feedback_fn(
                experience_id=str(uuid.uuid4()),
                helpful=True,
                comment="Worked great!",
            )

        data = json.loads(result)
        assert "recorded" in data["message"].lower()

    @pytest.mark.asyncio
    async def test_feedback_not_found(self):
        """feedback_experience should handle non-existent experience."""
        with patch("team_doc.server._get_service") as mock_get_service, \
             patch("team_doc.server.get_session") as mock_get_session, \
             patch("team_doc.server._get_current_user", return_value="alice"):

            mock_service = MagicMock()
            mock_service.feedback = AsyncMock(return_value=False)
            mock_get_service.return_value = mock_service

            mock_session = AsyncMock()
            mock_get_session.return_value.__aenter__ = AsyncMock(
                return_value=mock_session
            )
            mock_get_session.return_value.__aexit__ = AsyncMock(return_value=False)

            tools = await mcp.get_tools()
            feedback_fn = tools["feedback_experience"].fn
            result = await feedback_fn(
                experience_id=str(uuid.uuid4()),
                helpful=True,
            )

        data = json.loads(result)
        assert "not found" in data["message"].lower()

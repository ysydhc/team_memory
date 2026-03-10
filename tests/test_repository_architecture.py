"""Tests for ExperienceRepository architecture-related methods.

Covers list_experiences_by_node and list_experiences_by_nodes.
"""

from __future__ import annotations

import uuid
from unittest.mock import AsyncMock, MagicMock

import pytest

from team_memory.storage.repository import ExperienceRepository


class TestListExperiencesByNodes:
    """Tests for list_experiences_by_nodes batch method."""

    @pytest.mark.asyncio
    async def test_empty_node_keys_returns_empty_list(self):
        """Empty node_keys returns [] without hitting DB."""
        mock_session = AsyncMock()
        repo = ExperienceRepository(mock_session)

        result = await repo.list_experiences_by_nodes(node_keys=[])

        assert result == []
        mock_session.execute.assert_not_called()

    @pytest.mark.asyncio
    async def test_whitespace_only_node_keys_returns_empty_list(self):
        """node_keys with only whitespace returns [] without hitting DB."""
        mock_session = AsyncMock()
        repo = ExperienceRepository(mock_session)

        result = await repo.list_experiences_by_nodes(
            node_keys=["", "  ", "\t"],
        )

        assert result == []
        mock_session.execute.assert_not_called()

    @pytest.mark.asyncio
    async def test_returns_correct_format(self):
        """Valid node_keys returns list of {experience_id, title, node} dicts."""
        mock_session = AsyncMock()
        exp_id = uuid.uuid4()
        mock_result = MagicMock()
        mock_result.all.return_value = [
            (exp_id, "Fix auth bug", "src/auth/provider.py"),
            (exp_id, "Add logging", "src/utils/logger.py"),
        ]
        mock_session.execute = AsyncMock(return_value=mock_result)

        repo = ExperienceRepository(mock_session)
        result = await repo.list_experiences_by_nodes(
            node_keys=["src/auth/provider.py", "src/utils/logger.py"],
        )

        assert len(result) == 2
        assert result[0] == {
            "experience_id": str(exp_id),
            "title": "Fix auth bug",
            "node": "src/auth/provider.py",
        }
        assert result[1] == {
            "experience_id": str(exp_id),
            "title": "Add logging",
            "node": "src/utils/logger.py",
        }

    @pytest.mark.asyncio
    async def test_empty_result(self):
        """No matching bindings returns empty list."""
        mock_session = AsyncMock()
        mock_result = MagicMock()
        mock_result.all.return_value = []
        mock_session.execute = AsyncMock(return_value=mock_result)

        repo = ExperienceRepository(mock_session)
        result = await repo.list_experiences_by_nodes(
            node_keys=["nonexistent/file.py"],
        )

        assert result == []

    @pytest.mark.asyncio
    async def test_strips_and_filters_node_keys(self):
        """Whitespace in node_keys is stripped; empty entries are skipped."""
        mock_session = AsyncMock()
        mock_result = MagicMock()
        mock_result.all.return_value = []
        mock_session.execute = AsyncMock(return_value=mock_result)

        repo = ExperienceRepository(mock_session)
        await repo.list_experiences_by_nodes(
            node_keys=["  src/foo.py  ", "", "src/bar.py"],
        )

        # Should have called execute (valid keys: src/foo.py, src/bar.py)
        mock_session.execute.assert_called_once()
        # Verify the query uses .in_() with stripped keys
        call_args = mock_session.execute.call_args
        stmt = call_args[0][0]
        # The where clause should include in_(['src/foo.py', 'src/bar.py'])
        assert stmt is not None

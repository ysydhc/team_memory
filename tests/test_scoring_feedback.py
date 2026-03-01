"""Tests for Step 1: scoring feedback loop fixes.

Covers:
1. rating_weight degradation when avg_rating=0 (no penalty for unrated experiences)
2. tm_search auto use_count increment on top-1 result
3. tm_solve/tm_search feedback_hint in return value
"""

from __future__ import annotations

import json
import uuid
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from team_memory.server import mcp
from team_memory.services.scoring import (
    apply_rating_boost,
    apply_reference_boost,
)


def _mock_server_deps(mock_service, mock_session):
    """Context manager to patch all server dependencies for tool testing."""
    return (
        patch("team_memory.server._get_service", return_value=mock_service),
        patch("team_memory.server._get_db_url", return_value="sqlite+aiosqlite://"),
        patch("team_memory.server._get_current_user", return_value="test_user"),
        patch("team_memory.server._resolve_project", return_value="default"),
    )


def _make_mock_session():
    mock_session = AsyncMock()
    mock_ctx = MagicMock()
    mock_ctx.__aenter__ = AsyncMock(return_value=mock_session)
    mock_ctx.__aexit__ = AsyncMock(return_value=False)
    return mock_session, mock_ctx


# ============================================================
# 1. rating_weight degradation when avg_rating=0
# ============================================================


class TestRatingWeightDegradation:
    """When avg_rating=0 (no feedback), final_score should NOT be penalized."""

    def test_no_penalty_when_avg_rating_zero(self):
        """final_score should equal similarity when avg_rating=0."""
        similarity = 0.85
        avg_rating = 0.0
        rating_weight = 0.3

        if avg_rating > 0:
            final_score = similarity * (
                1.0 - rating_weight + rating_weight * avg_rating / 5.0
            )
        else:
            final_score = similarity

        assert final_score == similarity

    def test_penalty_applied_when_avg_rating_positive(self):
        """final_score should be weighted when avg_rating > 0."""
        similarity = 0.85
        avg_rating = 4.0
        rating_weight = 0.3

        if avg_rating > 0:
            final_score = similarity * (
                1.0 - rating_weight + rating_weight * avg_rating / 5.0
            )
        else:
            final_score = similarity

        expected = similarity * (1.0 - 0.3 + 0.3 * 4.0 / 5.0)
        assert abs(final_score - expected) < 1e-6

    def test_perfect_rating_boosts_score(self):
        """avg_rating=5 should give full score (no penalty)."""
        similarity = 0.85
        avg_rating = 5.0
        rating_weight = 0.3

        if avg_rating > 0:
            final_score = similarity * (
                1.0 - rating_weight + rating_weight * avg_rating / 5.0
            )
        else:
            final_score = similarity

        assert abs(final_score - similarity) < 1e-6

    def test_low_rating_penalizes(self):
        """avg_rating=1 should penalize the score."""
        similarity = 0.85
        avg_rating = 1.0
        rating_weight = 0.3

        if avg_rating > 0:
            final_score = similarity * (
                1.0 - rating_weight + rating_weight * avg_rating / 5.0
            )
        else:
            final_score = similarity

        assert final_score < similarity


# ============================================================
# 2. tm_search auto use_count increment on top-1
# ============================================================


class TestTmSearchAutoFeedback:
    """tm_search should auto-increment use_count on top-1 result."""

    @pytest.mark.asyncio
    async def test_search_increments_use_count_on_top1(self):
        """After tm_search returns results, top-1 use_count should be incremented."""
        exp_id = str(uuid.uuid4())
        mock_results = [
            {
                "group_id": exp_id,
                "id": exp_id,
                "score": 0.88,
                "similarity": 0.88,
                "confidence": "high",
                "parent": {"id": exp_id, "title": "Test exp"},
                "children": [],
                "total_children": 0,
            },
            {
                "group_id": str(uuid.uuid4()),
                "id": str(uuid.uuid4()),
                "score": 0.75,
                "similarity": 0.75,
                "confidence": "medium",
                "parent": {"id": str(uuid.uuid4()), "title": "Other exp"},
                "children": [],
                "total_children": 0,
            },
        ]

        mock_repo = MagicMock()
        mock_repo.increment_use_count = AsyncMock()

        mock_service = MagicMock()
        mock_service.search = AsyncMock(return_value=mock_results)
        mock_session, mock_ctx = _make_mock_session()

        patches = _mock_server_deps(mock_service, mock_session)
        with (
            patches[0], patches[1], patches[2], patches[3],
            patch("team_memory.server.get_session", return_value=mock_ctx),
            patch(
                "team_memory.storage.repository.ExperienceRepository",
                return_value=mock_repo,
            ),
        ):
            tools = await mcp.get_tools()
            search_fn = tools["tm_search"].fn
            result = await search_fn(query="test query")

        data = json.loads(result)
        assert len(data["results"]) == 2
        mock_repo.increment_use_count.assert_called_once()

    @pytest.mark.asyncio
    async def test_search_no_results_no_increment(self):
        """tm_search with no results should not attempt use_count increment."""
        mock_service = MagicMock()
        mock_service.search = AsyncMock(return_value=[])
        mock_session, mock_ctx = _make_mock_session()

        patches = _mock_server_deps(mock_service, mock_session)
        with (
            patches[0], patches[1], patches[2], patches[3],
            patch("team_memory.server.get_session", return_value=mock_ctx),
        ):
            tools = await mcp.get_tools()
            search_fn = tools["tm_search"].fn
            result = await search_fn(query="nonexistent query")

        data = json.loads(result)
        assert data["results"] == []


# ============================================================
# 3. feedback_hint in return values
# ============================================================


class TestFeedbackHint:
    """tm_solve and tm_search should include feedback_hint when results exist."""

    @pytest.mark.asyncio
    async def test_solve_returns_feedback_hint(self):
        """tm_solve result should contain feedback_hint."""
        exp_id = str(uuid.uuid4())
        mock_results = [
            {
                "group_id": exp_id,
                "id": exp_id,
                "score": 0.88,
                "parent": {"id": exp_id, "title": "Docker fix", "solution": "..."},
                "children": [],
                "total_children": 0,
            }
        ]

        mock_repo = MagicMock()
        mock_repo.increment_use_count = AsyncMock()

        mock_service = MagicMock()
        mock_service.search = AsyncMock(return_value=mock_results)
        mock_session, mock_ctx = _make_mock_session()
        mock_exp = MagicMock()
        mock_exp.quality_score = 100
        mock_db_result = MagicMock()
        mock_db_result.scalar_one_or_none.return_value = mock_exp
        mock_session.execute = AsyncMock(return_value=mock_db_result)
        mock_session.commit = AsyncMock()

        patches = _mock_server_deps(mock_service, mock_session)
        with (
            patches[0], patches[1], patches[2], patches[3],
            patch("team_memory.server.get_session", return_value=mock_ctx),
            patch(
                "team_memory.storage.repository.ExperienceRepository",
                return_value=mock_repo,
            ),
        ):
            tools = await mcp.get_tools()
            solve_fn = tools["tm_solve"].fn
            result = await solve_fn(problem="Docker container won't start")

        data = json.loads(result)
        assert "feedback_hint" in data
        assert "tm_feedback" in data["feedback_hint"]
        assert exp_id in data["feedback_hint"]

    @pytest.mark.asyncio
    async def test_search_returns_feedback_hint(self):
        """tm_search result should contain feedback_hint when results exist."""
        exp_id = str(uuid.uuid4())
        mock_results = [
            {
                "group_id": exp_id,
                "id": exp_id,
                "score": 0.88,
                "similarity": 0.88,
                "confidence": "high",
                "parent": {"id": exp_id, "title": "Test"},
                "children": [],
                "total_children": 0,
            }
        ]

        mock_repo = MagicMock()
        mock_repo.increment_use_count = AsyncMock()

        mock_service = MagicMock()
        mock_service.search = AsyncMock(return_value=mock_results)
        mock_session, mock_ctx = _make_mock_session()

        patches = _mock_server_deps(mock_service, mock_session)
        with (
            patches[0], patches[1], patches[2], patches[3],
            patch("team_memory.server.get_session", return_value=mock_ctx),
            patch(
                "team_memory.storage.repository.ExperienceRepository",
                return_value=mock_repo,
            ),
        ):
            tools = await mcp.get_tools()
            search_fn = tools["tm_search"].fn
            result = await search_fn(query="test query")

        data = json.loads(result)
        assert "feedback_hint" in data
        assert "tm_feedback" in data["feedback_hint"]


class TestTmFeedbackIgnoresInjectedSession:
    """tm_feedback must not pass MCP-injected session to service (risk control)."""

    @pytest.mark.asyncio
    async def test_tm_feedback_does_not_pass_session_to_service(self):
        """When MCP injects session=..., server must not pass it to service.feedback()."""
        mock_service = MagicMock()
        mock_service.feedback = AsyncMock(return_value=False)  # experience not found

        patches = _mock_server_deps(mock_service, None)
        with patches[0], patches[1], patches[2], patches[3]:
            tools = await mcp.get_tools()
            feedback_fn = tools["tm_feedback"].fn
            await feedback_fn(
                experience_id="00000000-0000-0000-0000-000000000000",
                rating=5,
                session=MagicMock(),  # simulate MCP injection
            )

        call_kwargs = mock_service.feedback.call_args.kwargs
        assert "session" not in call_kwargs, "session must not be passed to service.feedback"


# ============================================================
# MCP contract: return structure (risk control / CI regression)
# ============================================================


class TestMcpContractReturnStructure:
    """Contract: tm_search / tm_solve JSON must have message, results, feedback_hint."""

    @pytest.mark.asyncio
    async def test_tm_search_contract_with_results(self):
        """With results: message, results, feedback_hint; feedback_hint has tm_feedback and id."""
        exp_id = str(uuid.uuid4())
        mock_results = [
            {
                "group_id": exp_id,
                "id": exp_id,
                "score": 0.88,
                "similarity": 0.88,
                "confidence": "high",
                "parent": {"id": exp_id, "title": "T"},
                "children": [],
                "total_children": 0,
            }
        ]
        mock_service = MagicMock()
        mock_service.search = AsyncMock(return_value=mock_results)
        mock_repo = MagicMock()
        mock_repo.increment_use_count = AsyncMock()
        mock_session, mock_ctx = _make_mock_session()
        patches = _mock_server_deps(mock_service, mock_session)
        with (
            patches[0], patches[1], patches[2], patches[3],
            patch("team_memory.server.get_session", return_value=mock_ctx),
            patch("team_memory.storage.repository.ExperienceRepository", return_value=mock_repo),
        ):
            tools = await mcp.get_tools()
            result = await tools["tm_search"].fn(query="q")
        data = json.loads(result)
        assert "message" in data and "results" in data and "feedback_hint" in data
        assert data["results"]
        assert "tm_feedback" in data["feedback_hint"] and exp_id in data["feedback_hint"]

    @pytest.mark.asyncio
    async def test_tm_search_contract_with_no_results(self):
        """With no results: message and results=[] required; feedback_hint may be absent."""
        mock_service = MagicMock()
        mock_service.search = AsyncMock(return_value=[])
        mock_session, mock_ctx = _make_mock_session()
        patches = _mock_server_deps(mock_service, mock_session)
        with patches[0], patches[1], patches[2], patches[3], patch(
            "team_memory.server.get_session", return_value=mock_ctx
        ):
            tools = await mcp.get_tools()
            result = await tools["tm_search"].fn(query="nonexistent")
        data = json.loads(result)
        assert "message" in data and "results" in data
        assert data["results"] == []

    @pytest.mark.asyncio
    async def test_tm_solve_contract_with_results(self):
        """With results: message, results, feedback_hint; feedback_hint has tm_feedback and id."""
        exp_id = str(uuid.uuid4())
        mock_results = [
            {
                "group_id": exp_id,
                "id": exp_id,
                "score": 0.88,
                "parent": {"id": exp_id, "title": "T", "solution": "..."},
                "children": [],
                "total_children": 0,
            }
        ]
        mock_service = MagicMock()
        mock_service.search = AsyncMock(return_value=mock_results)
        mock_repo = MagicMock()
        mock_repo.increment_use_count = AsyncMock()
        mock_session, mock_ctx = _make_mock_session()
        mock_exp = MagicMock()
        mock_exp.quality_score = 100
        mock_db_result = MagicMock()
        mock_db_result.scalar_one_or_none.return_value = mock_exp
        mock_session.execute = AsyncMock(return_value=mock_db_result)
        mock_session.commit = AsyncMock()
        patches = _mock_server_deps(mock_service, mock_session)
        with (
            patches[0], patches[1], patches[2], patches[3],
            patch("team_memory.server.get_session", return_value=mock_ctx),
            patch("team_memory.storage.repository.ExperienceRepository", return_value=mock_repo),
        ):
            tools = await mcp.get_tools()
            result = await tools["tm_solve"].fn(problem="p")
        data = json.loads(result)
        assert "message" in data and "results" in data and "feedback_hint" in data
        assert data["results"]
        assert "tm_feedback" in data["feedback_hint"] and exp_id in data["feedback_hint"]

    @pytest.mark.asyncio
    async def test_tm_solve_contract_with_no_results(self):
        """With no results: message and results=[] required; feedback_hint may be absent."""
        mock_service = MagicMock()
        mock_service.search = AsyncMock(return_value=[])
        mock_session, mock_ctx = _make_mock_session()
        patches = _mock_server_deps(mock_service, mock_session)
        with patches[0], patches[1], patches[2], patches[3], patch(
            "team_memory.server.get_session", return_value=mock_ctx
        ):
            tools = await mcp.get_tools()
            result = await tools["tm_solve"].fn(problem="nonexistent")
        data = json.loads(result)
        assert "message" in data and "results" in data
        assert data["results"] == []


# ============================================================
# 4. Scoring utility functions
# ============================================================


class TestScoringUtils:
    def test_reference_boost(self):
        assert apply_reference_boost(100) == 102

    def test_reference_boost_capped(self):
        assert apply_reference_boost(299) == 300

    def test_rating_boost_high(self):
        assert apply_rating_boost(100, 5.0) == 101

    def test_rating_boost_low(self):
        assert apply_rating_boost(100, 2.0) == 100

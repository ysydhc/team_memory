"""Tests for 'promoted' status on Experience and source/status constraint.

Covers:
- Experience can be created with exp_status="promoted"
- source="pipeline" + exp_status="draft" works
- source="api" + exp_status="draft" raises ValueError
- source="manual" + exp_status="draft" raises ValueError
- Promoted experiences excluded from default search results
- Draft experiences appear in search with score * 0.7
"""

from __future__ import annotations

import uuid
from contextlib import asynccontextmanager
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from team_memory.services.experience import ExperienceService
from team_memory.services.search_orchestrator import SearchOrchestrator

# ============================================================
# Fixtures
# ============================================================

class _MockEmbedding:
    """Minimal mock embedding provider."""

    @property
    def dimension(self):
        return 768

    async def encode_single(self, text: str) -> list[float]:
        return [0.1] * 768

    async def encode(self, texts: list[str]) -> list[list[float]]:
        return [[0.1] * 768 for _ in texts]


class _NoAuth:
    async def authenticate(self, credentials):
        from team_memory.auth.provider import User

        return User(name="anonymous", role="admin")


@pytest.fixture
def mock_embedding():
    return _MockEmbedding()


@pytest.fixture
def no_auth():
    return _NoAuth()


@pytest.fixture
def service(mock_embedding, no_auth):
    return ExperienceService(
        embedding_provider=mock_embedding,
        auth_provider=no_auth,
        db_url="sqlite+aiosqlite://",
    )


def _make_session_and_repo():
    """Build mock session + repo pair for save tests."""
    mock_session = AsyncMock()
    mock_repo_instance = MagicMock()
    mock_experience = MagicMock()
    mock_experience.to_dict.return_value = {
        "id": str(uuid.uuid4()),
        "title": "Test",
        "status": "promoted",
        "created_at": "2026-01-01T00:00:00",
    }
    mock_repo_instance.create = AsyncMock(return_value=mock_experience)
    return mock_session, mock_repo_instance


@asynccontextmanager
async def _mock_get_session(mock_session):
    yield mock_session


# ============================================================
# Test: Experience can be created with exp_status="promoted"
# ============================================================


class TestPromotedStatus:
    """Tests for the promoted status value."""

    @pytest.mark.asyncio
    async def test_save_with_promoted_status(self, service):
        """exp_status='promoted' should be accepted and passed through."""
        mock_session, mock_repo_instance = _make_session_and_repo()

        with (
            patch(
                "team_memory.storage.database.get_session",
                lambda _url: _mock_get_session(mock_session),
            ),
            patch(
                "team_memory.services.experience.ExperienceRepository",
                return_value=mock_repo_instance,
            ),
        ):
            await service.save(
                title="Promoted experience",
                problem="Should be promoted",
                created_by="admin",
                source="manual",
                exp_status="promoted",
            )

        call_kwargs = mock_repo_instance.create.call_args
        assert call_kwargs.kwargs.get("exp_status") == "promoted"

    # ========================================================
    # Test: source="pipeline" + exp_status="draft" works
    # ========================================================

    @pytest.mark.asyncio
    async def test_pipeline_source_with_draft_status(self, service):
        """source='pipeline' + exp_status='draft' is valid."""
        mock_session, mock_repo_instance = _make_session_and_repo()

        with (
            patch(
                "team_memory.storage.database.get_session",
                lambda _url: _mock_get_session(mock_session),
            ),
            patch(
                "team_memory.services.experience.ExperienceRepository",
                return_value=mock_repo_instance,
            ),
        ):
            await service.save(
                title="Pipeline draft",
                problem="Auto-extracted from pipeline",
                created_by="pipeline",
                source="pipeline",
                exp_status="draft",
            )

        call_kwargs = mock_repo_instance.create.call_args
        assert call_kwargs.kwargs.get("exp_status") == "draft"
        assert call_kwargs.kwargs.get("source") == "pipeline"

    # ========================================================
    # Test: source="api" + exp_status="draft" raises ValueError
    # ========================================================

    @pytest.mark.asyncio
    async def test_api_source_with_draft_status_raises(self, service):
        """source='api' + exp_status='draft' should raise ValueError."""
        with pytest.raises(ValueError, match="draft"):
            await service.save(
                title="API draft",
                problem="Should fail",
                created_by="user",
                source="api",
                exp_status="draft",
            )

    # ========================================================
    # Test: source="manual" + exp_status="draft" raises ValueError
    # ========================================================

    @pytest.mark.asyncio
    async def test_manual_source_with_draft_status_raises(self, service):
        """source='manual' + exp_status='draft' should raise ValueError."""
        with pytest.raises(ValueError, match="draft"):
            await service.save(
                title="Manual draft",
                problem="Should fail",
                created_by="user",
                source="manual",
                exp_status="draft",
            )

    # ========================================================
    # Test: default source (auto_extract) + exp_status="draft" raises
    # ========================================================

    @pytest.mark.asyncio
    async def test_default_source_with_draft_status_raises(self, service):
        """Default source='auto_extract' + exp_status='draft' raises ValueError."""
        with pytest.raises(ValueError, match="draft"):
            await service.save(
                title="Auto draft",
                problem="Should fail",
                created_by="user",
                exp_status="draft",
            )


# ============================================================
# Test: Promoted experiences excluded from default search results
# ============================================================


class TestPromotedSearchExclusion:
    """Test that promoted experiences are excluded from default search."""

    @pytest.mark.asyncio
    async def test_promoted_excluded_from_default_search(self):
        """Promoted experiences should be filtered out of search results."""
        mock_embedding = _MockEmbedding()
        mock_pipeline = MagicMock()

        # Pipeline returns results including a promoted one
        mock_pipeline_result = MagicMock()
        mock_pipeline_result.results = [
            {"id": "pub-1", "title": "Published", "score": 0.95, "status": "published"},
            {"id": "prom-1", "title": "Promoted", "score": 0.90, "status": "promoted"},
            {"id": "pub-2", "title": "Published 2", "score": 0.85, "status": "published"},
        ]
        mock_pipeline_result.search_type = "hybrid"
        mock_pipeline_result.reranked = False
        mock_pipeline_result.cached = False
        mock_pipeline.search = AsyncMock(return_value=mock_pipeline_result)

        orchestrator = SearchOrchestrator(
            search_pipeline=mock_pipeline,
            embedding_provider=mock_embedding,
            db_url="sqlite://",
        )

        mock_session = AsyncMock()
        mock_repo = MagicMock()
        mock_repo.increment_recall_count = AsyncMock()
        mock_repo.increment_quality_score = AsyncMock()

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
                query="test",
                max_results=10,
            )

        # Promoted result should be excluded
        statuses = [r.get("status") for r in result.results]
        assert "promoted" not in statuses
        assert len(result.results) == 2

    @pytest.mark.asyncio
    async def test_promoted_included_when_explicitly_requested(self):
        """Promoted experiences should appear when include_promoted=True."""
        mock_embedding = _MockEmbedding()
        mock_pipeline = MagicMock()

        mock_pipeline_result = MagicMock()
        mock_pipeline_result.results = [
            {"id": "pub-1", "title": "Published", "score": 0.95, "status": "published"},
            {"id": "prom-1", "title": "Promoted", "score": 0.90, "status": "promoted"},
        ]
        mock_pipeline_result.search_type = "hybrid"
        mock_pipeline_result.reranked = False
        mock_pipeline_result.cached = False
        mock_pipeline.search = AsyncMock(return_value=mock_pipeline_result)

        orchestrator = SearchOrchestrator(
            search_pipeline=mock_pipeline,
            embedding_provider=mock_embedding,
            db_url="sqlite://",
        )

        mock_session = AsyncMock()
        mock_repo = MagicMock()
        mock_repo.increment_recall_count = AsyncMock()
        mock_repo.increment_quality_score = AsyncMock()

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
                query="test",
                max_results=10,
                include_promoted=True,
            )

        # Promoted result should be included
        statuses = [r.get("status") for r in result.results]
        assert "promoted" in statuses
        assert len(result.results) == 2


# ============================================================
# Test: Draft experiences appear in search with score * 0.7
# ============================================================


class TestDraftScorePenalty:
    """Test that draft experiences get their score multiplied by 0.7."""

    @pytest.mark.asyncio
    async def test_draft_score_multiplied_by_07(self):
        """Draft experiences should have their score multiplied by 0.7."""
        mock_embedding = _MockEmbedding()
        mock_pipeline = MagicMock()

        mock_pipeline_result = MagicMock()
        mock_pipeline_result.results = [
            {"id": "pub-1", "title": "Published", "score": 0.95, "status": "published"},
            {"id": "draft-1", "title": "Draft", "score": 0.90, "status": "draft"},
            {"id": "pub-2", "title": "Published 2", "score": 0.85, "status": "published"},
        ]
        mock_pipeline_result.search_type = "hybrid"
        mock_pipeline_result.reranked = False
        mock_pipeline_result.cached = False
        mock_pipeline.search = AsyncMock(return_value=mock_pipeline_result)

        orchestrator = SearchOrchestrator(
            search_pipeline=mock_pipeline,
            embedding_provider=mock_embedding,
            db_url="sqlite://",
        )

        mock_session = AsyncMock()
        mock_repo = MagicMock()
        mock_repo.increment_recall_count = AsyncMock()
        mock_repo.increment_quality_score = AsyncMock()

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
                query="test",
                max_results=10,
            )

        # Find the draft result
        draft_result = None
        for r in result.results:
            if r.get("status") == "draft":
                draft_result = r
                break

        assert draft_result is not None
        # Original score was 0.90, after * 0.7 it should be ~0.63
        assert abs(draft_result["score"] - 0.63) < 0.01

    @pytest.mark.asyncio
    async def test_published_score_not_penalized(self):
        """Published experiences should not have their score modified."""
        mock_embedding = _MockEmbedding()
        mock_pipeline = MagicMock()

        mock_pipeline_result = MagicMock()
        mock_pipeline_result.results = [
            {"id": "pub-1", "title": "Published", "score": 0.95, "status": "published"},
        ]
        mock_pipeline_result.search_type = "hybrid"
        mock_pipeline_result.reranked = False
        mock_pipeline_result.cached = False
        mock_pipeline.search = AsyncMock(return_value=mock_pipeline_result)

        orchestrator = SearchOrchestrator(
            search_pipeline=mock_pipeline,
            embedding_provider=mock_embedding,
            db_url="sqlite://",
        )

        mock_session = AsyncMock()
        mock_repo = MagicMock()
        mock_repo.increment_recall_count = AsyncMock()
        mock_repo.increment_quality_score = AsyncMock()

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
                query="test",
                max_results=10,
            )

        pub_result = result.results[0]
        assert pub_result["score"] == 0.95


# ============================================================
# Test: _validate_source_status directly
# ============================================================


class TestValidateSourceStatus:
    """Direct unit tests for the _validate_source_status helper."""

    def test_draft_with_pipeline_source_ok(self):
        """pipeline + draft should not raise."""
        from team_memory.services.experience import _validate_source_status

        _validate_source_status("pipeline", "draft")  # should not raise

    def test_draft_with_api_source_raises(self):
        from team_memory.services.experience import _validate_source_status

        with pytest.raises(ValueError, match="draft"):
            _validate_source_status("api", "draft")

    def test_draft_with_manual_source_raises(self):
        from team_memory.services.experience import _validate_source_status

        with pytest.raises(ValueError, match="draft"):
            _validate_source_status("manual", "draft")

    def test_published_with_any_source_ok(self):
        from team_memory.services.experience import _validate_source_status

        _validate_source_status("manual", "published")  # should not raise
        _validate_source_status("api", "published")  # should not raise

    def test_promoted_with_any_source_ok(self):
        from team_memory.services.experience import _validate_source_status

        _validate_source_status("manual", "promoted")  # should not raise
        _validate_source_status("api", "promoted")  # should not raise

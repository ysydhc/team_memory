"""Tests for ExperienceService business logic.

Uses mock embedding provider and real async SQLite database
to test the service layer without needing PostgreSQL + pgvector.
For full vector search tests, see test_integration.py (requires PostgreSQL).
"""

from __future__ import annotations

import uuid
from contextlib import asynccontextmanager
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from team_memory.services.experience import ExperienceService


class TestExperienceServiceUnit:
    """Unit tests for ExperienceService using mocks."""

    @pytest.fixture
    def service(self, mock_embedding, no_auth):
        return ExperienceService(
            embedding_provider=mock_embedding,
            auth_provider=no_auth,
            db_url="sqlite+aiosqlite://",
        )

    @pytest.mark.asyncio
    async def test_authenticate_no_auth(self, service):
        user = await service.authenticate({})
        assert user is not None
        assert user.name == "anonymous"

    @pytest.mark.asyncio
    async def test_authenticate_with_api_key(self, mock_embedding, api_key_auth):
        service = ExperienceService(
            embedding_provider=mock_embedding,
            auth_provider=api_key_auth,
            db_url="sqlite+aiosqlite://",
        )
        user = await service.authenticate({"api_key": "test_key_123"})
        assert user is not None
        assert user.name == "test_user"

    @pytest.mark.asyncio
    async def test_authenticate_invalid_api_key(self, mock_embedding, api_key_auth):
        service = ExperienceService(
            embedding_provider=mock_embedding,
            auth_provider=api_key_auth,
            db_url="sqlite+aiosqlite://",
        )
        user = await service.authenticate({"api_key": "wrong_key"})
        assert user is None

    @pytest.mark.asyncio
    async def test_save_calls_embedding(self, service, mock_embedding):
        """Verify that save generates an embedding from the content."""
        original_encode = mock_embedding.encode_single
        calls = []

        async def tracked_encode(text):
            calls.append(text)
            return await original_encode(text)

        mock_embedding.encode_single = tracked_encode

        # Create a mock session and repository
        mock_session = AsyncMock()
        mock_repo_instance = MagicMock()
        mock_experience = MagicMock()
        mock_experience.to_dict.return_value = {
            "id": str(uuid.uuid4()),
            "title": "Test",
            "created_at": "2026-01-01T00:00:00",
        }
        mock_repo_instance.create = AsyncMock(return_value=mock_experience)

        @asynccontextmanager
        async def mock_get_session(_db_url):
            yield mock_session

        with patch("team_memory.storage.database.get_session", mock_get_session), \
             patch("team_memory.services.experience.ExperienceRepository") as mock_repo:
            mock_repo.return_value = mock_repo_instance
            await service.save(
                title="Fix Docker issue",
                problem="Container won't start",
                solution="Check port conflicts",
                created_by="alice",
                tags=["docker"],
            )

        # Verify embedding was called with combined text
        assert len(calls) == 1
        assert "Fix Docker issue" in calls[0]
        assert "Container won't start" in calls[0]
        assert "Check port conflicts" in calls[0]

        # Verify repository create was called
        mock_repo_instance.create.assert_called_once()

    @pytest.mark.asyncio
    async def test_search_calls_embedding_and_repo(self, service, mock_embedding):
        """Verify that search generates query embedding and calls repo."""
        mock_session = AsyncMock()
        mock_repo_instance = MagicMock()
        mock_repo_instance.search_by_vector = AsyncMock(return_value=[
            {
                "id": "test-id",
                "title": "Existing solution",
                "similarity": 0.95,
            }
        ])

        @asynccontextmanager
        async def mock_get_session(_db_url):
            yield mock_session

        with patch("team_memory.storage.database.get_session", mock_get_session), \
             patch("team_memory.services.experience.ExperienceRepository") as mock_repo:
            mock_repo.return_value = mock_repo_instance
            results = await service.search(
                query="Docker container issue",
                max_results=3,
                min_similarity=0.7,
            )

        assert len(results) == 1
        assert results[0]["title"] == "Existing solution"
        mock_repo_instance.search_by_vector.assert_called_once()

    @pytest.mark.asyncio
    async def test_feedback_success(self, service):
        """Test successful feedback submission."""
        mock_session = AsyncMock()
        mock_repo_instance = MagicMock()
        mock_experience = MagicMock()
        mock_repo_instance.get_by_id = AsyncMock(return_value=mock_experience)
        mock_repo_instance.add_feedback = AsyncMock()

        exp_id = str(uuid.uuid4())

        @asynccontextmanager
        async def mock_get_session(_db_url):
            yield mock_session

        with patch("team_memory.storage.database.get_session", mock_get_session), \
             patch("team_memory.services.experience.ExperienceRepository") as mock_repo:
            mock_repo.return_value = mock_repo_instance
            result = await service.feedback(
                experience_id=exp_id,
                rating=5,
                feedback_by="alice",
                comment="Great solution!",
            )

        assert result is True
        mock_repo_instance.add_feedback.assert_called_once()

    @pytest.mark.asyncio
    async def test_feedback_not_found(self, service):
        """Test feedback for non-existent experience."""
        mock_session = AsyncMock()
        mock_repo_instance = MagicMock()
        mock_repo_instance.get_by_id = AsyncMock(return_value=None)

        exp_id = str(uuid.uuid4())

        @asynccontextmanager
        async def mock_get_session(_db_url):
            yield mock_session

        with patch("team_memory.storage.database.get_session", mock_get_session), \
             patch("team_memory.services.experience.ExperienceRepository") as mock_repo:
            mock_repo.return_value = mock_repo_instance
            result = await service.feedback(
                experience_id=exp_id,
                rating=4,
                feedback_by="alice",
            )

        assert result is False

    @pytest.mark.asyncio
    async def test_update_appends_solution(self, service):
        """Test that update appends solution addendum."""
        mock_session = AsyncMock()
        mock_repo_instance = MagicMock()
        mock_experience = MagicMock()
        mock_experience.solution = "Original solution"
        mock_experience.title = "Test"
        mock_experience.description = "Test desc"
        mock_experience.tags = ["python"]
        mock_experience.code_snippets = None
        mock_experience.to_dict.return_value = {"id": "test", "solution": "updated"}
        mock_repo_instance.get_by_id = AsyncMock(return_value=mock_experience)
        mock_repo_instance.update = AsyncMock(return_value=mock_experience)

        exp_id = str(uuid.uuid4())

        @asynccontextmanager
        async def mock_get_session(_db_url):
            yield mock_session

        with patch("team_memory.storage.database.get_session", mock_get_session), \
             patch("team_memory.services.experience.ExperienceRepository") as mock_repo:
            mock_repo.return_value = mock_repo_instance
            result = await service.update(
                experience_id=exp_id,
                solution_addendum="Additional fix",
            )

        assert result is not None
        # Check that update was called with merged solution
        call_kwargs = mock_repo_instance.update.call_args
        assert "Additional fix" in call_kwargs.kwargs.get("solution", "")

    @pytest.mark.asyncio
    async def test_update_replaces_tags(self, service):
        """Test that update replaces tags in-place (not merge)."""
        mock_session = AsyncMock()
        mock_repo_instance = MagicMock()
        mock_experience = MagicMock()
        mock_experience.solution = "Solution"
        mock_experience.title = "Test"
        mock_experience.description = "Desc"
        mock_experience.tags = ["python", "docker"]
        mock_experience.code_snippets = None
        mock_experience.to_dict.return_value = {
            "id": "test", "tags": ["docker", "linux"],
        }
        mock_repo_instance.get_by_id = AsyncMock(return_value=mock_experience)
        mock_repo_instance.update = AsyncMock(return_value=mock_experience)

        exp_id = str(uuid.uuid4())

        @asynccontextmanager
        async def mock_get_session(_db_url):
            yield mock_session

        with patch("team_memory.storage.database.get_session", mock_get_session), \
             patch("team_memory.services.experience.ExperienceRepository") as mock_repo:
            mock_repo.return_value = mock_repo_instance
            result = await service.update(
                experience_id=exp_id,
                tags=["docker", "linux"],
            )

        assert result is not None
        call_kwargs = mock_repo_instance.update.call_args
        new_tags = call_kwargs.kwargs.get("tags", [])
        # In-place update: tags are replaced, not merged
        assert new_tags == ["docker", "linux"]
        assert "python" not in new_tags


class TestPhase2Fixes:
    """Regression tests for phase-2 reliability fixes.

    Covers:
    - personal publish_status not overridden to draft
    - feedback() callable without external session
    - update() callable without external session
    """

    @pytest.fixture
    def service_with_review(self, mock_embedding, no_auth):
        """Service with require_review_for_ai=True (the risky config)."""
        from team_memory.config import ReviewConfig
        review_cfg = ReviewConfig(
            enabled=True,
            require_review_for_ai=True,
            auto_publish_threshold=0.0,
        )
        return ExperienceService(
            embedding_provider=mock_embedding,
            auth_provider=no_auth,
            review_config=review_cfg,
            db_url="sqlite+aiosqlite://",
        )

    @pytest.mark.asyncio
    async def test_personal_status_preserved_with_ai_review(self, service_with_review):
        """publish_status='personal' must NOT be overridden to 'draft' even when
        require_review_for_ai=True and source='auto_extract'."""
        mock_session = AsyncMock()
        mock_repo_instance = MagicMock()
        mock_experience = MagicMock()
        mock_experience.to_dict.return_value = {
            "id": str(uuid.uuid4()),
            "title": "Test",
            "publish_status": "personal",
            "created_at": "2026-01-01T00:00:00",
        }
        mock_repo_instance.create = AsyncMock(return_value=mock_experience)

        @asynccontextmanager
        async def mock_get_session(_db_url):
            yield mock_session

        with patch("team_memory.storage.database.get_session", mock_get_session), \
             patch("team_memory.services.experience.ExperienceRepository") as mock_repo:
            mock_repo.return_value = mock_repo_instance
            await service_with_review.save(
                title="Personal experience",
                problem="Should stay personal",
                created_by="user",
                source="auto_extract",
                publish_status="personal",
            )

        call_kwargs = mock_repo_instance.create.call_args
        assert call_kwargs.kwargs["publish_status"] == "personal", \
            "personal status must not be overridden to draft"

    @pytest.mark.asyncio
    async def test_published_status_becomes_draft_with_ai_review(self, service_with_review):
        """publish_status='published' SHOULD be overridden to 'draft' when
        require_review_for_ai=True and source='auto_extract'."""
        mock_session = AsyncMock()
        mock_repo_instance = MagicMock()
        mock_experience = MagicMock()
        mock_experience.to_dict.return_value = {
            "id": str(uuid.uuid4()),
            "title": "Test",
            "publish_status": "draft",
            "created_at": "2026-01-01T00:00:00",
        }
        mock_repo_instance.create = AsyncMock(return_value=mock_experience)

        @asynccontextmanager
        async def mock_get_session(_db_url):
            yield mock_session

        with patch("team_memory.storage.database.get_session", mock_get_session), \
             patch("team_memory.services.experience.ExperienceRepository") as mock_repo:
            mock_repo.return_value = mock_repo_instance
            await service_with_review.save(
                title="Published experience",
                problem="Should become draft for review",
                created_by="user",
                source="auto_extract",
                publish_status="published",
            )

        call_kwargs = mock_repo_instance.create.call_args
        assert call_kwargs.kwargs["publish_status"] == "draft", \
            "published status should be overridden to draft for AI review"

    @pytest.mark.asyncio
    async def test_feedback_without_session(self, service_with_review):
        """feedback() must work without passing session= (it manages its own)."""
        mock_session = AsyncMock()
        mock_repo_instance = MagicMock()
        mock_experience = MagicMock()
        mock_repo_instance.get_by_id = AsyncMock(return_value=mock_experience)
        mock_repo_instance.add_feedback = AsyncMock()

        @asynccontextmanager
        async def mock_get_session(_db_url):
            yield mock_session

        with patch("team_memory.storage.database.get_session", mock_get_session), \
             patch("team_memory.services.experience.ExperienceRepository") as mock_repo:
            mock_repo.return_value = mock_repo_instance
            result = await service_with_review.feedback(
                experience_id=str(uuid.uuid4()),
                rating=4,
                feedback_by="reviewer",
            )

        assert result is True

    @pytest.mark.asyncio
    async def test_feedback_accepts_extra_kwargs(self, service_with_review):
        """feedback() must accept MCP-injected session and extras via **kwargs without raising."""
        mock_session = AsyncMock()
        mock_repo_instance = MagicMock()
        mock_repo_instance.get_by_id = AsyncMock(return_value=None)  # not found

        @asynccontextmanager
        async def mock_get_session(_db_url):
            yield mock_session

        with patch("team_memory.storage.database.get_session", mock_get_session), \
             patch("team_memory.services.experience.ExperienceRepository") as mock_repo:
            mock_repo.return_value = mock_repo_instance
            result = await service_with_review.feedback(
                experience_id=str(uuid.uuid4()),
                rating=4,
                feedback_by="reviewer",
                session=object(),  # MCP may inject
                foo="bar",  # arbitrary extra
            )

        assert result is False  # experience not found

    @pytest.mark.asyncio
    async def test_update_without_session(self, service_with_review):
        """update() must work without passing session= (it manages its own)."""
        mock_session = AsyncMock()
        mock_repo_instance = MagicMock()
        mock_experience = MagicMock()
        mock_experience.solution = "Original"
        mock_experience.title = "Test"
        mock_experience.description = "Desc"
        mock_experience.tags = []
        mock_experience.code_snippets = None
        mock_experience.to_dict.return_value = {"id": "test", "solution": "Original + Addendum"}
        mock_repo_instance.get_by_id = AsyncMock(return_value=mock_experience)
        mock_repo_instance.update = AsyncMock(return_value=mock_experience)

        @asynccontextmanager
        async def mock_get_session(_db_url):
            yield mock_session

        with patch("team_memory.storage.database.get_session", mock_get_session), \
             patch("team_memory.services.experience.ExperienceRepository") as mock_repo:
            mock_repo.return_value = mock_repo_instance
            result = await service_with_review.update(
                experience_id=str(uuid.uuid4()),
                solution_addendum="Addendum",
            )

        assert result is not None

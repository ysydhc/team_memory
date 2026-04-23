"""Tests for source=obsidian Experience support.

Covers:
- source=obsidian + exp_status=draft → creation succeeds
- source=obsidian + exp_status=published → creation succeeds
- source=obsidian + draft → draft_publish upgrades to published
- source=api + exp_status=draft → fails (only pipeline/obsidian can draft)
- source=obsidian Experience returned normally in retrieval
- Existing tests should not regress
"""

from __future__ import annotations

import uuid
from contextlib import asynccontextmanager
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from team_memory.services.experience import ExperienceService, _validate_source_status


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
        "status": "draft",
        "created_at": "2026-01-01T00:00:00",
    }
    mock_repo_instance.create = AsyncMock(return_value=mock_experience)
    return mock_session, mock_repo_instance


@asynccontextmanager
async def _mock_get_session(mock_session):
    yield mock_session


# ============================================================
# Test: _validate_source_status with obsidian
# ============================================================


class TestValidateSourceStatusObsidian:
    """Direct unit tests for _validate_source_status with obsidian source."""

    def test_draft_with_obsidian_source_ok(self):
        """obsidian + draft should not raise."""
        _validate_source_status("obsidian", "draft")  # should not raise

    def test_draft_with_pipeline_source_ok(self):
        """pipeline + draft should not raise (regression)."""
        _validate_source_status("pipeline", "draft")  # should not raise

    def test_draft_with_api_source_raises(self):
        """api + draft should raise ValueError."""
        with pytest.raises(ValueError, match="draft"):
            _validate_source_status("api", "draft")

    def test_draft_with_manual_source_raises(self):
        """manual + draft should raise ValueError."""
        with pytest.raises(ValueError, match="draft"):
            _validate_source_status("manual", "draft")

    def test_published_with_obsidian_source_ok(self):
        """obsidian + published should not raise."""
        _validate_source_status("obsidian", "published")  # should not raise

    def test_promoted_with_obsidian_source_ok(self):
        """obsidian + promoted should not raise."""
        _validate_source_status("obsidian", "promoted")  # should not raise


# ============================================================
# Test: source=obsidian + exp_status=draft → creation succeeds
# ============================================================


class TestObsidianDraftCreation:
    """Test that source=obsidian + exp_status=draft can be created."""

    @pytest.mark.asyncio
    async def test_obsidian_source_with_draft_status(self, service):
        """source='obsidian' + exp_status='draft' is valid."""
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
            result = await service.save(
                title="Obsidian draft",
                problem="Auto-extracted from obsidian vault",
                created_by="obsidian",
                source="obsidian",
                exp_status="draft",
            )

        call_kwargs = mock_repo_instance.create.call_args
        assert call_kwargs.kwargs.get("exp_status") == "draft"
        assert call_kwargs.kwargs.get("source") == "obsidian"

    @pytest.mark.asyncio
    async def test_obsidian_source_with_published_status(self, service):
        """source='obsidian' + exp_status='published' is valid."""
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
            result = await service.save(
                title="Obsidian published",
                problem="Published from obsidian vault",
                created_by="obsidian",
                source="obsidian",
                exp_status="published",
            )

        call_kwargs = mock_repo_instance.create.call_args
        assert call_kwargs.kwargs.get("exp_status") == "published"
        assert call_kwargs.kwargs.get("source") == "obsidian"


# ============================================================
# Test: source=api + exp_status=draft → should fail
# ============================================================


class TestApiDraftStillFails:
    """Test that source=api + exp_status=draft still raises ValueError."""

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


# ============================================================
# Test: source=obsidian draft → draft_publish upgrades to published
# ============================================================


class TestObsidianDraftPublish:
    """Test that source=obsidian drafts can be published via op_draft_publish."""

    @pytest.mark.asyncio
    async def test_obsidian_draft_can_be_published(self):
        """source=obsidian draft should be publishable via op_draft_publish."""
        from team_memory.services.memory_operations import op_draft_publish

        mock_service = MagicMock()
        exp_id = str(uuid.uuid4())
        mock_exp = {
            "id": exp_id,
            "source": "obsidian",
            "status": "draft",
            "title": "Obsidian Draft",
        }
        mock_service.get_by_id = AsyncMock(return_value=mock_exp)
        mock_service.update = AsyncMock(
            return_value={**mock_exp, "status": "published"}
        )

        with patch(
            "team_memory.services.memory_operations._get_service",
            return_value=mock_service,
        ):
            result = await op_draft_publish(
                user="test_user",
                draft_id=exp_id,
            )

        assert result.get("error") is not True
        mock_service.update.assert_called_once()

    @pytest.mark.asyncio
    async def test_api_draft_cannot_be_published_via_op(self):
        """source=api draft should NOT be publishable via op_draft_publish."""
        from team_memory.services.memory_operations import op_draft_publish

        mock_service = MagicMock()
        exp_id = str(uuid.uuid4())
        mock_exp = {
            "id": exp_id,
            "source": "api",
            "status": "draft",
            "title": "API Draft",
        }
        mock_service.get_by_id = AsyncMock(return_value=mock_exp)

        with patch(
            "team_memory.services.memory_operations._get_service",
            return_value=mock_service,
        ):
            result = await op_draft_publish(
                user="test_user",
                draft_id=exp_id,
            )

        assert result.get("error") is True
        assert "validation_error" in result.get("code", "")


# ============================================================
# Test: source=obsidian Experience retrieval works normally
# ============================================================


class TestObsidianExperienceRetrieval:
    """Test that source=obsidian experiences are returned in retrieval."""

    @pytest.mark.asyncio
    async def test_obsidian_experience_retrieved_by_id(self, service):
        """source=obsidian experience should be retrievable."""
        mock_session = AsyncMock()
        mock_repo_instance = MagicMock()
        exp_id = uuid.uuid4()
        mock_experience = MagicMock()
        mock_experience.id = exp_id
        mock_experience.title = "Obsidian Note"
        mock_experience.source = "obsidian"
        mock_experience.exp_status = "published"
        mock_experience.to_dict.return_value = {
            "id": str(exp_id),
            "title": "Obsidian Note",
            "source": "obsidian",
            "exp_status": "published",
        }
        mock_repo_instance.get_by_id = AsyncMock(return_value=mock_experience)

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
            result = await service.get_by_id(str(exp_id))

        assert result is not None
        assert result.get("source") == "obsidian"

    @pytest.mark.asyncio
    async def test_obsidian_draft_experience_retrieved(self, service):
        """source=obsidian draft experience should be retrievable."""
        mock_session = AsyncMock()
        mock_repo_instance = MagicMock()
        exp_id = uuid.uuid4()
        mock_experience = MagicMock()
        mock_experience.id = exp_id
        mock_experience.title = "Obsidian Draft"
        mock_experience.source = "obsidian"
        mock_experience.exp_status = "draft"
        mock_experience.to_dict.return_value = {
            "id": str(exp_id),
            "title": "Obsidian Draft",
            "source": "obsidian",
            "exp_status": "draft",
        }
        mock_repo_instance.get_by_id = AsyncMock(return_value=mock_experience)

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
            result = await service.get_by_id(str(exp_id))

        assert result is not None
        assert result.get("source") == "obsidian"
        assert result.get("exp_status") == "draft"

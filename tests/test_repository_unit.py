"""Mock-based unit tests for ExperienceRepository.

Validates CRUD, search delegation, feedback, metrics, and dedup
without requiring a real database.
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from team_memory.storage.repository import ExperienceRepository

# ============================================================
# Helpers
# ============================================================


def _make_mock_session() -> AsyncMock:
    """Create a mock AsyncSession with common methods."""
    session = AsyncMock()
    session.add = MagicMock()
    session.flush = AsyncMock()
    session.execute = AsyncMock()
    session.delete = AsyncMock()
    return session


def _make_experience(**overrides) -> MagicMock:
    """Create a mock Experience object with sensible defaults."""
    exp = MagicMock()
    exp.id = overrides.get("id", uuid.uuid4())
    exp.title = overrides.get("title", "Test Experience")
    exp.description = overrides.get("description", "Test problem")
    exp.solution = overrides.get("solution", "Test solution")
    exp.tags = overrides.get("tags", ["python"])
    exp.group_key = overrides.get("group_key", None)
    exp.experience_type = overrides.get("experience_type", "general")
    exp.embedding = overrides.get("embedding", None)
    exp.source = overrides.get("source", "manual")
    exp.created_by = overrides.get("created_by", "test_user")
    exp.project = overrides.get("project", "default")
    exp.visibility = overrides.get("visibility", "project")
    exp.exp_status = overrides.get("exp_status", "published")
    exp.is_deleted = overrides.get("is_deleted", False)
    exp.deleted_at = overrides.get("deleted_at", None)
    exp.recall_count = overrides.get("recall_count", 0)
    exp.created_at = overrides.get("created_at", datetime.now(timezone.utc))
    exp.updated_at = overrides.get("updated_at", datetime.now(timezone.utc))
    exp.children = overrides.get("children", [])
    exp.feedbacks = overrides.get("feedbacks", [])
    exp.parent_id = overrides.get("parent_id", None)

    def to_dict(include_children: bool = False) -> dict:
        d = {
            "id": str(exp.id),
            "parent_id": str(exp.parent_id) if exp.parent_id else None,
            "title": exp.title,
            "description": exp.description,
            "solution": exp.solution,
            "tags": exp.tags or [],
            "group_key": exp.group_key,
            "experience_type": exp.experience_type,
            "source": exp.source,
            "created_by": exp.created_by,
            "visibility": exp.visibility,
            "status": exp.exp_status,
            "project": exp.project,
            "is_deleted": exp.is_deleted,
            "recall_count": exp.recall_count,
            "created_at": exp.created_at.isoformat() if exp.created_at else None,
            "updated_at": exp.updated_at.isoformat() if exp.updated_at else None,
        }
        if include_children and exp.children:
            d["children"] = [c.to_dict() for c in exp.children]
        return d

    exp.to_dict = to_dict
    return exp


# ============================================================
# CREATE tests
# ============================================================


class TestCreate:
    @pytest.mark.asyncio
    async def test_create_adds_to_session(self):
        """create() calls session.add and flush."""
        session = _make_mock_session()
        repo = ExperienceRepository(session)

        with patch("team_memory.storage.repository.Experience") as mock_exp_cls:
            mock_exp = _make_experience()
            mock_exp_cls.return_value = mock_exp

            result = await repo.create(
                title="Fix timeout",
                description="Pool exhausted",
                solution="Increase pool",
                created_by="admin",
            )

        session.add.assert_called_once_with(mock_exp)
        session.flush.assert_awaited_once()
        assert result == mock_exp

    @pytest.mark.asyncio
    async def test_create_with_embedding(self):
        """create() passes embedding to Experience model."""
        session = _make_mock_session()
        repo = ExperienceRepository(session)
        embedding = [0.1] * 768

        with patch("team_memory.storage.repository.Experience") as mock_exp_cls:
            mock_exp = _make_experience(embedding=embedding)
            mock_exp_cls.return_value = mock_exp

            result = await repo.create(
                title="Embed test",
                description="Test embedding",
                embedding=embedding,
                created_by="admin",
            )

        call_kwargs = mock_exp_cls.call_args
        assert call_kwargs.kwargs["embedding"] == embedding
        assert result.embedding == embedding

    @pytest.mark.asyncio
    async def test_create_without_embedding(self):
        """create() accepts embedding=None."""
        session = _make_mock_session()
        repo = ExperienceRepository(session)

        with patch("team_memory.storage.repository.Experience") as mock_exp_cls:
            mock_exp = _make_experience(embedding=None)
            mock_exp_cls.return_value = mock_exp

            result = await repo.create(
                title="No embed",
                description="No embedding needed",
                created_by="admin",
            )

        call_kwargs = mock_exp_cls.call_args
        assert call_kwargs.kwargs["embedding"] is None
        assert result.embedding is None


# ============================================================
# READ tests
# ============================================================


class TestGetById:
    @pytest.mark.asyncio
    async def test_get_by_id_found(self):
        """get_by_id() returns experience when found."""
        session = _make_mock_session()
        exp = _make_experience()

        mock_result = MagicMock()
        mock_result.scalar_one_or_none.return_value = exp
        session.execute.return_value = mock_result

        repo = ExperienceRepository(session)
        result = await repo.get_by_id(exp.id)

        assert result is not None
        assert result.id == exp.id
        session.execute.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_get_by_id_not_found(self):
        """get_by_id() returns None when not found."""
        session = _make_mock_session()

        mock_result = MagicMock()
        mock_result.scalar_one_or_none.return_value = None
        session.execute.return_value = mock_result

        repo = ExperienceRepository(session)
        result = await repo.get_by_id(uuid.uuid4())

        assert result is None

    @pytest.mark.asyncio
    async def test_get_by_id_deleted(self):
        """get_by_id() with default include_deleted=False filters out deleted records."""
        session = _make_mock_session()

        # Simulate DB returning None because the is_deleted filter excludes it
        mock_result = MagicMock()
        mock_result.scalar_one_or_none.return_value = None
        session.execute.return_value = mock_result

        repo = ExperienceRepository(session)
        result = await repo.get_by_id(uuid.uuid4())

        assert result is None
        session.execute.assert_awaited_once()


# ============================================================
# UPDATE tests
# ============================================================


class TestUpdate:
    @pytest.mark.asyncio
    async def test_update_modifies_fields(self):
        """update() calls setattr for each field."""
        session = _make_mock_session()
        exp = _make_experience(title="Old Title", solution="Old solution")

        mock_result = MagicMock()
        mock_result.scalar_one_or_none.return_value = exp
        session.execute.return_value = mock_result

        repo = ExperienceRepository(session)
        result = await repo.update(
            exp.id,
            title="New Title",
            solution="New solution",
        )

        assert result is not None
        assert exp.title == "New Title"
        assert exp.solution == "New solution"
        session.flush.assert_awaited_once()


# ============================================================
# DELETE tests
# ============================================================


class TestSoftDelete:
    @pytest.mark.asyncio
    async def test_delete_sets_is_deleted(self):
        """soft_delete() sets is_deleted=True and deleted_at."""
        session = _make_mock_session()
        exp = _make_experience(is_deleted=False)

        mock_result = MagicMock()
        mock_result.scalar_one_or_none.return_value = exp
        session.execute.return_value = mock_result

        repo = ExperienceRepository(session)
        result = await repo.soft_delete(exp.id)

        assert result is True
        assert exp.is_deleted is True
        assert exp.deleted_at is not None
        session.flush.assert_awaited_once()


# ============================================================
# SEARCH tests
# ============================================================


class TestSearchDelegation:
    @pytest.mark.asyncio
    async def test_search_by_vector_calls_query_builder(self):
        """search_by_vector() delegates to build_vector_search."""
        session = _make_mock_session()
        exp = _make_experience()
        similarity = 0.92

        mock_result = MagicMock()
        mock_result.all.return_value = [(exp, similarity)]
        session.execute.return_value = mock_result

        with patch("team_memory.storage.query_builders.build_vector_search") as mock_build:
            mock_build.return_value = MagicMock()

            repo = ExperienceRepository(session)
            results = await repo.search_by_vector(
                query_embedding=[0.1] * 768,
                max_results=5,
                project="default",
            )

        mock_build.assert_called_once()
        assert len(results) == 1
        assert results[0]["similarity"] == round(similarity, 4)

    @pytest.mark.asyncio
    async def test_search_by_fts_calls_query_builder(self):
        """search_by_fts() delegates to build_fts_search."""
        session = _make_mock_session()
        exp = _make_experience()
        rank = 0.75

        mock_result = MagicMock()
        mock_result.all.return_value = [(exp, rank)]
        session.execute.return_value = mock_result

        with (
            patch("team_memory.storage.query_builders.build_fts_search") as mock_build,
            patch("team_memory.services.tokenizer.tokenize", return_value="test query"),
        ):
            mock_build.return_value = MagicMock()

            repo = ExperienceRepository(session)
            results = await repo.search_by_fts(
                query_text="test query",
                max_results=5,
                project="default",
            )

        mock_build.assert_called_once()
        assert len(results) == 1
        assert results[0]["fts_rank"] == round(rank, 4)


# ============================================================
# FEEDBACK tests
# ============================================================


class TestFeedback:
    @pytest.mark.asyncio
    async def test_add_feedback_creates_record(self):
        """add_feedback() creates a feedback record in session."""
        session = _make_mock_session()

        with patch("team_memory.storage.repository.ExperienceFeedback") as mock_fb_cls:
            mock_fb = MagicMock()
            mock_fb_cls.return_value = mock_fb

            repo = ExperienceRepository(session)
            result = await repo.add_feedback(
                experience_id=uuid.uuid4(),
                rating=5,
                feedback_by="admin",
                comment="Helpful!",
            )

        session.add.assert_called_once_with(mock_fb)
        session.flush.assert_awaited_once()
        assert result == mock_fb


# ============================================================
# METRICS tests
# ============================================================


class TestMetrics:
    @pytest.mark.asyncio
    async def test_increment_recall_count(self):
        """increment_recall_count() executes an UPDATE statement."""
        session = _make_mock_session()
        exp_id = uuid.uuid4()

        repo = ExperienceRepository(session)
        await repo.increment_recall_count(exp_id)

        session.execute.assert_awaited_once()


# ============================================================
# PROJECTS tests
# ============================================================


class TestListProjects:
    @pytest.mark.asyncio
    async def test_list_projects_returns_distinct(self):
        """list_projects() returns distinct project names."""
        session = _make_mock_session()

        mock_result = MagicMock()
        mock_result.all.return_value = [("default",), ("project-a",), ("project-b",)]
        session.execute.return_value = mock_result

        repo = ExperienceRepository(session)
        projects = await repo.list_projects()

        assert projects == ["default", "project-a", "project-b"]
        session.execute.assert_awaited_once()


# ============================================================
# DEDUP / check_similar tests
# ============================================================


class TestCheckSimilar:
    @pytest.mark.asyncio
    async def test_check_similar_finds_duplicates(self):
        """check_similar() returns candidates above threshold."""
        session = _make_mock_session()
        exp = _make_experience(
            title="Existing fix",
            description="A known problem that was solved before",
        )
        similarity = 0.95

        mock_result = MagicMock()
        mock_result.all.return_value = [(exp, similarity)]
        session.execute.return_value = mock_result

        repo = ExperienceRepository(session)
        results = await repo.check_similar(
            embedding=[0.1] * 768,
            threshold=0.90,
            project="default",
        )

        assert len(results) == 1
        assert results[0]["title"] == "Existing fix"
        assert results[0]["similarity"] == round(similarity, 4)

    @pytest.mark.asyncio
    async def test_check_similar_no_duplicates(self):
        """check_similar() returns empty list when no candidates above threshold."""
        session = _make_mock_session()

        mock_result = MagicMock()
        mock_result.all.return_value = []
        session.execute.return_value = mock_result

        repo = ExperienceRepository(session)
        results = await repo.check_similar(
            embedding=[0.1] * 768,
            threshold=0.90,
            project="default",
        )

        assert results == []

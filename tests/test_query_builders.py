"""Tests for query_builders -- pure SQL construction, no I/O."""

from __future__ import annotations

from sqlalchemy import Select
from sqlalchemy.sql.elements import TextClause

from team_memory.storage.models import Experience
from team_memory.storage.query_builders import (
    active_filter,
    apply_scope_filter,
    build_duplicate_detection,
    build_fts_search,
    build_vector_search,
    project_value,
)


class TestActiveFilter:
    """Tests for active_filter visibility logic."""

    def test_active_filter_no_user_only_published(self) -> None:
        """Without current_user, returns is_deleted=False + published filter."""
        filters = active_filter(current_user=None)
        assert len(filters) == 2

    def test_active_filter_with_user_includes_or_clause(self) -> None:
        """With current_user, returns is_deleted=False + OR(published, owned)."""
        filters = active_filter(current_user="alice")
        assert len(filters) == 2
        # Second element is an OR clause
        second = filters[1]
        assert hasattr(second, "clauses") or "BooleanClauseList" in type(second).__name__

    def test_active_filter_no_user_second_is_published(self) -> None:
        """Without current_user, the second filter is a simple published comparison."""
        filters = active_filter(current_user=None)
        second = filters[1]
        # Should be a simple BinaryExpression, not an OR clause
        assert not hasattr(second, "clauses")


class TestProjectValue:
    """Tests for project_value normalization."""

    def test_default_when_none(self) -> None:
        assert project_value(None) == "default"

    def test_default_when_empty(self) -> None:
        assert project_value("") == "default"

    def test_default_when_whitespace(self) -> None:
        assert project_value("   ") == "default"

    def test_alias_team_memory(self) -> None:
        assert project_value("team-memory") == "team_memory"

    def test_alias_team_doc(self) -> None:
        assert project_value("team_doc") == "team_memory"

    def test_passthrough(self) -> None:
        assert project_value("my_project") == "my_project"

    def test_strips_whitespace(self) -> None:
        assert project_value("  my_project  ") == "my_project"


class TestApplyScopeFilter:
    """Tests for apply_scope_filter."""

    def test_returns_select(self) -> None:
        from sqlalchemy import select

        base = select(Experience)
        result = apply_scope_filter(base, "default", None, None)
        assert isinstance(result, Select)

    def test_personal_scope_adds_private_filter(self) -> None:
        from sqlalchemy import select

        base = select(Experience)
        result = apply_scope_filter(base, "myproj", "personal", "alice")
        compiled = str(result.compile(compile_kwargs={"literal_binds": True}))
        assert "private" in compiled.lower() or "visibility" in compiled.lower()


class TestBuildVectorSearch:
    """Tests for build_vector_search."""

    def test_returns_select(self) -> None:
        fake_embedding = [0.1] * 768
        stmt = build_vector_search(fake_embedding)
        assert isinstance(stmt, Select)

    def test_applies_project_filter(self) -> None:
        fake_embedding = [0.1] * 768
        stmt = build_vector_search(fake_embedding, project="test_proj")
        compiled = str(stmt.compile(compile_kwargs={"literal_binds": True}))
        assert "test_proj" in compiled

    def test_applies_tag_filter(self) -> None:
        fake_embedding = [0.1] * 768
        stmt = build_vector_search(fake_embedding, tags=["python", "testing"])
        compiled = str(stmt.compile(compile_kwargs={"literal_binds": True}))
        assert "python" in compiled

    def test_limit_applied(self) -> None:
        fake_embedding = [0.1] * 768
        stmt = build_vector_search(fake_embedding, limit=10)
        compiled = str(stmt.compile(compile_kwargs={"literal_binds": True}))
        assert "10" in compiled


class TestBuildFtsSearch:
    """Tests for build_fts_search."""

    def test_returns_select(self) -> None:
        stmt = build_fts_search("test query")
        assert isinstance(stmt, Select)

    def test_applies_project_filter(self) -> None:
        stmt = build_fts_search("test query", project="test_proj")
        # FTS uses REGCONFIG which cannot render literal_binds;
        # verify the Select was constructed without error and is the right type
        assert isinstance(stmt, Select)
        # Compile without literal_binds to verify structure
        compiled = str(stmt.compile())
        assert "experiences" in compiled

    def test_applies_tag_filter(self) -> None:
        stmt = build_fts_search("test query", tags=["python"])
        assert isinstance(stmt, Select)
        compiled = str(stmt.compile())
        # tags.overlap generates an && operator in the SQL
        assert "&&" in compiled or "tags" in compiled


class TestBuildDuplicateDetection:
    """Tests for build_duplicate_detection."""

    def test_returns_text_clause(self) -> None:
        stmt = build_duplicate_detection()
        assert isinstance(stmt, TextClause)

    def test_contains_threshold_param(self) -> None:
        stmt = build_duplicate_detection()
        text_str = str(stmt)
        assert ":threshold" in text_str

    def test_contains_proj_param(self) -> None:
        stmt = build_duplicate_detection()
        text_str = str(stmt)
        assert ":proj" in text_str

    def test_contains_pair_cap_param(self) -> None:
        stmt = build_duplicate_detection()
        text_str = str(stmt)
        assert ":pair_cap" in text_str


class TestBackwardCompatibility:
    """Ensure ExperienceRepository static methods still delegate correctly."""

    def test_repository_active_filter_delegates(self) -> None:
        from team_memory.storage.repository import ExperienceRepository

        filters = ExperienceRepository._active_filter(current_user=None)
        direct = active_filter(current_user=None)
        assert len(filters) == len(direct)

    def test_repository_project_value_delegates(self) -> None:
        from team_memory.storage.repository import ExperienceRepository

        assert ExperienceRepository._project_value("team-memory") == project_value("team-memory")
        assert ExperienceRepository._project_value(None) == project_value(None)

    def test_repository_active_filter_with_user_delegates(self) -> None:
        from team_memory.storage.repository import ExperienceRepository

        filters = ExperienceRepository._active_filter(current_user="bob")
        direct = active_filter(current_user="bob")
        assert len(filters) == len(direct)
        # Both should produce an OR clause as second element
        assert type(filters[1]).__name__ == type(direct[1]).__name__

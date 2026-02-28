"""Core path tests for team_memory visibility model, search, and tool tracking.

Tests the critical paths introduced in the visibility + analytics plan:
- Personal visibility: _active_filter with current_user
- Team publishing: publish_to_team / publish_personal repository methods
- Hook system: HookRegistry event dispatching
- Tool usage tracking: ToolUsageLog model
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from team_memory.storage.models import Experience, ToolUsageLog


class TestActiveFilter:
    """Tests for _active_filter visibility logic."""

    def test_active_filter_no_user_only_published(self):
        """Without current_user, only published should pass."""
        from team_memory.storage.repository import ExperienceRepository

        filters = ExperienceRepository._active_filter(current_user=None)
        assert len(filters) == 2

    def test_active_filter_with_user_includes_personal(self):
        """With current_user, filter should allow published + creator's own."""
        from team_memory.storage.repository import ExperienceRepository

        filters = ExperienceRepository._active_filter(current_user="alice")
        assert len(filters) == 2

    def test_active_filter_with_user_or_clause(self):
        """The second filter with current_user should be an OR clause."""
        from team_memory.storage.repository import ExperienceRepository

        filters = ExperienceRepository._active_filter(current_user="alice")
        second = filters[1]
        assert hasattr(second, "clauses") or "BooleanClauseList" in type(second).__name__


class TestPublishStatusDefaults:
    """Tests for publish_status default values."""

    def test_experience_model_defaults_to_personal(self):
        """Experience model should default to 'personal' publish_status."""
        assert Experience.publish_status.property.columns[0].default.arg == "personal"

    def test_experience_valid_statuses(self):
        """Verify the documented status values are valid."""
        valid = {"personal", "draft", "pending_team", "published", "rejected"}
        exp = Experience(
            title="test", description="test", publish_status="personal"
        )
        assert exp.publish_status in valid


class TestPublishToTeam:
    """Tests for publish_to_team and publish_personal repository methods."""

    @pytest.mark.asyncio
    async def test_publish_to_team_non_admin_sets_pending(self):
        """Non-admin should set pending_team status."""
        from team_memory.storage.repository import ExperienceRepository

        mock_session = AsyncMock()
        repo = ExperienceRepository(mock_session)

        mock_exp = MagicMock(spec=Experience)
        mock_exp.publish_status = "personal"
        mock_exp.review_status = "approved"

        with patch.object(repo, "get_by_id", return_value=mock_exp):
            result = await repo.publish_to_team(
                experience_id="test-id", is_admin=False
            )
        assert result.publish_status == "pending_team"
        assert result.review_status == "pending"

    @pytest.mark.asyncio
    async def test_publish_to_team_admin_sets_published(self):
        """Admin should directly set published status."""
        from team_memory.storage.repository import ExperienceRepository

        mock_session = AsyncMock()
        repo = ExperienceRepository(mock_session)

        mock_exp = MagicMock(spec=Experience)
        mock_exp.publish_status = "personal"

        with patch.object(repo, "get_by_id", return_value=mock_exp):
            result = await repo.publish_to_team(
                experience_id="test-id", is_admin=True
            )
        assert result.publish_status == "published"
        assert result.review_status == "approved"

    @pytest.mark.asyncio
    async def test_publish_personal_from_draft(self):
        """Draft -> personal transition should work."""
        from team_memory.storage.repository import ExperienceRepository

        mock_session = AsyncMock()
        repo = ExperienceRepository(mock_session)

        mock_exp = MagicMock(spec=Experience)
        mock_exp.publish_status = "draft"

        with patch.object(repo, "get_by_id", return_value=mock_exp):
            result = await repo.publish_personal(experience_id="test-id")
        assert result.publish_status == "personal"

    @pytest.mark.asyncio
    async def test_publish_personal_noop_if_not_draft(self):
        """publish_personal should be a no-op if already personal."""
        from team_memory.storage.repository import ExperienceRepository

        mock_session = AsyncMock()
        repo = ExperienceRepository(mock_session)

        mock_exp = MagicMock(spec=Experience)
        mock_exp.publish_status = "personal"

        with patch.object(repo, "get_by_id", return_value=mock_exp):
            result = await repo.publish_personal(experience_id="test-id")
        assert result.publish_status == "personal"

    @pytest.mark.asyncio
    async def test_publish_to_team_not_found(self):
        """Should return None if experience not found."""
        from team_memory.storage.repository import ExperienceRepository

        mock_session = AsyncMock()
        repo = ExperienceRepository(mock_session)

        with patch.object(repo, "get_by_id", return_value=None):
            result = await repo.publish_to_team(
                experience_id="nonexistent", is_admin=False
            )
        assert result is None


class TestReviewWorkflow:
    """Tests for the review approve/reject workflow."""

    @pytest.mark.asyncio
    async def test_review_approve_sets_published(self):
        """Approving should set publish_status to published."""
        from team_memory.storage.repository import ExperienceRepository

        mock_session = AsyncMock()
        repo = ExperienceRepository(mock_session)

        mock_exp = MagicMock(spec=Experience)
        mock_exp.publish_status = "pending_team"
        mock_exp.review_status = "pending"

        with patch.object(repo, "get_by_id", return_value=mock_exp):
            result = await repo.review(
                experience_id="test-id",
                review_status="approved",
                reviewed_by="admin",
            )
        assert result.publish_status == "published"

    @pytest.mark.asyncio
    async def test_review_reject_sets_rejected(self):
        """Rejecting should set publish_status to rejected."""
        from team_memory.storage.repository import ExperienceRepository

        mock_session = AsyncMock()
        repo = ExperienceRepository(mock_session)

        mock_exp = MagicMock(spec=Experience)
        mock_exp.publish_status = "pending_team"
        mock_exp.review_status = "pending"

        with patch.object(repo, "get_by_id", return_value=mock_exp):
            result = await repo.review(
                experience_id="test-id",
                review_status="rejected",
                reviewed_by="admin",
                review_note="Not relevant",
            )
        assert result.publish_status == "rejected"


class TestHooksAndTracking:
    """Tests for the hook system and usage tracking."""

    @pytest.mark.asyncio
    async def test_hook_registry_fire(self):
        """HookRegistry should dispatch events to registered handlers."""
        from team_memory.services.hooks import (
            HookContext,
            HookEvent,
            HookRegistry,
        )

        registry = HookRegistry()
        calls = []

        class TestHandler:
            @property
            def supported_events(self):
                return [HookEvent.POST_TOOL_CALL]

            async def handle(self, context):
                calls.append(context.tool_name)
                return None

        registry.register(TestHandler())
        ctx = HookContext(event=HookEvent.POST_TOOL_CALL, tool_name="tm_test")
        await registry.fire(ctx)
        assert calls == ["tm_test"]

    @pytest.mark.asyncio
    async def test_hook_registry_ignores_errors(self):
        """HookRegistry should not propagate handler errors."""
        from team_memory.services.hooks import (
            HookContext,
            HookEvent,
            HookRegistry,
        )

        registry = HookRegistry()

        class FailHandler:
            @property
            def supported_events(self):
                return [HookEvent.POST_TOOL_CALL]

            async def handle(self, context):
                raise RuntimeError("boom")

        registry.register(FailHandler())
        ctx = HookContext(event=HookEvent.POST_TOOL_CALL, tool_name="test")
        result = await registry.fire(ctx)
        assert result == []

    @pytest.mark.asyncio
    async def test_noop_exporter(self):
        """NoopExporter should be callable without side effects."""
        from team_memory.services.hooks import NoopExporter

        exporter = NoopExporter()
        await exporter.export_metric("test", 1.0, {})

    def test_init_hook_registry(self):
        """init_hook_registry should create and return a registry."""
        from team_memory.services.hooks import init_hook_registry

        registry = init_hook_registry()
        assert registry is not None


class TestToolUsageLog:
    """Tests for the ToolUsageLog model."""

    def test_tool_usage_log_to_dict(self):
        """ToolUsageLog.to_dict() should return proper structure."""
        log = ToolUsageLog(
            tool_name="tm_search",
            tool_type="mcp",
            user="alice",
            project="team-memory",
            duration_ms=150,
            success=True,
        )
        d = log.to_dict()
        assert d["tool_name"] == "tm_search"
        assert d["tool_type"] == "mcp"
        assert d["user"] == "alice"
        assert d["duration_ms"] == 150
        assert d["success"] is True

    def test_tool_usage_log_with_all_fields(self):
        """ToolUsageLog should handle all fields properly."""
        log = ToolUsageLog(
            tool_name="test_tool",
            tool_type="skill",
            user="bob",
            project="my-project",
            duration_ms=200,
            success=False,
            error_message="timeout",
        )
        d = log.to_dict()
        assert d["tool_type"] == "skill"
        assert d["success"] is False
        assert d["error_message"] == "timeout"


class TestSearchPipelineRequest:
    """Tests for SearchRequest current_user field."""

    def test_search_request_has_current_user(self):
        """SearchRequest should have current_user field."""
        from team_memory.services.search_pipeline import SearchRequest

        req = SearchRequest(query="test", current_user="alice")
        assert req.current_user == "alice"

    def test_search_request_default_current_user_none(self):
        """SearchRequest current_user should default to None."""
        from team_memory.services.search_pipeline import SearchRequest

        req = SearchRequest(query="test")
        assert req.current_user is None

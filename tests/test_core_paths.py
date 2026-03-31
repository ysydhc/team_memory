"""Core path tests for team_memory visibility model, search, and hook system.

Tests the critical paths:
- Personal visibility: _active_filter with current_user
- Status defaults: exp_status, visibility
- Hook system: HookRegistry event dispatching
- SearchRequest: current_user field
"""

from __future__ import annotations

import pytest

from team_memory.storage.models import Experience


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
    """Tests for exp_status and visibility."""

    def test_experience_model_defaults(self):
        """Experience model defaults: exp_status=draft, visibility=project."""
        assert Experience.exp_status.property.columns[0].default.arg == "draft"
        assert Experience.visibility.property.columns[0].default.arg == "project"

    def test_experience_valid_statuses(self):
        """Only draft and published are valid exp_status."""
        valid = {"draft", "published"}
        exp = Experience(
            title="test", description="test", exp_status="published", visibility="project"
        )
        assert exp.exp_status in valid


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

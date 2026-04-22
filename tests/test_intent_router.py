"""Tests for IntentRouter ABC + DefaultIntentRouter + SearchOrchestrator integration.

Covers:
- IntentResult dataclass creation
- DefaultIntentRouter.classify returns general
- DefaultIntentRouter.classify with context still returns general
- SearchOrchestrator accepts IntentRouter parameter
- SearchOrchestrator uses DefaultIntentRouter when none provided
- SearchOrchestrator passes intent through search flow
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from team_memory.services.intent_router import DefaultIntentRouter, IntentResult, IntentRouter
from team_memory.services.search_orchestrator import SearchOrchestrator

# ============================================================
# IntentResult dataclass
# ============================================================


class TestIntentResult:
    """Test IntentResult dataclass creation."""

    def test_create_with_intent_type(self):
        """IntentResult stores intent_type."""
        result = IntentResult(intent_type="factual")
        assert result.intent_type == "factual"
        assert result.params == {}

    def test_create_with_params(self):
        """IntentResult stores params dict."""
        result = IntentResult(
            intent_type="temporal",
            params={"time_range": "last_week"},
        )
        assert result.intent_type == "temporal"
        assert result.params == {"time_range": "last_week"}

    def test_params_default_is_independent(self):
        """Each IntentResult gets its own params dict (mutable default safety)."""
        r1 = IntentResult(intent_type="general")
        r2 = IntentResult(intent_type="general")
        r1.params["key"] = "value"
        assert "key" not in r2.params


# ============================================================
# DefaultIntentRouter
# ============================================================


class TestDefaultIntentRouter:
    """Test DefaultIntentRouter always returns general."""

    @pytest.mark.asyncio
    async def test_classify_returns_general(self):
        """DefaultIntentRouter.classify returns intent_type='general'."""
        router = DefaultIntentRouter()
        result = await router.classify("how to deploy?")
        assert result.intent_type == "general"
        assert result.params == {}

    @pytest.mark.asyncio
    async def test_classify_with_context_still_returns_general(self):
        """Even with context, DefaultIntentRouter still returns general."""
        router = DefaultIntentRouter()
        result = await router.classify(
            "why did the build fail?",
            context={"project": "team_doc"},
        )
        assert result.intent_type == "general"
        assert result.params == {}

    def test_is_intent_router_subclass(self):
        """DefaultIntentRouter is a subclass of IntentRouter."""
        assert issubclass(DefaultIntentRouter, IntentRouter)


# ============================================================
# IntentRouter ABC enforcement
# ============================================================


class TestIntentRouterABC:
    """Test IntentRouter cannot be instantiated directly."""

    def test_cannot_instantiate_abc(self):
        """IntentRouter ABC cannot be instantiated."""
        with pytest.raises(TypeError):
            IntentRouter()  # type: ignore[abstract]

    def test_custom_subclass_works(self):
        """A concrete subclass of IntentRouter can be instantiated."""

        class CustomRouter(IntentRouter):
            async def classify(self, query: str, context: dict | None = None) -> IntentResult:
                return IntentResult(intent_type="custom", params={"q": query})

        router = CustomRouter()
        assert isinstance(router, IntentRouter)


# ============================================================
# SearchOrchestrator integration
# ============================================================


class TestSearchOrchestratorIntentIntegration:
    """Test SearchOrchestrator integrates with IntentRouter."""

    def test_accepts_intent_router_parameter(self):
        """SearchOrchestrator.__init__ accepts intent_router."""
        mock_router = MagicMock(spec=IntentRouter)
        orchestrator = SearchOrchestrator(
            search_pipeline=None,
            embedding_provider=MagicMock(),
            db_url="sqlite://",
            intent_router=mock_router,
        )
        assert orchestrator._intent_router is mock_router

    def test_uses_default_intent_router_when_none_provided(self):
        """When no intent_router is provided, DefaultIntentRouter is used."""
        orchestrator = SearchOrchestrator(
            search_pipeline=None,
            embedding_provider=MagicMock(),
            db_url="sqlite://",
        )
        assert isinstance(orchestrator._intent_router, DefaultIntentRouter)

    @pytest.mark.asyncio
    async def test_intent_classify_called_during_search(self):
        """SearchOrchestrator.search() calls intent_router.classify()."""
        mock_router = MagicMock(spec=IntentRouter)
        mock_router.classify = AsyncMock(
            return_value=IntentResult(intent_type="factual", params={})
        )

        mock_pipeline = MagicMock()
        mock_pipeline_result = MagicMock()
        mock_pipeline_result.results = [
            {"id": "abc", "title": "Result 1", "score": 0.9},
        ]
        mock_pipeline_result.search_type = "hybrid"
        mock_pipeline.search = AsyncMock(return_value=mock_pipeline_result)

        orchestrator = SearchOrchestrator(
            search_pipeline=mock_pipeline,
            embedding_provider=MagicMock(),
            db_url="sqlite://",
            intent_router=mock_router,
        )

        mock_session = AsyncMock()
        mock_repo = MagicMock()
        mock_repo.increment_use_count = AsyncMock()
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
                query="test query",
                max_results=5,
                user_name="tester",
            )

        # Verify intent router was called with the query
        mock_router.classify.assert_awaited_once_with("test query")
        # Verify search still works
        assert len(result.results) == 1

    @pytest.mark.asyncio
    async def test_intent_stored_on_result(self):
        """SearchOrchestrator stores intent_type from classify on the result."""
        mock_router = MagicMock(spec=IntentRouter)
        mock_router.classify = AsyncMock(
            return_value=IntentResult(intent_type="exploratory", params={})
        )

        mock_pipeline = MagicMock()
        mock_pipeline_result = MagicMock()
        mock_pipeline_result.results = [
            {"id": "abc", "title": "Result 1", "score": 0.9},
        ]
        mock_pipeline_result.search_type = "hybrid"
        mock_pipeline.search = AsyncMock(return_value=mock_pipeline_result)

        orchestrator = SearchOrchestrator(
            search_pipeline=mock_pipeline,
            embedding_provider=MagicMock(),
            db_url="sqlite://",
            intent_router=mock_router,
        )

        mock_session = AsyncMock()
        mock_repo = MagicMock()
        mock_repo.increment_use_count = AsyncMock()
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
                query="explore options",
                max_results=5,
            )

        assert result.intent_type == "exploratory"

    @pytest.mark.asyncio
    async def test_default_intent_stored_when_no_router(self):
        """When no intent_router provided, result has intent_type='general'."""
        mock_pipeline = MagicMock()
        mock_pipeline_result = MagicMock()
        mock_pipeline_result.results = [
            {"id": "abc", "title": "Result 1", "score": 0.9},
        ]
        mock_pipeline_result.search_type = "hybrid"
        mock_pipeline.search = AsyncMock(return_value=mock_pipeline_result)

        orchestrator = SearchOrchestrator(
            search_pipeline=mock_pipeline,
            embedding_provider=MagicMock(),
            db_url="sqlite://",
        )

        mock_session = AsyncMock()
        mock_repo = MagicMock()
        mock_repo.increment_use_count = AsyncMock()
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
                query="test query",
                max_results=5,
            )

        assert result.intent_type == "general"

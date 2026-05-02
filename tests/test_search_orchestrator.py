"""Tests for SearchOrchestrator.

Validates that search delegates to the pipeline and that cache invalidation works.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from team_memory.services.search_orchestrator import SearchOrchestrator


class TestSearchOrchestratorSearch:
    """Test SearchOrchestrator.search delegation."""

    @pytest.mark.asyncio
    async def test_delegates_to_pipeline(self):
        """When pipeline is configured, search delegates to it."""
        mock_pipeline = MagicMock()
        mock_pipeline_result = MagicMock()
        mock_pipeline_result.results = [
            {"id": "abc", "title": "Result 1", "score": 0.9},
        ]
        mock_pipeline_result.search_type = "hybrid"
        mock_pipeline.search = AsyncMock(return_value=mock_pipeline_result)

        mock_embedding = MagicMock()
        orchestrator = SearchOrchestrator(
            search_pipeline=mock_pipeline,
            embedding_provider=mock_embedding,
            db_url="sqlite://",
        )

        mock_session = AsyncMock()
        mock_repo = MagicMock()
        mock_repo.increment_recall_count = AsyncMock()

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

            o = await orchestrator.search(
                query="test query",
                max_results=5,
                user_name="tester",
            )

        assert len(o.results) == 1
        assert o.results[0]["title"] == "Result 1"
        mock_pipeline.search.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_legacy_fallback_when_no_pipeline(self):
        """When no pipeline is configured, falls back to legacy search."""
        mock_embedding = MagicMock()
        mock_embedding.encode_single = AsyncMock(return_value=[0.1, 0.2, 0.3])

        orchestrator = SearchOrchestrator(
            search_pipeline=None,
            embedding_provider=mock_embedding,
            db_url="sqlite://",
        )

        mock_session = AsyncMock()
        mock_repo = MagicMock()
        mock_repo.search_by_vector = AsyncMock(
            return_value=[{"id": "xyz", "title": "Legacy result"}]
        )

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

            o = await orchestrator.search(
                query="test",
                max_results=3,
            )

        assert len(o.results) == 1
        assert o.results[0]["title"] == "Legacy result"
        assert o.reranked is False
        mock_embedding.encode_single.assert_awaited_once()
        mock_repo.search_by_vector.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_legacy_fallback_to_fts_on_vector_error(self):
        """When vector search fails, falls back to FTS."""
        mock_embedding = MagicMock()
        mock_embedding.encode_single = AsyncMock(side_effect=RuntimeError("embedding down"))

        orchestrator = SearchOrchestrator(
            search_pipeline=None,
            embedding_provider=mock_embedding,
            db_url="sqlite://",
        )

        mock_session = AsyncMock()
        mock_repo = MagicMock()
        mock_repo.search_by_fts = AsyncMock(return_value=[{"id": "fts1", "title": "FTS result"}])

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

            o = await orchestrator.search(query="fallback test")

        assert len(o.results) == 1
        assert o.results[0]["title"] == "FTS result"
        mock_repo.search_by_fts.assert_awaited_once()


class TestSearchOrchestratorInvalidateCache:
    """Test SearchOrchestrator.invalidate_cache."""

    @pytest.mark.asyncio
    async def test_invalidate_cache_delegates_to_pipeline(self):
        """invalidate_cache calls pipeline.invalidate_cache when pipeline exists."""
        mock_pipeline = MagicMock()
        mock_pipeline.invalidate_cache = AsyncMock()

        orchestrator = SearchOrchestrator(
            search_pipeline=mock_pipeline,
            embedding_provider=MagicMock(),
            db_url="sqlite://",
        )

        await orchestrator.invalidate_cache()
        mock_pipeline.invalidate_cache.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_invalidate_cache_noop_without_pipeline(self):
        """invalidate_cache is a no-op when no pipeline is configured."""
        orchestrator = SearchOrchestrator(
            search_pipeline=None,
            embedding_provider=MagicMock(),
            db_url="sqlite://",
        )

        # Should not raise
        await orchestrator.invalidate_cache()

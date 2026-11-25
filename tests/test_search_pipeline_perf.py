"""Unit tests for SearchPipeline performance optimizations, fallback-only query expansion,
embedding timeout degradation to FTS+Graph, full stage_metrics instrumentation, and concurrency.
"""

from __future__ import annotations

import asyncio
import time
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from team_memory.config import CacheConfig, LLMConfig, RetrievalConfig, SearchConfig
from team_memory.services.memory.ops.recall import op_recall
from team_memory.services.search.cache import SearchCache
from team_memory.services.search.search_orchestrator import (
    OrchestratedSearchResult,
    SearchOrchestrator,
)
from team_memory.services.search.search_pipeline import (
    SearchPipeline,
    SearchRequest,
    SearchResultItem,
)


def _make_item(eid: str, title: str, score: float, similarity: float = 0.0, source_type: str = "hybrid"):
    return SearchResultItem(
        data={"id": eid, "title": title, "description": title, "solution": "sol"},
        score=score,
        similarity=similarity,
        source_type=source_type,
    )


class TestStageMetricsCompleteness:
    """Verify all sub-stage timings are recorded in stage_metrics."""

    @pytest.mark.asyncio
    async def test_pipeline_records_all_stage_metrics(self):
        embedding = AsyncMock()
        embedding.encode_single = AsyncMock(return_value=[0.1] * 8)

        pipeline = SearchPipeline(
            embedding_provider=embedding,
            search_config=SearchConfig(
                mode="hybrid",
                adaptive_filter=False,
                enable_graph=True,
                enable_topic=False,
            ),
            retrieval_config=RetrievalConfig(),
            cache_config=CacheConfig(enabled=False),
            llm_config=LLMConfig(),
        )
        pipeline._retrieve_and_fuse = AsyncMock(
            return_value=[
                _make_item("11111111-1111-1111-1111-111111111111", "T1", 0.03, similarity=0.85),
                _make_item("22222222-2222-2222-2222-222222222222", "T2", 0.02, similarity=0.75),
                _make_item("33333333-3333-3333-3333-333333333333", "T3", 0.01, similarity=0.70),
            ]
        )

        mock_session = AsyncMock()
        with patch(
            "team_memory.services.search.search_pipeline.ArchiveRepository"
        ) as mock_arch_cls:
            mock_arch_cls.return_value.get_archive_ids_for_experiences = AsyncMock(return_value={})
            res = await pipeline.search(mock_session, SearchRequest(query="redis cache"))

        expected_keys = {
            "cache_check_ms",
            "query_expansion_ms",
            "graph_expand_ms",
            "embedding_ms",
            "parallel_fetch_wall_ms",
            "retrieve_fuse_ms",
            "popularity_boost_ms",
            "adaptive_filter_ms",
            "post_process_ms",
            "total_ms",
        }
        assert expected_keys.issubset(set(res.stage_metrics.keys()))
        assert res.stage_counts["query_expansion_triggered"] == 0
        assert res.stage_counts["embedding_degraded"] == 0
        assert res.degraded is False
        assert res.degraded_reason is None

    @pytest.mark.asyncio
    async def test_orchestrator_attaches_intent_route_ms_and_degraded(self):
        mock_pipeline = MagicMock()
        mock_pipeline_result = MagicMock()
        mock_pipeline_result.results = [{"id": "abc", "title": "R1", "score": 0.5}]
        mock_pipeline_result.search_type = "fts"
        mock_pipeline_result.reranked = False
        mock_pipeline_result.cached = False
        mock_pipeline_result.stage_metrics = {"total_ms": 120, "embedding_ms": 100}
        mock_pipeline_result.stage_counts = {"embedding_degraded": 1}
        mock_pipeline_result.degraded = True
        mock_pipeline_result.degraded_reason = "embedding_timeout"
        mock_pipeline.search = AsyncMock(return_value=mock_pipeline_result)

        orch = SearchOrchestrator(
            search_pipeline=mock_pipeline,
            embedding_provider=MagicMock(),
            db_url="sqlite://",
        )
        mock_session = AsyncMock()
        mock_repo = MagicMock()
        mock_repo.increment_recall_count = AsyncMock()

        with (
            patch("team_memory.storage.database.get_session") as mock_gs,
            patch(
                "team_memory.services.search.search_orchestrator.ExperienceRepository",
                return_value=mock_repo,
            ),
        ):
            mock_gs.return_value.__aenter__ = AsyncMock(return_value=mock_session)
            mock_gs.return_value.__aexit__ = AsyncMock(return_value=False)
            out = await orch.search(query="test")

        assert "intent_route_ms" in out.stage_metrics
        assert out.degraded is True
        assert out.degraded_reason == "embedding_timeout"


class TestFallbackOnlyQueryExpansion:
    """Verify query_expansion skips LLM on high-confidence 1st pass and triggers on weak 1st pass."""

    @pytest.mark.asyncio
    async def test_skips_llm_expansion_when_first_pass_is_strong(self):
        embedding = AsyncMock()
        embedding.encode_single = AsyncMock(return_value=[0.1] * 8)
        pipeline = SearchPipeline(
            embedding_provider=embedding,
            search_config=SearchConfig(
                mode="hybrid",
                adaptive_filter=False,
                enable_graph=False,
                query_expansion_enabled=True,
                query_expansion_mode="fallback",
                query_expansion_fallback_min_results=3,
                query_expansion_fallback_min_similarity=0.65,
            ),
            retrieval_config=RetrievalConfig(),
            cache_config=CacheConfig(enabled=False),
            llm_config=LLMConfig(),
        )
        pipeline._retrieve_and_fuse = AsyncMock(
            return_value=[
                _make_item("a", "A", 0.03, similarity=0.82),
                _make_item("b", "B", 0.02, similarity=0.71),
                _make_item("c", "C", 0.01, similarity=0.66),
            ]
        )

        with patch(
            "team_memory.services.search.search_pipeline._llm_expand_query",
            new_callable=AsyncMock,
            return_value="extra keywords",
        ) as mock_expand:
            res = await pipeline.search(AsyncMock(), SearchRequest(query="redis lock"))

        mock_expand.assert_not_awaited()
        assert res.stage_counts["query_expansion_triggered"] == 0
        assert len(res.results) == 3

    @pytest.mark.asyncio
    async def test_triggers_llm_expansion_when_first_pass_is_empty_or_weak(self):
        embedding = AsyncMock()
        embedding.encode_single = AsyncMock(return_value=[0.1] * 8)
        pipeline = SearchPipeline(
            embedding_provider=embedding,
            search_config=SearchConfig(
                mode="hybrid",
                adaptive_filter=False,
                enable_graph=False,
                query_expansion_enabled=True,
                query_expansion_mode="fallback",
                query_expansion_fallback_min_results=3,
                query_expansion_fallback_min_similarity=0.65,
            ),
            retrieval_config=RetrievalConfig(),
            cache_config=CacheConfig(enabled=False),
            llm_config=LLMConfig(),
        )
        pipeline._retrieve_and_fuse = AsyncMock(
            side_effect=[
                [_make_item("a", "Weak", 0.01, similarity=0.52)],  # 1st pass: < 3 and < 0.65
                [
                    _make_item("b", "Expanded Hit 1", 0.04, similarity=0.80),
                    _make_item("c", "Expanded Hit 2", 0.03, similarity=0.75),
                ],
            ]
        )

        with patch(
            "team_memory.services.search.search_pipeline._llm_expand_query",
            new_callable=AsyncMock,
            return_value="redis distributed lock",
        ) as mock_expand:
            res = await pipeline.search(AsyncMock(), SearchRequest(query="lock"))

        mock_expand.assert_awaited_once()
        assert res.stage_counts["query_expansion_triggered"] == 1
        ids = [r["id"] for r in res.results]
        assert "b" in ids and "c" in ids and "a" in ids


class TestEmbeddingTimeoutDegradation:
    """Verify hard timeout on embedding degrades gracefully to FTS + Entity-Graph."""

    @pytest.mark.asyncio
    async def test_embedding_timeout_degrades_to_fts_plus_graph(self):
        async def slow_encode(_text: str):
            await asyncio.sleep(5.0)
            return [0.1] * 8

        embedding = MagicMock()
        embedding.encode_single = AsyncMock(side_effect=slow_encode)

        pipeline = SearchPipeline(
            embedding_provider=embedding,
            search_config=SearchConfig(
                mode="hybrid",
                adaptive_filter=False,
                enable_graph=True,
                enable_topic=False,
                embedding_timeout_seconds=0.05,
            ),
            retrieval_config=RetrievalConfig(),
            cache_config=CacheConfig(enabled=True),
            db_url="postgresql+asyncpg://localhost/test_db",
        )

        fts_item = _make_item("fts-1", "FTS Hit", 0.6, source_type="fts")
        graph_item = _make_item("graph-1", "Graph Hit", 0.8, similarity=0.8, source_type="graph")

        pipeline._fts_search = AsyncMock(return_value=[fts_item])
        pipeline._graph_search = AsyncMock(return_value=[graph_item])

        with patch(
            "team_memory.services.entity.entity_search.find_weighted_experiences_by_entities",
            new_callable=AsyncMock,
            return_value=[("graph-1", 0.8)],
        ):
            t0 = time.monotonic()
            res = await pipeline.search(AsyncMock(), SearchRequest(query="pgvector timeout"))
            elapsed_ms = (time.monotonic() - t0) * 1000

        # Must finish well under the 5s slow_encode (bounded by 50ms timeout)
        assert elapsed_ms < 1000
        assert res.degraded is True
        assert res.degraded_reason == "embedding_timeout"
        assert res.stage_counts["embedding_degraded"] == 1
        assert res.stage_counts["fts_count"] == 1
        assert res.stage_counts["graph_count"] == 1
        ids = [r["id"] for r in res.results]
        assert "fts-1" in ids
        assert "graph-1" in ids

    @pytest.mark.asyncio
    async def test_cache_releases_lock_on_embedding_timeout(self):
        cache = SearchCache(enabled=True)

        async def slow_encode(_text: str):
            await asyncio.sleep(2.0)
            return [0.1]

        provider = MagicMock()
        provider.encode_single = AsyncMock(side_effect=slow_encode)

        with pytest.raises(asyncio.TimeoutError):
            await cache.get_or_compute_embedding("leak-check", provider, timeout_seconds=0.03)

        assert len(cache._locks) == 0


class TestParallelExecution:
    """Verify independent stages run concurrently via asyncio.gather."""

    @pytest.mark.asyncio
    async def test_op_recall_runs_search_and_personal_memory_in_parallel(self):
        async def slow_search(**_kwargs):
            await asyncio.sleep(0.12)
            return OrchestratedSearchResult(
                results=[{"id": "1", "title": "Doc 1", "score": 0.9}],
                stage_metrics={"total_ms": 120},
                degraded=True,
                degraded_reason="embedding_timeout",
            )

        async def slow_profile(**_kwargs):
            await asyncio.sleep(0.12)
            return {"static": ["likes pytest"], "dynamic": []}

        mock_orch = MagicMock()
        mock_orch.search = AsyncMock(side_effect=slow_search)
        mock_orch._search_pipeline = None

        mock_settings = MagicMock()
        mock_settings.mcp.profile_max_strings_per_side = 8
        mock_settings.mcp.profile_max_items = 8
        mock_settings.mcp.profile_max_tokens = 400
        mock_settings.search.embedding_timeout_seconds = 2.5

        captured_log_kwargs: dict = {}

        async def fake_write_log(**kwargs):
            captured_log_kwargs.update(kwargs)
            return "log-1"

        with (
            patch("team_memory.services.memory.personal_memory.PersonalMemoryService") as mock_pm_cls,
            patch("team_memory.services.memory.memory_operations.write_search_log", side_effect=fake_write_log),
        ):
            mock_pm_cls.return_value.build_profile_for_user = AsyncMock(side_effect=slow_profile)
            t0 = time.monotonic()
            res = await op_recall(
                "alice",
                search_orchestrator=mock_orch,
                db_url="sqlite://",
                settings=mock_settings,
                embedding_provider=MagicMock(),
                query="parallel test",
                include_user_profile=True,
                include_archives=False,
            )
            elapsed = time.monotonic() - t0

        # Sequential would be >= 0.24s; parallel should be ~0.12s
        assert elapsed < 0.21
        assert res["profile"] == {"static": ["likes pytest"], "dynamic": []}
        assert res["degraded"] is True
        assert res["degraded_reason"] == "embedding_timeout"
        assert "personal_memory_ms" in captured_log_kwargs["stage_metrics"]
        assert captured_log_kwargs["stage_metrics"]["personal_memory_ms"] >= 100

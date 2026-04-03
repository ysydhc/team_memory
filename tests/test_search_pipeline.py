"""Tests for the search pipeline (hybrid search, RRF fusion, adaptive filter)."""

import time
from unittest.mock import AsyncMock, patch

import pytest

from team_memory.config import (
    CacheConfig,
    LLMConfig,
    RetrievalConfig,
    SearchConfig,
)
from team_memory.services.cache import SearchCache
from team_memory.services.search_pipeline import (
    _EXPANSION_CACHE_TTL,
    SearchPipeline,
    SearchResultItem,
    _expansion_cache,
    _expansion_cache_clear,
    _llm_expand_query,
)


def _make_pipeline(
    search_mode="hybrid",
    adaptive_filter=True,
    cache_enabled=False,
    max_tokens=None,
) -> SearchPipeline:
    """Create a SearchPipeline with mock dependencies."""
    embedding = AsyncMock()
    embedding.encode_single = AsyncMock(return_value=[0.1] * 768)

    return SearchPipeline(
        embedding_provider=embedding,
        search_config=SearchConfig(
            mode=search_mode,
            adaptive_filter=adaptive_filter,
        ),
        retrieval_config=RetrievalConfig(max_tokens=max_tokens),
        cache_config=CacheConfig(enabled=cache_enabled),
        llm_config=LLMConfig(),
    )


# ======================== RRF Fusion ========================


class TestRRFFusion:
    def test_basic_rrf_fusion(self):
        """RRF should merge results from two sources."""
        pipeline = _make_pipeline()

        vector_results = [
            SearchResultItem(
                data={"id": "a", "title": "Result A"},
                score=0.9,
                similarity=0.9,
            ),
            SearchResultItem(
                data={"id": "b", "title": "Result B"},
                score=0.7,
                similarity=0.7,
            ),
        ]
        fts_results = [
            SearchResultItem(
                data={"id": "b", "title": "Result B"},
                score=0.5,
                fts_rank=0.5,
            ),
            SearchResultItem(
                data={"id": "c", "title": "Result C"},
                score=0.3,
                fts_rank=0.3,
            ),
        ]

        fused = pipeline._rrf_fuse(vector_results, fts_results, 10)

        # "b" should get the highest score (appears in both)
        ids = [r.data["id"] for r in fused]
        assert "b" in ids
        assert "a" in ids
        assert "c" in ids

    def test_rrf_with_empty_vector(self):
        """RRF should work with empty vector results."""
        pipeline = _make_pipeline()
        fts_results = [
            SearchResultItem(
                data={"id": "a"},
                score=0.5,
                fts_rank=0.5,
            )
        ]
        fused = pipeline._rrf_fuse([], fts_results, 10)
        assert len(fused) == 1

    def test_rrf_with_empty_fts(self):
        """RRF should work with empty FTS results."""
        pipeline = _make_pipeline()
        vector_results = [
            SearchResultItem(
                data={"id": "a"},
                score=0.9,
                similarity=0.9,
            )
        ]
        fused = pipeline._rrf_fuse(vector_results, [], 10)
        assert len(fused) == 1


# ======================== Adaptive Filter ========================


class TestAdaptiveFilter:
    def test_filters_low_scores(self):
        """Should filter results below dynamic threshold."""
        pipeline = _make_pipeline(adaptive_filter=True)
        # Use scores with small gaps between consecutive items
        # so elbow detection doesn't trigger, but dynamic threshold does
        candidates = [
            SearchResultItem(data={"id": "a"}, score=1.0),
            SearchResultItem(data={"id": "b"}, score=0.9),
            SearchResultItem(data={"id": "c"}, score=0.85),
            SearchResultItem(data={"id": "d"}, score=0.4),  # Below threshold (0.4 < 1.0 * 0.6)
        ]
        filtered = pipeline._apply_adaptive_filter(candidates)
        ids = [r.data["id"] for r in filtered]
        assert "a" in ids
        assert "b" in ids
        assert "c" in ids
        # "d" should be filtered (0.4 < 1.0 * 0.6 = 0.6)
        assert "d" not in ids

    def test_elbow_detection(self):
        """Should cut at large score gaps."""
        pipeline = _make_pipeline(adaptive_filter=True)
        pipeline._search_config.score_gap_threshold = 0.3

        candidates = [
            SearchResultItem(data={"id": "a"}, score=1.0),
            SearchResultItem(data={"id": "b"}, score=0.95),
            SearchResultItem(data={"id": "c"}, score=0.5),  # Big gap from 0.95
        ]
        filtered = pipeline._apply_adaptive_filter(candidates)
        assert len(filtered) == 2

    def test_empty_candidates(self):
        pipeline = _make_pipeline()
        assert pipeline._apply_adaptive_filter([]) == []

    def test_single_candidate(self):
        pipeline = _make_pipeline()
        candidates = [SearchResultItem(data={"id": "a"}, score=1.0)]
        filtered = pipeline._apply_adaptive_filter(candidates)
        assert len(filtered) == 1


# ======================== Confidence Labeling ========================


class TestConfidenceLabeling:
    def test_labels_high_medium_low(self):
        candidates = [
            SearchResultItem(data={"id": "a"}, score=1.0),
            SearchResultItem(data={"id": "b"}, score=0.7),
            SearchResultItem(data={"id": "c"}, score=0.3),
        ]
        labeled = SearchPipeline._label_confidence(candidates)
        assert labeled[0].confidence == "high"
        assert labeled[1].confidence == "medium"
        assert labeled[2].confidence == "low"

    def test_all_same_score(self):
        candidates = [
            SearchResultItem(data={"id": "a"}, score=1.0),
            SearchResultItem(data={"id": "b"}, score=1.0),
        ]
        labeled = SearchPipeline._label_confidence(candidates)
        assert all(r.confidence == "high" for r in labeled)

    def test_empty(self):
        assert SearchPipeline._label_confidence([]) == []


class TestCacheKeyIsolation:
    def test_cache_key_includes_project(self):
        key1 = SearchCache._make_key("same query", ["python"], project="proj-a")
        key2 = SearchCache._make_key("same query", ["python"], project="proj-b")
        assert key1 != key2


# ======================== LLM Expansion Cache ========================


class TestLLMExpansionCache:
    """Tests for caching of LLM query expansion results."""

    def setup_method(self):
        """Clear the module-level expansion cache before each test."""
        _expansion_cache_clear()

    def teardown_method(self):
        """Clean up after each test."""
        _expansion_cache_clear()

    @pytest.mark.asyncio
    async def test_empty_query_returns_none(self):
        result = await _llm_expand_query("", object(), 3.0)
        assert result is None

    @pytest.mark.asyncio
    async def test_none_llm_config_returns_none(self):
        result = await _llm_expand_query("some query", None, 3.0)
        assert result is None

    @pytest.mark.asyncio
    async def test_caches_expansion_result(self):
        """Same query should return cached result on second call."""
        mock_client = AsyncMock()
        mock_client.chat = AsyncMock(return_value="expanded keywords here")

        with patch("team_memory.services.llm_client.LLMClient") as mock_llm_cls:
            mock_llm_cls.from_config.return_value = mock_client

            llm_config = LLMConfig()

            result1 = await _llm_expand_query("database timeout", llm_config, 3.0)
            assert result1 == "expanded keywords here"
            assert mock_client.chat.call_count == 1

            # Second call should use cache
            result2 = await _llm_expand_query("database timeout", llm_config, 3.0)
            assert result2 == "expanded keywords here"
            # LLM should NOT have been called again
            assert mock_client.chat.call_count == 1

    @pytest.mark.asyncio
    async def test_cache_key_normalized(self):
        """Queries differing only in case/whitespace should share cache."""
        mock_client = AsyncMock()
        mock_client.chat = AsyncMock(return_value="expanded")

        with patch("team_memory.services.llm_client.LLMClient") as mock_llm_cls:
            mock_llm_cls.from_config.return_value = mock_client

            llm_config = LLMConfig()

            await _llm_expand_query("  Database Timeout  ", llm_config, 3.0)
            result = await _llm_expand_query("database timeout", llm_config, 3.0)
            assert result == "expanded"
            assert mock_client.chat.call_count == 1

    @pytest.mark.asyncio
    async def test_cache_expires_after_ttl(self):
        """Cached result should be evicted after TTL expires."""
        mock_client = AsyncMock()
        mock_client.chat = AsyncMock(return_value="result")

        with patch("team_memory.services.llm_client.LLMClient") as mock_llm_cls:
            mock_llm_cls.from_config.return_value = mock_client

            llm_config = LLMConfig()

            await _llm_expand_query("query", llm_config, 3.0)
            assert mock_client.chat.call_count == 1

            # Manually expire the cache entry by backdating the timestamp
            cache_key = "query"
            expanded, _ts = _expansion_cache[cache_key]
            _expansion_cache[cache_key] = (expanded, time.monotonic() - _EXPANSION_CACHE_TTL - 1)

            # Should call LLM again
            await _llm_expand_query("query", llm_config, 3.0)
            assert mock_client.chat.call_count == 2

    @pytest.mark.asyncio
    async def test_llm_failure_does_not_cache(self):
        """Failed LLM calls should not populate the cache."""
        mock_client = AsyncMock()
        mock_client.chat = AsyncMock(side_effect=RuntimeError("LLM down"))

        with patch("team_memory.services.llm_client.LLMClient") as mock_llm_cls:
            mock_llm_cls.from_config.return_value = mock_client

            llm_config = LLMConfig()

            result = await _llm_expand_query("failing query", llm_config, 3.0)
            assert result is None
            assert "failing query" not in _expansion_cache

    @pytest.mark.asyncio
    async def test_clear_function_empties_cache(self):
        """_expansion_cache_clear should remove all entries."""
        _expansion_cache["test"] = ("expanded", time.monotonic())
        assert len(_expansion_cache) == 1
        _expansion_cache_clear()
        assert len(_expansion_cache) == 0

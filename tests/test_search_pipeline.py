"""Tests for the search pipeline (hybrid search, RRF fusion, adaptive filter)."""

from unittest.mock import AsyncMock

import pytest

from team_memory.config import (
    CacheConfig,
    LLMConfig,
    PageIndexLiteConfig,
    RetrievalConfig,
    SearchConfig,
)
from team_memory.reranker.noop_provider import NoopRerankerProvider
from team_memory.services.cache import SearchCache
from team_memory.services.search_pipeline import (
    SearchPipeline,
    SearchRequest,
    SearchResultItem,
)


def _make_pipeline(
    search_mode="hybrid",
    adaptive_filter=True,
    cache_enabled=False,
    max_tokens=None,
    pageindex_enabled=True,
) -> SearchPipeline:
    """Create a SearchPipeline with mock dependencies."""
    embedding = AsyncMock()
    embedding.encode_single = AsyncMock(return_value=[0.1] * 768)

    reranker = NoopRerankerProvider()

    return SearchPipeline(
        embedding_provider=embedding,
        reranker_provider=reranker,
        search_config=SearchConfig(
            mode=search_mode,
            adaptive_filter=adaptive_filter,
        ),
        retrieval_config=RetrievalConfig(max_tokens=max_tokens),
        cache_config=CacheConfig(enabled=cache_enabled),
        pageindex_lite_config=PageIndexLiteConfig(enabled=pageindex_enabled),
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


class TestPageIndexLite:
    @pytest.mark.asyncio
    async def test_pageindex_boosts_score_and_attaches_nodes(self):
        pipeline = _make_pipeline(pageindex_enabled=True)
        repo = AsyncMock()
        exp_id = "11111111-1111-1111-1111-111111111111"
        repo.search_tree_nodes = AsyncMock(
            return_value={
                exp_id: [
                    {
                        "id": "n1",
                        "path": "1.2",
                        "node_title": "Docker Networking",
                        "content_summary": "Port conflict notes",
                        "score": 0.6,
                    }
                ]
            }
        )
        candidates = [
            SearchResultItem(
                data={"id": exp_id, "title": "Result A"},
                score=1.0,
            )
        ]

        boosted, tree_hits = await pipeline._apply_pageindex_lite(
            repo=repo,
            query="docker port",
            candidates=candidates,
        )

        assert tree_hits == 1
        assert boosted[0].score > 1.0
        assert boosted[0].data["matched_nodes"][0]["path"] == "1.2"
        assert boosted[0].data["tree_score"] == 0.6

    def test_should_use_pageindex_respects_request_override(self):
        pipeline = _make_pipeline(pageindex_enabled=True)
        req = SearchRequest(query="q", use_pageindex_lite=False)
        assert pipeline._should_use_pageindex(req) is False


class TestCacheKeyIsolation:
    def test_cache_key_includes_project(self):
        key1 = SearchCache._make_key("same query", ["python"], project="proj-a")
        key2 = SearchCache._make_key("same query", ["python"], project="proj-b")
        assert key1 != key2

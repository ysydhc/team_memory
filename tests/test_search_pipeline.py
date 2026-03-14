"""Tests for the search pipeline (hybrid search, RRF fusion, adaptive filter)."""

from unittest.mock import AsyncMock

import pytest

from team_memory.config import (
    CacheConfig,
    FileLocationBindingConfig,
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
from team_memory.utils.location_fingerprint import LOCATION_SCORE_EXACT


def _make_pipeline(
    search_mode="hybrid",
    adaptive_filter=True,
    cache_enabled=False,
    max_tokens=None,
    pageindex_enabled=True,
    location_weight=0.15,
    file_location_config=None,
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
            location_weight=location_weight,
        ),
        retrieval_config=RetrievalConfig(max_tokens=max_tokens),
        cache_config=CacheConfig(enabled=cache_enabled),
        pageindex_lite_config=PageIndexLiteConfig(enabled=pageindex_enabled),
        llm_config=LLMConfig(),
        file_location_config=file_location_config,
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


# ======================== Location score (current_file_locations) ========================


class TestLocationScoreInPipeline:
    """Location step: batch list_bindings_by_paths, in-memory location_score, final_score boost."""

    @pytest.mark.asyncio
    async def test_location_boost_raises_candidate_score_and_sets_location_score(self):
        """With current_file_locations and mock bindings, candidate gets location_score and final_score boost."""
        pipeline = _make_pipeline(
            location_weight=0.15,
            file_location_config=FileLocationBindingConfig(
                file_location_ttl_days=30,
                file_location_refresh_on_access=False,
            ),
        )
        exp_id = "eid-1111-1111-1111-111111111111"
        candidates = [
            SearchResultItem(
                data={"id": exp_id, "title": "E1"},
                score=0.02,
            )
        ]
        request = SearchRequest(
            query="q",
            current_file_locations=[
                {"path": "foo.py", "start_line": 10, "end_line": 20}
            ],
        )
        repo = AsyncMock()
        repo.list_bindings_by_paths = AsyncMock(
            return_value={
                "foo.py": [
                    {
                        "id": "bid-1",
                        "experience_id": exp_id,
                        "path": "foo.py",
                        "start_line": 10,
                        "end_line": 20,
                        "content_fingerprint": None,
                    }
                ]
            }
        )
        session = AsyncMock()

        await pipeline._apply_location_boost(session, request, repo, candidates)

        assert candidates[0].data.get("location_score") == LOCATION_SCORE_EXACT
        rrf_score = 0.02
        expected_final = rrf_score + 0.15 * LOCATION_SCORE_EXACT
        assert candidates[0].score == rrf_score  # RRF score unchanged
        final_score = candidates[0].score + pipeline._search_config.location_weight * candidates[0].data["location_score"]
        assert abs(final_score - expected_final) < 1e-6

    @pytest.mark.asyncio
    async def test_location_step_skipped_when_no_current_file_locations(self):
        """When current_file_locations is None or empty, list_bindings_by_paths is not called."""
        pipeline = _make_pipeline(
            location_weight=0.15,
            file_location_config=FileLocationBindingConfig(file_location_ttl_days=30),
        )
        candidates = [
            SearchResultItem(data={"id": "e1", "title": "E1"}, score=0.02)
        ]
        repo = AsyncMock()
        repo.list_bindings_by_paths = AsyncMock(return_value={})
        session = AsyncMock()

        for req in (
            SearchRequest(query="q", current_file_locations=None),
            SearchRequest(query="q", current_file_locations=[]),
        ):
            repo.list_bindings_by_paths.reset_mock()
            await pipeline._apply_location_boost(session, req, repo, candidates)
            repo.list_bindings_by_paths.assert_not_called()
            assert candidates[0].data.get("location_score", 0) == 0

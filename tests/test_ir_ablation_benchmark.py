"""Unit tests for Standard IR Benchmark Suite, 4-Stage Ablation Configs, and RRF4/Boost Calibration."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from dataclasses import dataclass
from typing import Any

from team_memory.daemon.quality.search_eval import (  # noqa: E402
    ABLATION_CONFIGS,
    IR_BENCHMARK_SUITE,
    compute_fama_at_k,
    compute_mrr,
    compute_ndcg_at_k,
    compute_recall_at_k,
    evaluate_benchmark_suite,
)
from team_memory.services.search.scoring import (  # noqa: E402
    apply_exact_match_boost,
    rrf_fuse4,
)


@dataclass
class SearchResultItem:
    """Lightweight SearchResultItem compatible with scoring functions."""

    data: dict[str, Any]
    score: float = 0.0
    similarity: float = 0.0
    source_type: str = "vector"


class TestIRBenchmarkSuiteAndAblationConfigs:
    """Verify the 24-query Ground-Truth IR Benchmark Suite and 4-stage ablation matrix."""

    def test_ir_benchmark_suite_structure(self) -> None:
        assert len(IR_BENCHMARK_SUITE) == 24
        categories: dict[str, int] = {}
        multihop_count = 0
        seen_ids: set[str] = set()

        for case in IR_BENCHMARK_SUITE:
            qid = case["id"]
            assert qid not in seen_ids, f"Duplicate query id: {qid}"
            seen_ids.add(qid)
            cat = case["category"]
            categories[cat] = categories.get(cat, 0) + 1
            assert case.get("query"), f"Empty query in {qid}"
            assert case.get("relevant_ids"), f"Missing relevant_ids in {qid}"
            if case.get("is_multihop"):
                multihop_count += 1
                assert case.get("hops", 1) >= 2

        assert categories == {
            "semantic_paraphrase": 6,
            "exact_identifier": 6,
            "multihop_graph": 6,
            "cross_project_rerank": 6,
        }
        assert multihop_count == 6

    def test_ablation_configs_progression(self) -> None:
        assert len(ABLATION_CONFIGS) == 4
        keys = [c["key"] for c in ABLATION_CONFIGS]
        assert keys == [
            "L1_vector_only",
            "L2_hybrid_rrf2",
            "L3_hybrid_rrf4_graph",
            "L4_full_pipeline",
        ]
        # L1: vector only
        assert ABLATION_CONFIGS[0]["mode"] == "vector"
        assert not ABLATION_CONFIGS[0]["enable_graph"]
        # L2: hybrid RRF2 without graph
        assert ABLATION_CONFIGS[1]["mode"] == "hybrid"
        assert not ABLATION_CONFIGS[1]["enable_graph"]
        # L3: hybrid RRF4 + 2-hop graph PPR
        assert ABLATION_CONFIGS[2]["mode"] == "hybrid"
        assert ABLATION_CONFIGS[2]["enable_graph"]
        assert not ABLATION_CONFIGS[2]["enable_exact_boost"]
        # L4: full pipeline with exact boost, popularity, and rerank
        assert ABLATION_CONFIGS[3]["mode"] == "hybrid"
        assert ABLATION_CONFIGS[3]["enable_graph"]
        assert ABLATION_CONFIGS[3]["enable_exact_boost"]
        assert ABLATION_CONFIGS[3]["popularity_enabled"]
        assert ABLATION_CONFIGS[3]["rerank_enabled"]

    def test_evaluate_benchmark_suite_with_recall_at_1(self) -> None:
        cases = [
            {
                "retrieved_ids": ["doc-1", "doc-2"],
                "relevant_ids": ["doc-1"],
                "active_relevant_ids": ["doc-1"],
                "superseded_ids": [],
                "is_multihop": False,
            },
            {
                "retrieved_ids": ["doc-x", "doc-hop"],
                "relevant_ids": ["doc-hop"],
                "active_relevant_ids": ["doc-hop"],
                "superseded_ids": [],
                "is_multihop": True,
                "hops": 2,
            },
        ]
        metrics = evaluate_benchmark_suite(cases, k=5, include_recall_at_1=True)
        assert metrics["recall_at_1"] == 0.5
        assert metrics["avg_recall_at_5"] == 1.0
        assert metrics["mrr"] == 0.75
        assert metrics["fama_at_5"] == 1.0
        assert metrics["multihop_recall_at_5"] == 1.0
        assert 0.8 < metrics["avg_ndcg_at_5"] <= 1.0

    def test_single_query_ir_metrics(self) -> None:
        retrieved = ["d2", "d1", "d3"]
        relevant = {"d1"}
        assert compute_recall_at_k(retrieved, relevant, k=1) == 0.0
        assert compute_recall_at_k(retrieved, relevant, k=5) == 1.0
        assert compute_mrr(retrieved, relevant) == 0.5
        assert round(compute_ndcg_at_k(retrieved, relevant, k=5), 4) == 0.6309
        assert compute_fama_at_k(retrieved, relevant, {"d_old"}, k=5) == 1.0
        assert compute_fama_at_k(retrieved, relevant, {"d2"}, k=5) == 0.0


class TestRRF4AndExactBoostCalibration:
    """Verify RRF4 preserves Rank-1 consensus hits while promoting novel 2-hop graph hits."""

    def test_rrf4_promotes_novel_graph_hit_without_demoting_consensus_rank1(self) -> None:
        vec_items = [
            SearchResultItem(
                data={"id": "exp-consensus-1", "title": "Rank 1 Vector+FTS"},
                score=0.92,
                similarity=0.92,
                source_type="vector",
            ),
            SearchResultItem(
                data={"id": "exp-vec-2", "title": "Rank 2 Vector"},
                score=0.81,
                similarity=0.81,
                source_type="vector",
            ),
        ]
        fts_items = [
            SearchResultItem(
                data={"id": "exp-consensus-1", "title": "Rank 1 Vector+FTS"},
                score=0.85,
                similarity=0.85,
                source_type="fts",
            ),
        ]
        graph_items = [
            # Generic hub overlaps with exp-vec-2 at graph rank 0
            SearchResultItem(
                data={"id": "exp-vec-2", "title": "Rank 2 Vector", "graph_score": 0.9},
                score=0.9,
                similarity=0.9,
                source_type="graph",
            ),
            # Novel 2-hop graph-only experience
            SearchResultItem(
                data={"id": "exp-novel-2hop", "title": "2-Hop Graph Only", "graph_score": 1.0},
                score=1.0,
                similarity=1.0,
                source_type="graph",
            ),
        ]

        fused = rrf_fuse4(
            vector_results=vec_items,
            fts_results=fts_items,
            topic_results=[],
            graph_results=graph_items,
            limit=5,
        )
        ids = [item.data["id"] for item in fused]
        # Consensus Vector+FTS hit stays #1, and novel 2-hop graph hit enters top-3
        assert ids[0] == "exp-consensus-1"
        assert "exp-novel-2hop" in ids[:3]

    def test_cjk_phrase_exact_match_boost(self) -> None:
        items = [
            SearchResultItem(
                data={
                    "id": "exp-generic",
                    "title": "iOS 通用视图生命周期说明",
                    "tags": ["ios", "uiview"],
                },
                score=0.0250,
                similarity=0.80,
                source_type="hybrid",
            ),
            SearchResultItem(
                data={
                    "id": "exp-target",
                    "title": "iOS 触摸事件穿透与手势命中失效排查",
                    "tags": ["手势事件", "页面按钮"],
                },
                score=0.0245,
                similarity=0.79,
                source_type="hybrid",
            ),
        ]
        boosted = apply_exact_match_boost(
            items,
            "苹果端点击页面按钮没反应或者手势事件直接透传到下层视图了怎么排查",
        )
        assert boosted[0].data["id"] == "exp-target"


if __name__ == "__main__":
    s1 = TestIRBenchmarkSuiteAndAblationConfigs()
    s1.test_ir_benchmark_suite_structure()
    s1.test_ablation_configs_progression()
    s1.test_evaluate_benchmark_suite_with_recall_at_1()
    s1.test_single_query_ir_metrics()

    s2 = TestRRF4AndExactBoostCalibration()
    s2.test_rrf4_promotes_novel_graph_hit_without_demoting_consensus_rank1()
    s2.test_cjk_phrase_exact_match_boost()
    print("All test_ir_ablation_benchmark tests passed!")

"""End-to-End Functional Acceptance Suite for Memory Mechanism Optimization & Telemetry."""

from __future__ import annotations

import sys
import types
from contextlib import asynccontextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))


def _ensure_sandbox_import_stubs() -> None:
    """Install lightweight module stubs only when running outside virtualenv."""
    try:
        import sqlalchemy  # noqa: F401

        return
    except ImportError:
        pass

    sa_mod = types.ModuleType("sqlalchemy")
    sa_mod.text = lambda sql: sql  # type: ignore[attr-defined]
    for name in (
        "JSON",
        "Boolean",
        "DateTime",
        "Float",
        "ForeignKey",
        "Integer",
        "String",
        "Text",
        "UniqueConstraint",
        "desc",
        "func",
        "select",
        "update",
    ):
        setattr(sa_mod, name, MagicMock())
    sys.modules["sqlalchemy"] = sa_mod

    sa_ext = types.ModuleType("sqlalchemy.ext")
    sa_asyncio = types.ModuleType("sqlalchemy.ext.asyncio")
    sa_asyncio.AsyncSession = Any  # type: ignore[attr-defined]
    sa_asyncio.async_sessionmaker = MagicMock()  # type: ignore[attr-defined]
    sa_asyncio.create_async_engine = MagicMock()  # type: ignore[attr-defined]
    sys.modules["sqlalchemy.ext"] = sa_ext
    sys.modules["sqlalchemy.ext.asyncio"] = sa_asyncio

    sa_orm = types.ModuleType("sqlalchemy.orm")
    sa_orm.DeclarativeBase = object  # type: ignore[attr-defined]
    sa_orm.Mapped = Any  # type: ignore[attr-defined]
    sa_orm.mapped_column = lambda *a, **kw: None  # type: ignore[attr-defined]
    sa_orm.relationship = lambda *a, **kw: None  # type: ignore[attr-defined]
    sa_orm.selectinload = MagicMock()  # type: ignore[attr-defined]
    sys.modules["sqlalchemy.orm"] = sa_orm

    db_mod = types.ModuleType("team_memory.storage.database")

    @asynccontextmanager
    async def _dummy_get_session(_db_url: str | None = None):
        yield AsyncMock()

    db_mod.get_session = _dummy_get_session  # type: ignore[attr-defined]
    sys.modules["team_memory.storage.database"] = db_mod

    repo_mod = types.ModuleType("team_memory.storage.repository")
    repo_mod.ExperienceRepository = MagicMock  # type: ignore[attr-defined]
    sys.modules["team_memory.storage.repository"] = repo_mod

    arch_mod = types.ModuleType("team_memory.storage.archive_repository")
    arch_mod.ArchiveRepository = MagicMock  # type: ignore[attr-defined]
    sys.modules["team_memory.storage.archive_repository"] = arch_mod

    cfg_mod = types.ModuleType("team_memory.config")

    @dataclass
    class _DummySearchConfig:
        mode: str = "hybrid"
        rrf_k: int = 60
        vector_weight: float = 0.7
        fts_weight: float = 0.3
        adaptive_filter: bool = True
        min_confidence_ratio: float = 0.6
        score_gap_threshold: float = 0.15
        adaptive_min_keep: int = 3
        popularity_enabled: bool = True
        popularity_recall_weight: float = 0.02
        popularity_used_weight: float = 0.1
        popularity_decay_lambda: float = 0.05
        popularity_age_floor_ratio: float = 0.3
        popularity_scale: float = 0.3
        query_expansion_enabled: bool = False
        query_expansion_timeout_seconds: float = 3.0
        short_query_max_chars: int = 20
        min_similarity_short: float = 0.45

    cfg_mod.SearchConfig = _DummySearchConfig  # type: ignore[attr-defined]
    cfg_mod.RetrievalConfig = MagicMock  # type: ignore[attr-defined]
    cfg_mod.CacheConfig = MagicMock  # type: ignore[attr-defined]
    cfg_mod.LLMConfig = MagicMock  # type: ignore[attr-defined]
    cfg_mod.Settings = Any  # type: ignore[attr-defined]
    cfg_mod.get_settings = MagicMock()  # type: ignore[attr-defined]
    cfg_mod.load_settings = MagicMock()  # type: ignore[attr-defined]
    sys.modules["team_memory.config"] = cfg_mod


_ensure_sandbox_import_stubs()

from team_memory.daemon.observe.metrics_collector import MetricsCollector  # noqa: E402
from team_memory.daemon.observe.tm_stats import format_telemetry_panel_data  # noqa: E402
from team_memory.daemon.quality.search_eval import (  # noqa: E402
    compute_fama_at_k,
    compute_ndcg_at_k,
    evaluate_benchmark_suite,
)
from team_memory.services.entity.entity_search import (  # noqa: E402
    compute_ppr_edge_weight,
    extract_entities_from_query,
)
from team_memory.services.memory.mutation_resolver import MutationResolver  # noqa: E402
from team_memory.services.search.progressive_disclosure import (  # noqa: E402
    apply_progressive_disclosure,
)
from team_memory.services.search.scoring import (  # noqa: E402
    apply_popularity_boost,
    rrf_fuse4,
)
from team_memory.services.search.search_pipeline import SearchResultItem  # noqa: E402


def test_direction1_anti_noise_and_progressive_disclosure_acceptance() -> None:
    """Acceptance test for Direction 1: Anti-Noise Utility Ranking & L0/L1/L2 Progressive Disclosure."""
    noise_candidate = SearchResultItem(
        data={
            "id": "exp-noisy-high-recall",
            "title": "Generic database troubleshooting",
            "recall_count": 30,
            "used_count": 0,
            "negative_feedback_count": 2,
        },
        score=0.50,
        similarity=0.80,
    )
    verified_candidate = SearchResultItem(
        data={
            "id": "exp-verified-utility",
            "title": "Exact asyncpg connection pool fix",
            "recall_count": 4,
            "used_count": 4,
            "negative_feedback_count": 0,
        },
        score=0.48,
        similarity=0.79,
    )

    ranked = apply_popularity_boost([noise_candidate, verified_candidate])
    ranked_ids = [item.data["id"] for item in ranked]
    assert ranked_ids[0] == "exp-verified-utility"
    assert compute_ndcg_at_k(ranked_ids, {"exp-verified-utility": 3.0, "exp-noisy-high-recall": 0.0}, k=5) == 1.0

    # Progressive disclosure on 5 verbose experiences
    long_results = [
        {
            "id": f"exp-{i}",
            "title": f"Architecture guide #{i}",
            "description": ("Detailed background problem context and root cause analysis. " * 25),
            "solution": ("Step-by-step production remediation commands and verification checklist. " * 30),
            "tags": ["architecture", "ops"],
            "children": [
                {"id": f"child-{i}-1", "title": f"Sub-step {i}.1", "solution": "Long child detail " * 20},
                {"id": f"child-{i}-2", "title": f"Sub-step {i}.2", "solution": "Long child detail " * 20},
            ],
        }
        for i in range(5)
    ]
    disclosed, telemetry = apply_progressive_disclosure(long_results, detail_level="progressive", full_top_k=2)
    assert len(disclosed) == 5
    assert disclosed[0]["disclosure_level"] == "full"
    assert disclosed[1]["disclosure_level"] == "full"
    assert all(item["disclosure_level"] == "summary" for item in disclosed[2:])
    assert telemetry["saved_tokens_pct"] >= 50.0


def test_direction2_weighted_ppr_and_4way_rrf_multihop_acceptance() -> None:
    """Acceptance test for Direction 2: Entity Lexicon + Weighted 2-Hop PPR + 4-Way RRF."""
    entities = extract_entities_from_query("如何优化 pgvector 的连接池超时问题")
    assert "pgvector" in entities
    assert "连接池超时" in entities

    specific_weight = compute_ppr_edge_weight(1.0, source_degree=1, target_degree=2, hop_depth=1, damping=0.5)
    hub_weight = compute_ppr_edge_weight(1.0, source_degree=1, target_degree=80, hop_depth=1, damping=0.5)
    assert specific_weight > hub_weight * 5.0

    vec_results = [
        SearchResultItem(
            data={"id": "exp-hop1-direct", "title": "pgvector HNSW index tuning"},
            score=0.89,
            similarity=0.89,
            source_type="vector",
        )
    ]
    graph_results = [
        SearchResultItem(
            data={
                "id": "exp-hop2-graph-only",
                "title": "asyncpg statement_timeout under index rebuild",
                "graph_score": specific_weight,
            },
            score=specific_weight,
            similarity=specific_weight,
            source_type="graph",
        )
    ]
    fused = rrf_fuse4(vec_results, [], [], graph_results, limit=5)
    fused_ids = [item.data["id"] for item in fused]
    assert "exp-hop1-direct" in fused_ids
    assert "exp-hop2-graph-only" in fused_ids

    suite_metrics = evaluate_benchmark_suite(
        [
            {
                "retrieved_ids": fused_ids,
                "relevant_ids": {"exp-hop1-direct", "exp-hop2-graph-only"},
                "active_relevant_ids": {"exp-hop1-direct", "exp-hop2-graph-only"},
                "superseded_ids": set(),
                "is_multihop": True,
            }
        ],
        k=5,
    )
    assert suite_metrics["multihop_recall_at_5"] == 1.0


def test_direction3_bitemporal_mutation_and_fama_acceptance() -> None:
    """Acceptance test for Direction 3: Bi-Temporal Mutation State Machine & FAMA@5 = 100%."""
    resolver = MutationResolver(noop_threshold=0.96, mutation_threshold=0.85)

    # 1. ADD
    d_add = resolver.resolve_by_rules(
        title="PostgreSQL Pool v1",
        problem="Pool exhaustion",
        solution="Set pool_size=5",
        candidates=[],
    )
    assert d_add.action == "ADD"

    # 2. NOOP
    d_noop = resolver.resolve_by_rules(
        title="PostgreSQL Pool v1",
        problem="Pool exhaustion",
        solution="Set pool_size=5",
        candidates=[{"id": "exp-v1", "similarity": 0.98, "solution": "Set pool_size=5"}],
    )
    assert d_noop.action == "NOOP"

    # 3. MERGE
    d_merge = resolver.resolve_by_rules(
        title="PostgreSQL Pool v1+",
        problem="Pool exhaustion under firewall idle drop",
        solution="Also set pool_recycle=1800",
        candidates=[{"id": "exp-v1", "similarity": 0.90, "solution": "Set pool_size=5"}],
    )
    assert d_merge.action == "MERGE"
    assert "pool_size=5" in (d_merge.merged_solution or "")
    assert "pool_recycle=1800" in (d_merge.merged_solution or "")

    # 4. SUPERSEDE
    d_supersede = resolver.resolve_by_rules(
        title="PostgreSQL Pool v2",
        problem="Pool exhaustion under high concurrency",
        solution="Set pool_size=25",
        candidates=[{"id": "exp-v1", "similarity": 0.92, "solution": "Set pool_size=5"}],
    )
    assert d_supersede.action == "SUPERSEDE"
    assert d_supersede.target_id == "exp-v1"

    # Verify FAMA@5 before vs after bi-temporal valid_until filtering
    fama_unfiltered = compute_fama_at_k(
        retrieved_ids=["exp-v2", "exp-v1"],
        active_relevant_ids={"exp-v2"},
        superseded_ids={"exp-v1"},
        k=5,
    )
    fama_filtered = compute_fama_at_k(
        retrieved_ids=["exp-v2"],
        active_relevant_ids={"exp-v2"},
        superseded_ids={"exp-v1"},
        k=5,
    )
    assert fama_unfiltered == 0.0
    assert fama_filtered == 1.0


def test_telemetry_collector_and_stats_aggregation_acceptance() -> None:
    """Acceptance test for Telemetry Collector & Dashboard Stats Aggregation."""
    collector = MetricsCollector(":memory:")
    collector.record_search_telemetry(
        returned_tokens=620,
        saved_tokens_pct=54.2,
        stage_metrics={"total_ms": 28.4, "vec_ms": 11.2, "graph_ms": 6.1},
        stage_counts={"vec_count": 3, "graph_unique_added": 1},
    )
    collector.record_mutation_action("SUPERSEDE", project="team_memory")
    collector.record_mutation_action("MERGE", project="team_memory")

    summary = collector.summary(hours=1)
    assert summary["recall_returned_tokens"]["_total"] == 620.0
    assert summary["recall_saved_tokens_pct"]["_total"] == 54.2
    assert summary["pipeline_total_ms"]["_total"] == 28.4
    assert summary["graph_unique_added"]["_total"] == 1.0
    assert summary["memory_mutation_action"]["action=SUPERSEDE_project=team_memory"] == 1.0
    assert summary["memory_mutation_action"]["action=MERGE_project=team_memory"] == 1.0

    panel_data = format_telemetry_panel_data(
        {"avg_returned_tokens": 620.0, "avg_total_ms": 28.44, "graph_assisted_searches": 7},
        {"active_count": 142, "superseded_count": 18, "verified_count": 65},
    )
    assert panel_data == {
        "avg_returned_tokens": 620.0,
        "avg_total_ms": 28.4,
        "graph_assisted_searches": 7,
        "active_experiences": 142,
        "superseded_experiences": 18,
        "verified_experiences": 65,
    }


if __name__ == "__main__":
    test_direction1_anti_noise_and_progressive_disclosure_acceptance()
    test_direction2_weighted_ppr_and_4way_rrf_multihop_acceptance()
    test_direction3_bitemporal_mutation_and_fama_acceptance()
    test_telemetry_collector_and_stats_aggregation_acceptance()
    print("ALL END-TO-END MEMORY OPTIMIZATION ACCEPTANCE TESTS PASSED!")

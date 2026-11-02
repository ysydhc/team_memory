"""Unit tests for Phase 2: Entity Lexicon + Specificity-Weighted 2-Hop PPR Graph Search + 4-Way RRF Fusion."""

from __future__ import annotations

import asyncio
import sys
import types
from contextlib import asynccontextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

try:
    import pytest

    _async_mark = pytest.mark.asyncio
except ImportError:  # pragma: no cover
    pytest = None  # type: ignore[assignment]

    def _async_mark(fn: Any) -> Any:
        return fn


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

    class _DummyExperienceRepository:
        def __init__(self, session: Any = None) -> None:
            self._session = session

        async def search_by_vector(self, **kwargs: Any) -> list[dict[str, Any]]:
            return []

        async def search_by_fts(self, **kwargs: Any) -> list[dict[str, Any]]:
            return []

        async def get_by_id(self, experience_id: Any, include_deleted: bool = False) -> Any:
            return None

    repo_mod.ExperienceRepository = _DummyExperienceRepository  # type: ignore[attr-defined]
    sys.modules["team_memory.storage.repository"] = repo_mod

    arch_mod = types.ModuleType("team_memory.storage.archive_repository")

    class _DummyArchiveRepository:
        def __init__(self, session: Any = None) -> None:
            self._session = session

        async def search_archives(self, *args: Any, **kwargs: Any) -> list[dict[str, Any]]:
            return []

        async def get_archive_ids_for_experiences(self, exp_ids: list[Any]) -> dict[str, list[str]]:
            return {}

    arch_mod.ArchiveRepository = _DummyArchiveRepository  # type: ignore[attr-defined]
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
        popularity_enabled: bool = False
        popularity_recall_weight: float = 0.02
        popularity_used_weight: float = 0.1
        popularity_decay_lambda: float = 0.05
        popularity_age_floor_ratio: float = 0.3
        popularity_scale: float = 0.3
        query_expansion_enabled: bool = False
        query_expansion_timeout_seconds: float = 3.0
        short_query_max_chars: int = 20
        min_similarity_short: float = 0.45

    @dataclass
    class _DummyRetrievalConfig:
        max_tokens: int | None = None

    @dataclass
    class _DummyCacheConfig:
        enabled: bool = False
        max_size: int = 128
        ttl_seconds: int = 300
        embedding_cache_size: int = 256
        backend: str = "memory"
        redis_url: str = "redis://localhost:6379/0"

    @dataclass
    class _DummyLLMConfig:
        pass

    cfg_mod.SearchConfig = _DummySearchConfig  # type: ignore[attr-defined]
    cfg_mod.RetrievalConfig = _DummyRetrievalConfig  # type: ignore[attr-defined]
    cfg_mod.CacheConfig = _DummyCacheConfig  # type: ignore[attr-defined]
    cfg_mod.LLMConfig = _DummyLLMConfig  # type: ignore[attr-defined]
    cfg_mod.Settings = Any  # type: ignore[attr-defined]
    cfg_mod.get_settings = MagicMock()  # type: ignore[attr-defined]
    sys.modules["team_memory.config"] = cfg_mod


_ensure_sandbox_import_stubs()

from team_memory.config import CacheConfig, LLMConfig, RetrievalConfig, SearchConfig  # noqa: E402
from team_memory.services.entity.entity_search import (  # noqa: E402
    compute_ppr_edge_weight,
    extract_entities_from_query,
    find_weighted_experiences_by_entities,
)
from team_memory.services.search.scoring import rrf_fuse4  # noqa: E402
from team_memory.services.search.search_pipeline import (  # noqa: E402
    SearchPipeline,
    SearchRequest,
    SearchResultItem,
)


class TestEntityLexiconAndCompoundExtraction:
    """Test upgraded rule-based entity extraction for lowercase tech terms & CJK compounds."""

    def test_extracts_lowercase_tech_terms(self) -> None:
        query = "how to tune pgvector and asyncpg with alembic"
        entities = extract_entities_from_query(query)
        assert "pgvector" in entities
        assert "asyncpg" in entities
        assert "alembic" in entities

    def test_extracts_chinese_compound_tech_terms(self) -> None:
        query = "排查连接池超时与向量检索延迟"
        entities = extract_entities_from_query(query)
        assert "连接池超时" in entities
        assert "向量检索" in entities

        query2 = "如何解决连接池超时与向量检索慢的问题"
        entities2 = extract_entities_from_query(query2)
        assert "连接池超时" in entities2
        assert "向量检索" in entities2

    def test_extracts_camel_upper_hyphen_snake_and_backtick(self) -> None:
        query = "Use LiteLLM with MCP on litellm-proxy and search_pipeline or `custom_token`plus custom_lex_item"
        entities = extract_entities_from_query(query, lexicon=["custom_lex_item"])
        assert "LiteLLM" in entities
        assert "MCP" in entities
        assert "litellm-proxy" in entities
        assert "search_pipeline" in entities
        assert "custom_token" in entities
        assert "custom_lex_item" in entities


class TestSpecificityWeightedPPR:
    """Test degree-penalized PPR edge weight calculation and hop decay."""

    def test_degree_penalty_specific_leaf_beats_generic_hub(self) -> None:
        leaf_weight = compute_ppr_edge_weight(
            edge_weight=1.0,
            source_degree=1,
            target_degree=2,
            hop_depth=1,
            damping=0.5,
        )
        hub_weight = compute_ppr_edge_weight(
            edge_weight=1.0,
            source_degree=1,
            target_degree=100,
            hop_depth=1,
            damping=0.5,
        )
        assert leaf_weight > hub_weight
        assert leaf_weight / hub_weight >= 5.0

    def test_hop_decay(self) -> None:
        hop1 = compute_ppr_edge_weight(
            edge_weight=1.0,
            source_degree=2,
            target_degree=2,
            hop_depth=1,
            damping=0.5,
        )
        hop2 = compute_ppr_edge_weight(
            edge_weight=1.0,
            source_degree=2,
            target_degree=2,
            hop_depth=2,
            damping=0.5,
        )
        assert hop1 > hop2
        assert abs(hop1 - 2.0 * hop2) < 1e-9

    @_async_mark
    async def test_find_weighted_experiences_fallback_on_cte_error(self) -> None:
        with (
            patch(
                "team_memory.services.entity.entity_search.get_session",
                side_effect=RuntimeError("SQLite does not support PG jsonb_array_elements_text"),
            ),
            patch(
                "team_memory.services.entity.entity_search.find_experience_ids_by_entities",
                new_callable=AsyncMock,
                return_value=["exp-fallback-1", "exp-fallback-2"],
            ),
        ):
            hits = await find_weighted_experiences_by_entities(
                db_url="sqlite+aiosqlite://",
                entity_names=["pgvector"],
                projects=["default"],
                max_depth=2,
            )
        assert hits == [("exp-fallback-1", 0.5), ("exp-fallback-2", 0.5)]


class TestFourWayRRFFusion:
    """Test rrf_fuse4 recalls graph-only experiences and boosts multi-source matches."""

    def test_rrf_fuse4_recalls_graph_only_experience(self) -> None:
        vec_items = [
            SearchResultItem(
                data={"id": "exp-vec-1", "title": "Vector Hit 1"},
                score=0.88,
                similarity=0.88,
                source_type="vector",
            )
        ]
        graph_items = [
            SearchResultItem(
                data={"id": "exp-graph-only-2", "title": "Graph Multi-Hop Hit", "graph_score": 0.85},
                score=0.85,
                similarity=0.85,
                source_type="graph",
            )
        ]

        fused = rrf_fuse4(
            vector_results=vec_items,
            fts_results=[],
            topic_results=[],
            graph_results=graph_items,
            limit=10,
        )
        fused_ids = [item.data["id"] for item in fused]
        assert "exp-vec-1" in fused_ids
        assert "exp-graph-only-2" in fused_ids
        assert len(fused) == 2


class TestSearchPipelineGraphIndependentRecall:
    """Test SearchPipeline 4-way RRF independent graph recall and stage_counts."""

    @_async_mark
    async def test_pipeline_recalls_graph_only_experience_and_records_stage_counts(self) -> None:
        embedding = AsyncMock()
        embedding.encode_single = AsyncMock(return_value=[0.1] * 768)

        pipeline = SearchPipeline(
            embedding_provider=embedding,
            search_config=SearchConfig(mode="hybrid", adaptive_filter=True),
            retrieval_config=RetrievalConfig(),
            cache_config=CacheConfig(enabled=False),
            llm_config=LLMConfig(),
            db_url="postgresql+asyncpg://localhost/test_db",
        )

        mock_repo = MagicMock()
        mock_repo.search_by_vector = AsyncMock(
            return_value=[
                {
                    "id": "exp-vec-1",
                    "title": "Direct pgvector tuning guide",
                    "description": "Tune HNSW index parameters",
                    "solution": "Set ef_search=100",
                    "similarity": 0.90,
                    "tags": ["pgvector"],
                }
            ]
        )
        mock_repo.search_by_fts = AsyncMock(return_value=[])
        mock_repo.get_by_id = AsyncMock(
            return_value={
                "id": "exp-graph-only-2",
                "title": "Asyncpg pool exhaustion under HNSW rebuild",
                "description": "2-hop graph connected experience with zero lexical overlap",
                "solution": "Increase max_overflow and statement_timeout",
                "tags": ["asyncpg"],
            }
        )

        @asynccontextmanager
        async def _mock_get_session(_db_url: str):
            sess = AsyncMock()
            sess.execute = AsyncMock(side_effect=RuntimeError("use repo.get_by_id fallback in unit test"))
            yield sess

        with (
            patch(
                "team_memory.services.search.search_pipeline.ExperienceRepository",
                return_value=mock_repo,
            ),
            patch(
                "team_memory.services.search.search_pipeline.get_session",
                _mock_get_session,
            ),
            patch(
                "team_memory.services.entity.entity_search.find_weighted_experiences_by_entities",
                new_callable=AsyncMock,
                return_value=[("exp-graph-only-2", 0.85)],
            ),
        ):
            req = SearchRequest(
                query="how to tune pgvector latency",
                project="default",
                max_results=10,
            )
            result = await pipeline.search(AsyncMock(), req)

        returned_ids = [r["id"] for r in result.results]
        assert "exp-vec-1" in returned_ids
        assert "exp-graph-only-2" in returned_ids
        assert result.stage_counts["vec_count"] == 1
        assert result.stage_counts["fts_count"] == 0
        assert result.stage_counts["graph_count"] == 1
        assert result.stage_counts["graph_unique_added"] == 1
        assert result.stage_counts["fused_count"] == 2


if __name__ == "__main__":
    s1 = TestEntityLexiconAndCompoundExtraction()
    s1.test_extracts_lowercase_tech_terms()
    s1.test_extracts_chinese_compound_tech_terms()
    s1.test_extracts_camel_upper_hyphen_snake_and_backtick()

    s2 = TestSpecificityWeightedPPR()
    s2.test_degree_penalty_specific_leaf_beats_generic_hub()
    s2.test_hop_decay()
    asyncio.run(s2.test_find_weighted_experiences_fallback_on_cte_error())

    s3 = TestFourWayRRFFusion()
    s3.test_rrf_fuse4_recalls_graph_only_experience()

    s4 = TestSearchPipelineGraphIndependentRecall()
    asyncio.run(s4.test_pipeline_recalls_graph_only_experience_and_records_stage_counts())
    print("All test_weighted_graph_ppr tests passed!")

"""Unit tests for Task 10: SearchPipeline DB session reuse & server.py _guard_output fast path."""

from __future__ import annotations

import asyncio
import json
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
    except ImportError:
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

        class _DummyAsyncSession:
            async def execute(self, *args: Any, **kwargs: Any) -> Any:
                return MagicMock()

        sa_ext = types.ModuleType("sqlalchemy.ext")
        sa_asyncio = types.ModuleType("sqlalchemy.ext.asyncio")
        sa_asyncio.AsyncSession = _DummyAsyncSession  # type: ignore[attr-defined]
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

        @dataclass
        class _DummyMCPConfig:
            max_output_tokens: int = 4000
            truncate_solution_at: int = 2000

        cfg_mod.SearchConfig = _DummySearchConfig  # type: ignore[attr-defined]
        cfg_mod.RetrievalConfig = _DummyRetrievalConfig  # type: ignore[attr-defined]
        cfg_mod.CacheConfig = _DummyCacheConfig  # type: ignore[attr-defined]
        cfg_mod.LLMConfig = _DummyLLMConfig  # type: ignore[attr-defined]
        cfg_mod.Settings = Any  # type: ignore[attr-defined]
        cfg_mod.get_settings = MagicMock()  # type: ignore[attr-defined]
        sys.modules["team_memory.config"] = cfg_mod

        cfg_mcp_mod = types.ModuleType("team_memory.config.mcp")
        cfg_mcp_mod.MCPConfig = _DummyMCPConfig  # type: ignore[attr-defined]
        sys.modules["team_memory.config.mcp"] = cfg_mcp_mod

    try:
        import fastmcp  # noqa: F401
    except ImportError:
        fm_mod = types.ModuleType("fastmcp")

        class _DummyFastMCP:
            def __init__(self, *args: Any, **kwargs: Any) -> None:
                self._tools: list[Any] = []

            def tool(self, name: str | None = None, description: str | None = None):
                def _decorator(fn: Any) -> Any:
                    t = types.SimpleNamespace(name=name or fn.__name__, description=description or "", fn=fn)
                    self._tools.append(t)
                    return fn

                return _decorator

            async def list_tools(self) -> list[Any]:
                return list(self._tools)

        fm_mod.FastMCP = _DummyFastMCP  # type: ignore[attr-defined]
        sys.modules["fastmcp"] = fm_mod

        boot_mod = types.ModuleType("team_memory.bootstrap")
        boot_mod.bootstrap = MagicMock()  # type: ignore[attr-defined]
        boot_mod.get_context = MagicMock()  # type: ignore[attr-defined]
        sys.modules["team_memory.bootstrap"] = boot_mod

        mem_mod = types.ModuleType("team_memory.services.memory")
        mem_mod.memory_operations = MagicMock()  # type: ignore[attr-defined]
        sys.modules["team_memory.services.memory"] = mem_mod


_ensure_sandbox_import_stubs()

from sqlalchemy.ext.asyncio import AsyncSession  # noqa: E402

from team_memory.config import CacheConfig, LLMConfig, RetrievalConfig, SearchConfig  # noqa: E402
from team_memory.server import _guard_output, mcp  # noqa: E402
from team_memory.services.entity.entity_search import (  # noqa: E402
    find_experience_ids_by_entities,
    find_weighted_experiences_by_entities,
)
from team_memory.services.search.search_pipeline import (  # noqa: E402
    SearchPipeline,
    SearchRequest,
)
from team_memory.storage.repository import ExperienceRepository  # noqa: E402


class TestEntitySearchSessionReuse:
    """Verify entity_search reuses a provided AsyncSession without opening get_session."""

    @_async_mark
    async def test_find_weighted_experiences_reuses_passed_session(self) -> None:
        mock_session = AsyncMock(spec=AsyncSession)
        mock_result = MagicMock()
        mock_result.fetchall.return_value = [("exp-1", 0.9), ("exp-2", 0.45)]
        mock_session.execute = AsyncMock(return_value=mock_result)

        with patch("team_memory.services.entity.entity_search.get_session") as mock_get_session:
            hits = await find_weighted_experiences_by_entities(
                db_url="postgresql+asyncpg://localhost/test_db",
                entity_names=["pgvector"],
                projects=["default"],
                max_depth=2,
                session=mock_session,
            )

        assert hits == [("exp-1", 0.9), ("exp-2", 0.45)]
        mock_session.execute.assert_awaited_once()
        mock_get_session.assert_not_called()

    @_async_mark
    async def test_find_experience_ids_reuses_passed_session(self) -> None:
        mock_session = AsyncMock(spec=AsyncSession)
        mock_result = MagicMock()
        mock_result.fetchall.return_value = [("exp-a",), ("exp-b",)]
        mock_session.execute = AsyncMock(return_value=mock_result)

        with patch("team_memory.services.entity.entity_search.get_session") as mock_get_session:
            ids = await find_experience_ids_by_entities(
                db_url="postgresql+asyncpg://localhost/test_db",
                entity_names=["redis"],
                projects=["default"],
                max_depth=2,
                session=mock_session,
            )

        assert ids == ["exp-a", "exp-b"]
        mock_session.execute.assert_awaited_once()
        mock_get_session.assert_not_called()


class TestSearchPipelineGraphSessionAndDeduplication:
    """Verify SearchPipeline._graph_search skips already-fetched IDs and reuses repo._session."""

    @_async_mark
    async def test_graph_search_skips_db_query_for_existing_by_id(self) -> None:
        embedding = AsyncMock()
        pipeline = SearchPipeline(
            embedding_provider=embedding,
            search_config=SearchConfig(mode="hybrid", adaptive_filter=True),
            retrieval_config=RetrievalConfig(),
            cache_config=CacheConfig(enabled=False),
            llm_config=LLMConfig(),
            db_url="postgresql+asyncpg://localhost/test_db",
        )

        mock_session = AsyncMock(spec=AsyncSession)
        mock_session.execute = AsyncMock()
        repo = ExperienceRepository(mock_session)

        existing_by_id = {
            "exp-overlap-1": {
                "id": "exp-overlap-1",
                "title": "Already retrieved by vector search",
                "description": "Desc",
                "solution": "Sol",
                "tags": ["pgvector"],
            }
        }

        with patch("team_memory.services.search.search_pipeline.get_session") as mock_get_session:
            items = await pipeline._graph_search(
                repo,
                [("exp-overlap-1", 0.88)],
                limit=10,
                request=SearchRequest(query="pgvector", project="default"),
                existing_by_id=existing_by_id,
            )

        assert len(items) == 1
        assert items[0].data["id"] == "exp-overlap-1"
        assert items[0].data["graph_score"] == 0.88
        mock_session.execute.assert_not_called()
        mock_get_session.assert_not_called()

    @_async_mark
    async def test_graph_search_reuses_repo_session_for_missing_ids(self) -> None:
        embedding = AsyncMock()
        pipeline = SearchPipeline(
            embedding_provider=embedding,
            search_config=SearchConfig(mode="hybrid", adaptive_filter=True),
            retrieval_config=RetrievalConfig(),
            cache_config=CacheConfig(enabled=False),
            llm_config=LLMConfig(),
            db_url="postgresql+asyncpg://localhost/test_db",
        )

        mock_session = AsyncMock(spec=AsyncSession)
        mock_row = (
            "exp-graph-unique-2",
            "Graph Unique Title",
            "Graph Unique Desc",
            "Graph Unique Sol",
            ["asyncpg"],
            "default",
            "general",
            "high",
            3,
            1,
        )
        mock_result = MagicMock()
        mock_result.fetchall.return_value = [mock_row]
        mock_session.execute = AsyncMock(return_value=mock_result)
        repo = ExperienceRepository(mock_session)

        existing_by_id = {
            "exp-overlap-1": {
                "id": "exp-overlap-1",
                "title": "Already retrieved by vector search",
                "description": "Desc",
                "solution": "Sol",
                "tags": ["pgvector"],
            }
        }

        with patch("team_memory.services.search.search_pipeline.get_session") as mock_get_session:
            items = await pipeline._graph_search(
                repo,
                [("exp-overlap-1", 0.90), ("exp-graph-unique-2", 0.75)],
                limit=10,
                request=SearchRequest(query="pgvector", project="default"),
                existing_by_id=existing_by_id,
            )

        assert [it.data["id"] for it in items] == ["exp-overlap-1", "exp-graph-unique-2"]
        mock_session.execute.assert_awaited_once()
        # Only the missing ID should be queried from DB
        called_params = mock_session.execute.await_args.args[1]
        assert called_params == {"ids": ["exp-graph-unique-2"]}
        mock_get_session.assert_not_called()

    @_async_mark
    async def test_pipeline_search_passes_reusable_session_to_entity_graph_cte(self) -> None:
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

        mock_session = AsyncMock(spec=AsyncSession)
        mock_repo = MagicMock()
        mock_repo._session = mock_session
        mock_repo.search_by_vector = AsyncMock(
            return_value=[
                {
                    "id": "exp-vec-overlap-1",
                    "title": "pgvector index tuning",
                    "description": "Tune HNSW",
                    "solution": "Set ef_search=100",
                    "similarity": 0.91,
                    "tags": ["pgvector"],
                }
            ]
        )
        mock_repo.search_by_fts = AsyncMock(return_value=[])

        with (
            patch(
                "team_memory.services.search.search_pipeline.ExperienceRepository",
                return_value=mock_repo,
            ),
            patch(
                "team_memory.services.entity.entity_search.find_weighted_experiences_by_entities",
                new_callable=AsyncMock,
                return_value=[("exp-vec-overlap-1", 0.85)],
            ) as mock_find_weighted,
            patch.object(pipeline, "_fts_search", new_callable=AsyncMock, return_value=[]),
            patch.object(pipeline, "_topic_search", new_callable=AsyncMock, return_value=[]),
            patch("team_memory.services.search.search_pipeline.get_session") as mock_get_session,
        ):
            req = SearchRequest(query="how to tune pgvector", project="default", max_results=5)
            res = await pipeline.search(mock_session, req)

        mock_find_weighted.assert_awaited_once()
        assert mock_find_weighted.await_args.kwargs.get("session") is mock_session
        mock_get_session.assert_not_called()
        assert len(res.results) == 1
        assert res.results[0]["id"] == "exp-vec-overlap-1"


class TestGuardOutputFastPath:
    """Verify _guard_output fast path skips json.dumps when no truncation is needed."""

    def test_fast_path_skips_json_dumps_when_within_budget_and_not_truncated(self) -> None:
        payload = {
            "message": "Found 2 results",
            "results": [
                {"id": "1", "title": "A", "confidence": "high", "solution": "short solution"},
                {"id": "2", "title": "B", "confidence": "medium", "solution": "another short solution"},
            ],
        }
        raw_json = json.dumps(payload, ensure_ascii=False)

        with patch("team_memory.server.json.dumps", wraps=json.dumps) as spy_dumps:
            out = _guard_output(raw_json, max_tokens=4000)

        assert out is raw_json
        spy_dumps.assert_not_called()

    def test_slow_path_calls_json_dumps_when_low_confidence_or_long_solution_truncated(self) -> None:
        payload = {
            "message": "Found 2 results",
            "results": [
                {"id": "1", "title": "A", "confidence": "high", "solution": "x" * 7000},
                {"id": "2", "title": "B", "confidence": "low", "solution": "low confidence item"},
            ],
        }
        raw_json = json.dumps(payload, ensure_ascii=False)

        with patch("team_memory.server.json.dumps", wraps=json.dumps) as spy_dumps:
            out = _guard_output(raw_json, max_tokens=4000)

        spy_dumps.assert_called()
        parsed = json.loads(out)
        assert parsed.get("truncated") is True
        assert len(parsed["results"]) == 1
        assert parsed["results"][0]["solution"].endswith("... [truncated]")

    @_async_mark
    async def test_fastmcp_get_tools_compat_present(self) -> None:
        assert hasattr(mcp, "get_tools")
        tools = await mcp.get_tools()
        assert "memory_recall" in tools
        assert "memory_save" in tools


if __name__ == "__main__":
    t1 = TestEntitySearchSessionReuse()
    asyncio.run(t1.test_find_weighted_experiences_reuses_passed_session())
    asyncio.run(t1.test_find_experience_ids_reuses_passed_session())

    t2 = TestSearchPipelineGraphSessionAndDeduplication()
    asyncio.run(t2.test_graph_search_skips_db_query_for_existing_by_id())
    asyncio.run(t2.test_graph_search_reuses_repo_session_for_missing_ids())
    asyncio.run(t2.test_pipeline_search_passes_reusable_session_to_entity_graph_cte())

    t3 = TestGuardOutputFastPath()
    t3.test_fast_path_skips_json_dumps_when_within_budget_and_not_truncated()
    t3.test_slow_path_calls_json_dumps_when_low_confidence_or_long_solution_truncated()
    asyncio.run(t3.test_fastmcp_get_tools_compat_present())
    print("All test_pipeline_session_and_guard_opt tests passed!")

"""Unit tests for Phase 0: Telemetry Schema Extension & Offline IR/FAMA Benchmark Upgrade."""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from team_memory.daemon.quality.search_eval import (
    compute_fama_at_k,
    compute_mrr,
    compute_ndcg_at_k,
    compute_recall_at_k,
    evaluate_benchmark_suite,
)
from team_memory.services.memory.memory_operations import write_search_log
from team_memory.services.search.search_log_writer import SearchLogWriter
from team_memory.storage.models import Base, Experience, Relationship, SearchLog
from team_memory.storage.search_log_repository import SearchLogRepository

TEST_DB_URL = "sqlite+aiosqlite://"


@pytest.fixture
async def engine():
    """Create a fresh SQLite async engine with SearchLog table."""
    eng = create_async_engine(TEST_DB_URL, echo=False)
    async with eng.begin() as conn:
        await conn.run_sync(
            lambda sync_conn: Base.metadata.create_all(sync_conn, tables=[SearchLog.__table__], checkfirst=True)
        )
    yield eng
    async with eng.begin() as conn:
        await conn.run_sync(
            lambda sync_conn: Base.metadata.drop_all(sync_conn, tables=[SearchLog.__table__], checkfirst=True)
        )
    await eng.dispose()


@pytest.fixture
async def session(engine) -> AsyncSession:
    factory = async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)
    async with factory() as sess:
        yield sess


class TestOfflineIRMetrics:
    """Test pure offline IR and FAMA@K metrics in search_eval.py."""

    def test_recall_at_k(self):
        assert compute_recall_at_k(["a", "b", "c"], {"a", "c"}, k=5) == 1.0
        assert compute_recall_at_k(["a", "b", "c"], {"a", "d"}, k=2) == 0.5
        assert compute_recall_at_k([], {"a"}, k=5) == 0.0
        assert compute_recall_at_k(["a"], set(), k=5) == 0.0
        assert compute_recall_at_k(["a"], {"a"}, k=0) == 0.0

    def test_mrr(self):
        assert compute_mrr(["a", "b", "c"], {"a"}) == 1.0
        assert compute_mrr(["x", "b", "c"], {"b", "c"}) == 0.5
        assert compute_mrr(["x", "y", "z"], {"a"}) == 0.0
        assert compute_mrr([], {"a"}) == 0.0
        assert compute_mrr(["a"], set()) == 0.0

    def test_ndcg_at_k_binary_and_graded(self):
        # Perfect rank 1 hit
        assert compute_ndcg_at_k(["a", "b"], {"a"}, k=5) == pytest.approx(1.0)
        # Rank 2 hit vs ideal rank 1 hit: (1/log2(3)) / (1/log2(2)) = 1/log2(3)
        lower_rank = compute_ndcg_at_k(["x", "a"], {"a"}, k=5)
        assert 0.6 < lower_rank < 0.65
        # Graded relevance: putting higher grade first beats putting lower grade first
        perfect_graded = compute_ndcg_at_k(["high", "low"], {"high": 3.0, "low": 1.0}, k=5)
        swapped_graded = compute_ndcg_at_k(["low", "high"], {"high": 3.0, "low": 1.0}, k=5)
        assert perfect_graded == pytest.approx(1.0)
        assert swapped_graded < perfect_graded
        assert compute_ndcg_at_k([], {"a"}, k=5) == 0.0

    def test_fama_at_k(self):
        # Active relevant present and no superseded present -> 1.0
        assert compute_fama_at_k(["act1", "other"], {"act1"}, {"old1"}, k=5) == 1.0
        # Active relevant present BUT superseded also in top-K -> penalized to 0.0
        assert compute_fama_at_k(["act1", "old1"], {"act1"}, {"old1"}, k=5) == 0.0
        # Superseded outside top-K (at rank k+1) does not penalize top-K
        assert compute_fama_at_k(["act1", "x", "old1"], {"act1"}, {"old1"}, k=2) == 1.0
        # No active relevant in top-K -> 0.0
        assert compute_fama_at_k(["x", "y"], {"act1"}, {"old1"}, k=5) == 0.0
        assert compute_fama_at_k([], {"act1"}, set(), k=5) == 0.0

    def test_evaluate_benchmark_suite(self):
        assert evaluate_benchmark_suite([]) == {
            "avg_recall_at_5": 0.0,
            "mrr": 0.0,
            "avg_ndcg_at_5": 0.0,
            "fama_at_5": 0.0,
            "multihop_recall_at_5": 0.0,
        }

        cases = [
            {
                "retrieved_ids": ["exp1", "exp2"],
                "relevant_ids": {"exp1"},
                "superseded_ids": {"exp_old"},
                "is_multihop": False,
            },
            {
                "retrieved_ids": ["exp_old", "exp3"],
                "relevant_ids": {"exp3"},
                "superseded_ids": {"exp_old"},
                "is_multihop": True,
            },
        ]
        summary = evaluate_benchmark_suite(cases, k=5)
        assert summary["avg_recall_at_5"] == 1.0
        assert summary["mrr"] == 0.75  # (1.0 + 0.5) / 2
        assert summary["fama_at_5"] == 0.5  # case 1 = 1.0, case 2 = 0.0 (polluted by exp_old)
        assert summary["multihop_recall_at_5"] == 1.0


class TestBitemporalAndTelemetryModels:
    """Test new ORM columns on Experience, SearchLog, and Relationship."""

    def test_experience_bitemporal_fields_and_to_dict(self):
        now = datetime.now(timezone.utc)
        superseded_id = uuid.uuid4()
        exp = Experience(
            id=uuid.uuid4(),
            title="Use connection pool v2",
            description="Pool v1 leaks sockets",
            solution="Upgrade to v2",
            created_by="alice",
            project="team_memory",
            negative_feedback_count=2,
            valid_from=now,
            valid_until=now,
            superseded_by_id=superseded_id,
            last_verified_at=now,
            mutation_history=[{"op": "supersede", "at": now.isoformat()}],
        )
        d = exp.to_dict()
        assert d["negative_feedback_count"] == 2
        assert d["valid_from"] == now.isoformat()
        assert d["valid_until"] == now.isoformat()
        assert d["superseded_by_id"] == str(superseded_id)
        assert d["last_verified_at"] == now.isoformat()
        assert len(exp.mutation_history) == 1

    def test_relationship_temporal_fields(self):
        now = datetime.now(timezone.utc)
        rel = Relationship(
            id=uuid.uuid4(),
            source_entity_id=uuid.uuid4(),
            target_entity_id=uuid.uuid4(),
            relation_type="depends_on",
            valid_from=now,
            valid_until=None,
        )
        assert rel.valid_from == now
        assert rel.valid_until is None


class TestSearchLogTelemetryPlumbing:
    """Test SearchLogRepository, write_search_log, and SearchLogWriter telemetry support."""

    @pytest.mark.asyncio
    async def test_repository_create_with_telemetry(self, session: AsyncSession):
        repo = SearchLogRepository(session)
        metrics = {"vector_ms": 12.4, "fts_ms": 4.1, "rrf_ms": 0.8, "total_ms": 18.2}
        counts = {"vector": 10, "fts": 5, "graph": 2, "graph_unique_added": 1, "final": 5}
        log = await repo.create(
            query="pgvector index tuning",
            intent_type="search",
            project="team_memory",
            source="mcp",
            result_ids=[{"id": "exp-1", "score": 0.91}],
            stage_metrics=metrics,
            stage_counts=counts,
            returned_tokens=420,
        )
        await session.commit()

        res = await session.execute(select(SearchLog).where(SearchLog.id == log.id))
        saved = res.scalar_one()
        assert saved.stage_metrics == metrics
        assert saved.stage_counts == counts
        assert saved.returned_tokens == 420

    @pytest.mark.asyncio
    async def test_write_search_log_forwards_telemetry(self):
        mock_session = AsyncMock()
        mock_repo = MagicMock()
        mock_log = MagicMock()
        mock_log.id = uuid.uuid4()
        mock_repo.create = AsyncMock(return_value=mock_log)

        with (
            patch("team_memory.services.memory.memory_operations.get_session") as mock_gs,
            patch(
                "team_memory.services.memory.memory_operations.SearchLogRepository",
                return_value=mock_repo,
            ),
        ):
            mock_gs.return_value.__aenter__ = AsyncMock(return_value=mock_session)
            mock_gs.return_value.__aexit__ = AsyncMock(return_value=False)

            log_id = await write_search_log(
                db_url="sqlite://",
                query="test telemetry",
                stage_metrics={"total_ms": 25.0},
                stage_counts={"final": 3},
                returned_tokens=310,
            )

        assert log_id == str(mock_log.id)
        mock_repo.create.assert_awaited_once_with(
            query="test telemetry",
            intent_type="unknown",
            project="default",
            source="mcp",
            result_ids=None,
            stage_metrics={"total_ms": 25.0},
            stage_counts={"final": 3},
            returned_tokens=310,
        )

    @pytest.mark.asyncio
    async def test_search_log_writer_telemetry_summary(self):
        writer = SearchLogWriter(dsn="postgresql://localhost/test")
        mock_conn = AsyncMock()
        mock_conn.fetchrow = AsyncMock(
            return_value={
                "avg_returned_tokens": 512.34,
                "avg_total_ms": 42.678,
                "graph_assisted_searches": 7,
            }
        )
        mock_pool = MagicMock()
        mock_pool._closed = False
        mock_acquire = MagicMock()
        mock_acquire.__aenter__ = AsyncMock(return_value=mock_conn)
        mock_acquire.__aexit__ = AsyncMock(return_value=False)
        mock_pool.acquire.return_value = mock_acquire
        writer._pool = mock_pool

        summary = await writer.get_telemetry_summary(days=7)
        assert summary == {
            "days": 7,
            "avg_returned_tokens": 512.3,
            "avg_total_ms": 42.68,
            "graph_assisted_searches": 7,
        }

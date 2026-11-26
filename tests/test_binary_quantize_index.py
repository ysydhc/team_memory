"""Tests for pgvector 4096-dim binary-quantized HNSW indexes and two-stage reranking.

Covers:
  1. SearchConfig toggles (binary_quantize_index, binary_quantize_oversample).
  2. Two-stage SQL generation (CTE bq_candidates, binary_quantize + <~>, exact cosine <=> rerank).
  3. Toggle switching (binary_quantize=True vs False, config-driven resolution).
  4. HNSW iterative_scan GUC execution on PostgreSQL sessions (and no-op on non-PG sessions).
  5. Live PostgreSQL integration:
     - Migration 028 HNSW expression indexes exist on experiences / archives / personal_memories.
     - EXPLAIN (ANALYZE, BUFFERS) confirms HNSW index scan is used.
     - Multi-tenant RLS isolation + iterative_scan under tm_app_user without under-recall.
"""

from __future__ import annotations

import math
import os
import uuid
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from sqlalchemy import Select, text
from sqlalchemy.dialects import postgresql

from team_memory.config.search import SearchConfig
from team_memory.storage.database import close_db, get_session
from team_memory.storage.models import DB_VECTOR_DIM, Experience, Tenant
from team_memory.storage.query_builders import (
    MIN_BQ_STAGE1_CANDIDATES,
    build_vector_search,
    compute_bq_stage1_limit,
    enable_hnsw_iterative_scan,
    resolve_binary_quantize_settings,
)
from team_memory.storage.repository import ExperienceRepository

PG_TEST_URL = os.environ.get(
    "TEAM_MEMORY_DB_URL",
    "postgresql+asyncpg://developer:devpass@localhost:5432/team_memory",
)


def _make_unit_vector(seed: int, dim: int = DB_VECTOR_DIM) -> list[float]:
    """Generate a deterministic non-zero unit vector of length ``dim``."""
    raw = [math.sin((i + 1) * (seed + 1) * 0.173) for i in range(dim)]
    norm = math.sqrt(sum(v * v for v in raw)) or 1.0
    return [v / norm for v in raw]


# ============================================================
# 1. Config & Helper Unit Tests
# ============================================================


class TestBinaryQuantizeConfig:
    """Verify SearchConfig defaults and bounds for binary-quantized HNSW search."""

    def test_search_config_defaults(self) -> None:
        cfg = SearchConfig()
        assert cfg.binary_quantize_index is True
        assert cfg.binary_quantize_oversample == 10

    def test_search_config_custom_values(self) -> None:
        cfg = SearchConfig(binary_quantize_index=False, binary_quantize_oversample=20)
        assert cfg.binary_quantize_index is False
        assert cfg.binary_quantize_oversample == 20

    def test_compute_bq_stage1_limit_enforces_floor(self) -> None:
        assert compute_bq_stage1_limit(limit=3, oversample=10) == MIN_BQ_STAGE1_CANDIDATES
        assert compute_bq_stage1_limit(limit=5, oversample=10) == 50
        assert compute_bq_stage1_limit(limit=10, oversample=10) == 100
        assert compute_bq_stage1_limit(limit=5, oversample=10, offset=5) == 100

    def test_resolve_binary_quantize_settings_from_config(self) -> None:
        mock_settings = MagicMock()
        mock_settings.search.binary_quantize_index = False
        mock_settings.search.binary_quantize_oversample = 15
        with patch("team_memory.config.get_settings", return_value=mock_settings):
            assert resolve_binary_quantize_settings() == (False, 15)
            assert resolve_binary_quantize_settings(binary_quantize=True) == (True, 15)
            assert resolve_binary_quantize_settings(oversample=8) == (False, 8)
            assert resolve_binary_quantize_settings(binary_quantize=True, oversample=12) == (
                True,
                12,
            )


# ============================================================
# 2. Two-Stage SQL Generation & Toggle Switching Tests
# ============================================================


class TestBuildVectorSearchSQL:
    """Verify SQL structure when binary_quantize is enabled vs disabled."""

    def test_two_stage_sql_structure(self) -> None:
        q_vec = [0.1] * DB_VECTOR_DIM
        stmt = build_vector_search(
            q_vec,
            project="team_memory",
            limit=5,
            min_similarity=0.5,
            binary_quantize=True,
            oversample=10,
        )
        assert isinstance(stmt, Select)
        compiled = str(stmt.compile(dialect=postgresql.dialect()))
        assert "WITH bq_candidates AS" in compiled
        assert "binary_quantize(experiences.embedding)" in compiled
        assert f"BIT({DB_VECTOR_DIM})" in compiled
        assert "<~>" in compiled
        assert "<=>" in compiled
        assert "experiences.embedding IS NOT NULL" in compiled

    def test_two_stage_stage1_limit_scales_with_oversample(self) -> None:
        q_vec = [0.1] * DB_VECTOR_DIM
        stmt = build_vector_search(
            q_vec,
            project="team_memory",
            limit=8,
            offset=2,
            binary_quantize=True,
            oversample=12,
        )
        compiled = str(
            stmt.compile(dialect=postgresql.dialect(), compile_kwargs={"literal_binds": True})
        )
        # Stage 1 limit = max((8 + 2) * 12, 50) = 120
        assert "LIMIT 120" in compiled
        # Stage 2 limit = 8, offset = 2
        assert "LIMIT 8" in compiled
        assert "OFFSET 2" in compiled

    def test_single_stage_sql_when_disabled(self) -> None:
        q_vec = [0.1] * DB_VECTOR_DIM
        stmt = build_vector_search(
            q_vec,
            project="team_memory",
            limit=5,
            binary_quantize=False,
        )
        assert isinstance(stmt, Select)
        compiled = str(stmt.compile(dialect=postgresql.dialect()))
        assert "WITH bq_candidates AS" not in compiled
        assert "binary_quantize" not in compiled
        assert "<~>" not in compiled
        assert "<=>" in compiled

    @pytest.mark.asyncio
    async def test_enable_hnsw_iterative_scan_pg_vs_mock(self) -> None:
        # Non-PG / plain AsyncMock session -> skipped
        mock_session = AsyncMock()
        assert await enable_hnsw_iterative_scan(mock_session) is False
        mock_session.execute.assert_not_awaited()

        # PostgreSQL-bound session -> executes SET LOCAL hnsw.iterative_scan = relaxed_order
        pg_session = AsyncMock()
        pg_session.sync_session = MagicMock()
        pg_session.sync_session.bind = MagicMock()
        pg_session.sync_session.bind.dialect = MagicMock()
        pg_session.sync_session.bind.dialect.name = "postgresql"

        assert await enable_hnsw_iterative_scan(pg_session) is True
        pg_session.execute.assert_awaited_once()
        sql_arg = str(pg_session.execute.await_args.args[0])
        assert "SET LOCAL hnsw.iterative_scan = relaxed_order" in sql_arg

    @pytest.mark.asyncio
    async def test_repository_respects_binary_quantize_toggle(self) -> None:
        pg_session = AsyncMock()
        pg_session.sync_session = MagicMock()
        pg_session.sync_session.bind = MagicMock()
        pg_session.sync_session.bind.dialect = MagicMock()
        pg_session.sync_session.bind.dialect.name = "postgresql"
        mock_result = MagicMock()
        mock_result.all.return_value = []
        pg_session.execute.return_value = mock_result

        repo = ExperienceRepository(pg_session)
        q_vec = [0.1] * DB_VECTOR_DIM

        # Enabled -> 2 calls: SET LOCAL + two-stage SELECT
        await repo.search_by_vector(q_vec, binary_quantize=True, oversample=10)
        assert pg_session.execute.await_count == 2
        first_sql = str(pg_session.execute.await_args_list[0].args[0])
        second_sql = str(
            pg_session.execute.await_args_list[1].args[0].compile(dialect=postgresql.dialect())
        )
        assert "hnsw.iterative_scan = relaxed_order" in first_sql
        assert "bq_candidates" in second_sql

        # Disabled -> 1 call: single-stage SELECT only
        pg_session.execute.reset_mock()
        await repo.search_by_vector(q_vec, binary_quantize=False)
        assert pg_session.execute.await_count == 1
        only_sql = str(
            pg_session.execute.await_args_list[0].args[0].compile(dialect=postgresql.dialect())
        )
        assert "bq_candidates" not in only_sql
        assert "binary_quantize" not in only_sql


# ============================================================
# 3. Live PostgreSQL Integration & RLS + EXPLAIN Verification
# ============================================================


async def _pg_available() -> bool:
    try:
        async with get_session(PG_TEST_URL, bypass_rls=True) as session:
            val = (await session.execute(text("SELECT 1"))).scalar()
            return val == 1
    except Exception:
        return False


class TestLivePostgresBinaryQuantizeHNSW:
    """Integration tests against native PostgreSQL (:5432) with pgvector 0.8+."""

    @pytest.mark.asyncio
    async def test_hnsw_indexes_present(self) -> None:
        await close_db()
        if not await _pg_available():
            pytest.skip("Native PostgreSQL not reachable")

        try:
            async with get_session(PG_TEST_URL, bypass_rls=True) as session:
                rows = (
                    await session.execute(
                        text(
                            """
                            SELECT tablename, indexname, indexdef
                            FROM pg_indexes
                            WHERE indexname IN (
                                'ix_experiences_embedding_bq_hnsw',
                                'ix_archives_embedding_bq_hnsw',
                                'ix_personal_memories_embedding_bq_hnsw'
                            )
                            ORDER BY tablename
                            """
                        )
                    )
                ).all()
            found = {r[1]: r[2] for r in rows}
            assert "ix_experiences_embedding_bq_hnsw" in found
            assert "ix_archives_embedding_bq_hnsw" in found
            assert "ix_personal_memories_embedding_bq_hnsw" in found
            for idx_def in found.values():
                assert "USING hnsw" in idx_def
                assert "binary_quantize" in idx_def
                assert "bit_hamming_ops" in idx_def
                assert "WHERE (embedding IS NOT NULL)" in idx_def
        finally:
            await close_db()

    @pytest.mark.asyncio
    async def test_explain_analyze_uses_hnsw_index(self) -> None:
        await close_db()
        if not await _pg_available():
            pytest.skip("Native PostgreSQL not reachable")

        try:
            q_vec = _make_unit_vector(42)
            vec_literal = "[" + ",".join(f"{x:.6f}" for x in q_vec) + "]"
            async with get_session(PG_TEST_URL, bypass_rls=True) as session:
                await session.execute(text("SET LOCAL hnsw.iterative_scan = relaxed_order"))
                plan_rows = (
                    await session.execute(
                        text(
                            """
                            EXPLAIN (ANALYZE, BUFFERS)
                            WITH bq_candidates AS (
                                SELECT id, embedding
                                FROM experiences
                                WHERE embedding IS NOT NULL
                                ORDER BY binary_quantize(embedding)::bit(4096)
                                         <~> binary_quantize(CAST(:q AS vector(4096)))::bit(4096)
                                LIMIT 50
                            )
                            SELECT id, (1 - (embedding <=> CAST(:q AS vector(4096)))) AS similarity
                            FROM bq_candidates
                            ORDER BY similarity DESC
                            LIMIT 5
                            """
                        ),
                        {"q": vec_literal},
                    )
                ).scalars().all()
            plan_text = "\n".join(plan_rows)
            assert "ix_experiences_embedding_bq_hnsw" in plan_text, (
                f"Expected HNSW index scan in EXPLAIN output:\n{plan_text}"
            )
            assert "Index Scan" in plan_text
        finally:
            await close_db()

    @pytest.mark.asyncio
    async def test_multitenant_rls_two_stage_search_and_isolation(self) -> None:
        """Verify two-stage binary-quantized search under RLS (tm_app_user).

        Creates two ephemeral tenants with orthogonal 4096-dim vectors, verifies:
          - SET LOCAL hnsw.iterative_scan = relaxed_order succeeds under tm_app_user.
          - Tenant A recalls its own exact match with high cosine similarity and
            never sees Tenant B's records.
          - Both binary_quantize=True and binary_quantize=False return the same top-1.
        """
        await close_db()
        if not await _pg_available():
            pytest.skip("Native PostgreSQL not reachable")

        tid_a = uuid.uuid4()
        tid_b = uuid.uuid4()
        slug_suffix = uuid.uuid4().hex[:8]
        proj = f"bq_rls_test_{slug_suffix}"
        vec_a = _make_unit_vector(101)
        vec_b = _make_unit_vector(202)

        try:
            # 1. Seed two tenants + one experience each (bypass_rls=True for tenant creation)
            async with get_session(PG_TEST_URL, bypass_rls=True) as session:
                session.add(Tenant(id=tid_a, slug=f"bq-a-{slug_suffix}", name="BQ Tenant A"))
                session.add(Tenant(id=tid_b, slug=f"bq-b-{slug_suffix}", name="BQ Tenant B"))
                await session.flush()

            async with get_session(PG_TEST_URL, tenant_id=tid_a) as session:
                exp_a = Experience(
                    id=uuid.uuid4(),
                    tenant_id=tid_a,
                    title="BQ RLS Tenant A Target",
                    description="Pgvector binary quantization test record A",
                    solution="Use HNSW iterative_scan relaxed_order",
                    created_by="tester_a",
                    project=proj,
                    exp_status="published",
                    embedding=vec_a,
                )
                session.add(exp_a)

            async with get_session(PG_TEST_URL, tenant_id=tid_b) as session:
                exp_b = Experience(
                    id=uuid.uuid4(),
                    tenant_id=tid_b,
                    title="BQ RLS Tenant B Secret",
                    description="Pgvector binary quantization test record B",
                    solution="Secret tenant B solution",
                    created_by="tester_b",
                    project=proj,
                    exp_status="published",
                    embedding=vec_b,
                )
                session.add(exp_b)

            # 2. Query as Tenant A with binary_quantize=True
            async with get_session(PG_TEST_URL, tenant_id=tid_a) as session:
                repo = ExperienceRepository(session)
                bq_results = await repo.search_by_vector(
                    vec_a,
                    max_results=5,
                    min_similarity=0.8,
                    project=proj,
                    binary_quantize=True,
                    oversample=10,
                )
                exact_results = await repo.search_by_vector(
                    vec_a,
                    max_results=5,
                    min_similarity=0.8,
                    project=proj,
                    binary_quantize=False,
                )
                # Querying Tenant B's vector under Tenant A RLS must NOT leak Tenant B's row
                cross_results = await repo.search_by_vector(
                    vec_b,
                    max_results=5,
                    min_similarity=0.0,
                    project=proj,
                    binary_quantize=True,
                    oversample=10,
                )

            assert len(bq_results) == 1
            assert bq_results[0]["title"] == "BQ RLS Tenant A Target"
            assert bq_results[0]["similarity"] >= 0.99
            assert [r["id"] for r in bq_results] == [r["id"] for r in exact_results]
            assert all(r["title"] != "BQ RLS Tenant B Secret" for r in cross_results)
        finally:
            # Clean up ephemeral tenants (CASCADE deletes their experiences)
            try:
                async with get_session(PG_TEST_URL, bypass_rls=True) as session:
                    await session.execute(
                        text("DELETE FROM tenants WHERE id IN (:a, :b)"),
                        {"a": str(tid_a), "b": str(tid_b)},
                    )
            finally:
                await close_db()

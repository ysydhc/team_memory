"""Tests for Refinement Quality Gate auto-repair & Multi-Tenant LLM/Embedding Usage Ledger.

Covers:
  1. Front-loaded Quality Gate constraints in refinement prompt & structural validation
  2. 1-pass lightweight auto-repair for high-value drafts (e.g. short title / missing tags)
  3. Discard classification: ``true_noise`` vs ``format_error`` vs ``format_repaired``
  4. Non-blocking async Usage Ledger in LLMClient & Embedding providers (fault-isolated)
  5. Multi-tenant PostgreSQL RLS isolation on ``llm_usage_logs``
  6. 7d / 30d per-caller observability aggregation & `/api/observability/overview`
  7. Janitor 90-day TTL cleanup (``prune_llm_usage_logs``)
"""

from __future__ import annotations

import os
import uuid
from datetime import datetime, timedelta, timezone
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest
from fastapi import FastAPI
from sqlalchemy import text

from team_memory.auth.provider import User
from team_memory.daemon.refinement.draft_refiner import (
    MAX_TITLE_LEN,
    MIN_TITLE_LEN,
    auto_repair_structured_draft,
    classify_discard_category,
    is_empty_shell_draft,
    validate_structured_draft,
)
from team_memory.daemon.refinement.refinement_worker import (
    _REFINEMENT_PROMPT,
    RefinementWorker,
)
from team_memory.daemon.runtime.config import DaemonConfig, RefinementSettings
from team_memory.embedding.openai_provider import OpenAIEmbedding
from team_memory.services.analysis.observability import get_observability_overview
from team_memory.services.analysis.usage_ledger import (
    configure_usage_ledger,
    estimate_tokens,
    flush_usage_ledger,
    record_usage_async,
    write_usage_log,
)
from team_memory.services.llm.llm_client import LLMClient
from team_memory.services.maintenance.janitor import MemoryJanitor
from team_memory.storage.database import close_db, get_session
from team_memory.storage.tenant_context import tenant_scope
from team_memory.web.auth_session import get_current_user
from team_memory.web.routes import observability as obs_route

PG_TEST_URL = os.environ.get(
    "TEAM_MEMORY_DB_URL",
    "postgresql+asyncpg://developer:devpass@localhost:5432/team_memory",
)


def _make_worker_config(**overrides: Any) -> DaemonConfig:
    defaults = dict(
        enabled=True,
        provider="litellm",
        model="test-model",
        base_url="http://localhost:9999/v1",
        api_key_env="TEST_API_KEY",
        timeout=5,
        scan_interval_seconds=1,
        max_input_chars=4000,
        max_retries=1,
        fallback_on_failure=True,
        quality_gate_enabled=True,
        quality_min_score=60,
    )
    defaults.update(overrides)
    cfg = DaemonConfig()
    cfg.refinement = RefinementSettings(**defaults)
    return cfg


def _rich_problem() -> str:
    return (
        "在生产环境部署 FastAPI 服务连接本机 PostgreSQL 数据库时，高并发下频繁出现 "
        "asyncpg connection pool timeout 报错，导致部分检索请求返回 500 错误。"
    )


def _rich_solution() -> str:
    return (
        "调整 SQLAlchemy create_async_engine 的 pool_size=10、max_overflow=20、pool_pre_ping=True，"
        "并在 PostgreSQL postgresql.conf 中将 max_connections 提升至 200 后重启验证恢复。"
    )


# ============================================================
# 1. Refinement Prompt, Format Validation & Auto-Repair
# ============================================================


class TestRefinementFormatGovernance:
    def test_prompt_frontloads_quality_gate_constraints(self) -> None:
        assert "5~120" in _REFINEMENT_PROMPT
        assert "50" in _REFINEMENT_PROMPT
        assert "tags" in _REFINEMENT_PROMPT
        assert "空壳草稿" in _REFINEMENT_PROMPT

    def test_validate_hard_format_constraints(self) -> None:
        valid = {
            "title": "PostgreSQL 连接池超时排查与参数调优",
            "problem": _rich_problem(),
            "solution": _rich_solution(),
            "tags": ["postgresql", "asyncpg"],
            "experience_type": "bugfix",
        }
        ok, reason = validate_structured_draft(valid)
        assert ok is True
        assert reason == ""

        # Title too long
        too_long = dict(valid, title="A" * (MAX_TITLE_LEN + 5))
        ok, reason = validate_structured_draft(too_long)
        assert ok is False
        assert "title too long" in reason

        # Generic title
        generic = dict(valid, title="调试")
        ok, reason = validate_structured_draft(generic)
        assert ok is False
        assert "title too short" in reason or "title too generic" in reason

        # Empty tags
        no_tags = dict(valid, tags=[])
        ok, reason = validate_structured_draft(no_tags)
        assert ok is False
        assert "tags empty" in reason

        # Placeholder problem
        placeholder = dict(valid, problem="暂无")
        ok, reason = validate_structured_draft(placeholder)
        assert ok is False
        assert "problem too short" in reason

    def test_auto_repair_fixes_short_title_and_missing_tags(self) -> None:
        short_title_draft = {
            "title": "PG",  # too short (< 5)
            "problem": _rich_problem(),
            "solution": _rich_solution(),
            "tags": [],
            "experience_type": "",
        }
        repaired, fields = auto_repair_structured_draft(short_title_draft)
        assert set(fields) == {"title", "tags", "experience_type"}
        assert MIN_TITLE_LEN <= len(repaired["title"]) <= MAX_TITLE_LEN
        assert len(repaired["tags"]) >= 1
        assert repaired["experience_type"] == "bugfix"

        ok, reason = validate_structured_draft(repaired)
        assert ok is True, reason

    def test_auto_repair_does_not_fabricate_empty_shell_noise(self) -> None:
        empty_shell = {
            "title": "",
            "problem": "",
            "solution": "",
            "tags": [],
            "experience_type": "",
        }
        assert is_empty_shell_draft(empty_shell) is True
        repaired, fields = auto_repair_structured_draft(empty_shell)
        assert fields == []
        assert repaired == empty_shell
        assert classify_discard_category(empty_shell, "discarded: title too short") == "true_noise"

    def test_discard_category_distinguishes_noise_vs_format_error(self) -> None:
        noise = {"title": "hi", "problem": "hello", "solution": "", "tags": [], "experience_type": ""}
        assert classify_discard_category(noise, "discarded: title too short") == "true_noise"

        format_err = {
            "title": "PostgreSQL 索引配置问题",
            "problem": _rich_problem(),
            "solution": "重启了一下",  # too short, cannot auto-fabricate
            "tags": ["postgresql"],
            "experience_type": "bugfix",
        }
        assert classify_discard_category(format_err, "discarded: solution too short") == "format_error"

    @pytest.mark.asyncio
    async def test_refinement_worker_auto_repairs_and_tracks_discard_breakdown(self) -> None:
        cfg = _make_worker_config(quality_gate_enabled=True, quality_min_score=60)
        buf = AsyncMock()
        sink = AsyncMock()
        sink.recall = AsyncMock(return_value=[])
        sink.draft_save = AsyncMock(return_value={"id": "exp-auto-1"})
        sink.draft_publish = AsyncMock(return_value={})
        sink.update_experience = AsyncMock(return_value={})
        metrics = MagicMock()

        worker = RefinementWorker(cfg, buf, sink, metrics=metrics)

        # Case A: High-value draft with short title & missing tags -> auto-repaired & published
        draft_a = {
            "id": "d-repair-1",
            "title": "Draft d-repair",
            "content": _rich_problem() + "\n" + _rich_solution(),
            "project": "team_doc",
            "conversation_id": "conv-a",
        }
        with (
            patch.object(
                worker,
                "_call_llm",
                new_callable=AsyncMock,
                return_value={
                    "title": "PG",
                    "problem": _rich_problem(),
                    "solution": _rich_solution(),
                    "tags": [],
                    "experience_type": "bugfix",
                },
            ),
            patch.object(worker, "_quality_score", new_callable=AsyncMock, return_value=85),
        ):
            await worker._refine_one(draft_a)

        sink.draft_publish.assert_awaited_once()
        sink.update_experience.assert_awaited_once()
        updated_title = sink.update_experience.call_args.kwargs["title"]
        updated_tags = sink.update_experience.call_args.kwargs["tags"]
        assert MIN_TITLE_LEN <= len(updated_title) <= MAX_TITLE_LEN
        assert len(updated_tags) >= 1

        # Case B: Pure noise empty shell -> discarded as true_noise
        draft_b = {
            "id": "d-noise-1",
            "title": "Draft d-noise",
            "content": "hello world",
            "project": "team_doc",
            "conversation_id": "conv-b",
        }
        with patch.object(
            worker,
            "_call_llm",
            new_callable=AsyncMock,
            return_value={"title": "", "problem": "", "solution": "", "tags": [], "experience_type": ""},
        ):
            await worker._refine_one(draft_b)

        # Case C: Non-trivial problem but unfixably short solution -> discarded as format_error
        draft_c = {
            "id": "d-fmt-1",
            "title": "Draft d-fmt",
            "content": _rich_problem(),
            "project": "team_doc",
            "conversation_id": "conv-c",
        }
        with patch.object(
            worker,
            "_call_llm",
            new_callable=AsyncMock,
            return_value={
                "title": "FastAPI 连接池问题排查",
                "problem": _rich_problem(),
                "solution": "重启解决",
                "tags": ["fastapi"],
                "experience_type": "bugfix",
            },
        ):
            await worker._refine_one(draft_c)

        stats = worker.get_discard_stats()
        assert stats["format_repaired"] == 1
        assert stats["true_noise"] == 1
        assert stats["format_error"] == 1


# ============================================================
# 2. Usage Ledger Unit & Fault-Isolation Tests
# ============================================================


class TestUsageLedgerUnit:
    def test_estimate_tokens(self) -> None:
        assert estimate_tokens("") == 0
        assert estimate_tokens(None) == 0
        assert estimate_tokens("hello world from team memory") >= 5
        assert estimate_tokens("这是一个中文技术经验标题测试") >= 8
        assert estimate_tokens([{"role": "user", "content": "SELECT * FROM llm_usage_logs"}]) >= 5

    @pytest.mark.asyncio
    async def test_llm_client_records_usage_and_ignores_ledger_errors(self) -> None:
        fake_resp = MagicMock()
        fake_resp.raise_for_status = MagicMock()
        fake_resp.json.return_value = {
            "choices": [{"message": {"content": "ok reply"}}],
            "usage": {"prompt_tokens": 120, "completion_tokens": 35},
        }
        fake_http = AsyncMock()
        fake_http.post = AsyncMock(return_value=fake_resp)
        fake_http.__aenter__ = AsyncMock(return_value=fake_http)
        fake_http.__aexit__ = AsyncMock(return_value=False)

        recorded: list[dict[str, Any]] = []

        def _capture(**kwargs: Any) -> None:
            recorded.append(kwargs)

        client = LLMClient(base_url="http://mock-llm/v1", model="test-llm", provider="openai")
        with (
            patch("team_memory.services.llm.llm_client.httpx.AsyncClient", return_value=fake_http),
            patch("team_memory.services.llm.llm_client.record_usage_async", side_effect=_capture),
        ):
            out = await client.chat("sys", "user prompt", caller="unit_llm")

        assert out == "ok reply"
        assert len(recorded) == 1
        assert recorded[0]["caller"] == "unit_llm"
        assert recorded[0]["prompt_tokens"] == 120
        assert recorded[0]["completion_tokens"] == 35
        assert recorded[0]["success"] is True

        # Ledger write failure must NEVER break main LLM business path
        configure_usage_ledger("postgresql+asyncpg://invalid:invalid@127.0.0.1:1/nonexistent")
        try:
            with patch("team_memory.services.llm.llm_client.httpx.AsyncClient", return_value=fake_http):
                out2 = await client.chat("sys", "user prompt", caller="fault_test")
            assert out2 == "ok reply"
            await flush_usage_ledger(timeout=2.0)
        finally:
            configure_usage_ledger(None)

    @pytest.mark.asyncio
    async def test_embedding_provider_records_usage_non_blocking(self) -> None:
        fake_resp = MagicMock()
        fake_resp.raise_for_status = MagicMock()
        fake_resp.json.return_value = {
            "data": [{"index": 0, "embedding": [0.1, 0.2, 0.3]}],
            "usage": {"prompt_tokens": 42},
        }
        fake_http = AsyncMock()
        fake_http.post = AsyncMock(return_value=fake_resp)
        fake_http.is_closed = False

        recorded: list[dict[str, Any]] = []

        provider = OpenAIEmbedding(api_key="test-key", model="text-embedding-3-small", dim=3)
        with (
            patch.object(provider, "_get_client", return_value=fake_http),
            patch(
                "team_memory.embedding.openai_provider.record_usage_async",
                side_effect=lambda **kw: recorded.append(kw),
            ),
        ):
            vecs = await provider.encode(["hello embedding world"])

        assert vecs == [[0.1, 0.2, 0.3]]
        assert len(recorded) == 1
        assert recorded[0]["caller"] == "embedding"
        assert recorded[0]["provider"] == "openai"
        assert recorded[0]["prompt_tokens"] == 42
        assert recorded[0]["completion_tokens"] == 0
        assert recorded[0]["success"] is True


# ============================================================
# 3. Native PostgreSQL RLS Isolation, Observability & Janitor TTL
# ============================================================


@pytest.mark.asyncio
async def test_usage_ledger_rls_observability_and_janitor_ttl() -> None:
    """Integration test against local native PostgreSQL (:5432)."""
    tenant_a = uuid.uuid4()
    tenant_b = uuid.uuid4()
    now = datetime.now(timezone.utc)

    await close_db()
    configure_usage_ledger(PG_TEST_URL)
    try:
        # Seed two isolated test tenants (superuser session)
        async with get_session(PG_TEST_URL, bypass_rls=True) as session:
            await session.execute(
                text(
                    "INSERT INTO tenants (id, slug, name) VALUES (:id, :slug, :name)"
                ),
                {"id": tenant_a, "slug": f"test-usage-a-{tenant_a.hex[:8]}", "name": "Test Tenant A"},
            )
            await session.execute(
                text(
                    "INSERT INTO tenants (id, slug, name) VALUES (:id, :slug, :name)"
                ),
                {"id": tenant_b, "slug": f"test-usage-b-{tenant_b.hex[:8]}", "name": "Test Tenant B"},
            )

        # 1. Async fire-and-forget write under Tenant A scope
        with tenant_scope(tenant_a):
            task = record_usage_async(
                caller="refinement",
                provider="openai",
                model="gbu-gemini-3.8-flash",
                prompt_tokens=300,
                completion_tokens=100,
                latency_ms=420,
                success=True,
            )
            assert task is not None
        await flush_usage_ledger(timeout=5.0)

        # Direct writes: Tenant A (15d ago, 95d ago) + Tenant B (1d ago)
        assert await write_usage_log(
            PG_TEST_URL,
            caller="embedding",
            provider="openai",
            model="text-embedding-3-small",
            prompt_tokens=500,
            completion_tokens=0,
            latency_ms=80,
            success=True,
            tenant_id=tenant_a,
            created_at=now - timedelta(days=15),
        )
        assert await write_usage_log(
            PG_TEST_URL,
            caller="legacy_job",
            provider="openai",
            model="old-model",
            prompt_tokens=999,
            completion_tokens=1,
            latency_ms=100,
            success=True,
            tenant_id=tenant_a,
            created_at=now - timedelta(days=95),
        )
        assert await write_usage_log(
            PG_TEST_URL,
            caller="refinement",
            provider="openai",
            model="gbu-gemini-3.8-flash",
            prompt_tokens=7777,
            completion_tokens=2222,
            latency_ms=500,
            success=True,
            tenant_id=tenant_b,
            created_at=now - timedelta(days=1),
        )

        # 2. Verify strict PostgreSQL RLS isolation + 7d/30d aggregation in observability
        with tenant_scope(tenant_a):
            overview_a = await get_observability_overview(
                PG_TEST_URL,
                days=30,
                daemon_status_url="http://127.0.0.1:1/unreachable",
            )
        usage_a = overview_a["llm_usage"]
        assert usage_a["window_7d"]["total_calls"] == 1
        assert usage_a["window_7d"]["total_tokens"] == 400
        assert usage_a["window_7d"]["by_caller"][0]["caller"] == "refinement"

        # 30d includes both 0d (400 tok) and 15d (500 tok), excludes 95d and Tenant B (9999 tok)
        assert usage_a["window_30d"]["total_calls"] == 2
        assert usage_a["window_30d"]["total_tokens"] == 900
        callers_30d = {item["caller"]: item for item in usage_a["window_30d"]["by_caller"]}
        assert callers_30d["refinement"]["total_tokens"] == 400
        assert callers_30d["embedding"]["total_tokens"] == 500
        assert "legacy_job" not in callers_30d

        # Verify Tenant B only sees its own row
        with tenant_scope(tenant_b):
            overview_b = await get_observability_overview(
                PG_TEST_URL,
                days=30,
                daemon_status_url="http://127.0.0.1:1/unreachable",
            )
        usage_b = overview_b["llm_usage"]
        assert usage_b["window_7d"]["total_calls"] == 1
        assert usage_b["window_7d"]["total_tokens"] == 9999

        # 3. Verify `/api/observability/overview` route exposes llm_usage
        app = FastAPI()
        app.include_router(obs_route.router, prefix="/api")
        app.dependency_overrides[get_current_user] = lambda: User(
            name="alice", role="admin", tenant_id=str(tenant_a)
        )
        fake_ctx = MagicMock(db_url=PG_TEST_URL)
        with (
            patch.object(obs_route, "get_context", return_value=fake_ctx),
            tenant_scope(tenant_a),
        ):
            async with httpx.AsyncClient(
                transport=httpx.ASGITransport(app=app), base_url="http://test"
            ) as api_client:
                resp = await api_client.get("/api/observability/overview?days=30")
        assert resp.status_code == 200
        body = resp.json()
        assert "llm_usage" in body
        assert body["llm_usage"]["window_7d"]["total_tokens"] == 400
        assert body["subsystems"]["infra"]["llm_usage"]["window_30d"]["total_tokens"] == 900

        # 4. Verify Janitor 90-day TTL pruning (dry_run + real delete)
        janitor = MemoryJanitor(PG_TEST_URL)
        with tenant_scope(tenant_a):
            dry_res = await janitor.prune_llm_usage_logs(older_than_days=90, dry_run=True)
            assert dry_res["would_delete_count"] == 1
            assert dry_res["deleted_count"] == 0

            del_res = await janitor.prune_llm_usage_logs(older_than_days=90, dry_run=False)
            assert del_res["deleted_count"] == 1

            # Verify only the 95-day-old row was deleted; 2 recent rows remain
            async with get_session(PG_TEST_URL, tenant_id=tenant_a) as session:
                remaining = int(
                    (await session.execute(text("SELECT COUNT(*) FROM llm_usage_logs"))).scalar() or 0
                )
            assert remaining == 2
    finally:
        configure_usage_ledger(None)
        # Clean up test tenants (ON DELETE CASCADE removes their llm_usage_logs rows)
        async with get_session(PG_TEST_URL, bypass_rls=True) as session:
            await session.execute(
                text("DELETE FROM tenants WHERE id IN (:ta, :tb)"),
                {"ta": tenant_a, "tb": tenant_b},
            )
        await close_db()

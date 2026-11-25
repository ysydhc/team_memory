"""Comprehensive unit tests for upgraded AutoDreamWorker."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from team_memory.daemon.integration import auto_dream as ad_mod
from team_memory.daemon.integration.auto_dream import AutoDreamWorker
from team_memory.daemon.runtime.config import AutoDreamSettings, DaemonConfig, RefinementSettings
from team_memory.services.llm.llm_client import LLMError

pytestmark = pytest.mark.unit


def _make_worker(
    *,
    llm_client: AsyncMock | None = None,
    dream_overrides: dict | None = None,
    refinement_overrides: dict | None = None,
) -> tuple[AutoDreamWorker, AsyncMock]:
    cfg = DaemonConfig()
    cfg.auto_dream = AutoDreamSettings(**(dream_overrides or {}))
    cfg.refinement = RefinementSettings(**(refinement_overrides or {}))
    sink = AsyncMock()
    sink.recall = AsyncMock(return_value=[])
    sink.update_experience = AsyncMock(return_value={"id": "exp-1", "status": "published"})
    sink.supersede_experience = AsyncMock(return_value={"resolved": True})
    sink.save = AsyncMock(return_value={"id": "playbook-1", "status": "published"})
    worker = AutoDreamWorker(config=cfg, sink=sink, llm_client=llm_client)
    return worker, sink


def _exp(
    eid: str,
    *,
    title: str = "Exp",
    description: str = "Problem",
    solution: str = "Solution",
    project: str = "team_doc",
    quality_tier: str = "Silver",
    quality_score: float = 80.0,
    is_pinned: bool = False,
    recall_count: int = 1,
    used_count: int = 0,
    created_at: str = "2026-11-20T10:00:00+00:00",
    tags: list[str] | None = None,
    embedding: list[float] | None = None,
) -> dict:
    return {
        "id": eid,
        "title": title,
        "description": description,
        "solution": solution,
        "project": project,
        "quality_tier": quality_tier,
        "quality_score": quality_score,
        "is_pinned": is_pinned,
        "recall_count": recall_count,
        "used_count": used_count,
        "valid_until": None,
        "created_at": created_at,
        "tags": tags or [],
        "experience_type": "general",
        "embedding": embedding,
    }


@pytest.fixture
def isolated_cache(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(ad_mod, "_CACHE_DIR", tmp_path)
    monkeypatch.setattr(ad_mod, "_LAST_DREAM_FILE", tmp_path / "last_dream.txt")
    monkeypatch.setattr(ad_mod, "_LAST_DREAM_STATS_FILE", tmp_path / "last_dream_stats.json")
    monkeypatch.setattr(ad_mod, "_LOCK_FILE", tmp_path / "auto_dream.lock")
    return tmp_path


class TestConsolidateActions:
    """Tests for MERGE, SUPERSEDE, KEEP_BOTH, protection guardrails, and LLM error fallback."""

    @pytest.mark.asyncio
    async def test_merge_rewrites_and_soft_deletes_secondary(self):
        llm = AsyncMock()
        llm.chat_json = AsyncMock(
            return_value={
                "action": "MERGE",
                "reason": "Complementary steps for PG port setup",
                "primary_id": "exp-1",
                "secondary_ids": ["exp-2"],
                "merged": {
                    "title": "PostgreSQL Docker Port 5433 Setup",
                    "problem": "Local PG container conflicts with default 5432",
                    "solution": "As of 2026-11-20, map Docker PG to host port 5433 and run alembic upgrade head.",
                    "tags": ["postgres", "docker"],
                },
            }
        )
        worker, sink = _make_worker(llm_client=llm)
        cluster = [
            _exp("exp-1", title="PG port 5433", solution="Yesterday we moved PG to 5433"),
            _exp("exp-2", title="PG alembic after port move", solution="Run alembic upgrade head on 5433"),
        ]

        with patch.object(worker, "_soft_delete", new_callable=AsyncMock) as mock_del:
            res = await worker._phase_consolidate([cluster], dry_run=False)

        assert res["merged"] == 1
        assert res["errors"] == 0
        sink.update_experience.assert_called_once()
        kw = sink.update_experience.call_args.kwargs
        assert kw["experience_id"] == "exp-1"
        assert "---" not in kw["solution"]
        assert "2026-11-20" in kw["solution"]
        mock_del.assert_called_once_with("exp-2")

    @pytest.mark.asyncio
    async def test_supersede_calls_bitemporal_interface(self):
        llm = AsyncMock()
        llm.chat_json = AsyncMock(
            return_value={
                "action": "SUPERSEDE",
                "reason": "Embedding migrated from nomic 768d to qwen3-embedding-8b 4096d",
                "winner_id": "exp-new",
                "superseded_ids": ["exp-old"],
            }
        )
        worker, sink = _make_worker(llm_client=llm)
        cluster = [
            _exp("exp-old", title="Use nomic-embed-text 768d", created_at="2026-01-10T00:00:00+00:00"),
            _exp("exp-new", title="Use Qwen3-Embedding-8B 4096d", created_at="2026-11-20T00:00:00+00:00"),
        ]

        with patch.object(worker, "_soft_delete", new_callable=AsyncMock) as mock_del:
            res = await worker._phase_consolidate([cluster], dry_run=False)

        assert res["superseded"] == 1
        assert res["merged"] == 0
        assert res["errors"] == 0
        sink.supersede_experience.assert_called_once_with(
            old_experience_id="exp-old",
            new_experience_id="exp-new",
            reason="Embedding migrated from nomic 768d to qwen3-embedding-8b 4096d",
        )
        mock_del.assert_not_called()

    @pytest.mark.asyncio
    async def test_keep_both_makes_no_writes(self):
        llm = AsyncMock()
        llm.chat_json = AsyncMock(
            return_value={
                "action": "KEEP_BOTH",
                "reason": "One applies to Android SDK, the other to iOS SDK",
            }
        )
        worker, sink = _make_worker(llm_client=llm)
        cluster = [_exp("exp-a"), _exp("exp-b")]

        with patch.object(worker, "_soft_delete", new_callable=AsyncMock) as mock_del:
            res = await worker._phase_consolidate([cluster], dry_run=False)

        assert res["kept"] == 1
        assert res["merged"] == 0
        assert res["superseded"] == 0
        sink.update_experience.assert_not_called()
        sink.supersede_experience.assert_not_called()
        mock_del.assert_not_called()

    @pytest.mark.asyncio
    async def test_pinned_and_gold_cannot_be_superseded_or_deleted(self):
        # Case 1: SUPERSEDE targeting a pinned experience → downgraded to KEEP_BOTH
        llm = AsyncMock()
        llm.chat_json = AsyncMock(
            return_value={
                "action": "SUPERSEDE",
                "reason": "Newer note",
                "winner_id": "exp-silver",
                "superseded_ids": ["exp-pinned"],
            }
        )
        worker, sink = _make_worker(llm_client=llm)
        cluster = [
            _exp("exp-pinned", is_pinned=True, quality_tier="Silver"),
            _exp("exp-silver", is_pinned=False, quality_tier="Silver"),
        ]

        res = await worker._phase_consolidate([cluster], dry_run=False)
        assert res["kept"] == 1
        assert res["superseded"] == 0
        sink.supersede_experience.assert_not_called()

        # Case 2: MERGE trying to soft-delete a Gold experience → forces Gold as primary
        llm.chat_json = AsyncMock(
            return_value={
                "action": "MERGE",
                "reason": "Duplicate",
                "primary_id": "exp-silver",
                "secondary_ids": ["exp-gold"],
                "merged": {"title": "Merged", "problem": "P", "solution": "S", "tags": ["t"]},
            }
        )
        cluster2 = [
            _exp("exp-gold", quality_tier="Gold"),
            _exp("exp-silver", quality_tier="Silver"),
        ]
        with patch.object(worker, "_soft_delete", new_callable=AsyncMock) as mock_del:
            res2 = await worker._phase_consolidate([cluster2], dry_run=False)
        assert res2["merged"] == 1
        assert sink.update_experience.call_args.kwargs["experience_id"] == "exp-gold"
        mock_del.assert_called_once_with("exp-silver")

        # Case 3: Two protected items (Gold + pinned) → downgraded to KEEP_BOTH
        sink.update_experience.reset_mock()
        cluster3 = [
            _exp("exp-gold", quality_tier="Gold"),
            _exp("exp-pinned", is_pinned=True),
        ]
        llm.chat_json = AsyncMock(
            return_value={
                "action": "MERGE",
                "reason": "Duplicate",
                "primary_id": "exp-gold",
                "secondary_ids": ["exp-pinned"],
                "merged": {"title": "Merged", "problem": "P", "solution": "S", "tags": ["t"]},
            }
        )
        with patch.object(worker, "_soft_delete", new_callable=AsyncMock) as mock_del3:
            res3 = await worker._phase_consolidate([cluster3], dry_run=False)
        assert res3["kept"] == 1
        assert res3["merged"] == 0
        sink.update_experience.assert_not_called()
        mock_del3.assert_not_called()

    @pytest.mark.asyncio
    async def test_llm_error_skips_safely_without_data_loss(self):
        llm = AsyncMock()
        llm.chat_json = AsyncMock(side_effect=LLMError("Gateway 503"))
        worker, sink = _make_worker(llm_client=llm)
        cluster = [_exp("exp-1"), _exp("exp-2")]

        with patch.object(worker, "_soft_delete", new_callable=AsyncMock) as mock_del:
            res = await worker._phase_consolidate([cluster], dry_run=False)

        assert res["errors"] == 1
        assert res["skipped"] == 1
        assert res["merged"] == 0
        assert res["superseded"] == 0
        sink.update_experience.assert_not_called()
        sink.supersede_experience.assert_not_called()
        mock_del.assert_not_called()


class TestWatermarkLockAndDryRun:
    """Tests for watermark safety on DB failure, concurrency locking, and dry-run zero side effects."""

    @pytest.mark.asyncio
    async def test_db_failure_does_not_advance_watermark(self, isolated_cache: Path):
        worker, _ = _make_worker()
        mock_ctx = MagicMock()
        mock_ctx.service.list_published = AsyncMock(side_effect=RuntimeError("PG connection refused"))

        with patch("team_memory.daemon.integration.auto_dream._daemon_context", return_value=mock_ctx):
            summary = await worker.run_once(dry_run=False, force=True, trigger="scheduled")

        assert summary["success"] is False
        assert summary["watermark_updated"] is False
        assert not (isolated_cache / "last_dream.txt").exists()
        stats_file = isolated_cache / "last_dream_stats.json"
        assert stats_file.exists()
        saved = json.loads(stats_file.read_text(encoding="utf-8"))
        assert saved["success"] is False
        assert saved["watermark_updated"] is False

    @pytest.mark.asyncio
    async def test_dry_run_has_zero_side_effects(self, isolated_cache: Path):
        llm = AsyncMock()
        llm.chat_json = AsyncMock(
            return_value={
                "action": "MERGE",
                "reason": "Duplicate",
                "primary_id": "exp-1",
                "secondary_ids": ["exp-2"],
                "merged": {
                    "title": "Merged Title",
                    "problem": "Merged Problem",
                    "solution": "Merged Solution",
                    "tags": ["tag1"],
                },
            }
        )
        worker, sink = _make_worker(llm_client=llm, dream_overrides={"distill_enabled": False})

        e1 = _exp("exp-1", embedding=[0.1, 0.2])
        e2 = _exp("exp-2", embedding=[0.1, 0.2])
        e2_hit = {**e2, "similarity": 0.91}

        mock_ctx = MagicMock()
        mock_ctx.service.list_published = AsyncMock(side_effect=[[e1, e2], []])
        mock_ctx.service.list_low_adoption = AsyncMock(return_value=[])
        mock_ctx.service.find_similar_active = AsyncMock(return_value=[e2_hit])
        mock_ctx.service.soft_delete_by_id = AsyncMock()

        with patch("team_memory.daemon.integration.auto_dream._daemon_context", return_value=mock_ctx):
            summary = await worker.run_once(dry_run=True, force=True, trigger="manual")

        assert summary["dry_run"] is True
        assert summary["success"] is True
        assert summary["watermark_updated"] is False
        assert summary["clusters"] == 1
        assert summary["merged"] == 1
        assert len(summary["plan"]) == 1
        assert summary["estimated_tokens"]["input"] > 0

        # Strictly zero side effects
        sink.update_experience.assert_not_called()
        sink.supersede_experience.assert_not_called()
        sink.save.assert_not_called()
        mock_ctx.service.soft_delete_by_id.assert_not_called()
        assert not (isolated_cache / "last_dream.txt").exists()
        assert not (isolated_cache / "last_dream_stats.json").exists()

    @pytest.mark.asyncio
    async def test_concurrent_run_blocked_by_in_process_lock(self, isolated_cache: Path):
        worker, _ = _make_worker()
        gate = asyncio.Event()

        async def slow_dream(*args, **kwargs):
            await gate.wait()
            return {
                "candidates": 0,
                "low_adoption_candidates": 0,
                "clusters": 0,
                "merged": 0,
                "superseded": 0,
                "kept": 0,
                "distilled": 0,
                "pruned": 0,
                "skipped": 0,
                "errors": 0,
                "estimated_tokens": {"input": 0, "output": 0},
                "plan": [],
                "distill_plan": [],
            }

        with patch.object(worker, "_dream", side_effect=slow_dream):
            t1 = asyncio.create_task(worker.run_once(dry_run=True, force=True))
            await asyncio.sleep(0.02)
            second = await worker.run_once(dry_run=True, force=True)
            gate.set()
            first = await t1

        assert second["skipped_run"] is True
        assert second["skipped_reason"] == "locked"
        assert first["skipped_run"] is False

    @pytest.mark.asyncio
    async def test_all_clusters_llm_failure_does_not_advance_watermark(self, isolated_cache: Path):
        llm = AsyncMock()
        llm.chat_json = AsyncMock(side_effect=LLMError("Gateway down"))
        worker, _ = _make_worker(llm_client=llm, dream_overrides={"distill_enabled": False})

        e1 = _exp("exp-1", embedding=[0.1, 0.2])
        e2 = _exp("exp-2", embedding=[0.1, 0.2])
        e2_hit = {**e2, "similarity": 0.91}

        mock_ctx = MagicMock()
        mock_ctx.service.list_published = AsyncMock(side_effect=[[e1, e2], []])
        mock_ctx.service.list_low_adoption = AsyncMock(return_value=[])
        mock_ctx.service.find_similar_active = AsyncMock(return_value=[e2_hit])

        with patch("team_memory.daemon.integration.auto_dream._daemon_context", return_value=mock_ctx):
            summary = await worker.run_once(dry_run=False, force=True, trigger="scheduled")

        assert summary["success"] is False
        assert summary["watermark_updated"] is False
        assert not (isolated_cache / "last_dream.txt").exists()


class TestOrientDistillAndModelConfig:
    """Tests for Phase 1 low-adoption sampling, Phase 4 Playbook distillation, and model resolution."""

    @pytest.mark.asyncio
    async def test_orient_merges_recent_and_low_adoption_signals(self):
        worker, _ = _make_worker()
        mock_ctx = MagicMock()
        mock_ctx.service.list_published = AsyncMock(return_value=[_exp("recent-1", quality_tier="Silver")])
        mock_ctx.service.list_low_adoption = AsyncMock(
            return_value=[_exp("stale-high-recall", recall_count=12, used_count=0)]
        )

        with patch("team_memory.daemon.integration.auto_dream._daemon_context", return_value=mock_ctx):
            cands, low_cnt = await worker._phase_orient(None)

        assert [c["id"] for c in cands] == ["recent-1", "stale-high-recall"]
        assert low_cnt == 1
        assert cands[1]["_signal"] == "low_adoption"

    @pytest.mark.asyncio
    async def test_distill_creates_playbook_with_token_cap(self):
        llm = AsyncMock()
        long_text = "word " * 4000
        llm.chat_json = AsyncMock(
            return_value={
                "should_distill": True,
                "reason": "Coherent deployment theme",
                "title": "[Playbook] team_doc: PG Migrations",
                "problem": "When updating ORM models",
                "solution": long_text,
                "tags": ["alembic"],
            }
        )
        worker, sink = _make_worker(
            llm_client=llm,
            dream_overrides={"distill_min_cluster_size": 3, "distill_max_tokens": 300},
        )
        cluster = [_exp("e1"), _exp("e2"), _exp("e3")]

        res = await worker._phase_distill([cluster], dry_run=False)

        assert res["distilled"] == 1
        sink.save.assert_called_once()
        kw = sink.save.call_args.kwargs
        assert kw["experience_type"] == "playbook"
        assert "playbook" in kw["tags"]
        assert "auto-dream" in kw["tags"]
        assert ad_mod._estimate_tokens(kw["solution"]) <= 350

    @pytest.mark.asyncio
    async def test_distill_excludes_retired_ids_from_phase3(self):
        llm = AsyncMock()
        worker, sink = _make_worker(
            llm_client=llm,
            dream_overrides={"distill_min_cluster_size": 3},
        )
        cluster = [_exp("e1"), _exp("e2"), _exp("e3")]

        # If e2 was superseded/soft-deleted in Phase 3, only 2 active items remain (< 3)
        res = await worker._phase_distill([cluster], dry_run=False, retired_ids={"e2"})
        assert res["distilled"] == 0
        llm.chat_json.assert_not_called()
        sink.save.assert_not_called()

    def test_model_resolution_priority(self, monkeypatch: pytest.MonkeyPatch):
        worker, _ = _make_worker(
            dream_overrides={"llm_model": "cfg-dream-model", "llm_fallback_model": "cfg-fallback"},
            refinement_overrides={"model": "ref-model", "fallback_model": "ref-fallback"},
        )
        monkeypatch.delenv("TM_AUTO_DREAM_MODEL", raising=False)
        monkeypatch.delenv("TM_AUTO_DREAM_FALLBACK_MODEL", raising=False)

        # 1. Config auto_dream.llm_model overrides refinement.model
        m, fb, *_ = worker._resolve_llm_settings()
        assert m == "cfg-dream-model"
        assert fb == "cfg-fallback"

        # 2. Env var overrides config
        monkeypatch.setenv("TM_AUTO_DREAM_MODEL", "env-dream-model")
        m, *_ = worker._resolve_llm_settings()
        assert m == "env-dream-model"

        # 3. CLI override wins over everything
        m, *_ = worker._resolve_llm_settings(model_override="cli-model")
        assert m == "cli-model"

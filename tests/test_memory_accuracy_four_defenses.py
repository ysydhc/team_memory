"""Comprehensive tests for the 4-Defense Memory Accuracy Architecture.

Covers:
  - Defense 1 (Write-Time): ADD / UPDATE / SUPERSEDE / NOOP, LLM timeout degradation,
    epistemic tier inference, protection guardrails, draft_publish zombie cleanup.
  - Defense 2 (Code-Grounded): 0ms anchor extraction + regex boundary guards,
    file deletion -> possibly_stale, anchor-free immunity, symbol-aware churn check,
    GitWatcher.on_code_change integration.
  - Defense 3 (Offline): Janitor verified/pinned decay exemption, AutoDream Phase 1
    possibly_stale priority queue, explicit vs implicit feedback staleness loop.
  - Defense 4 (Read-Time CRAG): epistemic & staleness scoring calibration, multi-signal
    low_confidence detection, recall/context/progressive_disclosure metadata & stale_warning,
    MCP grounding contract, and multi-tenant RLS isolation.
"""

from __future__ import annotations

import asyncio
import uuid
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from team_memory.config.lifecycle import LifecycleConfig
from team_memory.config.search import CacheConfig, RetrievalConfig, SearchConfig
from team_memory.hooks.integrations.git_watcher import GitWatcher
from team_memory.services.maintenance.janitor import MemoryJanitor
from team_memory.services.maintenance.staleness_verifier import StalenessVerifier
from team_memory.services.memory.evidence_anchors import (
    compact_evidence_anchors,
    extract_evidence_anchors,
    infer_epistemic_tier,
)
from team_memory.services.memory.experience import ExperienceService
from team_memory.services.memory.mutation_resolver import MutationResolver
from team_memory.services.search.progressive_disclosure import apply_progressive_disclosure
from team_memory.services.search.scoring import (
    STALE_WARNING_TEXT,
    apply_epistemic_and_staleness_adjustment,
    detect_low_confidence,
)
from team_memory.services.search.search_pipeline import (
    SearchPipeline,
    SearchRequest,
    SearchResultItem,
)

pytestmark = pytest.mark.unit


# =============================================================================
# Defense 1: Write-Time Gate (MutationResolver + Epistemic Tiers)
# =============================================================================


class TestDefense1WriteTimeGate:
    @pytest.mark.asyncio
    async def test_add_update_supersede_noop_transitions(self):
        resolver = MutationResolver(
            noop_threshold=0.95,
            mutation_threshold=0.82,
            update_action_name="UPDATE",
        )

        # 1. ADD when no candidate >= 0.82
        d_add = await resolver.resolve(
            title="Redis cache setup",
            problem="Need Redis caching",
            solution="Configure redis_url",
            candidates=[{"id": "c1", "similarity": 0.70, "description": "other", "solution": "other"}],
        )
        assert d_add.action == "ADD"

        # 2. NOOP when similarity >= 0.95
        d_noop = await resolver.resolve(
            title="Redis cache setup",
            problem="Need Redis caching",
            solution="Configure redis_url",
            candidates=[{"id": "c1", "similarity": 0.97, "description": "Need Redis", "solution": "Configure redis_url"}],
        )
        assert d_noop.action == "NOOP"
        assert d_noop.target_id == "c1"

        # 3. UPDATE when complementary solution in [0.82, 0.95)
        d_upd = await resolver.resolve(
            title="Redis cache setup",
            problem="Need Redis caching",
            solution="Also set max_connections=50",
            candidates=[{"id": "c1", "similarity": 0.86, "description": "Need Redis", "solution": "Configure redis_url"}],
        )
        assert d_upd.action == "UPDATE"
        assert d_upd.target_id == "c1"
        assert "max_connections=50" in (d_upd.merged_solution or "")

        # 4. SUPERSEDE when strong deprecation/replacement signal and sim >= 0.88
        d_sup = await resolver.resolve(
            title="Migrate from Graphiti",
            problem="Graphiti is deprecated and replaced by pgvector entity graph",
            solution="Remove Graphiti and use entity_search.py",
            candidates=[
                {
                    "id": "c2",
                    "similarity": 0.90,
                    "description": "Use Graphiti for graph search",
                    "solution": "Run Graphiti service",
                    "is_pinned": False,
                    "epistemic_tier": "auto_extract",
                    "staleness_status": "fresh",
                }
            ],
        )
        assert d_sup.action == "SUPERSEDE"
        assert d_sup.target_id == "c2"

    @pytest.mark.asyncio
    async def test_protection_guardrail_blocks_supersede_on_verified_or_pinned(self):
        resolver = MutationResolver(mutation_threshold=0.82, update_action_name="UPDATE")
        for protected_meta in (
            {"is_pinned": True, "epistemic_tier": "auto_extract", "staleness_status": "fresh"},
            {"is_pinned": False, "epistemic_tier": "verified", "staleness_status": "fresh"},
        ):
            decision = await resolver.resolve(
                title="Replace rule",
                problem="Old rule is deprecated",
                solution="Use new rule",
                candidates=[{"id": "prot-1", "similarity": 0.91, "description": "Old rule", "solution": "Keep", **protected_meta}],
            )
            assert decision.action == "ADD"

    @pytest.mark.asyncio
    async def test_llm_timeout_degrades_safely_to_add(self):
        fake_llm = SimpleNamespace(base_url="https://example.invalid/v1", model="fast-model", api_key="k")
        resolver = MutationResolver(
            llm_config=fake_llm,
            mutation_threshold=0.82,
            llm_timeout_seconds=0.05,
            llm_enabled=True,
            llm_gray_zone_min=0.84,
            update_action_name="UPDATE",
        )

        async def _slow_call(**_kwargs):
            await asyncio.sleep(1.0)
            return {"action": "SUPERSEDE", "target_id": "c1", "reason": "late"}

        with patch.object(resolver, "_call_llm_adjudicate", side_effect=_slow_call):
            decision = await resolver.resolve(
                title="Borderline update",
                problem="Refining search pipeline",
                solution="Add new fallback stage",
                candidates=[{"id": "c1", "similarity": 0.87, "description": "Search pipeline", "solution": "Old stage"}],
            )
        assert decision.action == "ADD"
        assert decision.degraded is True
        assert "timed out" in decision.reason.lower()

    def test_epistemic_tier_inference_hierarchy(self):
        assert infer_epistemic_tier(explicit_tier="verified") == "verified"
        assert infer_epistemic_tier(is_pinned=True) == "verified"
        assert infer_epistemic_tier(source="manual") == "verified"
        assert infer_epistemic_tier(evidence="pytest tests/ -q -> 42 passed") == "tool_grounded"
        assert infer_epistemic_tier(text="Fixed bug; pytest passed with exit code 0") == "tool_grounded"
        assert infer_epistemic_tier(source="auto_extract", text="Remember to keep logs clean") == "auto_extract"

    @pytest.mark.asyncio
    async def test_publish_draft_noop_cleans_up_zombie_draft(self):
        draft_id = uuid.uuid4()
        target_id = uuid.uuid4()

        fake_draft = SimpleNamespace(
            id=draft_id,
            title="Cached config",
            description="Check src/team_memory/server.py",
            solution="Use ctx.settings",
            tags=["mcp"],
            project="team_memory",
            embedding=[0.1, 0.2],
            evidence_anchors={},
        )
        fake_target = MagicMock()
        fake_target.to_dict.return_value = {"id": str(target_id), "title": "Cached config", "status": "published"}

        mock_repo = MagicMock()
        mock_repo.get_by_id = AsyncMock(return_value=fake_draft)
        mock_repo.check_similar = AsyncMock(
            return_value=[
                {
                    "id": str(target_id),
                    "title": "Cached config",
                    "description": "Check src/team_memory/server.py",
                    "solution": "Use ctx.settings",
                    "similarity": 0.98,
                }
            ]
        )
        mock_repo.touch_verified = AsyncMock(return_value=fake_target)
        mock_repo.update = AsyncMock(return_value=fake_draft)

        mock_embed = MagicMock()
        mock_embed.encode_single = AsyncMock(return_value=[0.1, 0.2])
        svc = ExperienceService(
            embedding_provider=mock_embed,
            auth_provider=MagicMock(),
            lifecycle_config=LifecycleConfig(auto_resolve_mutations=True, mutation_llm_enabled=False),
        )

        with patch("team_memory.services.memory.experience.ExperienceRepository", return_value=mock_repo):
            res = await svc.publish_draft(str(draft_id), user="alice", session=MagicMock())

        assert res is not None
        assert res["mutation_action"] == "NOOP"
        assert res["target_id"] == str(target_id)
        assert res["retired_draft_id"] == str(draft_id)
        mock_repo.update.assert_awaited_once_with(draft_id, is_deleted=True, superseded_by_id=target_id)


# =============================================================================
# Defense 2: Code-Grounded Evidence Anchors & Staleness Verification
# =============================================================================


class TestDefense2CodeGroundedAnchors:
    def test_extract_anchors_and_regex_boundaries(self):
        text = (
            "Updated `SearchPipeline` in src/team_memory/services/search/search_pipeline.py "
            "and Makefile (commit a1b2c3d4). See https://example.com/src/fake.py and "
            "ignore placeholder src/.../*.py, path/to/file.py, v0.3.3, 2026/11/25.json, "
            "application/json, and bare config.py. Call `apply_epistemic_and_staleness_adjustment`."
        )
        anchors = extract_evidence_anchors(text)
        assert "src/team_memory/services/search/search_pipeline.py" in anchors["file_paths"]
        assert "Makefile" in anchors["file_paths"]
        assert all("fake.py" not in p for p in anchors["file_paths"])
        assert all("..." not in p and "*" not in p for p in anchors["file_paths"])
        assert all("path/to" not in p for p in anchors["file_paths"])
        assert all(p != "config.py" for p in anchors["file_paths"])
        assert anchors["commit_sha"] == "a1b2c3d4"
        assert "SearchPipeline" in anchors["symbols"]
        assert "apply_epistemic_and_staleness_adjustment" in anchors["symbols"]

    def test_staleness_verifier_file_deleted_vs_anchor_free_vs_symbol_intact(self, tmp_path: Path):
        # Create a mock repo directory with one intact file
        src_dir = tmp_path / "src" / "pkg"
        src_dir.mkdir(parents=True)
        existing_file = src_dir / "service.py"
        existing_file.write_text("class ActiveService:\n    def run_task(self):\n        return 1\n", encoding="utf-8")

        verifier = StalenessVerifier(db_url="postgresql://unused", churn_lines_with_symbols=50)
        repo_files = ["src/pkg/service.py"]

        # 1. Deleted file triggers possibly_stale
        r_del = verifier.evaluate_anchors(
            experience_id="exp-del",
            title="Old helper guide",
            current_status="fresh",
            anchors={"file_paths": ["src/pkg/deleted_helper.py"], "symbols": ["DeletedHelper"]},
            repo_root=tmp_path,
            repo_files=repo_files,
        )
        assert r_del.new_status == "possibly_stale"
        assert r_del.changed is True
        assert "src/pkg/deleted_helper.py" in r_del.missing_files

        # 2. Anchor-free general principle is NEVER marked stale
        r_free = verifier.evaluate_anchors(
            experience_id="exp-principle",
            title="Always write idempotent migrations",
            current_status="fresh",
            anchors={},
            repo_root=tmp_path,
            repo_files=repo_files,
        )
        assert r_free.new_status == "fresh"
        assert r_free.changed is False

        # 3. Heavy churn on hotspot file, but anchored symbol STILL exists -> stays fresh
        with patch.object(StalenessVerifier, "_git_file_churn_lines", return_value=200):
            r_intact = verifier.evaluate_anchors(
                experience_id="exp-intact",
                title="ActiveService usage",
                current_status="fresh",
                anchors={"file_paths": ["src/pkg/service.py"], "symbols": ["ActiveService", "run_task"]},
                repo_root=tmp_path,
                repo_files=repo_files,
            )
        assert r_intact.new_status == "fresh"

        # 4. Heavy churn AND anchored symbol removed -> possibly_stale
        with patch.object(StalenessVerifier, "_git_file_churn_lines", return_value=200):
            r_sym_gone = verifier.evaluate_anchors(
                experience_id="exp-gone",
                title="Removed method",
                current_status="fresh",
                anchors={"file_paths": ["src/pkg/service.py"], "symbols": ["legacy_removed_fn"]},
                repo_root=tmp_path,
                repo_files=repo_files,
            )
        assert r_sym_gone.new_status == "possibly_stale"
        assert "legacy_removed_fn" in r_sym_gone.missing_symbols

    @pytest.mark.asyncio
    async def test_git_watcher_on_code_change_triggers_verifier(self):
        watcher = GitWatcher(config={}, indexer=MagicMock(), tm_client=MagicMock())
        mock_verifier = MagicMock()
        mock_verifier.verify_project = AsyncMock(
            return_value={"project": "team_memory", "checked": 3, "marked_possibly_stale": 1}
        )
        res = await watcher.on_code_change(
            "/tmp/repo",
            changed_files=["src/a.py"],
            deleted_files=["src/old.py"],
            verifier=mock_verifier,
            project="team_memory",
        )
        assert res["marked_possibly_stale"] == 1
        mock_verifier.verify_project.assert_awaited_once_with(
            project="team_memory",
            repo_path="/tmp/repo",
            deleted_files=["src/old.py"],
            dry_run=False,
        )


# =============================================================================
# Defense 3: Offline Coordination (Janitor + AutoDream + Feedback Loop)
# =============================================================================


class TestDefense3OfflineCoordination:
    @pytest.mark.asyncio
    async def test_janitor_exempts_verified_and_pinned_from_decay(self):
        old_ts = datetime.now(timezone.utc) - timedelta(days=30)
        exp_verified = SimpleNamespace(
            id=uuid.uuid4(),
            is_pinned=False,
            epistemic_tier="verified",
            staleness_status="fresh",
            quality_score=100.0,
            quality_tier="Silver",
            last_scored_at=old_ts,
            created_at=old_ts,
        )
        exp_pinned = SimpleNamespace(
            id=uuid.uuid4(),
            is_pinned=True,
            epistemic_tier="auto_extract",
            staleness_status="fresh",
            quality_score=100.0,
            quality_tier="Silver",
            last_scored_at=old_ts,
            created_at=old_ts,
        )
        exp_auto = SimpleNamespace(
            id=uuid.uuid4(),
            is_pinned=False,
            epistemic_tier="auto_extract",
            staleness_status="fresh",
            quality_score=100.0,
            quality_tier="Silver",
            last_scored_at=old_ts,
            created_at=old_ts,
        )

        mock_result = MagicMock()
        mock_result.scalars.return_value.all.return_value = [exp_verified, exp_pinned, exp_auto]
        mock_session = MagicMock()
        mock_session.execute = AsyncMock(return_value=mock_result)
        mock_session.commit = AsyncMock()

        janitor = MemoryJanitor(db_url="postgresql://unused")

        from contextlib import asynccontextmanager

        @asynccontextmanager
        async def _fake_session():
            yield mock_session

        with patch.object(janitor, "_session", _fake_session):
            stats = await janitor.run_score_decay()

        assert stats["updated_count"] == 1
        assert exp_verified.quality_score == 100.0
        assert exp_pinned.quality_score == 100.0
        assert exp_auto.quality_score < 100.0

    @pytest.mark.asyncio
    async def test_explicit_feedback_updates_staleness_but_implicit_does_not(self):
        exp_id = uuid.uuid4()
        mock_repo = MagicMock()
        mock_repo.get_by_id = AsyncMock(return_value=SimpleNamespace(id=exp_id))
        mock_repo.add_feedback = AsyncMock()
        mock_repo.increment_quality_score = AsyncMock()
        mock_repo.increment_used_count = AsyncMock()
        mock_repo.increment_negative_feedback_count = AsyncMock()
        mock_repo.increment_implicit_negative_count = AsyncMock()
        mock_repo.update = AsyncMock()

        svc = ExperienceService(embedding_provider=MagicMock(), auth_provider=MagicMock())
        with patch("team_memory.services.memory.experience.ExperienceRepository", return_value=mock_repo):
            # Explicit rating=1 -> possibly_stale
            await svc.feedback(str(exp_id), rating=1, feedback_by="alice", source="explicit_user", session=MagicMock())
            assert mock_repo.update.await_args.kwargs["staleness_status"] == "possibly_stale"

            mock_repo.update.reset_mock()
            # Implicit judge rating=1 -> MUST NOT touch staleness_status
            await svc.feedback(str(exp_id), rating=1, feedback_by="judge", source="implicit_judge", session=MagicMock())
            mock_repo.update.assert_not_awaited()

            # Explicit rating=5 -> fresh + last_verified_at refreshed
            await svc.feedback(str(exp_id), rating=5, feedback_by="alice", source="explicit_user", session=MagicMock())
            assert mock_repo.update.await_args.kwargs["staleness_status"] == "fresh"
            assert mock_repo.update.await_args.kwargs["last_verified_at"] is not None


# =============================================================================
# Defense 4: Read-Time CRAG, Metadata Exposure, Contract & RLS
# =============================================================================


class TestDefense4ReadTimeCRAG:
    def test_epistemic_and_staleness_scoring_order(self):
        items = [
            SearchResultItem(
                data={"id": "stale", "epistemic_tier": "auto_extract", "staleness_status": "possibly_stale"},
                score=0.020,
                similarity=0.85,
            ),
            SearchResultItem(
                data={"id": "auto", "epistemic_tier": "auto_extract", "staleness_status": "fresh"},
                score=0.020,
                similarity=0.85,
            ),
            SearchResultItem(
                data={"id": "grounded", "epistemic_tier": "tool_grounded", "staleness_status": "fresh"},
                score=0.020,
                similarity=0.85,
            ),
            SearchResultItem(
                data={"id": "verified", "epistemic_tier": "verified", "staleness_status": "fresh"},
                score=0.020,
                similarity=0.85,
            ),
        ]
        ranked = apply_epistemic_and_staleness_adjustment(items)
        assert [i.data["id"] for i in ranked] == ["verified", "grounded", "auto", "stale"]

    def test_crag_multi_signal_low_confidence_detection(self):
        weak = [SearchResultItem(data={"id": "1"}, score=0.01, similarity=0.45, fts_rank=0.01)]
        assert detect_low_confidence(weak, sim_threshold=0.58) is True

        fts_strong = [SearchResultItem(data={"id": "2"}, score=0.02, similarity=0.0, fts_rank=0.35)]
        assert detect_low_confidence(fts_strong, sim_threshold=0.58) is False

        exact_hit = [SearchResultItem(data={"id": "3", "exact_title_match": "exact"}, score=0.02, similarity=0.40)]
        assert detect_low_confidence(exact_hit, sim_threshold=0.58) is False

    @pytest.mark.asyncio
    async def test_pipeline_exposes_staleness_warning_and_trims_l1_anchors(self):
        mock_embed = MagicMock()
        mock_embed.encode_single = AsyncMock(return_value=[0.1, 0.2])

        pipeline = SearchPipeline(
            embedding_provider=mock_embed,
            search_config=SearchConfig(mode="vector", enable_graph=False, enable_topic=False),
            retrieval_config=RetrievalConfig(),
            cache_config=CacheConfig(enabled=False),
        )

        many_paths = [f"src/mod/f{i}.py" for i in range(12)]
        many_syms = [f"SymbolName{i}" for i in range(12)]
        raw_hits = [
            SearchResultItem(
                data={
                    "id": "e1",
                    "title": "Stale memory",
                    "description": "Old path",
                    "solution": "Do X",
                    "epistemic_tier": "tool_grounded",
                    "staleness_status": "possibly_stale",
                    "created_at": "2026-10-01T00:00:00+00:00",
                    "evidence_anchors": {"file_paths": many_paths, "symbols": many_syms},
                },
                score=0.52,
                similarity=0.52,
                source_type="vector",
            ),
            SearchResultItem(
                data={
                    "id": "e2",
                    "title": "Fresh memory",
                    "description": "Current",
                    "solution": "Do Y",
                    "epistemic_tier": "verified",
                    "staleness_status": "fresh",
                    "created_at": "2026-11-01T00:00:00+00:00",
                    "evidence_anchors": {"file_paths": many_paths, "symbols": many_syms},
                },
                score=0.50,
                similarity=0.50,
                source_type="vector",
            ),
        ]

        with patch.object(pipeline, "_retrieve_and_fuse", new_callable=AsyncMock, return_value=raw_hits):
            res = await pipeline.search(MagicMock(), SearchRequest(query="obscure query", max_results=5))

        assert res.low_confidence is True
        # Fresh verified (0.50 * 1.04 = 0.52) beats possibly_stale tool_grounded (0.52 * 1.02 * 0.90 = 0.4774)
        assert res.results[0]["id"] == "e2"
        stale_item = next(r for r in res.results if r["id"] == "e1")
        assert stale_item["staleness_status"] == "possibly_stale"
        assert stale_item["epistemic_tier"] == "tool_grounded"
        assert stale_item["stale_warning"] == STALE_WARNING_TEXT
        assert stale_item["last_verified_at"] == "2026-10-01T00:00:00+00:00"
        assert len(stale_item["evidence_anchors"]["file_paths"]) == 8
        assert stale_item["low_confidence"] is True

        # Progressive disclosure L1 trims evidence_anchors to file_paths[:3] and keeps stale_warning
        disclosed, _ = apply_progressive_disclosure(res.results, detail_level="summary")
        stale_l1 = next(r for r in disclosed if r["id"] == "e1")
        assert stale_l1["stale_warning"] == STALE_WARNING_TEXT
        assert stale_l1["epistemic_tier"] == "tool_grounded"
        assert stale_l1["staleness_status"] == "possibly_stale"
        assert stale_l1["evidence_anchors"] == {"file_paths": many_paths[:3]}

    def test_compact_evidence_anchors_caps(self):
        trimmed = compact_evidence_anchors(
            {"file_paths": [f"a/f{i}.py" for i in range(15)], "symbols": [f"Sym{i}" for i in range(15)]},
            max_paths=3,
            max_symbols=2,
        )
        assert len(trimmed["file_paths"]) == 3
        assert len(trimmed["symbols"]) == 2

    @pytest.mark.asyncio
    async def test_mcp_server_grounding_contract_in_descriptions(self):
        from team_memory.server import mcp

        tools = {t.name: t for t in (await mcp.get_tools()).values()}
        assert "memory_recall" in tools
        assert "memory_context" in tools
        assert "override historical memory" in (tools["memory_recall"].description or "").lower()
        assert "override historical memory" in (tools["memory_context"].description or "").lower()
        assert "NEVER blindly trust memory" in (mcp.instructions or "")

    def test_epistemic_tier_monotonic_promotion(self):
        from team_memory.services.memory.evidence_anchors import promote_epistemic_tier

        assert promote_epistemic_tier("auto_extract", "tool_grounded") == "tool_grounded"
        assert promote_epistemic_tier("tool_grounded", "verified") == "verified"
        assert promote_epistemic_tier("verified", "auto_extract") == "verified"
        assert promote_epistemic_tier("tool_grounded", "auto_extract") == "tool_grounded"

    @pytest.mark.asyncio
    async def test_possibly_stale_keeps_semantic_confidence_not_dropped_to_low(self):
        mock_embed = MagicMock()
        mock_embed.encode_single = AsyncMock(return_value=[0.1, 0.2])
        pipeline = SearchPipeline(
            embedding_provider=mock_embed,
            search_config=SearchConfig(mode="vector", enable_graph=False, enable_topic=False),
            retrieval_config=RetrievalConfig(),
            cache_config=CacheConfig(enabled=False),
        )
        raw_hits = [
            SearchResultItem(
                data={"id": "top", "title": "Fresh Top", "epistemic_tier": "auto_extract", "staleness_status": "fresh"},
                score=1.0,
                similarity=0.90,
                source_type="vector",
            ),
            SearchResultItem(
                data={
                    "id": "mid-stale",
                    "title": "Borderline Medium Stale",
                    "epistemic_tier": "auto_extract",
                    "staleness_status": "possibly_stale",
                },
                score=0.52,
                similarity=0.70,
                source_type="vector",
            ),
        ]
        with patch.object(pipeline, "_retrieve_and_fuse", new_callable=AsyncMock, return_value=raw_hits):
            res = await pipeline.search(MagicMock(), SearchRequest(query="test query", max_results=5))

        stale_item = next(r for r in res.results if r["id"] == "mid-stale")
        # 0.52 / 1.0 = 0.52 (medium) BEFORE 0.90 staleness adjustment -> must remain 'medium' so _guard_output keeps it
        assert stale_item["confidence"] == "medium"
        assert stale_item["score"] == round(0.52 * 0.90, 4)
        assert stale_item["stale_warning"] == STALE_WARNING_TEXT

    @pytest.mark.asyncio
    async def test_crag_verifier_optional_filter_and_timeout_fallback(self):
        mock_embed = MagicMock()
        mock_embed.encode_single = AsyncMock(return_value=[0.1, 0.2])
        fake_llm = SimpleNamespace(base_url="https://example.invalid/v1", model="fast", api_key="k")
        pipeline = SearchPipeline(
            embedding_provider=mock_embed,
            search_config=SearchConfig(
                mode="vector",
                enable_graph=False,
                enable_topic=False,
                crag_verifier_enabled=True,
                crag_verifier_timeout_seconds=0.1,
            ),
            retrieval_config=RetrievalConfig(),
            cache_config=CacheConfig(enabled=False),
            llm_config=fake_llm,
        )
        raw_hits = [
            SearchResultItem(data={"id": "a", "title": "Hit A"}, score=0.50, similarity=0.50, source_type="vector"),
            SearchResultItem(data={"id": "b", "title": "Hit B"}, score=0.48, similarity=0.48, source_type="vector"),
        ]

        # Verifier keeps only index 1
        with (
            patch.object(pipeline, "_retrieve_and_fuse", new_callable=AsyncMock, return_value=raw_hits),
            patch.object(pipeline, "_crag_verify_candidates", new_callable=AsyncMock, return_value=[raw_hits[1]]),
        ):
            res = await pipeline.search(MagicMock(), SearchRequest(query="borderline", max_results=5))
        assert res.low_confidence is True
        assert [r["id"] for r in res.results] == ["b"]

    def test_staleness_verifier_word_boundary_and_churn_cache(self, tmp_path: Path):
        src_dir = tmp_path / "src" / "pkg"
        src_dir.mkdir(parents=True)
        (src_dir / "mod.py").write_text("def saved_tokens_helper():\n    return 1\n", encoding="utf-8")

        verifier = StalenessVerifier(db_url="postgresql://unused", churn_lines_with_symbols=50)
        churn_cache: dict[tuple[str, str], int] = {}

        with patch.object(StalenessVerifier, "_git_file_churn_lines", return_value=120) as mock_git:
            # Symbol 'saved_tokens' should NOT match 'saved_tokens_helper' under word-boundary check
            r1 = verifier.evaluate_anchors(
                experience_id="e1",
                title="T1",
                current_status="fresh",
                anchors={"file_paths": ["src/pkg/mod.py"], "symbols": ["saved_tokens"]},
                repo_root=tmp_path,
                repo_files=["src/pkg/mod.py"],
                churn_cache=churn_cache,
            )
            r2 = verifier.evaluate_anchors(
                experience_id="e2",
                title="T2",
                current_status="fresh",
                anchors={"file_paths": ["src/pkg/mod.py"], "symbols": ["saved_tokens_helper"]},
                repo_root=tmp_path,
                repo_files=["src/pkg/mod.py"],
                churn_cache=churn_cache,
            )
        assert r1.new_status == "possibly_stale"
        assert r2.new_status == "fresh"
        # git log called only ONCE across both experiences thanks to churn_cache
        assert mock_git.call_count == 1


# =============================================================================
# Multi-Tenant RLS Isolation for 4-Defense Columns & StalenessVerifier
# =============================================================================


class TestFourDefensesMultiTenantRLS:
    @pytest.mark.asyncio
    async def test_staleness_verifier_respects_tenant_rls(self, tmp_path: Path):
        """Verify Tenant A StalenessVerifier never sees or mutates Tenant B experiences."""
        import os

        from sqlalchemy import select, text

        from dotenv import load_dotenv

        load_dotenv(".env")
        from team_memory.bootstrap import _resolve_db_url
        from team_memory.config import load_settings
        from team_memory.storage.database import close_db, get_session
        from team_memory.storage.models import Experience, Tenant
        from team_memory.storage.tenant_context import tenant_scope

        db_url = os.environ.get("TEAM_MEMORY_DB_URL") or _resolve_db_url(load_settings())
        # Skip gracefully if PostgreSQL is unreachable
        try:
            async with get_session(db_url) as s:
                await s.execute(text("SELECT 1"))
        except Exception:
            pytest.skip("PostgreSQL not available for RLS integration test")

        tid_a = uuid.uuid4()
        tid_b = uuid.uuid4()
        proj = f"rls_acc_{uuid.uuid4().hex[:8]}"
        exp_a_id = uuid.uuid4()
        exp_b_id = uuid.uuid4()

        try:
            # Create two tenants
            async with get_session(db_url) as s:
                s.add(Tenant(id=tid_a, slug=f"t-a-{tid_a.hex[:6]}", name="Tenant A"))
                s.add(Tenant(id=tid_b, slug=f"t-b-{tid_b.hex[:6]}", name="Tenant B"))
                await s.commit()

            # Insert anchored experiences in both tenants pointing to a missing file
            with tenant_scope(tid_a):
                async with get_session(db_url) as s:
                    s.add(
                        Experience(
                            id=exp_a_id,
                            tenant_id=tid_a,
                            title="Tenant A anchored exp",
                            description="Uses src/missing_a.py",
                            solution="Fix A",
                            created_by="alice",
                            project=proj,
                            exp_status="published",
                            epistemic_tier="tool_grounded",
                            evidence_anchors={"file_paths": ["src/missing_a.py"]},
                            staleness_status="fresh",
                        )
                    )
                    await s.commit()

            with tenant_scope(tid_b):
                async with get_session(db_url) as s:
                    s.add(
                        Experience(
                            id=exp_b_id,
                            tenant_id=tid_b,
                            title="Tenant B anchored exp",
                            description="Uses src/missing_b.py",
                            solution="Fix B",
                            created_by="bob",
                            project=proj,
                            exp_status="published",
                            epistemic_tier="verified",
                            evidence_anchors={"file_paths": ["src/missing_b.py"]},
                            staleness_status="fresh",
                        )
                    )
                    await s.commit()

            # Run StalenessVerifier strictly inside Tenant A scope
            verifier = StalenessVerifier(db_url)
            with tenant_scope(tid_a):
                report = await verifier.verify_project(project=proj, repo_path=tmp_path)

            assert report["checked"] == 1
            assert report["marked_possibly_stale"] == 1

            # Verify Tenant A is now possibly_stale while Tenant B remains untouched (fresh)
            with tenant_scope(tid_a):
                async with get_session(db_url) as s:
                    row_a = (await s.execute(select(Experience).where(Experience.id == exp_a_id))).scalar_one()
                    assert row_a.staleness_status == "possibly_stale"
                    assert row_a.epistemic_tier == "tool_grounded"

            with tenant_scope(tid_b):
                async with get_session(db_url) as s:
                    row_b = (await s.execute(select(Experience).where(Experience.id == exp_b_id))).scalar_one()
                    assert row_b.staleness_status == "fresh"
                    assert row_b.epistemic_tier == "verified"
        finally:
            try:
                async with get_session(db_url) as s:
                    await s.execute(text("DELETE FROM tenants WHERE id = ANY(:ids)"), {"ids": [tid_a, tid_b]})
                    await s.commit()
            except Exception:
                pass
            await close_db()

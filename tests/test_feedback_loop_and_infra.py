"""Unit tests for dual-layer feedback loop, Redis fallback cache, and IntentGate 0ms screen."""

from __future__ import annotations

import uuid
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from team_memory.daemon.integration.intent_gate import (
    IntentGate,
    IntentResult,
    fast_screen_intent,
)
from team_memory.daemon.quality.faithfulness_judge import FaithfulnessJudge
from team_memory.services.analysis.implicit_feedback import (
    evaluate_negative_candidates,
    record_implicit_positive_feedback,
    resolve_positive_target_ids,
    scan_and_record_implicit_negative_feedback,
)
from team_memory.services.memory.ops.feedback import op_feedback
from team_memory.services.memory.ops.recall import build_feedback_instruction
from team_memory.services.search.cache import RedisCacheBackend, SearchCache
from team_memory.services.search.scoring import (
    apply_popularity_boost,
    compute_bayesian_utility_rate,
)
from team_memory.services.search.search_pipeline import SearchResultItem
from team_memory.storage.tenant_context import reset_current_tenant_id, set_current_tenant_id

# ============================================================
# 1. MCP Layer 1 — Strong Compact Feedback Contract
# ============================================================


class TestMcpFeedbackContract:
    @pytest.mark.asyncio
    async def test_mcp_tool_descriptions_include_feedback_contract(self) -> None:
        from team_memory.server import mcp

        tools = await mcp.get_tools()
        assert "memory_feedback" in tools["memory_recall"].description
        assert "memory_feedback" in tools["memory_context"].description
        assert "rating=5" in tools["memory_recall"].description

    def test_build_feedback_instruction_skips_archives_and_caps_top3(self) -> None:
        results = [
            {"id": "arch-1", "type": "archive"},
            {"id": "exp-1", "title": "A"},
            {"group_id": "exp-2", "id": "child-2", "title": "B"},
            {"id": "exp-3", "title": "C"},
            {"id": "exp-4", "title": "D"},
        ]
        inst = build_feedback_instruction(results)
        assert inst is not None
        assert inst["action"] == "memory_feedback"
        assert inst["top_experience_ids"] == ["exp-1", "exp-2", "exp-3"]
        assert "exp-1" in inst["hint"]

    def test_build_feedback_instruction_returns_none_for_archives_only(self) -> None:
        assert build_feedback_instruction([{"id": "arch-1", "type": "archive"}]) is None

    @pytest.mark.asyncio
    async def test_op_feedback_marks_explicit_agent_source(self) -> None:
        svc = AsyncMock()
        svc.feedback = AsyncMock(return_value=True)
        res = await op_feedback(
            "alice",
            service=svc,
            experience_id=str(uuid.uuid4()),
            rating=5,
            comment="solved it",
        )
        assert "error" not in res
        svc.feedback.assert_awaited_once()
        assert svc.feedback.call_args.kwargs["source"] == "explicit_agent"


# ============================================================
# 2. Layer 2 — Implicit Feedback & Weighted Scoring
# ============================================================


class TestFaithfulnessJudgeAttribution:
    def test_parse_result_extracts_used_indices(self) -> None:
        judge = FaithfulnessJudge(api_key="test")
        raw = '{"claims": [{"claim": "x", "supported": true, "reason": "ok"}], "used_indices": [2, 1, 2, "bad"], "score": 0.9}'
        res = judge._parse_result(raw)
        assert res.score == pytest.approx(0.9)
        assert res.used_indices == [2, 1]

    def test_resolve_positive_target_ids_uses_indices_else_top1(self) -> None:
        id1, id2, id3 = str(uuid.uuid4()), str(uuid.uuid4()), str(uuid.uuid4())
        rids = [{"id": id1}, {"id": id2}, {"id": id3}]
        # Precise attribution when judge specifies used_indices
        assert resolve_positive_target_ids(rids, [2]) == [id2]
        assert resolve_positive_target_ids(rids, [1, 3]) == [id1, id3]
        # Conservative fallback to Top-1 only (never broadcasts to Top-3)
        assert resolve_positive_target_ids(rids, []) == [id1]
        assert resolve_positive_target_ids(rids, None) == [id1]


class TestImplicitNegativeAggregation:
    def test_high_relevance_repeated_non_adoption_triggers_per_tenant(self) -> None:
        t1, t2 = str(uuid.uuid4()), str(uuid.uuid4())
        e1 = str(uuid.uuid4())
        rows = [
            # Tenant 1: 5 high-relevance unused Top-3 recalls -> SHOULD trigger
            *[
                {
                    "tenant_id": t1,
                    "result_ids": [{"id": e1, "score": 0.8}],
                    "was_used": False,
                    "relevance_score": 0.72,
                    "is_noise": False,
                }
                for _ in range(5)
            ],
            # Tenant 2: only 3 recalls for same exp_id -> must NOT leak across tenants
            *[
                {
                    "tenant_id": t2,
                    "result_ids": [{"id": e1, "score": 0.8}],
                    "was_used": False,
                    "relevance_score": 0.8,
                    "is_noise": False,
                }
                for _ in range(3)
            ],
        ]
        cands = evaluate_negative_candidates(rows, min_recalls=5)
        assert len(cands) == 1
        assert cands[0][0] == t1
        assert cands[0][1] == e1
        assert cands[0][2] == 5

    def test_low_relevance_false_positives_do_not_penalize_experience(self) -> None:
        t1, e1 = str(uuid.uuid4()), str(uuid.uuid4())
        rows = [
            {
                "tenant_id": t1,
                "result_ids": [{"id": e1, "score": 0.2}],
                "was_used": False,
                "relevance_score": 0.2,
                "is_noise": False,
            }
            for _ in range(6)
        ]
        assert evaluate_negative_candidates(rows, min_recalls=5) == []

    def test_any_used_true_blocks_implicit_negative(self) -> None:
        t1, e1 = str(uuid.uuid4()), str(uuid.uuid4())
        rows = [
            {
                "tenant_id": t1,
                "result_ids": [{"id": e1, "score": 0.8}],
                "was_used": i == 4,
                "relevance_score": 0.8,
                "is_noise": False,
            }
            for i in range(5)
        ]
        assert evaluate_negative_candidates(rows, min_recalls=5) == []


class TestImplicitFeedbackDbGuards:
    @pytest.mark.asyncio
    async def test_implicit_positive_below_threshold_skips_db(self) -> None:
        res = await record_implicit_positive_feedback(
            db_url="postgresql+asyncpg://unused",
            result_ids=[{"id": str(uuid.uuid4())}],
            faithfulness_score=0.65,
            was_used=True,
        )
        assert res["recorded"] == 0
        assert res["skipped_reason"] == "below_threshold"

    @pytest.mark.asyncio
    async def test_implicit_negative_respects_daily_cap(self) -> None:
        t1, e1 = str(uuid.uuid4()), str(uuid.uuid4())
        mock_session = AsyncMock()
        # 1) search_logs fetch -> 5 unused high-relevance logs
        # 2) experience check -> exists, is_pinned=False
        # 3) explicit positive check -> None
        # 4) cap check -> cnt_24h=1 (hits daily_cap=1)
        log_rows = [(t1, [{"id": e1, "score": 0.85}], False, 0.8, False) for _ in range(5)]
        res1 = MagicMock(fetchall=MagicMock(return_value=log_rows))
        res2 = MagicMock(fetchone=MagicMock(return_value=(False,)))
        res3 = MagicMock(fetchone=MagicMock(return_value=None))
        res4 = MagicMock(fetchone=MagicMock(return_value=(1, 1)))
        mock_session.execute = AsyncMock(side_effect=[res1, res2, res3, res4])

        cm = AsyncMock()
        cm.__aenter__.return_value = mock_session
        cm.__aexit__.return_value = False

        with patch("team_memory.services.analysis.implicit_feedback.get_session", return_value=cm):
            out = await scan_and_record_implicit_negative_feedback(
                db_url="postgresql+asyncpg://test",
                daily_cap=1,
                weekly_cap=2,
            )
        assert out["recorded"] == 0
        assert out["skipped_capped"] == 1


class TestWeightedScoring:
    def test_explicit_has_full_weight_and_implicit_has_half_weight(self) -> None:
        u_explicit = compute_bayesian_utility_rate(10, used_count=2, negative_feedback_count=0)
        u_implicit = compute_bayesian_utility_rate(
            10, used_count=0, negative_feedback_count=0, implicit_positive_count=4
        )
        # 4 implicit positives * 0.5 == 2 explicit positives
        assert u_implicit == pytest.approx(u_explicit)

        u_neg_explicit = compute_bayesian_utility_rate(10, used_count=0, negative_feedback_count=1)
        u_neg_implicit = compute_bayesian_utility_rate(
            10, used_count=0, negative_feedback_count=0, implicit_negative_count=2
        )
        assert u_neg_implicit == pytest.approx(u_neg_explicit)

    def test_apply_popularity_boost_ranks_implicit_positive_above_implicit_negative(self) -> None:
        pos = SearchResultItem(
            data={"id": "pos", "recall_count": 5, "used_count": 0, "implicit_positive_count": 4},
            score=0.5,
            similarity=0.8,
        )
        neg = SearchResultItem(
            data={"id": "neg", "recall_count": 5, "used_count": 0, "implicit_negative_count": 4},
            score=0.5,
            similarity=0.8,
        )
        ranked = apply_popularity_boost([neg, pos])
        assert ranked[0].data["id"] == "pos"
        assert ranked[0].score > ranked[1].score


# ============================================================
# 3. Redis Shared Cache — Tenant Prefix & Graceful Memory Fallback
# ============================================================


class TestRedisCacheTenantAndFallback:
    def test_result_and_embedding_keys_include_tenant_prefix(self) -> None:
        t1 = uuid.uuid4()
        t2 = uuid.uuid4()
        provider_a = SimpleNamespace(model="qwen3-8b", dimension=4096)
        provider_b = SimpleNamespace(model="qwen3-0.6b", dimension=1024)

        tok1 = set_current_tenant_id(t1)
        try:
            res_k1 = SearchCache._make_key("pgvector index", project="team_memory")
            emb_k1a = SearchCache._make_embedding_key("pgvector index", provider_a)
            emb_k1b = SearchCache._make_embedding_key("pgvector index", provider_b)
        finally:
            reset_current_tenant_id(tok1)

        tok2 = set_current_tenant_id(t2)
        try:
            res_k2 = SearchCache._make_key("pgvector index", project="team_memory")
            emb_k2a = SearchCache._make_embedding_key("pgvector index", provider_a)
        finally:
            reset_current_tenant_id(tok2)

        assert res_k1.startswith(f"tenant:{t1}:res:")
        assert res_k2.startswith(f"tenant:{t2}:res:")
        assert res_k1 != res_k2

        assert emb_k1a.startswith(f"tenant:{t1}:emb:")
        assert emb_k2a.startswith(f"tenant:{t2}:emb:")
        assert emb_k1a != emb_k2a
        # Switching model/dimension produces a distinct embedding cache key
        assert emb_k1a != emb_k1b

    @pytest.mark.asyncio
    async def test_redis_backend_degrades_gracefully_to_memory_when_unreachable(self) -> None:
        backend = RedisCacheBackend(
            redis_url="redis://127.0.0.1:1/0",
            ttl_seconds=60,
            connect_timeout=0.05,
            retry_cooldown_seconds=30.0,
        )
        await backend.put("tenant:t1:res:abc", {"ok": True})
        assert backend.is_degraded is True
        cached = await backend.get("tenant:t1:res:abc")
        assert cached == {"ok": True}
        assert await backend.size() == 1
        await backend.clear()
        assert await backend.get("tenant:t1:res:abc") is None


# ============================================================
# 4. IntentGate — 0ms Local Regex Fast-Screen
# ============================================================


class TestIntentGateFastScreen:
    @pytest.mark.parametrize(
        "prompt",
        [
            "",
            "   ",
            "继续",
            "确认",
            "好的！",
            "commit+push",
            "commit and push",
            "重试",
            "LGTM",
            "ok",
            "https://example.com/docs/api?v=1",
            "src/team_memory/server.py",
            "/Users/dev/project/main.go:42",
            "看下吧",
        ],
    )
    def test_fast_screen_blocks_obvious_non_retrieval_prompts(self, prompt: str) -> None:
        assert fast_screen_intent(prompt) == IntentResult.NONE

    @pytest.mark.parametrize(
        "prompt",
        [
            "OOM",
            "pgvector",
            "RLS",
            "4096",
            "为什么 Redis 缓存断连后没有降级？",
            "how does search_pipeline handle multi-tenant isolation",
        ],
    )
    def test_fast_screen_preserves_tech_terms_and_real_queries(self, prompt: str) -> None:
        assert fast_screen_intent(prompt) is None

    @pytest.mark.asyncio
    async def test_classify_skips_llm_on_fast_screen_hit(self) -> None:
        gate = IntentGate(api_key="test")
        gate._call_llm = AsyncMock(side_effect=AssertionError("LLM must not be called"))
        assert await gate.classify("commit+push") == IntentResult.NONE
        assert await gate.classify("继续") == IntentResult.NONE
        assert await gate.classify("https://github.com/org/repo") == IntentResult.NONE
        gate._call_llm.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_classify_with_history_allows_bare_file_path_in_multiturn(self) -> None:
        gate = IntentGate(api_key="test")
        gate._call_llm = AsyncMock(return_value="experience")
        res = await gate.classify_with_history(
            "src/team_memory/server.py",
            history=[{"role": "user", "content": "帮我查下这个文件的历史坑"}],
        )
        assert res == IntentResult.EXPERIENCE
        gate._call_llm.assert_awaited_once()

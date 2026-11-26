"""Unit and native PostgreSQL FORCE-RLS integration tests for Wave 1:
- Per-Experience AttributionJudge & Evidence-Required hard gate
- Anti-pollution guardrails: daily +/-2 cap, verified/pinned exemption,
  contradicted -> possibly_stale
- Batch skip-noise, relevance_score/was_used/judgment_source writeback, daily cap
- Persistent AttributionEvalWorker retry & zero DEVNULL subprocess in hooks
- End-to-end native PG (:5432) FORCE-RLS writeback with full test-data cleanup
"""

from __future__ import annotations

import json
import os
import uuid
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from sqlalchemy import text as sa_text

from team_memory.daemon.buffers.response_buffer import ResponseBuffer
from team_memory.daemon.quality.faithfulness_batch import AttributionEvalWorker, run_batch
from team_memory.daemon.quality.faithfulness_judge import AttributionJudge, AttributionResult
from team_memory.services.analysis.implicit_feedback import (
    handle_contradicted_experiences,
    record_implicit_positive_feedback,
    scan_and_record_implicit_negative_feedback,
)
from team_memory.storage.database import close_db, get_session
from team_memory.storage.tenant_context import (
    DEFAULT_TENANT_ID,
    reset_current_tenant_id,
    set_current_tenant_id,
)

PG_ASYNC_URL = os.environ.get(
    "TEAM_MEMORY_TEST_DB_URL",
    "postgresql+asyncpg://developer@localhost:5432/team_memory",
)
PG_RAW_URL = PG_ASYNC_URL.replace("postgresql+asyncpg://", "postgresql://", 1)
DEFAULT_TID_STR = str(DEFAULT_TENANT_ID)


# ===========================================================================
# 1. Per-Experience AttributionJudge & Evidence-Required Hard Gate
# ===========================================================================


class TestAttributionJudge:
    def setup_method(self) -> None:
        self.judge = AttributionJudge(
            base_url="http://fake:4000/v1",
            model="test-flash",
            api_key="sk-test",
            retry_base_delay=0.01,
        )

    @pytest.mark.asyncio
    async def test_evidence_required_hard_gate_downgrades_empty_or_short_quote(self) -> None:
        exp_a = str(uuid.uuid4())
        exp_b = str(uuid.uuid4())
        exp_c = str(uuid.uuid4())
        llm_payload = json.dumps(
            [
                {
                    "experience_id": exp_a,
                    "relevance": 0.9,
                    "adopted": True,
                    "evidence_quote": "",  # empty -> MUST downgrade
                    "contradicted": False,
                },
                {
                    "experience_id": exp_b,
                    "relevance": 0.85,
                    "adopted": True,
                    "evidence_quote": "ok!",  # len=3 < 4 -> MUST downgrade
                    "contradicted": False,
                },
                {
                    "experience_id": exp_c,
                    "relevance": 0.88,
                    "adopted": True,
                    "evidence_quote": "SET LOCAL app.current_tenant_id",  # valid quote -> stays adopted
                    "contradicted": False,
                },
            ]
        )

        contexts = [
            {"id": exp_a, "title": "A", "problem": "pA", "solution": "sA"},
            {"id": exp_b, "title": "B", "problem": "pB", "solution": "sB"},
            {"id": exp_c, "title": "C", "problem": "pC", "solution": "sC"},
        ]

        with patch.object(self.judge, "_call_llm", new_callable=AsyncMock, return_value=llm_payload):
            res = await self.judge.evaluate(
                query="fix postgres RLS writeback",
                contexts=contexts,
                response="Added SET LOCAL app.current_tenant_id inside get_session.",
            )

        assert isinstance(res, AttributionResult)
        assert res.adopted_ids == [exp_c]
        assert res.used_indices == [3]
        assert res.score >= 0.8
        assert res.score == pytest.approx(0.88)
        assert res.max_relevance == pytest.approx(0.9)
        assert res.avg_relevance == pytest.approx((0.9 + 0.85 + 0.88) / 3, abs=1e-3)

        by_id = {it.experience_id: it for it in res.items}
        assert by_id[exp_a].adopted is False
        assert by_id[exp_a].downgraded is True
        assert by_id[exp_b].adopted is False
        assert by_id[exp_b].downgraded is True
        assert by_id[exp_c].adopted is True
        assert by_id[exp_c].downgraded is False

    @pytest.mark.asyncio
    async def test_no_adoption_caps_score_below_0_7_and_preserves_relevance(self) -> None:
        exp_a = str(uuid.uuid4())
        llm_payload = json.dumps(
            [
                {
                    "experience_id": exp_a,
                    "relevance": 0.92,
                    "adopted": True,
                    "evidence_quote": "   ",  # downgraded to False
                    "contradicted": True,
                }
            ]
        )
        with patch.object(self.judge, "_call_llm", new_callable=AsyncMock, return_value=llm_payload):
            res = await self.judge.evaluate(
                query="docker port",
                contexts=[{"id": exp_a, "title": "Old port", "problem": "5433", "solution": "use 5433"}],
                response="Docker is not installed; native Postgres runs on 5432.",
            )

        assert res.adopted_ids == []
        assert res.contradicted_ids == [exp_a]
        assert res.max_relevance == pytest.approx(0.92)
        assert 0.0 <= res.score < 0.7
        assert res.score == pytest.approx(0.69)

    @pytest.mark.asyncio
    async def test_adopted_low_relevance_boosts_score_at_least_0_8(self) -> None:
        exp_a = str(uuid.uuid4())
        llm_payload = json.dumps(
            [
                {
                    "experience_id": exp_a,
                    "relevance": 0.6,
                    "adopted": True,
                    "evidence_quote": "uv run pytest -q",
                    "contradicted": False,
                }
            ]
        )
        with patch.object(self.judge, "_call_llm", new_callable=AsyncMock, return_value=llm_payload):
            res = await self.judge.evaluate(
                query="run tests",
                contexts=[{"id": exp_a, "title": "uv pytest", "problem": "venv", "solution": "uv run pytest"}],
                response="Ran uv run pytest -q and all tests passed.",
            )

        assert res.adopted_ids == [exp_a]
        assert res.score == pytest.approx(0.8)


# ===========================================================================
# 2. Batch Evaluation: Skip Noise, Relevance Writeback, Daily Cap
# ===========================================================================


class TestAttributionBatch:
    @pytest.mark.asyncio
    async def test_skips_noise_and_writes_relevance_was_used_attribution(self) -> None:
        exp_id = str(uuid.uuid4())
        entry_noise = {
            "id": str(uuid.uuid4()),
            "query": "ok",
            "agent_response": "done",
            "result_ids": [{"id": exp_id}],
            "search_log_id": str(uuid.uuid4()),
            "is_noise": True,
        }
        entry_real = {
            "id": str(uuid.uuid4()),
            "query": "how to fix postgres rls",
            "agent_response": "Use get_session(db_url, tenant_id=tid) to set app.current_tenant_id.",
            "result_ids": [{"id": exp_id, "score": 0.82}],
            "search_log_id": str(uuid.uuid4()),
            "is_noise": False,
        }

        mock_buf = MagicMock()
        mock_buf._batch_threshold = 1
        mock_buf.pending_count = AsyncMock(return_value=2)
        mock_buf.fetch_pending = AsyncMock(return_value=[entry_noise, entry_real])
        mock_buf.mark_evaluated = AsyncMock()
        mock_buf.sync_to_search_log = AsyncMock()
        mock_buf.close = AsyncMock()

        eval_res = AttributionResult(
            score=0.88,
            max_relevance=0.88,
            avg_relevance=0.88,
            adopted_ids=[exp_id],
            contradicted_ids=[],
            items=[],
            raw_response="[]",
            used_indices=[1],
        )
        mock_judge = MagicMock()
        mock_judge.evaluate = AsyncMock(return_value=eval_res)

        with (
            patch("team_memory.daemon.buffers.response_buffer.ResponseBuffer", return_value=mock_buf),
            patch("team_memory.daemon.quality.faithfulness_judge.AttributionJudge", return_value=mock_judge),
            patch(
                "team_memory.daemon.quality.faithfulness_batch.load_structured_contexts",
                new_callable=AsyncMock,
                return_value=[{"id": exp_id, "title": "RLS", "problem": "p", "solution": "s"}],
            ),
            patch(
                "team_memory.daemon.quality.faithfulness_batch.get_daily_eval_count",
                new_callable=AsyncMock,
                return_value=0,
            ),
            patch(
                "team_memory.services.analysis.implicit_feedback.record_implicit_positive_feedback",
                new_callable=AsyncMock,
                return_value={"recorded": 1},
            ) as mock_pos,
            patch(
                "team_memory.services.analysis.implicit_feedback.scan_and_record_implicit_negative_feedback",
                new_callable=AsyncMock,
                return_value={"recorded": 0},
            ),
        ):
            summary = await run_batch(dsn="postgresql://unused", force=True, daily_limit=200)

        # Noise entry skipped LLM, real entry evaluated once
        mock_judge.evaluate.assert_awaited_once()
        mock_buf.sync_to_search_log.assert_awaited_once_with(
            entry_real["id"],
            faithfulness_score=0.88,
            relevance_score=0.88,
            was_used=True,
            judgment_source="attribution",
        )
        mock_pos.assert_awaited_once()
        assert summary["evaluated"] == 1
        assert summary["implicit_positive_recorded"] == 1

    @pytest.mark.asyncio
    async def test_daily_budget_exhausted_skips_llm(self) -> None:
        mock_buf = MagicMock()
        mock_buf._batch_threshold = 1
        mock_buf.pending_count = AsyncMock(return_value=5)
        mock_buf.fetch_pending = AsyncMock(
            return_value=[
                {
                    "id": str(uuid.uuid4()),
                    "query": "real query here",
                    "agent_response": "some answer",
                    "result_ids": [{"id": str(uuid.uuid4())}],
                    "search_log_id": str(uuid.uuid4()),
                    "is_noise": False,
                }
            ]
        )
        mock_buf.close = AsyncMock()

        with (
            patch("team_memory.daemon.buffers.response_buffer.ResponseBuffer", return_value=mock_buf),
            patch(
                "team_memory.daemon.quality.faithfulness_batch.get_daily_eval_count",
                new_callable=AsyncMock,
                return_value=200,
            ),
        ):
            summary = await run_batch(dsn="postgresql://unused", force=True, daily_limit=200)

        assert summary["skipped"] is True
        assert summary["budget_exhausted"] is True
        assert summary["evaluated"] == 0


# ===========================================================================
# 3. Persistent Async Worker & Zero DEVNULL Subprocess
# ===========================================================================


class TestAttributionEvalWorker:
    @pytest.mark.asyncio
    async def test_worker_retries_up_to_3_times_on_failure(self) -> None:
        calls = 0

        async def _flaky_batch(**kwargs):
            nonlocal calls
            calls += 1
            if calls < 3:
                raise RuntimeError(f"transient failure #{calls}")
            return {"evaluated": 2, "avg_faithfulness": 0.85}

        worker = AttributionEvalWorker(
            dsn="postgresql://unused",
            max_retries=3,
            retry_base_delay=0.01,
        )
        with patch("team_memory.daemon.quality.faithfulness_batch.run_batch", side_effect=_flaky_batch):
            res = await worker.run_once_with_retry(force=True)

        assert calls == 3
        assert res is not None
        assert res["evaluated"] == 2

    def test_hooks_has_no_devnull_subprocess_popen(self) -> None:
        hooks_path = (
            Path(__file__).resolve().parent.parent
            / "src"
            / "team_memory"
            / "daemon"
            / "runtime"
            / "routes"
            / "hooks.py"
        )
        content = hooks_path.read_text(encoding="utf-8")
        assert "subprocess.Popen" not in content
        assert "DEVNULL" not in content
        assert "evaluation_worker" in content


# ===========================================================================
# 4. Native PostgreSQL (:5432) Forced RLS End-to-End Writeback Verification
# ===========================================================================


@pytest.mark.asyncio
async def test_native_pg_rls_end_to_end_writeback_and_guardrails() -> None:
    """Connect to native PG 16 (:5432) with ContextVar unset (standalone mode)
    and verify under FORCE ROW LEVEL SECURITY:
    1. ResponseBuffer.sync_to_search_log writes relevance_score, was_used,
       faithfulness_score, judgment_source='attribution'.
    2. record_implicit_positive_feedback writes experience_feedbacks and
       increments implicit_positive_count, capped at +2/day.
    3. handle_contradicted_experiences marks normal experience 'possibly_stale'
       while exempting 'verified' and 'is_pinned' experiences.
    4. scan_and_record_implicit_negative_feedback writes negative feedback for
       normal experience (5 unused high-relevance recalls) while exempting
       'verified' and 'is_pinned' experiences.
    """
    # Ensure ContextVar is explicitly None to simulate standalone worker / CLI
    tok = set_current_tenant_id(None)

    exp_normal = uuid.uuid4()
    exp_verified = uuid.uuid4()
    exp_pinned = uuid.uuid4()
    sl_pos_ids = [uuid.uuid4() for _ in range(3)]
    sl_neg_ids = [uuid.uuid4() for _ in range(5)]
    rb_id = uuid.uuid4()
    test_tag = f"test_wave1_attr_{uuid.uuid4().hex[:8]}"

    try:
        # Seed test records under default tenant RLS
        async with get_session(PG_ASYNC_URL, tenant_id=DEFAULT_TID_STR) as session:
            now = datetime.now(timezone.utc)
            for eid, title, pinned, tier in (
                (exp_normal, f"{test_tag}_normal", False, "auto_extract"),
                (exp_verified, f"{test_tag}_verified", False, "verified"),
                (exp_pinned, f"{test_tag}_pinned", True, "auto_extract"),
            ):
                await session.execute(
                    sa_text(
                        """
                        INSERT INTO experiences (
                            id, tenant_id, title, description, solution,
                            source, created_by, recall_count, created_at, updated_at,
                            is_deleted, project, experience_type, visibility, exp_status,
                            quality_score, quality_tier, is_pinned, used_count,
                            negative_feedback_count, implicit_positive_count,
                            implicit_negative_count, epistemic_tier, evidence_anchors,
                            staleness_status
                        ) VALUES (
                            :id, :tid, :title, 'test desc', 'test sol',
                            'test', 'pytest', 0, :now, :now,
                            false, 'team_memory', 'general', 'project', 'published',
                            80.0, 'Silver', :pinned, 0,
                            0, 0, 0, :tier, '{}'::jsonb, 'fresh'
                        )
                        """
                    ),
                    {
                        "id": eid,
                        "tid": DEFAULT_TENANT_ID,
                        "title": title,
                        "now": now,
                        "pinned": pinned,
                        "tier": tier,
                    },
                )

            # 3 search_logs for positive feedback (+2/day cap test)
            for sl_id in sl_pos_ids:
                await session.execute(
                    sa_text(
                        """
                        INSERT INTO search_logs (
                            id, tenant_id, query, intent_type, project, source,
                            result_ids, created_at, updated_at
                        ) VALUES (
                            :id, :tid, :q, 'experience', 'team_memory', 'mcp',
                            CAST(:rids AS json), :now, :now
                        )
                        """
                    ),
                    {
                        "id": sl_id,
                        "tid": DEFAULT_TENANT_ID,
                        "q": f"{test_tag} positive query",
                        "rids": json.dumps([{"id": str(exp_normal), "score": 0.9}]),
                        "now": now,
                    },
                )

            # 5 search_logs for negative feedback scan (normal + verified + pinned in Top-3)
            neg_rids = json.dumps(
                [
                    {"id": str(exp_normal), "score": 0.85},
                    {"id": str(exp_verified), "score": 0.85},
                    {"id": str(exp_pinned), "score": 0.85},
                ]
            )
            for sl_id in sl_neg_ids:
                await session.execute(
                    sa_text(
                        """
                        INSERT INTO search_logs (
                            id, tenant_id, query, intent_type, project, source,
                            result_ids, was_used, relevance_score, is_noise,
                            judgment_source, created_at, updated_at
                        ) VALUES (
                            :id, :tid, :q, 'experience', 'team_memory', 'mcp',
                            CAST(:rids AS json), false, 0.82, false,
                            'attribution', :now, :now
                        )
                        """
                    ),
                    {
                        "id": sl_id,
                        "tid": DEFAULT_TENANT_ID,
                        "q": f"{test_tag} unused high-rel query",
                        "rids": neg_rids,
                        "now": now,
                    },
                )

            # 1 response_buffer row linked to sl_pos_ids[0]
            await session.execute(
                sa_text(
                    """
                    INSERT INTO response_buffer (
                        id, query, agent_response, result_ids, search_log_id, evaluated, created_at
                    ) VALUES (
                        :id, :q, 'Used the fix', CAST(:rids AS json), :sl_id, false, :now
                    )
                    """
                ),
                {
                    "id": rb_id,
                    "q": f"{test_tag} positive query",
                    "rids": json.dumps([{"id": str(exp_normal), "score": 0.9}]),
                    "sl_id": sl_pos_ids[0],
                    "now": now,
                },
            )

        # --- Step A: ResponseBuffer.sync_to_search_log under FORCE RLS ---
        buf = ResponseBuffer(dsn=PG_RAW_URL)
        try:
            await buf.sync_to_search_log(
                str(rb_id),
                faithfulness_score=0.91,
                relevance_score=0.87,
                was_used=True,
                judgment_source="attribution",
            )
        finally:
            await buf.close()

        async with get_session(PG_ASYNC_URL, tenant_id=DEFAULT_TID_STR) as session:
            sl_row = (
                await session.execute(
                    sa_text(
                        "SELECT faithfulness_score, relevance_score, was_used, judgment_source "
                        "FROM search_logs WHERE id = :id"
                    ),
                    {"id": sl_pos_ids[0]},
                )
            ).fetchone()
        assert sl_row is not None
        assert sl_row[0] == pytest.approx(0.91)
        assert sl_row[1] == pytest.approx(0.87)
        assert sl_row[2] is True
        assert sl_row[3] == "attribution"

        # --- Step B: Implicit positive feedback under FORCE RLS + daily +2 cap ---
        r1 = await record_implicit_positive_feedback(
            db_url=PG_ASYNC_URL,
            result_ids=[{"id": str(exp_normal)}],
            adopted_ids=[str(exp_normal)],
            faithfulness_score=0.9,
            relevance_score=0.85,
            was_used=True,
            search_log_id=str(sl_pos_ids[0]),
        )
        r2 = await record_implicit_positive_feedback(
            db_url=PG_ASYNC_URL,
            result_ids=[{"id": str(exp_normal)}],
            adopted_ids=[str(exp_normal)],
            faithfulness_score=0.88,
            relevance_score=0.82,
            was_used=True,
            search_log_id=str(sl_pos_ids[1]),
        )
        r3 = await record_implicit_positive_feedback(
            db_url=PG_ASYNC_URL,
            result_ids=[{"id": str(exp_normal)}],
            adopted_ids=[str(exp_normal)],
            faithfulness_score=0.92,
            relevance_score=0.9,
            was_used=True,
            search_log_id=str(sl_pos_ids[2]),
        )
        assert r1["recorded"] == 1
        assert r2["recorded"] == 1
        assert r3["recorded"] == 0
        assert r3["skipped_capped"] == 1

        # --- Step C: Contradiction -> possibly_stale (with verified & pinned exemption) ---
        contra_res = await handle_contradicted_experiences(
            db_url=PG_ASYNC_URL,
            contradicted_ids=[str(exp_normal), str(exp_verified), str(exp_pinned)],
            search_log_id=str(sl_pos_ids[0]),
        )
        assert contra_res["marked_stale"] == 1
        assert contra_res["warned_protected"] == 2

        # --- Step D: Implicit negative feedback under FORCE RLS (with verified & pinned exemption) ---
        # Note: sl_pos_ids[0] has was_used=True for exp_normal, so clear its was_used to isolate neg scan
        async with get_session(PG_ASYNC_URL, tenant_id=DEFAULT_TID_STR) as session:
            await session.execute(
                sa_text("UPDATE search_logs SET was_used = NULL WHERE id = :id"),
                {"id": sl_pos_ids[0]},
            )

        neg_res = await scan_and_record_implicit_negative_feedback(
            db_url=PG_ASYNC_URL,
            min_recalls=5,
            daily_cap=2,
            weekly_cap=2,
        )
        assert neg_res["recorded"] >= 1
        assert neg_res["skipped_protected"] >= 2

        # Verify final DB state for all 3 experiences under RLS
        async with get_session(PG_ASYNC_URL, tenant_id=DEFAULT_TID_STR) as session:
            rows = (
                await session.execute(
                    sa_text(
                        """
                        SELECT id, implicit_positive_count, implicit_negative_count, staleness_status
                        FROM experiences
                        WHERE id IN (:e1, :e2, :e3)
                        """
                    ),
                    {"e1": exp_normal, "e2": exp_verified, "e3": exp_pinned},
                )
            ).fetchall()
            state = {r[0]: (r[1], r[2], r[3]) for r in rows}

            fb_count = (
                await session.execute(
                    sa_text(
                        "SELECT COUNT(*) FROM experience_feedbacks "
                        "WHERE experience_id = :eid AND source = 'implicit_judge'"
                    ),
                    {"eid": exp_normal},
                )
            ).scalar()

        # exp_normal: +2 positive (capped), +1 negative, marked possibly_stale
        assert state[exp_normal] == (2, 1, "possibly_stale")
        assert fb_count == 3  # 2 positive + 1 negative
        # exp_verified & exp_pinned: protected from negative feedback & contradiction staleness
        assert state[exp_verified] == (0, 0, "fresh")
        assert state[exp_pinned] == (0, 0, "fresh")

    finally:
        # Clean up all temporary test rows
        try:
            async with get_session(PG_ASYNC_URL, tenant_id=DEFAULT_TID_STR) as session:
                await session.execute(
                    sa_text("DELETE FROM response_buffer WHERE id = :id"),
                    {"id": rb_id},
                )
                for sl_id in sl_pos_ids + sl_neg_ids:
                    await session.execute(
                        sa_text("DELETE FROM search_logs WHERE id = :id"),
                        {"id": sl_id},
                    )
                for eid in (exp_normal, exp_verified, exp_pinned):
                    await session.execute(
                        sa_text("DELETE FROM experience_feedbacks WHERE experience_id = :eid"),
                        {"eid": eid},
                    )
                    await session.execute(
                        sa_text("DELETE FROM experiences WHERE id = :eid"),
                        {"eid": eid},
                    )
        finally:
            await close_db()
            reset_current_tenant_id(tok)

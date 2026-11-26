"""Unit & integration tests for Scoring Loop Observability SLA redlines.

Covers:
  1. Normal green state (healthy adoption rate + active implicit feedback)
  2. Redline 1 — Anti-empty-run (RED): >20 non-noise searches in recent 3 workdays
     with 0 implicit feedback writebacks (plus boundary at == 20 -> green)
  3. Redline 2 — Anti-bias/drift (YELLOW): >=10 judged non-noise searches in 7d
     with adopted_rate < 1% (too strict) or > 80% (rubber-stamp), plus boundaries
     (judged == 9 -> green; rate == 1% / 80% -> green)
  4. Combined red + yellow alerts (overall escalates to RED, both reasons preserved)
  5. Redline 3 — ops-digest text & /api/observability/overview JSON surface scoring
     loop status, metrics, and alert reasons
  6. Workday helper & contradicted reasoning parser & live DB overview integration
"""

from __future__ import annotations

import json
from datetime import date
from unittest.mock import AsyncMock, patch

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from team_memory.auth.provider import User
from team_memory.services.analysis.observability import (
    _count_contradicted_from_reasonings,
    evaluate_scoring_sla,
    evaluate_subsystem_statuses,
    format_ops_digest,
    get_observability_overview,
    recent_workday_dates,
)
from team_memory.web.auth_session import get_current_user
from team_memory.web.middleware import api_version_compat
from team_memory.web.routes import observability as obs_route


def _healthy_base_subsystems() -> tuple[dict, dict, dict, dict, dict]:
    return (
        {
            "today_total": 5,
            "avg_7d": 10.0,
            "prev_7d_avg": 8.0,
            "zero_write_workdays": 0,
            "queue_pending": 0,
            "queue_failed": 0,
        },
        {
            "autodream": {"age_hours": 4.0, "total_output": 3, "zero_output_streak": 0},
            "janitor": {"age_hours": 10.0},
        },
        {
            "entities": 4000,
            "relationships": 7000,
            "isolated_rate": 0.1,
            "graph_assist_rate_7d": 0.25,
            "latest_graph_lift": 0.03,
        },
        {
            "searches_7d": 200,
            "hit_rate_7d": 0.85,
            "p50_ms_7d": 400.0,
            "p95_ms_7d": 1500.0,
            "avg_tokens_7d": 1200.0,
            "eval_recall_delta": 0.01,
            "latest_eval": {"graph_lift": 0.03, "modes": {"full": {"recall_at_5": 0.85, "mrr_at_5": 0.78}}},
        },
        {
            "total_experiences": 1000,
            "embedded_experiences": 1000,
            "embedding_coverage": 1.0,
            "pg_status": "up",
            "pg_latency_ms": 2.0,
            "daemon_status": "up",
        },
    )


def _healthy_scoring(**overrides) -> dict:
    base = {
        "implicit_positive_7d": 8,
        "implicit_negative_7d": 2,
        "explicit_feedback_7d": 3,
        "adopted_rate_7d": 0.25,
        "adopted_7d": 10,
        "judged_non_noise_7d": 40,
        "non_noise_searches_3wd": 35,
        "implicit_feedback_3wd": 4,
        "contradicted_7d": 1,
        "avg_relevance_7d": 0.72,
        "pending_eval_buffer": 3,
    }
    base.update(overrides)
    return base


class TestScoringSLARedlines:
    def test_green_healthy_scoring_loop(self):
        ing, maint, kg, ret, inf = _healthy_base_subsystems()
        scoring = _healthy_scoring()

        overall, alerts = evaluate_subsystem_statuses(
            ingestion=ing,
            maintenance=maint,
            knowledge_graph=kg,
            retrieval=ret,
            infra=inf,
            scoring=scoring,
        )
        assert overall == "green"
        assert scoring["status"] == "green"
        assert alerts == []
        assert "25.0%" in scoring["reason"]
        assert "隐式+8/-2" in scoring["reason"]

    def test_redline1_empty_run_triggers_red_and_boundary(self):
        # Boundary: exactly 20 non-noise searches -> NOT triggered (must be > 20)
        scoring_at_boundary = _healthy_scoring(
            non_noise_searches_3wd=20,
            implicit_feedback_3wd=0,
            implicit_positive_7d=0,
            implicit_negative_7d=0,
        )
        status_b, alerts_b = evaluate_scoring_sla(scoring_at_boundary)
        assert status_b == "green"
        assert alerts_b == []

        # Triggered: 21 non-noise searches in 3 workdays and 0 implicit feedback -> RED
        ing, maint, kg, ret, inf = _healthy_base_subsystems()
        scoring_red = _healthy_scoring(
            non_noise_searches_3wd=21,
            implicit_feedback_3wd=0,
            implicit_positive_7d=0,
            implicit_negative_7d=0,
        )
        overall, alerts = evaluate_subsystem_statuses(
            ingestion=ing,
            maintenance=maint,
            knowledge_graph=kg,
            retrieval=ret,
            infra=inf,
            scoring=scoring_red,
        )
        assert overall == "red"
        assert scoring_red["status"] == "red"
        assert len(alerts) == 1
        assert alerts[0]["subsystem"] == "scoring"
        assert alerts[0]["level"] == "red"
        assert "空跑" in alerts[0]["message"]
        assert "21" in alerts[0]["message"]

    def test_redline1_fallback_to_7d_implicit_when_3wd_omitted(self):
        scoring = _healthy_scoring(
            non_noise_searches_3wd=30,
            implicit_positive_7d=0,
            implicit_negative_7d=0,
        )
        scoring.pop("implicit_feedback_3wd")
        status, alerts = evaluate_scoring_sla(scoring)
        assert status == "red"
        assert any(a["level"] == "red" and "空跑" in a["message"] for a in alerts)

    def test_redline2_adopted_rate_too_low_triggers_yellow_and_boundaries(self):
        # Boundary 1: judged_non_noise_7d == 9 (< 10) -> NOT triggered even if rate == 0.0
        s_few = _healthy_scoring(judged_non_noise_7d=9, adopted_rate_7d=0.0)
        st, al = evaluate_scoring_sla(s_few)
        assert st == "green"
        assert al == []

        # Boundary 2: adopted_rate_7d == 0.01 (1.0%) -> NOT triggered (must be < 1%)
        s_exact = _healthy_scoring(judged_non_noise_7d=100, adopted_rate_7d=0.01)
        st, al = evaluate_scoring_sla(s_exact)
        assert st == "green"
        assert al == []

        # Triggered: judged_non_noise_7d == 10, adopted_rate_7d == 0.005 (< 1%) -> YELLOW
        ing, maint, kg, ret, inf = _healthy_base_subsystems()
        s_low = _healthy_scoring(judged_non_noise_7d=10, adopted_rate_7d=0.005)
        overall, alerts = evaluate_subsystem_statuses(
            ingestion=ing,
            maintenance=maint,
            knowledge_graph=kg,
            retrieval=ret,
            infra=inf,
            scoring=s_low,
        )
        assert overall == "yellow"
        assert s_low["status"] == "yellow"
        assert len(alerts) == 1
        assert alerts[0]["subsystem"] == "scoring"
        assert alerts[0]["level"] == "yellow"
        assert "过低" in alerts[0]["message"]

    def test_redline2_adopted_rate_too_high_triggers_yellow_and_boundary(self):
        # Boundary: adopted_rate_7d == 0.80 (80.0%) -> NOT triggered (must be > 80%)
        s_exact = _healthy_scoring(judged_non_noise_7d=20, adopted_rate_7d=0.80)
        st, al = evaluate_scoring_sla(s_exact)
        assert st == "green"
        assert al == []

        # Triggered: judged_non_noise_7d == 15, adopted_rate_7d == 0.85 (> 80%) -> YELLOW
        ing, maint, kg, ret, inf = _healthy_base_subsystems()
        s_high = _healthy_scoring(judged_non_noise_7d=15, adopted_rate_7d=0.85)
        overall, alerts = evaluate_subsystem_statuses(
            ingestion=ing,
            maintenance=maint,
            knowledge_graph=kg,
            retrieval=ret,
            infra=inf,
            scoring=s_high,
        )
        assert overall == "yellow"
        assert s_high["status"] == "yellow"
        assert len(alerts) == 1
        assert alerts[0]["subsystem"] == "scoring"
        assert alerts[0]["level"] == "yellow"
        assert "过高" in alerts[0]["message"]

    def test_combined_empty_run_and_low_adopted_rate_both_reported(self):
        scoring = _healthy_scoring(
            non_noise_searches_3wd=42,
            implicit_feedback_3wd=0,
            implicit_positive_7d=0,
            implicit_negative_7d=0,
            judged_non_noise_7d=25,
            adopted_rate_7d=0.0,
        )
        status, alerts = evaluate_scoring_sla(scoring)
        assert status == "red"
        assert len(alerts) == 2
        levels = {a["level"] for a in alerts}
        assert levels == {"red", "yellow"}
        assert "空跑" in scoring["reason"]
        assert "过低" in scoring["reason"]


class TestRedline3DigestAndOverviewAPI:
    def test_ops_digest_surfaces_scoring_status_and_alert_reason(self):
        ing, maint, kg, ret, inf = _healthy_base_subsystems()
        scoring = _healthy_scoring(
            non_noise_searches_3wd=35,
            implicit_feedback_3wd=0,
            implicit_positive_7d=0,
            implicit_negative_7d=0,
            judged_non_noise_7d=20,
            adopted_rate_7d=0.0,
            contradicted_7d=2,
            avg_relevance_7d=0.61,
            pending_eval_buffer=7,
        )
        overall, alerts = evaluate_subsystem_statuses(
            ingestion=ing,
            maintenance=maint,
            knowledge_graph=kg,
            retrieval=ret,
            infra=inf,
            scoring=scoring,
        )
        overview = {
            "overall_status": overall,
            "alerts": alerts,
            "subsystems": {
                "ingestion": ing,
                "maintenance": maint,
                "knowledge_graph": kg,
                "retrieval": ret,
                "infra": inf,
                "scoring": scoring,
            },
        }

        alert_text = format_ops_digest(overview, alert_only=True)
        assert "[scoring]" in alert_text
        assert "空跑" in alert_text
        assert "过低" in alert_text

        weekly_text = format_ops_digest(overview, weekly=True)
        assert len(weekly_text.splitlines()) <= 10
        assert "🔴评分" in weekly_text
        assert "评分:" in weekly_text
        assert "矛盾 2" in weekly_text
        assert "待评 7" in weekly_text
        assert "空跑" in weekly_text

    def test_api_observability_overview_exposes_scoring_subsystem(self):
        test_app = FastAPI()
        test_app.middleware("http")(api_version_compat)
        test_app.include_router(obs_route.router, prefix="/api/v1")
        test_app.dependency_overrides[get_current_user] = lambda: User(name="tester", role="admin")

        ing, maint, kg, ret, inf = _healthy_base_subsystems()
        scoring = _healthy_scoring(judged_non_noise_7d=12, adopted_rate_7d=0.92)
        overall, alerts = evaluate_subsystem_statuses(
            ingestion=ing,
            maintenance=maint,
            knowledge_graph=kg,
            retrieval=ret,
            infra=inf,
            scoring=scoring,
        )
        fake_overview = {
            "generated_at": "2026-11-26T08:00:00+00:00",
            "days": 30,
            "overall_status": overall,
            "alerts": alerts,
            "subsystems": {
                "ingestion": ing,
                "maintenance": maint,
                "knowledge_graph": kg,
                "retrieval": ret,
                "infra": inf,
                "scoring": scoring,
            },
        }
        fake_ctx = type("Ctx", (), {"db_url": "postgresql+asyncpg://localhost/test"})()

        with (
            patch.object(obs_route, "get_context", return_value=fake_ctx),
            patch.object(obs_route, "get_observability_overview", new=AsyncMock(return_value=fake_overview)),
        ):
            client = TestClient(test_app)
            resp = client.get("/api/observability/overview?days=30")
            assert resp.status_code == 200
            payload = resp.json()
            assert payload["overall_status"] == "yellow"
            sc = payload["subsystems"]["scoring"]
            assert sc["status"] == "yellow"
            assert "过高" in sc["reason"]
            assert sc["adopted_rate_7d"] == pytest.approx(0.92)
            assert any(a["subsystem"] == "scoring" and a["level"] == "yellow" for a in payload["alerts"])


class TestHelpersAndLiveDBOverview:
    def test_recent_workday_dates_skips_weekend(self):
        # 2026-11-23 is Monday -> recent 3 workdays: Thu Nov 19, Fri Nov 20, Mon Nov 23
        assert recent_workday_dates(3, today=date(2026, 11, 23)) == [
            date(2026, 11, 19),
            date(2026, 11, 20),
            date(2026, 11, 23),
        ]
        # 2026-11-22 is Sunday -> recent 3 workdays: Wed Nov 18, Thu Nov 19, Fri Nov 20
        assert recent_workday_dates(3, today=date(2026, 11, 22)) == [
            date(2026, 11, 18),
            date(2026, 11, 19),
            date(2026, 11, 20),
        ]

    def test_count_contradicted_from_reasonings(self):
        r1 = json.dumps([{"experience_id": "1", "contradicted": True}, {"experience_id": "2", "contradicted": False}])
        r2 = json.dumps({"experience_id": "3", "contradicted": True})
        r3 = '[{"experience_id": "4", "contradicted": true}, {"truncated...'
        assert _count_contradicted_from_reasonings([r1, r2, r3, None, ""]) == 3

    @pytest.mark.asyncio
    async def test_live_db_get_observability_overview_contains_all_scoring_fields(self):
        db_url = "postgresql+asyncpg://developer:developer@localhost:5432/team_memory"
        overview = await get_observability_overview(db_url, days=30)
        sc = overview["subsystems"]["scoring"]
        expected_keys = {
            "implicit_positive_7d",
            "implicit_negative_7d",
            "explicit_feedback_7d",
            "adopted_rate_7d",
            "adopted_7d",
            "judged_non_noise_7d",
            "non_noise_searches_3wd",
            "implicit_feedback_3wd",
            "contradicted_7d",
            "avg_relevance_7d",
            "pending_eval_buffer",
            "status",
            "reason",
        }
        assert expected_keys.issubset(sc.keys())
        assert sc["status"] in {"green", "yellow", "red"}
        assert isinstance(sc["pending_eval_buffer"], int)
        assert isinstance(sc["contradicted_7d"], int)

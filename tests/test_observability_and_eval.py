"""Unit tests for Golden Eval metrics, observability redlines, ops-digest, and API."""

from __future__ import annotations

from datetime import date, timedelta
from unittest.mock import AsyncMock, patch

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from team_memory.auth.provider import User
from team_memory.services.analysis.golden_eval import (
    build_ablation_specs,
    compute_mrr_at_k,
    compute_ndcg_at_k,
    compute_percentile,
    compute_recall_at_k,
    is_noise_query,
    summarize_mode_metrics,
)
from team_memory.services.analysis.observability import (
    count_consecutive_zero_write_workdays,
    evaluate_subsystem_statuses,
    format_ops_digest,
)
from team_memory.web.auth_session import get_current_user
from team_memory.web.middleware import api_version_compat
from team_memory.web.routes import observability as obs_route


class TestGoldenEvalMetrics:
    def test_recall_at_k(self):
        assert compute_recall_at_k(["a", "b", "c"], {"a", "d"}, k=5) == 0.5
        assert compute_recall_at_k(["x", "y", "a"], {"a"}, k=2) == 0.0
        assert compute_recall_at_k(["a", "b"], set(), k=5) == 0.0

    def test_mrr_at_k(self):
        assert compute_mrr_at_k(["a", "b", "c"], {"a"}, k=5) == 1.0
        assert compute_mrr_at_k(["x", "a", "b"], {"a", "b"}, k=5) == 0.5
        assert compute_mrr_at_k(["x", "y", "z", "a"], {"a"}, k=3) == 0.0

    def test_ndcg_at_k(self):
        assert compute_ndcg_at_k(["a", "b"], {"a", "b"}, k=5) == pytest.approx(1.0)
        score = compute_ndcg_at_k(["x", "a"], {"a"}, k=5)
        assert 0.6 < score < 0.65
        assert compute_ndcg_at_k(["x", "y"], {"a"}, k=5) == 0.0

    def test_percentile_and_summary(self):
        assert compute_percentile([100, 200, 300, 400, 500], 50) == 300.0
        assert compute_percentile([100, 200, 300, 400, 500], 95) == pytest.approx(480.0)
        summary = summarize_mode_metrics(
            [
                {"retrieved_ids": ["e1", "e2"], "expected_ids": ["e1"], "latency_ms": 100, "tokens": 500},
                {"retrieved_ids": ["e3", "e2"], "expected_ids": ["e2"], "latency_ms": 300, "tokens": 700},
            ],
            k=5,
        )
        assert summary["recall_at_5"] == 1.0
        assert summary["mrr_at_5"] == 0.75
        assert summary["p50_latency_ms"] == 200.0
        assert summary["avg_tokens"] == 600.0

    def test_controlled_ablation_specs(self):
        specs_on = build_ablation_specs(prod_rerank_enabled=True)
        by_name = {s.name: s for s in specs_on}
        assert by_name["vector_only"].mode == "vector"
        assert by_name["vector_only"].enable_graph is False
        assert by_name["hybrid"].enable_graph is False
        assert by_name["hybrid"].rerank_enabled is True
        assert by_name["full"].enable_graph is True
        assert by_name["full"].rerank_enabled is True

    def test_noise_query_filter(self):
        assert is_noise_query("ping") is True
        assert is_noise_query("https://example.com/foo") is True
        assert is_noise_query("team_doc", known_projects={"team_doc"}) is True
        assert is_noise_query("abc") is True
        assert is_noise_query("pgvector 4096 维向量检索超时怎么排查") is False

    def test_intent_adaptive_rank_profiles(self):
        from team_memory.services.infra.intent_router import classify_rank_intent
        from team_memory.services.search.scoring import resolve_rank_profile

        assert classify_rank_intent("InAppDSP-4466 llog ReportManager") == "exact_identifier"
        assert classify_rank_intent("phone_pilot v0.9.0 发布验证") == "recency_release"
        assert classify_rank_intent("iOS 广告 SDK 全生命周期状态机与监控漏斗") == "multihop_arch"
        assert classify_rank_intent("终端里搜索框按粘贴快捷键没反应怎么解决") == "semantic_qa"

        p_exact = resolve_rank_profile("InAppDSP-4466 ReportManager")
        p_arch = resolve_rank_profile("全链路状态机架构对比")
        assert p_exact.fts_weight > p_arch.fts_weight
        assert p_arch.graph_weight > p_exact.graph_weight


class TestObservabilityRedlinesAndDigest:
    def _base_subsystems(self):
        return (
            {"today_total": 5, "avg_7d": 10.0, "prev_7d_avg": 8.0, "zero_write_workdays": 0, "queue_pending": 0, "queue_failed": 0},
            {"autodream": {"age_hours": 4.0, "total_output": 3, "zero_output_streak": 0}, "janitor": {"age_hours": 10.0}},
            {"entities": 4000, "relationships": 7000, "isolated_rate": 0.1, "graph_assist_rate_7d": 0.25, "latest_graph_lift": 0.03},
            {
                "searches_7d": 200,
                "hit_rate_7d": 0.85,
                "p50_ms_7d": 400.0,
                "p95_ms_7d": 1500.0,
                "avg_tokens_7d": 1200.0,
                "eval_recall_delta": 0.01,
                "latest_eval": {"graph_lift": 0.03, "modes": {"full": {"recall_at_5": 0.85, "mrr_at_5": 0.78}}},
            },
            {"total_experiences": 1000, "embedded_experiences": 1000, "embedding_coverage": 1.0, "pg_status": "up", "pg_latency_ms": 2.0, "daemon_status": "up"},
        )

    def test_all_green_and_alert_only_silent(self):
        ing, maint, kg, ret, inf = self._base_subsystems()
        overall, alerts = evaluate_subsystem_statuses(
            ingestion=ing, maintenance=maint, knowledge_graph=kg, retrieval=ret, infra=inf
        )
        assert overall == "green"
        assert alerts == []
        overview = {
            "overall_status": overall,
            "alerts": alerts,
            "subsystems": {
                "ingestion": ing,
                "maintenance": maint,
                "knowledge_graph": kg,
                "retrieval": ret,
                "infra": inf,
            },
        }
        assert format_ops_digest(overview, alert_only=True) == ""
        weekly = format_ops_digest(overview, weekly=True)
        assert len(weekly.splitlines()) <= 10
        assert "GREEN" in weekly

    def test_five_redlines_trigger_alerts(self):
        ing, maint, kg, ret, inf = self._base_subsystems()
        ing["zero_write_workdays"] = 2
        maint["autodream"]["age_hours"] = 50.0
        kg["latest_graph_lift"] = -0.02
        ret["p95_ms_7d"] = 9200.0
        ret["eval_recall_delta"] = -0.08
        inf["embedded_experiences"] = 990
        inf["embedding_coverage"] = 0.99

        overall, alerts = evaluate_subsystem_statuses(
            ingestion=ing, maintenance=maint, knowledge_graph=kg, retrieval=ret, infra=inf
        )
        assert overall == "red"
        subsystems_alerted = {a["subsystem"] for a in alerts}
        assert subsystems_alerted == {"ingestion", "maintenance", "knowledge_graph", "retrieval", "infra"}

        overview = {
            "overall_status": overall,
            "alerts": alerts,
            "subsystems": {
                "ingestion": ing,
                "maintenance": maint,
                "knowledge_graph": kg,
                "retrieval": ret,
                "infra": inf,
            },
        }
        alert_text = format_ops_digest(overview, alert_only=True)
        assert "异常告警" in alert_text
        assert "Graph Lift" in alert_text

    def test_consecutive_zero_write_workdays_skips_weekend(self):
        # 2026-11-23 is Monday; Thu Nov 19 & Fri Nov 20 are workdays with 0 writes, Sat/Sun skipped
        monday = date(2026, 11, 23)
        series = [
            {"date": (monday - timedelta(days=i)).isoformat(), "total": 5 if i >= 5 else 0}
            for i in range(7, 0, -1)
        ]
        assert count_consecutive_zero_write_workdays(series, today=monday) == 2


class TestObservabilityAPI:
    def test_overview_route_v1_and_compat_path(self):
        test_app = FastAPI()
        test_app.middleware("http")(api_version_compat)
        test_app.include_router(obs_route.router, prefix="/api/v1")
        test_app.dependency_overrides[get_current_user] = lambda: User(name="tester", role="admin")

        fake_overview = {
            "generated_at": "2026-11-25T08:00:00+00:00",
            "days": 30,
            "overall_status": "green",
            "alerts": [],
            "subsystems": {},
        }
        fake_ctx = type("Ctx", (), {"db_url": "postgresql+asyncpg://localhost/test"})()

        with (
            patch.object(obs_route, "get_context", return_value=fake_ctx),
            patch.object(obs_route, "get_observability_overview", new=AsyncMock(return_value=fake_overview)),
        ):
            client = TestClient(test_app)
            r1 = client.get("/api/observability/overview?days=30")
            r2 = client.get("/api/v1/observability/overview?days=180")
            assert r1.status_code == 200
            assert r1.json()["overall_status"] == "green"
            assert r2.status_code == 200

"""Tests for MetricsCollector — SQLite-backed metrics storage."""

from __future__ import annotations

from pathlib import Path

import pytest

from team_memory.daemon.observe.metrics_collector import MetricsCollector


class TestMetricsCollector:
    @pytest.fixture
    def mc(self, tmp_path: Path) -> MetricsCollector:
        return MetricsCollector(db_path=str(tmp_path / "test_metrics.db"))

    def test_record_and_query(self, mc: MetricsCollector) -> None:
        mc.record("hook_calls_total", 1.0, {"source": "cursor"})
        rows = mc.query(name="hook_calls_total")
        assert len(rows) == 1
        assert rows[0]["value"] == 1.0

    def test_summary_aggregates(self, mc: MetricsCollector) -> None:
        mc.record("hook_calls_total", 1.0, {"source": "cursor"})
        mc.record("hook_calls_total", 1.0, {"source": "hermes"})
        mc.record("hook_calls_total", 1.0, {"source": "cursor"})
        s = mc.summary()
        assert s["hook_calls_total"]["source=cursor"] == 2.0
        assert s["hook_calls_total"]["source=hermes"] == 1.0

    def test_latest_value(self, mc: MetricsCollector) -> None:
        mc.record("sessions_active", 5.0, {"project": "team_doc"})
        mc.record("sessions_active", 3.0, {"project": "team_doc"})
        rows = mc.query(name="sessions_active")
        assert rows[0]["value"] == 3.0

    def test_empty_summary(self, mc: MetricsCollector) -> None:
        assert mc.summary() == {}

    @pytest.mark.failure_injection
    def test_record_and_query_multiple(self, mc: MetricsCollector) -> None:
        """Multiple records return correct counts."""
        for _ in range(3):
            mc.record("hook_calls_total", 1.0)
        s = mc.summary()
        assert s["hook_calls_total"]["_total"] == 3.0

    def test_cleanup_old_records(self, mc: MetricsCollector) -> None:
        mc._db.execute(
            "INSERT INTO metrics (name, value, recorded_at) VALUES (?, ?, ?)",
            ("old_metric", 1.0, "2020-01-01T00:00:00"),
        )
        mc._db.execute(
            "INSERT INTO metrics (name, value, recorded_at) VALUES (?, ?, ?)",
            ("new_metric", 2.0, "2099-01-01T00:00:00"),
        )
        mc._db.commit()
        mc.cleanup()
        rows = mc.query()
        names = {r["name"] for r in rows}
        assert "old_metric" not in names
        assert "new_metric" in names

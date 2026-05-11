"""Tests for HealthCheck — in-process health monitoring."""
from __future__ import annotations
import asyncio
from pathlib import Path
from unittest.mock import AsyncMock, patch
import pytest
from team_memory.daemon.health_check import HealthCheck


def _make_alert_callback():
    return AsyncMock()


class TestHealthCheck:
    @pytest.fixture
    def alert_cb(self):
        return _make_alert_callback()

    @pytest.fixture
    def hc(self, tmp_path: Path, alert_cb) -> HealthCheck:
        db_path = str(tmp_path / "test.db")
        Path(db_path).write_text("x" * 100)
        return HealthCheck(
            db_path=db_path, pg_url=None,
            worker_task=None, alert_callback=alert_cb,
        )

    @pytest.mark.asyncio
    async def test_check_all_healthy(self, hc: HealthCheck, alert_cb) -> None:
        anomalies = await hc.check_all()
        assert anomalies == []
        alert_cb.assert_not_called()

    @pytest.mark.asyncio
    async def test_check_sqlite_over_limit(self, tmp_path: Path, alert_cb) -> None:
        db_path = str(tmp_path / "big.db")
        with patch("os.path.getsize", return_value=600 * 1024 * 1024):
            hc = HealthCheck(db_path=db_path, pg_url=None, alert_callback=alert_cb)
            anomalies = await hc.check_all()
            assert "sqlite_size" in anomalies
            alert_cb.assert_called()

    @pytest.mark.asyncio
    async def test_backoff_skips_alert(self, tmp_path: Path, alert_cb) -> None:
        db_path = str(tmp_path / "big.db")
        with patch("os.path.getsize", return_value=600 * 1024 * 1024):
            hc = HealthCheck(db_path=db_path, pg_url=None, alert_callback=alert_cb)
            await hc.check_all()
            call_count = alert_cb.call_count
            alert_cb.reset_mock()
            await hc.check_all()
            assert alert_cb.call_count == 0

    @pytest.mark.asyncio
    async def test_recovery_sends_notification(self, tmp_path: Path, alert_cb) -> None:
        db_path = str(tmp_path / "recover.db")
        Path(db_path).write_text("x" * 100)
        with patch("os.path.getsize", return_value=600 * 1024 * 1024):
            hc = HealthCheck(db_path=db_path, pg_url=None, alert_callback=alert_cb)
            await hc.check_all()
            assert alert_cb.call_count > 0
            alert_cb.reset_mock()
        await hc.check_all()
        recovery_found = False
        for call in alert_cb.call_args_list:
            title = call.kwargs.get("title", "")
            if call.args:
                title = call.args[0]
            if "recover" in title.lower():
                recovery_found = True
                break
        assert recovery_found, f"Expected recovery, got: {alert_cb.call_args_list}"

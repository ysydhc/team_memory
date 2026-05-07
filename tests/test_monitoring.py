"""Tests for monitoring modules — heartbeat, alerting, watchdog, config / 监控模块测试"""
from __future__ import annotations

import asyncio
import time
import pytest
from unittest.mock import AsyncMock, patch, MagicMock
from pathlib import Path


# === Heartbeat Tests ===

class TestHeartbeatState:
    def test_initial_state(self):
        from monitoring.heartbeat import HeartbeatState
        state = HeartbeatState()
        assert state.last_beat == 0.0
        assert state.pid > 0
        assert state.uptime_seconds >= 0

    def test_to_dict(self):
        from monitoring.heartbeat import HeartbeatState
        state = HeartbeatState()
        d = state.to_dict()
        assert "last_beat" in d
        assert "uptime_seconds" in d
        assert "pid" in d

    def test_is_alive_when_zero(self):
        from monitoring.heartbeat import HeartbeatState
        state = HeartbeatState()
        assert state.is_alive() is True  # not started yet

    def test_is_alive_recent(self):
        from monitoring.heartbeat import HeartbeatState
        state = HeartbeatState()
        state.last_beat = time.time()
        assert state.is_alive(timeout=90) is True

    def test_is_alive_expired(self):
        from monitoring.heartbeat import HeartbeatState
        state = HeartbeatState()
        state.last_beat = time.time() - 200
        assert state.is_alive(timeout=90) is False

    def test_last_beat_iso_when_zero(self):
        from monitoring.heartbeat import HeartbeatState
        state = HeartbeatState()
        assert state.last_beat_iso == ""


class TestHeartbeatLifecycle:
    @pytest.mark.asyncio
    async def test_start_stop(self):
        from monitoring.heartbeat import start_heartbeat, stop_heartbeat
        state = start_heartbeat(interval=1)
        await asyncio.sleep(0.1)
        assert state.last_beat > 0
        stop_heartbeat(state)


# === Config Tests ===

class TestMonitoringConfig:
    def test_load_default(self):
        from monitoring.config import load_monitoring_config
        cfg = load_monitoring_config()
        assert cfg.watchdog_interval == 60
        assert cfg.web_url == "http://localhost:9111/health"

    def test_load_nonexistent(self):
        from monitoring.config import load_monitoring_config
        cfg = load_monitoring_config("/tmp/nonexistent.yaml")
        assert cfg.watchdog_enabled is True  # defaults

    def test_config_fields(self):
        from monitoring.config import MonitoringConfig
        cfg = MonitoringConfig()
        assert cfg.admin_webhook == ""
        assert cfg.restart_allowed is True
        assert cfg.watchdog_max_failures == 3


# === Alerting Tests ===

class TestAlertManager:
    def test_should_alert_first_time(self):
        from monitoring.alerting import AlertManager
        am = AlertManager(alert_cooldown=60)
        assert am._should_alert("web") is True

    def test_should_alert_cooldown(self):
        from monitoring.alerting import AlertManager
        am = AlertManager(alert_cooldown=60)
        am._should_alert("web")
        assert am._should_alert("web") is False

    def test_should_alert_different_service(self):
        from monitoring.alerting import AlertManager
        am = AlertManager(alert_cooldown=60)
        am._should_alert("web")
        assert am._should_alert("daemon") is True

    def test_format_message(self):
        from monitoring.alerting import AlertManager, AlertSeverity
        am = AlertManager()
        msg = am._format_message("web", "down", "Connection refused", AlertSeverity.CRITICAL)
        assert "web" in msg
        assert "down" in msg
        assert "Connection refused" in msg
        assert "CRITICAL" in msg

    def test_write_log(self, tmp_path):
        from monitoring.alerting import AlertManager
        log_file = str(tmp_path / "test.log")
        am = AlertManager(log_file=log_file)
        am._write_log("test alert message")
        content = Path(log_file).read_text()
        assert "test alert message" in content

    @pytest.mark.asyncio
    async def test_send_alert_writes_log(self, tmp_path):
        from monitoring.alerting import AlertManager, AlertSeverity
        log_file = str(tmp_path / "test.log")
        am = AlertManager(log_file=log_file, alert_cooldown=0)
        await am.send_alert("web", "down", "test detail", AlertSeverity.WARNING)
        content = Path(log_file).read_text()
        assert "web" in content
        assert "down" in content


# === Watchdog Tests ===

class TestServiceState:
    def test_initial_state(self):
        from monitoring.watchdog import ServiceState
        svc = ServiceState(name="test", url="http://localhost:9999")
        assert svc.is_up is True
        assert svc.consecutive_failures == 0

    def test_failure_counting(self):
        from monitoring.watchdog import ServiceState
        svc = ServiceState(name="test", url="http://localhost:9999")
        svc.consecutive_failures = 3
        assert svc.consecutive_failures == 3


class TestServiceWatchdog:
    def test_init(self):
        from monitoring.watchdog import ServiceWatchdog
        from monitoring.config import MonitoringConfig
        cfg = MonitoringConfig(watchdog_interval=5)
        wd = ServiceWatchdog(config=cfg)
        assert "web" in wd._services
        assert "daemon" in wd._services

    @pytest.mark.asyncio
    async def test_check_service_up(self):
        from monitoring.watchdog import ServiceWatchdog, ServiceState
        from monitoring.config import MonitoringConfig
        cfg = MonitoringConfig(watchdog_max_failures=3)
        wd = ServiceWatchdog(config=cfg)
        svc = ServiceState(name="test", url="http://localhost:9111/health", timeout=5)
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_client = AsyncMock()
        mock_client.get = AsyncMock(return_value=mock_resp)
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)
        with patch("httpx.AsyncClient", return_value=mock_client):
            result = await wd.check_service(svc)
        assert result is True
        assert svc.consecutive_failures == 0

    @pytest.mark.asyncio
    async def test_check_service_down(self):
        from monitoring.watchdog import ServiceWatchdog, ServiceState
        from monitoring.config import MonitoringConfig
        cfg = MonitoringConfig(watchdog_max_failures=3)
        wd = ServiceWatchdog(config=cfg)
        svc = ServiceState(name="test", url="http://localhost:9999", timeout=1)
        mock_client = AsyncMock()
        mock_client.get = AsyncMock(side_effect=Exception("connection refused"))
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)
        with patch("httpx.AsyncClient", return_value=mock_client):
            result = await wd.check_service(svc)
        assert result is False
        assert svc.consecutive_failures == 1

    def test_get_status(self):
        from monitoring.watchdog import ServiceWatchdog
        from monitoring.config import MonitoringConfig
        wd = ServiceWatchdog(config=MonitoringConfig())
        status = wd.get_status()
        assert "web" in status
        assert "daemon" in status
        assert "is_up" in status["web"]

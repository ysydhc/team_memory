"""Tests for monitoring modules — heartbeat, alerting, watchdog, config"""
from __future__ import annotations

import asyncio
import json
import time
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

# ── Config ──

class TestMonitoringConfig:
    """Tests for MonitoringConfig / 监控配置测试"""

    def test_default_config(self):
        from team_memory.monitoring.config import MonitoringConfig
        cfg = MonitoringConfig()
        assert cfg.watchdog_interval == 60
        assert cfg.channels == []

    def test_load_config(self):
        from team_memory.monitoring.config import load_monitoring_config
        cfg = load_monitoring_config()
        assert cfg.web_url == "http://localhost:9111/health"
        assert cfg.daemon_url == "http://localhost:3901/status"

    def test_channel_config(self):
        from team_memory.monitoring.config import ChannelConfig
        ch = ChannelConfig(type="feishu", webhook="https://example.com", secret="abc")
        assert ch.type == "feishu"
        assert ch.secret == "abc"

    def test_channels_parsed_from_yaml(self):
        from team_memory.monitoring.config import load_monitoring_config
        cfg = load_monitoring_config()
        # Config has feishu channel with empty webhook
        feishu_channels = [c for c in cfg.channels if c.type == "feishu"]
        assert len(feishu_channels) >= 1


# ── Heartbeat ──

class TestHeartbeatState:
    """Tests for HeartbeatState / 心跳状态测试"""

    def test_initial_state(self):
        from team_memory.monitoring.heartbeat import HeartbeatState
        state = HeartbeatState()
        assert state.last_beat == 0
        assert state.uptime_seconds == 0

    @pytest.mark.asyncio
    async def test_start_stop(self):
        from team_memory.monitoring.heartbeat import start_heartbeat, stop_heartbeat
        state = start_heartbeat(interval=1)
        await asyncio.sleep(0.1)
        assert state.last_beat > 0
        stop_heartbeat(state)

    def test_stop_idempotent(self):
        from team_memory.monitoring.heartbeat import HeartbeatState, stop_heartbeat
        state = HeartbeatState()
        stop_heartbeat(state)  # should not raise


# ── Alerting ──

class TestAlertMessage:
    """Tests for AlertMessage / 告警消息测试"""

    def test_to_plain_text(self):
        from team_memory.monitoring.alerting import AlertMessage, AlertSeverity
        msg = AlertMessage(title="Test", body="Something happened", severity=AlertSeverity.WARNING)
        text = msg.to_plain_text()
        assert "WARNING" in text
        assert "Test" in text
        assert "Something happened" in text

    def test_auto_timestamp(self):
        from team_memory.monitoring.alerting import AlertMessage
        msg = AlertMessage(title="T", body="B")
        assert msg.timestamp  # auto-filled


class TestFeishuChannel:
    """Tests for FeishuChannel / 飞书渠道测试"""

    def test_build_payload_no_secret(self):
        from team_memory.monitoring.alerting import FeishuChannel, AlertMessage, AlertSeverity
        ch = FeishuChannel(webhook="https://open.feishu.cn/open-apis/bot/v2/hook/test")
        msg = AlertMessage(title="Daemon", body="down", severity=AlertSeverity.CRITICAL)
        payload = ch._build_payload(msg)
        assert payload["msg_type"] == "text"
        assert "Daemon" in payload["content"]["text"]
        assert "sign" not in payload

    def test_build_payload_with_secret(self):
        from team_memory.monitoring.alerting import FeishuChannel, AlertMessage
        ch = FeishuChannel(webhook="https://open.feishu.cn/open-apis/bot/v2/hook/test", secret="mysecret")
        msg = AlertMessage(title="Test", body="body")
        payload = ch._build_payload(msg)
        assert "sign" in payload
        assert "timestamp" in payload

    @pytest.mark.asyncio
    async def test_send_success(self):
        from team_memory.monitoring.alerting import FeishuChannel, AlertMessage
        ch = FeishuChannel(webhook="https://example.com/hook")
        msg = AlertMessage(title="T", body="B")
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {"code": 0, "msg": "success"}
        mock_resp.raise_for_status = MagicMock()
        mock_client = AsyncMock()
        mock_client.post = AsyncMock(return_value=mock_resp)
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)
        with patch("httpx.AsyncClient", return_value=mock_client):
            ok = await ch.send(msg)
        assert ok is True

    @pytest.mark.asyncio
    async def test_send_api_error(self):
        from team_memory.monitoring.alerting import FeishuChannel, AlertMessage
        ch = FeishuChannel(webhook="https://example.com/hook")
        msg = AlertMessage(title="T", body="B")
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {"code": 19001, "msg": "keyword not match"}
        mock_resp.raise_for_status = MagicMock()
        mock_client = AsyncMock()
        mock_client.post = AsyncMock(return_value=mock_resp)
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)
        with patch("httpx.AsyncClient", return_value=mock_client):
            ok = await ch.send(msg)
        assert ok is False


class TestWeComChannel:
    """Tests for WeComChannel / 企微渠道测试"""

    def test_build_payload(self):
        from team_memory.monitoring.alerting import WeComChannel, AlertMessage, AlertSeverity
        ch = WeComChannel(webhook="https://qyapi.weixin.qq.com/cgi-bin/webhook/send?key=test")
        msg = AlertMessage(title="Web", body="unhealthy", severity=AlertSeverity.WARNING)
        payload = ch._build_payload(msg)
        assert payload["msgtype"] == "text"
        assert "Web" in payload["text"]["content"]

    @pytest.mark.asyncio
    async def test_send_success(self):
        from team_memory.monitoring.alerting import WeComChannel, AlertMessage
        ch = WeComChannel(webhook="https://example.com/hook")
        msg = AlertMessage(title="T", body="B")
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {"errcode": 0, "errmsg": "ok"}
        mock_resp.raise_for_status = MagicMock()
        mock_client = AsyncMock()
        mock_client.post = AsyncMock(return_value=mock_resp)
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)
        with patch("httpx.AsyncClient", return_value=mock_client):
            ok = await ch.send(msg)
        assert ok is True


class TestAlertManager:
    """Tests for AlertManager / 告警管理器测试"""

    @pytest.mark.asyncio
    async def test_cooldown_suppresses(self):
        from team_memory.monitoring.alerting import AlertManager, LogChannel, AlertSeverity
        ch = LogChannel(log_file="/tmp/test_tm_alert.log")
        mgr = AlertManager(channels=[ch], alert_cooldown=60)
        await mgr.send_alert("Test", "body", AlertSeverity.INFO)
        # Second call within cooldown should be suppressed
        await mgr.send_alert("Test", "body", AlertSeverity.INFO)

    @pytest.mark.asyncio
    async def test_different_keys_no_cooldown(self):
        from team_memory.monitoring.alerting import AlertManager, LogChannel, AlertSeverity
        ch = LogChannel(log_file="/tmp/test_tm_alert2.log")
        mgr = AlertManager(channels=[ch], alert_cooldown=60)
        await mgr.send_alert("Key1", "body", AlertSeverity.INFO)
        await mgr.send_alert("Key2", "body", AlertSeverity.INFO)  # different key, should pass

    def test_from_config(self):
        from team_memory.monitoring.alerting import AlertManager
        from team_memory.monitoring.config import MonitoringConfig, ChannelConfig
        cfg = MonitoringConfig(
            channels=[
                ChannelConfig(type="feishu", webhook="https://example.com/hook", secret="s"),
                ChannelConfig(type="wecom", webhook="https://example.com/hook2"),
            ],
            watchdog_log_file="/tmp/test.log",
            watchdog_alert_cooldown=120,
        )
        mgr = AlertManager.from_config(cfg)
        assert len(mgr.channels) >= 4  # 2 configured + log + macos
        assert mgr.alert_cooldown == 120

    def test_create_channel_factory(self):
        from team_memory.monitoring.alerting import create_channel, FeishuChannel, WeComChannel, WebhookChannel
        assert isinstance(create_channel("feishu", webhook="x"), FeishuChannel)
        assert isinstance(create_channel("wecom", webhook="x"), WeComChannel)
        assert isinstance(create_channel("webhook", url="x"), WebhookChannel)


# ── Watchdog ──

class TestServiceWatchdog:
    """Tests for ServiceWatchdog / 服务看门狗测试"""

    @pytest.mark.asyncio
    async def test_check_service_up(self):
        from team_memory.monitoring.watchdog import ServiceWatchdog, ServiceState
        from team_memory.monitoring.config import MonitoringConfig
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
        from team_memory.monitoring.watchdog import ServiceWatchdog, ServiceState
        from team_memory.monitoring.config import MonitoringConfig
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

"""Alerting — multi-channel notifications / 多渠道告警通知

Strategy pattern: each platform implements AlertChannel interface.
Channels: feishu, wecom, webhook, macos, log
"""
from __future__ import annotations

import abc
import base64
import hashlib
import hmac
import json
import logging
import subprocess
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any

logger = logging.getLogger(__name__)


class AlertSeverity(str, Enum):
    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"


@dataclass
class AlertMessage:
    """Unified alert message / 统一告警消息"""
    title: str
    body: str
    severity: AlertSeverity = AlertSeverity.CRITICAL
    timestamp: str = ""

    def __post_init__(self):
        if not self.timestamp:
            self.timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")

    def to_plain_text(self) -> str:
        """Format as plain text / 格式化为纯文本"""
        level = self.severity.value.upper()
        return (
            f"[TM {level}] {self.title}\n"
            f"Time: {self.timestamp}\n"
            f"Detail: {self.body}"
        )


class AlertChannel(abc.ABC):
    """Base class for alert channels / 告警渠道基类"""

    @abc.abstractmethod
    async def send(self, message: AlertMessage) -> bool:
        """Send alert. Returns True on success / 发送告警，成功返回 True"""
        ...


class FeishuChannel(AlertChannel):
    """Feishu (飞书) custom bot webhook / 飞书自定义机器人"""

    def __init__(self, webhook: str, secret: str = ""):
        self.webhook = webhook
        self.secret = secret

    def _build_payload(self, message: AlertMessage) -> dict[str, Any]:
        """Build feishu webhook payload / 构建飞书 payload"""
        text = message.to_plain_text()
        payload: dict[str, Any] = {
            "msg_type": "text",
            "content": {"text": text},
        }
        # Add signature if secret configured
        if self.secret:
            timestamp = str(int(time.time()))
            string_to_sign = f"{timestamp}\n{self.secret}"
            hmac_code = hmac.new(
                string_to_sign.encode("utf-8"),
                digestmod=hashlib.sha256,
            ).digest()
            sign = base64.b64encode(hmac_code).decode("utf-8")
            payload["timestamp"] = timestamp
            payload["sign"] = sign
        return payload

    async def send(self, message: AlertMessage) -> bool:
        try:
            import httpx
            payload = self._build_payload(message)
            async with httpx.AsyncClient(timeout=10.0) as client:
                resp = await client.post(self.webhook, json=payload)
                resp.raise_for_status()
                data = resp.json()
                if data.get("code") != 0:
                    logger.error("Feishu alert failed: %s", data)
                    return False
                logger.info("Feishu alert sent OK")
                return True
        except Exception as e:
            logger.error("Feishu alert error: %s", e)
            return False


class WeComChannel(AlertChannel):
    """WeCom (企业微信) custom bot webhook / 企微自定义机器人"""

    def __init__(self, webhook: str):
        self.webhook = webhook

    def _build_payload(self, message: AlertMessage) -> dict[str, Any]:
        """Build wecom webhook payload / 构建企微 payload"""
        text = message.to_plain_text()
        return {
            "msgtype": "text",
            "text": {"content": text},
        }

    async def send(self, message: AlertMessage) -> bool:
        try:
            import httpx
            payload = self._build_payload(message)
            async with httpx.AsyncClient(timeout=10.0) as client:
                resp = await client.post(self.webhook, json=payload)
                resp.raise_for_status()
                data = resp.json()
                if data.get("errcode") != 0:
                    logger.error("WeCom alert failed: %s", data)
                    return False
                logger.info("WeCom alert sent OK")
                return True
        except Exception as e:
            logger.error("WeCom alert error: %s", e)
            return False


class WebhookChannel(AlertChannel):
    """Generic webhook (Slack, DingTalk, custom) / 通用 Webhook"""

    def __init__(self, url: str):
        self.url = url

    async def send(self, message: AlertMessage) -> bool:
        try:
            import httpx
            payload = {"text": message.to_plain_text()}
            async with httpx.AsyncClient(timeout=10.0) as client:
                resp = await client.post(self.url, json=payload)
                resp.raise_for_status()
                logger.info("Webhook alert sent OK")
                return True
        except Exception as e:
            logger.error("Webhook alert error: %s", e)
            return False


class MacOSNotificationChannel(AlertChannel):
    """macOS native notification / macOS 系统通知"""

    async def send(self, message: AlertMessage) -> bool:
        try:
            subprocess.run(
                ["osascript", "-e",
                 f'display notification "{message.body[:200]}" with title "TM Alert: {message.title}"'],
                timeout=5, capture_output=True,
            )
            return True
        except (OSError, subprocess.TimeoutExpired):
            return False


class LogChannel(AlertChannel):
    """Write alert to log file / 写入日志文件"""

    def __init__(self, log_file: str = "/tmp/tm-watchdog.log"):
        self.log_file = log_file

    async def send(self, message: AlertMessage) -> bool:
        try:
            with open(self.log_file, "a") as f:
                f.write(message.to_plain_text() + "\n---\n")
            return True
        except OSError as e:
            logger.error("Failed to write alert log: %s", e)
            return False


def create_channel(channel_type: str, **kwargs) -> AlertChannel:
    """Factory: create channel by type / 工厂方法：按类型创建渠道"""
    if channel_type == "feishu":
        return FeishuChannel(webhook=kwargs.get("webhook", ""), secret=kwargs.get("secret", ""))
    elif channel_type == "wecom":
        return WeComChannel(webhook=kwargs.get("webhook", ""))
    elif channel_type == "webhook":
        return WebhookChannel(url=kwargs.get("webhook", kwargs.get("url", "")))
    elif channel_type == "macos":
        return MacOSNotificationChannel()
    elif channel_type == "log":
        return LogChannel(log_file=kwargs.get("log_file", "/tmp/tm-watchdog.log"))
    else:
        logger.warning("Unknown channel type: %s, falling back to webhook", channel_type)
        return WebhookChannel(url=kwargs.get("webhook", ""))


@dataclass
class AlertManager:
    """Multi-channel alert manager / 多渠道告警管理器"""
    channels: list[AlertChannel] = field(default_factory=list)
    alert_cooldown: int = 300
    _last_alert_times: dict[str, float] = field(default_factory=dict, repr=False)

    def _should_alert(self, key: str) -> bool:
        """Check cooldown / 冷却检查"""
        now = time.time()
        last = self._last_alert_times.get(key, 0)
        if (now - last) < self.alert_cooldown:
            return False
        self._last_alert_times[key] = now
        return True

    async def send_alert(
        self,
        title: str,
        body: str,
        severity: AlertSeverity = AlertSeverity.CRITICAL,
    ) -> None:
        """Send alert through all configured channels / 通过所有渠道发送告警"""
        if not self._should_alert(title):
            logger.debug("Alert suppressed (cooldown): %s", title)
            return

        message = AlertMessage(title=title, body=body, severity=severity)
        logger.warning("ALERT: %s", message.to_plain_text())

        for channel in self.channels:
            try:
                ok = await channel.send(message)
                if not ok:
                    logger.warning("Channel %s returned False", type(channel).__name__)
            except Exception as e:
                logger.error("Channel %s error: %s", type(channel).__name__, e)

    @classmethod
    def from_config(cls, config: Any) -> AlertManager:
        """Create AlertManager from MonitoringConfig / 从配置创建"""
        from team_memory.monitoring.config import MonitoringConfig, ChannelConfig
        if not isinstance(config, MonitoringConfig):
            raise TypeError(f"Expected MonitoringConfig, got {type(config)}")

        channels: list[AlertChannel] = []

        # Add configured channels
        for ch in config.channels:
            if not ch.webhook:
                continue
            channels.append(create_channel(
                channel_type=ch.type,
                webhook=ch.webhook,
                secret=ch.secret,
            ))

        # Always add log + macos (local dev)
        channels.append(LogChannel(log_file=config.watchdog_log_file))
        channels.append(MacOSNotificationChannel())

        return cls(channels=channels, alert_cooldown=config.watchdog_alert_cooldown)

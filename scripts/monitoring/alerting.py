"""Alerting — webhook + local notification / 告警通知"""
from __future__ import annotations

import json
import logging
import subprocess
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum

logger = logging.getLogger(__name__)


class AlertSeverity(str, Enum):
    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"


@dataclass
class AlertManager:
    webhook_url: str = ""
    email: str = ""
    log_file: str = "/tmp/tm-watchdog.log"
    alert_cooldown: int = 300  # seconds between repeated alerts

    _last_alert_times: dict[str, float] = field(default_factory=dict, repr=False)

    def _should_alert(self, service: str) -> bool:
        """Check cooldown to avoid alert storms."""
        now = time.time()
        last = self._last_alert_times.get(service, 0)
        if (now - last) < self.alert_cooldown:
            return False
        self._last_alert_times[service] = now
        return True

    def _format_message(self, service: str, status: str, detail: str, severity: AlertSeverity) -> str:
        now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
        return (
            f"[TM {severity.value.upper()}] {service} {status}\n"
            f"Time: {now}\n"
            f"Detail: {detail}"
        )

    async def send_alert(self, service: str, status: str, detail: str, severity: AlertSeverity = AlertSeverity.CRITICAL) -> None:
        """Send alert through configured channels."""
        if not self._should_alert(service):
            logger.debug("Alert suppressed (cooldown): %s", service)
            return

        message = self._format_message(service, status, detail, severity)
        logger.warning("ALERT: %s", message)

        # Log file
        self._write_log(message)

        # Webhook
        if self.webhook_url:
            await self._send_webhook(message)

        # macOS notification (local dev)
        self._send_macos_notification(f"TM Alert: {service}", detail)

    def _write_log(self, message: str) -> None:
        try:
            with open(self.log_file, "a") as f:
                f.write(message + "\n---\n")
        except OSError as e:
            logger.error("Failed to write alert log: %s", e)

    async def _send_webhook(self, message: str) -> None:
        try:
            import httpx
            payload = {"text": message}
            async with httpx.AsyncClient(timeout=10.0) as client:
                resp = await client.post(self.webhook_url, json=payload)
                resp.raise_for_status()
                logger.info("Webhook alert sent OK")
        except Exception as e:
            logger.error("Webhook alert failed: %s", e)

    def _send_macos_notification(self, title: str, message: str) -> None:
        try:
            subprocess.run(
                ["osascript", "-e", f'display notification "{message}" with title "{title}"'],
                timeout=5, capture_output=True,
            )
        except (OSError, subprocess.TimeoutExpired):
            pass  # not on macOS or osascript not available

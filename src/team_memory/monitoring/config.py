"""Monitoring configuration loader"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path


def _find_monitoring_yaml() -> Path:
    """Find monitoring.yaml in project config/ or CWD."""
    candidates = [
        Path(__file__).resolve().parents[2] / "config" / "monitoring.yaml",
        Path("config/monitoring.yaml"),
    ]
    for p in candidates:
        if p.exists():
            return p
    raise FileNotFoundError("monitoring.yaml not found")


@dataclass
class ChannelConfig:
    """Single alert channel configuration / 单个告警渠道配置"""

    type: str = ""  # feishu | wecom | webhook
    webhook: str = ""  # full webhook URL
    secret: str = ""  # signing secret (feishu only, optional)


@dataclass
class MonitoringConfig:
    # Admin
    channels: list[ChannelConfig] = field(default_factory=list)
    restart_allowed: bool = True

    # Health check targets
    web_url: str = "http://localhost:9111/health"
    web_timeout: int = 10
    web_expected_status: int = 200
    daemon_url: str = "http://localhost:3901/status"
    daemon_timeout: int = 5
    daemon_expected_status: int = 200

    # Watchdog
    watchdog_enabled: bool = True
    watchdog_interval: int = 60
    watchdog_max_failures: int = 3
    watchdog_restart_cooldown: int = 60
    watchdog_alert_cooldown: int = 300
    watchdog_log_file: str = "/tmp/tm-watchdog.log"
    restart_web_cmd: str = ""
    restart_daemon_cmd: str = ""
    workspace: str = ""


def load_monitoring_config(config_path: str | None = None) -> MonitoringConfig:
    """Load monitoring.yaml into MonitoringConfig dataclass."""
    import yaml

    path = Path(config_path) if config_path else _find_monitoring_yaml()
    if not path.exists():
        return MonitoringConfig()

    with open(path) as f:
        content = f.read()
    # Substitute ${VAR} patterns with environment variables
    import re

    def _env_sub(match: object) -> str:
        var = match.group(1)  # type: ignore[union-attr]
        return os.environ.get(var, "")

    content = re.sub(r"\$\{(\w+)\}", _env_sub, content)
    raw = yaml.safe_load(content) or {}

    admin = raw.get("admin", {})
    hc = raw.get("health_check", {})
    web_cfg = hc.get("web", {})
    daemon_cfg = hc.get("daemon", {})
    wd = raw.get("watchdog", {})

    # Parse channels
    channels: list[ChannelConfig] = []
    for ch in admin.get("channels", []):
        if not ch:
            continue
        channels.append(
            ChannelConfig(
                type=ch.get("type", "webhook"),
                webhook=ch.get("webhook", ch.get("url", "")),
                secret=ch.get("secret", ""),
            )
        )

    # Backward compat: if old-style webhook field exists, add as generic channel
    legacy_webhook = admin.get("webhook", "")
    if legacy_webhook and not any(c.webhook == legacy_webhook for c in channels):
        channels.append(ChannelConfig(type="webhook", webhook=legacy_webhook))

    return MonitoringConfig(
        channels=channels,
        restart_allowed=admin.get("restart_allowed", True),
        web_url=web_cfg.get("url", "http://localhost:9111/health"),
        web_timeout=web_cfg.get("timeout", 10),
        web_expected_status=web_cfg.get("expected_status", 200),
        daemon_url=daemon_cfg.get("url", "http://localhost:3901/status"),
        daemon_timeout=daemon_cfg.get("timeout", 5),
        daemon_expected_status=daemon_cfg.get("expected_status", 200),
        watchdog_enabled=wd.get("enabled", True),
        watchdog_interval=wd.get("interval", 60),
        watchdog_max_failures=wd.get("max_failures", 3),
        watchdog_restart_cooldown=wd.get("restart_cooldown", 60),
        watchdog_alert_cooldown=wd.get("alert_cooldown", 300),
        watchdog_log_file=wd.get("log_file", "/tmp/tm-watchdog.log"),
        restart_web_cmd=wd.get("restart_web", ""),
        restart_daemon_cmd=wd.get("restart_daemon", ""),
        workspace=wd.get("workspace", ""),
    )

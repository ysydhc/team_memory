"""Monitoring configuration loader"""
from __future__ import annotations

import os
from pathlib import Path
from dataclasses import dataclass


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
class MonitoringConfig:
    # Admin
    admin_webhook: str = ""
    admin_email: str = ""
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
        raw = yaml.safe_load(f) or {}

    admin = raw.get("admin", {})
    hc = raw.get("health_check", {})
    web_cfg = hc.get("web", {})
    daemon_cfg = hc.get("daemon", {})
    wd = raw.get("watchdog", {})

    return MonitoringConfig(
        admin_webhook=admin.get("webhook", ""),
        admin_email=admin.get("email", ""),
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

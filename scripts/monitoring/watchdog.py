"""Watchdog — external health probe + auto-restart / 外部探活 + 自动重启"""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass, field

from monitoring.alerting import AlertManager, AlertSeverity
from monitoring.config import MonitoringConfig, load_monitoring_config

logger = logging.getLogger(__name__)


@dataclass
class ServiceState:
    name: str
    url: str
    timeout: int = 10
    expected_status: int = 200
    consecutive_failures: int = 0
    last_check: float = 0.0
    last_ok: float = 0.0
    is_up: bool = True


@dataclass
class ServiceWatchdog:
    config: MonitoringConfig = field(default_factory=load_monitoring_config)
    alert_manager: AlertManager = field(default=None, repr=False)
    _services: dict[str, ServiceState] = field(default_factory=dict, repr=False)
    _running: bool = False
    _task: asyncio.Task | None = field(default=None, repr=False)

    def __post_init__(self):
        if self.alert_manager is None:
            self.alert_manager = AlertManager.from_config(self.config)
        self._services["web"] = ServiceState(
            name="team-memory-web",
            url=self.config.web_url,
            timeout=self.config.web_timeout,
            expected_status=self.config.web_expected_status,
        )
        self._services["daemon"] = ServiceState(
            name="team-memory-daemon",
            url=self.config.daemon_url,
            timeout=self.config.daemon_timeout,
            expected_status=self.config.daemon_expected_status,
        )

    async def check_service(self, svc: ServiceState) -> bool:
        """Probe a single service. Returns True if healthy."""
        import httpx

        try:
            async with httpx.AsyncClient(timeout=svc.timeout) as client:
                resp = await client.get(svc.url)
                ok = resp.status_code == svc.expected_status
                # Parse health JSON for degraded checks
                degraded_checks: list[str] = []
                if ok and svc.name == "team-memory-web":
                    try:
                        data = resp.json()
                        for check_name, check_val in data.get("checks", {}).items():
                            if isinstance(check_val, dict) and check_val.get("status") == "down":
                                err = check_val.get("error", "unknown")
                                degraded_checks.append(f"{check_name}: {err[:60]}")
                    except Exception:
                        pass
        except Exception:
            ok = False
            degraded_checks = []

        now = time.time()
        svc.last_check = now

        if ok and not degraded_checks:
            if not svc.is_up:
                logger.info("Service %s recovered", svc.name)
                await self.alert_manager.send_alert(
                    f"{svc.name} recovered", "Service is back online", AlertSeverity.INFO
                )
            svc.consecutive_failures = 0
            svc.is_up = True
            svc.last_ok = now
        elif ok and degraded_checks:
            # HTTP OK but some checks degraded
            svc.last_ok = now
            svc.is_up = True
            svc.consecutive_failures = 0
            detail = "; ".join(degraded_checks)
            logger.warning("Service %s degraded: %s", svc.name, detail)
            await self._send_degraded_alert(svc, detail)
        else:
            svc.consecutive_failures += 1
            svc.is_up = False
            logger.warning(
                "Service %s check failed (%d/%d)",
                svc.name,
                svc.consecutive_failures,
                self.config.watchdog_max_failures,
            )
            if svc.consecutive_failures >= self.config.watchdog_max_failures:
                await self._handle_failure(svc)

        return ok

    async def _send_degraded_alert(self, svc: ServiceState, detail: str) -> None:
        """Send a WARNING alert for degraded checks (respects alert_cooldown)."""
        await self.alert_manager.send_alert(
            f"{svc.name} degraded",
            f"Some checks are down: {detail}",
            AlertSeverity.WARNING,
        )

    async def _handle_failure(self, svc: ServiceState) -> None:
        """Handle consecutive failure threshold reached."""
        logger.critical("Service %s reached failure threshold, attempting restart", svc.name)

        # Alert
        await self.alert_manager.send_alert(
            f"{svc.name} down",
            f"Consecutive failures: {svc.consecutive_failures}. Attempting restart.",
            AlertSeverity.CRITICAL,
        )

        # Restart
        if self.config.restart_allowed:
            restarted = await self._restart_service(svc)
            if restarted:
                svc.consecutive_failures = 0  # reset after restart
                logger.info("Service %s restart initiated", svc.name)
            else:
                await self.alert_manager.send_alert(
                    f"{svc.name} restart_failed",
                    "Auto-restart failed. Manual intervention required.",
                    AlertSeverity.CRITICAL,
                )

    async def _restart_service(self, svc: ServiceState) -> bool:
        """Execute restart command for a service."""
        if svc.name == "team-memory-web":
            cmd = self.config.restart_web_cmd
        else:
            cmd = self.config.restart_daemon_cmd

        if not cmd:
            logger.warning("No restart command configured for %s", svc.name)
            return False

        # Substitute {workspace}
        cmd = cmd.replace("{workspace}", self.config.workspace)

        try:
            proc = await asyncio.create_subprocess_shell(
                cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=30)
            if proc.returncode == 0:
                logger.info("Restart command OK: %s", cmd)
                # Wait for service to come up
                await asyncio.sleep(self.config.watchdog_restart_cooldown)
                return True
            else:
                logger.error("Restart failed (rc=%d): %s", proc.returncode, stderr.decode()[:200])
                return False
        except Exception as e:
            logger.error("Restart error: %s", e)
            return False

    async def _loop(self) -> None:
        """Main watchdog loop."""
        logger.info("Watchdog started (interval=%ds)", self.config.watchdog_interval)
        while self._running:
            for svc in self._services.values():
                await self.check_service(svc)
            await asyncio.sleep(self.config.watchdog_interval)

    def start(self) -> None:
        """Start watchdog background task."""
        self._running = True
        self._task = asyncio.create_task(self._loop())

    def stop(self) -> None:
        """Stop watchdog."""
        self._running = False
        if self._task and not self._task.done():
            self._task.cancel()

    def get_status(self) -> dict:
        return {
            name: {
                "is_up": svc.is_up,
                "consecutive_failures": svc.consecutive_failures,
                "last_ok": svc.last_ok,
            }
            for name, svc in self._services.items()
        }


def main():
    """Entry point for standalone watchdog."""
    import sys

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    cfg = load_monitoring_config(sys.argv[1] if len(sys.argv) > 1 else None)
    watchdog = ServiceWatchdog(config=cfg)

    loop = asyncio.new_event_loop()
    watchdog._running = True
    try:
        loop.run_until_complete(watchdog._loop())
    except KeyboardInterrupt:
        watchdog.stop()
        logger.info("Watchdog stopped")


if __name__ == "__main__":
    main()

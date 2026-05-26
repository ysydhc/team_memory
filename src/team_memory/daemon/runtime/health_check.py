"""HealthCheck — in-process health monitoring with backoff alerting.

在 daemon._tick 末尾运行，检查进程内健康状态。
与服务 ServiceWatchdog（外部 HTTP probe）互补。
"""

from __future__ import annotations

import asyncio
import logging
import os
import time
from typing import Callable

logger = logging.getLogger(__name__)

# Backoff: 首次立即告警，后续按退避时间递增
BACKOFF_MINUTES = [10, 20, 40]
SQLITE_MAX_MB = 500
PG_TIMEOUT = 3.0


class HealthCheck:
    """In-process health check / 进程内健康检查。

    Checks: worker task status, SQLite file size, PG connectivity.
    Alerts with backoff via the provided callback (usually send_daemon_alert).
    """

    def __init__(
        self,
        db_path: str,
        pg_url: str | None = None,
        worker_task: asyncio.Task | None = None,
        alert_callback: Callable | None = None,
        pending_queue: Any | None = None,
    ) -> None:
        self._db_path = db_path
        self._pg_url = pg_url
        self._worker_task = worker_task
        self._alert_cb = alert_callback
        self._pending_queue = pending_queue
        # {name: {"last_alert": float, "backoff_idx": int, "started": float}}
        self._alerts: dict[str, dict[str, float | int]] = {}
        # Stall detection: track last time worker produced output
        self._last_worker_activity: float = time.time()
        self._stall_threshold: float = 600.0  # 10 minutes of inactivity → alert

    async def check_all(self) -> list[str]:
        """Run all checks. Returns list of anomaly names. 返回异常列表。"""
        anomalies: list[str] = []

        checks = [
            ("sqlite_size", self._check_sqlite_size),
            ("worker_task", self._check_worker_task),
            ("pending_failures", self._check_pending_failures),
            ("worker_stall", self._check_worker_stall),
        ]
        if self._pg_url:
            checks.append(("pg_connect", self._check_pg_connect))

        for name, check_fn in checks:
            detail = await check_fn()
            if detail:
                anomalies.append(name)
                await self._handle_alert(name, detail)
            elif name in self._alerts:
                await self._handle_recovery(name)

        return anomalies

    async def _check_sqlite_size(self) -> str | None:
        """Check SQLite file size / 检查 SQLite 文件大小。"""
        try:
            size_mb = os.path.getsize(self._db_path) / (1024 * 1024)
            if size_mb > SQLITE_MAX_MB:
                return f"SQLite {size_mb:.0f}MB > {SQLITE_MAX_MB}MB limit"
        except OSError as e:
            return f"SQLite file access error: {e}"
        return None

    async def _check_worker_task(self) -> str | None:
        """Check RefinementWorker task / 检查 worker asyncio task。"""
        if self._worker_task and self._worker_task.done():
            try:
                exc = self._worker_task.exception()
                return f"Worker task stopped (exception={exc})" if exc else "Worker task stopped unexpectedly"
            except asyncio.InvalidStateError:
                return "Worker task in invalid state"
        return None

    async def _check_pg_connect(self) -> str | None:
        """Check PG connectivity / 检查 PG 连通。"""
        try:
            import asyncpg

            conn = await asyncio.wait_for(
                asyncpg.connect(self._pg_url),
                timeout=PG_TIMEOUT,
            )
            await conn.close()
        except asyncio.TimeoutError:
            return f"PG timeout ({PG_TIMEOUT}s)"
        except Exception as e:
            return f"PG error: {e}"
        return None

    async def _check_pending_failures(self) -> str | None:
        """Check if PendingQueue has excessive failures."""
        if not self._pending_queue:
            return None
        try:
            s = self._pending_queue.stats()
            failed = s.get("failed", 0)
            if failed >= 3:
                return f"PendingQueue has {failed} permanently failed ops"
        except Exception as e:
            return f"PendingQueue stats error: {e}"
        return None

    async def _check_worker_stall(self) -> str | None:
        """Check if RefinementWorker has been inactive too long."""
        idle = time.time() - self._last_worker_activity
        if idle > self._stall_threshold:
            return (
                f"RefinementWorker appears stalled — no activity for {idle:.0f}s "
                f"(threshold={self._stall_threshold:.0f}s)"
            )
        return None

    def report_worker_activity(self) -> None:
        """Call from RefinementWorker after each successful tick."""
        self._last_worker_activity = time.time()

    def _should_alert(self, name: str) -> bool:
        """Check backoff: should we alert now? 检查退避：现在该告警吗？"""
        now = time.time()
        state = self._alerts.get(name)
        if not state:
            return True  # first time

        idx = int(state.get("backoff_idx", 0))
        if idx >= len(BACKOFF_MINUTES):
            return False  # exhausted all retries

        elapsed = now - float(state.get("last_alert", 0))
        backoff_sec = BACKOFF_MINUTES[idx] * 60
        return elapsed >= backoff_sec

    async def _handle_alert(self, name: str, detail: str) -> None:
        """Send alert if backoff permits / 如果退避允许则发告警。"""
        if not self._should_alert(name):
            return
        if not self._alert_cb:
            logger.warning("HealthCheck: %s — %s (no alert callback)", name, detail)
            return

        now = time.time()
        state = self._alerts.get(name, {})
        is_new = name not in self._alerts

        logger.warning("HealthCheck ALERT: %s — %s", name, detail)
        try:
            await self._alert_cb(
                title=f"TM {name}",
                body=detail,
                severity="warning",
            )
        except Exception as e:
            logger.error("HealthCheck alert send failed: %s", e)

        self._alerts[name] = {
            "last_alert": now,
            "backoff_idx": state.get("backoff_idx", 0) + (0 if is_new else 1),
            "started": state.get("started", now),
        }

    async def _handle_recovery(self, name: str) -> None:
        """Send recovery notification / 发送恢复通知。"""
        state = self._alerts.pop(name, None)
        if not state or not self._alert_cb:
            return

        now = time.time()
        duration_min = int((now - float(state.get("started", now))) / 60) or 1

        logger.info("HealthCheck RECOVERY: %s (down %d min)", name, duration_min)
        try:
            await self._alert_cb(
                title=f"TM {name} recovered",
                body=f"Was down for ~{duration_min} minute(s)",
                severity="info",
            )
        except Exception as e:
            logger.error("HealthCheck recovery send failed: %s", e)

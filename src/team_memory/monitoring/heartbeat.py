"""Process-level heartbeat with sleep/wake detection."""

from __future__ import annotations

import asyncio
import logging
import os
import time
from collections.abc import Callable, Awaitable
from dataclasses import dataclass, field
from datetime import datetime, timezone

logger = logging.getLogger(__name__)

HEARTBEAT_INTERVAL = 30  # seconds
SLEEP_THRESHOLD = 300  # 5 minutes — gap larger than this → sleep detected


@dataclass
class HeartbeatState:
    last_beat: float = 0.0
    start_time: float = field(default_factory=time.time)
    pid: int = field(default_factory=os.getpid)
    _task: asyncio.Task | None = field(default=None, repr=False)
    # Sleep detection fields
    _last_monotonic: float = field(default_factory=time.monotonic, repr=False)
    _wake_callback: Callable[[float], Awaitable[None]] | None = field(
        default=None, repr=False
    )

    @property
    def last_beat_iso(self) -> str:
        if self.last_beat == 0:
            return ""
        return datetime.fromtimestamp(self.last_beat, tz=timezone.utc).isoformat()

    @property
    def uptime_seconds(self) -> int:
        return int(time.time() - self.start_time)

    def to_dict(self) -> dict:
        return {
            "last_beat": self.last_beat_iso,
            "uptime_seconds": self.uptime_seconds,
            "pid": self.pid,
        }

    def is_alive(self, timeout: int = 90) -> bool:
        if self.last_beat == 0:
            return True
        return (time.time() - self.last_beat) < timeout

    def on_wake(self, callback: Callable[[float], Awaitable[None]]) -> None:
        """Register a callback fired when sleep is detected (arg = sleep_seconds)."""
        self._wake_callback = callback


async def _beat_loop(state: HeartbeatState, interval: int) -> None:
    """Background heartbeat with sleep detection via monotonic clock."""
    while True:
        state.last_beat = time.time()

        # --- Sleep detection via monotonic clock ---
        now_mono = time.monotonic()
        gap = now_mono - state._last_monotonic
        state._last_monotonic = now_mono

        if gap > SLEEP_THRESHOLD:
            sleep_sec = gap - interval  # subtract normal interval
            logger.warning(
                "[WAKE]  Sleep detected — %.0f sec gap (%.1f min). Triggering recovery.",
                sleep_sec,
                sleep_sec / 60,
            )
            if state._wake_callback:
                try:
                    await state._wake_callback(sleep_sec)
                except Exception as exc:
                    logger.error("[WAKE]  Recovery callback failed: %s", exc)
        elif gap > interval * 2:
            logger.info("[WAKE]  Minor gap %.0fs (likely GC/load), no recovery", gap)
        else:
            logger.debug("Heartbeat beat (pid=%d)", state.pid)

        await asyncio.sleep(interval)


def start_heartbeat(interval: int = HEARTBEAT_INTERVAL) -> HeartbeatState:
    """Start heartbeat background task."""
    state = HeartbeatState()
    state._task = asyncio.create_task(_beat_loop(state, interval))
    logger.info("Heartbeat started (interval=%ds, pid=%d)", interval, state.pid)
    return state


def stop_heartbeat(state: HeartbeatState) -> None:
    """Stop heartbeat background task."""
    if state._task and not state._task.done():
        state._task.cancel()
        logger.info("Heartbeat stopped")

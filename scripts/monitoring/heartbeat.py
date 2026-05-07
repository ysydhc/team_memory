"""Process-level heartbeat / 进程级心跳"""
from __future__ import annotations

import asyncio
import os
import time
import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone

logger = logging.getLogger(__name__)

HEARTBEAT_INTERVAL = 30  # seconds


@dataclass
class HeartbeatState:
    last_beat: float = 0.0
    start_time: float = field(default_factory=time.time)
    pid: int = field(default_factory=os.getpid)
    _task: asyncio.Task | None = field(default=None, repr=False)

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
        """Check if heartbeat is within timeout window."""
        if self.last_beat == 0:
            return True  # not started yet
        return (time.time() - self.last_beat) < timeout


async def _beat_loop(state: HeartbeatState, interval: int) -> None:
    """Background loop that updates heartbeat timestamp."""
    while True:
        state.last_beat = time.time()
        logger.debug("Heartbeat beat (pid=%d)", state.pid)
        await asyncio.sleep(interval)


def start_heartbeat(interval: int = HEARTBEAT_INTERVAL) -> HeartbeatState:
    """Start heartbeat background task. Call from app startup."""
    state = HeartbeatState()
    state._task = asyncio.create_task(_beat_loop(state, interval))
    logger.info("Heartbeat started (interval=%ds, pid=%d)", interval, state.pid)
    return state


def stop_heartbeat(state: HeartbeatState) -> None:
    """Stop heartbeat background task."""
    if state._task and not state._task.done():
        state._task.cancel()
        logger.info("Heartbeat stopped")

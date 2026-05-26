"""SessionTimeoutManager — 30-minute draft safety net.

If a session hasn't been updated for 30 minutes, force-trigger session_end
so the daemon can flush and refine its draft. This prevents memory loss from
sessions that end without explicit closure.

No daemon internals are imported — all communication goes through the HTTP
hook endpoints (daemon_http.async_post_hook).
"""

from __future__ import annotations

import asyncio
import logging

from team_memory.hooks.core.daemon_http import async_post_hook

logger = logging.getLogger(__name__)


class SessionTimeoutManager:
    """Force-triggers session_end for sessions idle beyond the timeout threshold.

    Periodically polls /hooks/session_timeout_check to discover idle sessions,
    then fires /hooks/session_end for each one.

    Args:
        daemon_url: Base URL of the daemon (unused directly — daemon_http reads
            TM_DAEMON_URL; kept for backwards-compatible construction).
        timeout_minutes: Number of minutes after which a session is considered
            timed out. Informational only — the daemon enforces the actual
            cutoff in /hooks/session_timeout_check.
    """

    def __init__(
        self,
        daemon_url: str = "http://localhost:3901",
        timeout_minutes: int = 30,
    ) -> None:
        self._daemon_url = daemon_url  # retained for potential future use
        self._timeout_minutes = timeout_minutes
        self._task: asyncio.Task | None = None
        self._running = False

    async def start(self) -> None:
        """Start background check loop."""
        self._running = True
        self._task = asyncio.create_task(self._check_loop())

    async def stop(self) -> None:
        """Stop background check loop."""
        self._running = False
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
            self._task = None

    async def _check_loop(self) -> None:
        """Check every 5 minutes for timed-out sessions."""
        while self._running:
            await asyncio.sleep(300)  # 5 min check interval
            await self._check_timeouts()

    async def _check_timeouts(self) -> None:
        """Ask the daemon for timed-out sessions, then fire session_end for each."""
        try:
            result = await async_post_hook("session_timeout_check", {})
        except Exception as exc:
            logger.warning("session_timeout_check failed: %s", exc)
            return

        timed_out: list[str] = result.get("timed_out_sessions", [])
        for sid in timed_out:
            try:
                await async_post_hook(
                    "session_end",
                    {"conversation_id": sid, "reason": "timeout"},
                )
                logger.info(
                    "session_timeout: fired session_end for %s (idle > %d min)",
                    sid,
                    self._timeout_minutes,
                )
            except Exception as exc:
                logger.warning("session_end for %s failed: %s", sid, exc)

    async def check_and_publish(self, session_id: str) -> dict | None:
        """Manually trigger timeout check for a specific session.

        Fires session_end for *session_id* if the daemon considers it timed out.

        Args:
            session_id: The conversation / session identifier.

        Returns:
            The daemon response dict, or None if the session was not timed out
            or the daemon was unreachable.
        """
        try:
            result = await async_post_hook("session_timeout_check", {})
        except Exception as exc:
            logger.warning("session_timeout_check failed: %s", exc)
            return None

        timed_out: list[str] = result.get("timed_out_sessions", [])
        if session_id not in timed_out:
            return None

        try:
            return await async_post_hook(
                "session_end",
                {"conversation_id": session_id, "reason": "timeout"},
            )
        except Exception as exc:
            logger.warning("session_end for %s failed: %s", session_id, exc)
            return None

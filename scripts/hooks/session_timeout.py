"""SessionTimeoutManager — 30-minute draft safety net.

If a draft hasn't been updated for 30 minutes, force-publish it regardless
of convergence signals. This prevents memory loss from sessions that end
without explicit closure.
"""
from __future__ import annotations

import asyncio
from datetime import datetime, timedelta, timezone
from typing import Any, Callable

from draft_buffer import DraftBuffer
from draft_refiner import DraftRefiner


def _utcnow() -> datetime:
    """Default clock — returns current UTC time. Override for testing."""
    return datetime.now(timezone.utc)


class SessionTimeoutManager:
    """Force-publishes drafts that have been pending beyond the timeout threshold.

    Args:
        draft_refiner: DraftRefiner instance for publishing drafts.
        draft_buffer: DraftBuffer instance for querying pending drafts.
        timeout_minutes: Number of minutes after which a draft is considered
            timed out. Defaults to 30.
        clock: Optional callable returning a timezone-aware datetime.
            Defaults to ``datetime.now(timezone.utc)``. Override in tests
            to simulate time passing.
    """

    def __init__(
        self,
        draft_refiner: DraftRefiner,
        draft_buffer: DraftBuffer,
        timeout_minutes: int = 30,
        clock: Callable[[], datetime] | None = None,
    ) -> None:
        self._refiner = draft_refiner
        self._buffer = draft_buffer
        self._timeout = timedelta(minutes=timeout_minutes)
        self._clock = clock or _utcnow
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
        """Check every 5 minutes for timed-out drafts."""
        while self._running:
            await asyncio.sleep(300)  # 5 min check interval
            await self._check_timeouts()

    async def _check_timeouts(self) -> None:
        """Find and force-publish timed-out drafts."""
        pending = await self._buffer.get_all_pending()
        now = self._clock()
        # Group by session to avoid duplicate publish calls
        timed_out_sessions: set[str] = set()
        for draft in pending:
            updated_at = draft.get("updated_at")
            if updated_at is None:
                continue
            parsed = _parse_updated_at(updated_at)
            if now - parsed >= self._timeout:
                session_id = draft.get("conversation_id", "")
                if session_id and session_id not in timed_out_sessions:
                    timed_out_sessions.add(session_id)
                    await self._refiner.refine_and_publish(session_id)

    async def check_and_publish(self, session_id: str) -> dict[str, Any] | None:
        """Manually trigger timeout check for a specific session.

        If the most recent draft for *session_id* has been pending longer
        than the timeout threshold, force-publish it via DraftRefiner.

        Args:
            session_id: The conversation / session identifier.

        Returns:
            The result dict from refine_and_publish, or None if the session
            has no pending drafts or the drafts have not timed out.
        """
        draft = await self._buffer.get_pending(session_id)
        if not draft:
            return None
        # Get the most recent draft's updated_at
        latest = max(
            draft,
            key=lambda d: _parse_updated_at(d.get("updated_at")),
        )
        updated_at = latest.get("updated_at")
        if updated_at is None:
            return None
        parsed = _parse_updated_at(updated_at)
        if self._clock() - parsed >= self._timeout:
            return await self._refiner.refine_and_publish(session_id)
        return None


def _parse_updated_at(value: Any) -> datetime:
    """Parse updated_at value into a timezone-aware datetime.

    Args:
        value: Either a datetime object or an ISO format string.

    Returns:
        Timezone-aware datetime in UTC.
    """
    if isinstance(value, datetime):
        if value.tzinfo is None:
            return value.replace(tzinfo=timezone.utc)
        return value
    if isinstance(value, str):
        dt = datetime.fromisoformat(value)
        if dt.tzinfo is None:
            return dt.replace(tzinfo=timezone.utc)
        return dt
    # Fallback — treat as epoch 0
    return datetime.min.replace(tzinfo=timezone.utc)

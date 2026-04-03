"""In-process async event bus for decoupling modules.

Provides a simple publish/subscribe mechanism based on asyncio.
Used primarily for cache invalidation on data changes.

Usage:
    bus = EventBus()
    bus.on(Events.EXPERIENCE_CREATED, my_handler)
    await bus.emit(Events.EXPERIENCE_CREATED, {"id": "..."})
"""

from __future__ import annotations

import asyncio
import logging
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Callable, Coroutine

logger = logging.getLogger("team_memory.events")

# Type alias for async event handler
EventHandler = Callable[[dict[str, Any]], Coroutine[Any, Any, None]]


class Events:
    """Event type constants.

    Naming convention: <entity>.<action>
    """

    # Experience lifecycle events
    EXPERIENCE_CREATED = "experience.created"
    EXPERIENCE_UPDATED = "experience.updated"
    EXPERIENCE_DELETED = "experience.deleted"
    EXPERIENCE_RESTORED = "experience.restored"
    EXPERIENCE_PUBLISHED = "experience.published"
    EXPERIENCE_REVIEWED = "experience.reviewed"

    # Feedback events
    FEEDBACK_ADDED = "feedback.added"

    # Archive lifecycle events
    ARCHIVE_CREATED = "archive.created"


@dataclass
class Event:
    """Structured event payload."""

    type: str
    payload: dict[str, Any]
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


class EventBus:
    """Simple in-process async event bus.

    Handlers are registered per event type and called concurrently
    when an event is emitted. Handler exceptions are caught and logged
    so that one failing handler does not block others.
    """

    def __init__(self) -> None:
        self._handlers: dict[str, list[EventHandler]] = defaultdict(list)

    def on(self, event_type: str, handler: EventHandler) -> None:
        """Register an async handler for an event type."""
        self._handlers[event_type].append(handler)
        logger.debug("Registered handler %s for event '%s'", handler.__name__, event_type)

    async def emit(self, event_type: str, payload: dict[str, Any] | None = None) -> None:
        """Emit an event, calling all registered handlers concurrently."""
        if payload is None:
            payload = {}

        handlers = self._handlers.get(event_type, [])
        if not handlers:
            return

        logger.debug("Emitting event '%s' to %d handler(s)", event_type, len(handlers))
        tasks = [self._safe_call(h, payload, event_type) for h in handlers]
        await asyncio.gather(*tasks)

    async def _safe_call(
        self, handler: EventHandler, payload: dict[str, Any], event_type: str
    ) -> None:
        """Call a handler with error isolation."""
        try:
            await handler(payload)
        except Exception:
            logger.warning(
                "Event handler %s failed for '%s'",
                handler.__name__,
                event_type,
                exc_info=True,
            )

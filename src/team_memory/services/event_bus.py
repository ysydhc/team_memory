"""In-process async event bus for decoupling modules.

Provides a simple publish/subscribe mechanism based on asyncio.
Event handlers are called asynchronously when events are emitted.
Failures in one handler do not affect others.

Usage:
    bus = EventBus()
    bus.on(Events.EXPERIENCE_CREATED, my_handler)
    await bus.emit(Events.EXPERIENCE_CREATED, {"id": "..."})

Future upgrade path:
    - Redis Pub/Sub for multi-instance event broadcasting
    - Event sourcing with persistent event log
    - Webhook notifications
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
    EXPERIENCE_MERGED = "experience.merged"
    EXPERIENCE_ROLLED_BACK = "experience.rolled_back"
    EXPERIENCE_IMPORTED = "experience.imported"
    EXPERIENCE_PUBLISHED = "experience.published"
    EXPERIENCE_REVIEWED = "experience.reviewed"

    # Search events
    SEARCH_EXECUTED = "search.executed"

    # Embedding events
    EMBEDDING_COMPLETED = "embedding.completed"
    EMBEDDING_FAILED = "embedding.failed"

    # Feedback events
    FEEDBACK_ADDED = "feedback.added"


@dataclass
class Event:
    """Structured event payload."""

    type: str
    payload: dict[str, Any]
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    def to_dict(self) -> dict:
        return {
            "type": self.type,
            "payload": self.payload,
            "timestamp": self.timestamp.isoformat(),
        }


class EventBus:
    """Simple in-process async event bus.

    Handlers are registered per event type and called concurrently
    when an event is emitted. Handler exceptions are caught and logged
    so that one failing handler does not block others.
    """

    def __init__(self) -> None:
        self._handlers: dict[str, list[EventHandler]] = defaultdict(list)
        self._event_log: list[Event] = []  # Optional: keep recent events for debugging
        self._max_log_size = 100

    def on(self, event_type: str, handler: EventHandler) -> None:
        """Register an async handler for an event type.

        Args:
            event_type: Event type string (use Events constants).
            handler: Async callable receiving a dict payload.
        """
        self._handlers[event_type].append(handler)
        logger.debug("Registered handler %s for event '%s'", handler.__name__, event_type)

    def off(self, event_type: str, handler: EventHandler) -> None:
        """Unregister a handler."""
        handlers = self._handlers.get(event_type, [])
        if handler in handlers:
            handlers.remove(handler)

    async def emit(self, event_type: str, payload: dict[str, Any] | None = None) -> None:
        """Emit an event, calling all registered handlers concurrently.

        Args:
            event_type: Event type string.
            payload: Event data dict (default empty).
        """
        if payload is None:
            payload = {}

        event = Event(type=event_type, payload=payload)

        # Keep event log for debugging (bounded)
        self._event_log.append(event)
        if len(self._event_log) > self._max_log_size:
            self._event_log = self._event_log[-self._max_log_size:]

        handlers = self._handlers.get(event_type, [])
        if not handlers:
            logger.debug("No handlers for event '%s'", event_type)
            return

        logger.debug(
            "Emitting event '%s' to %d handler(s)", event_type, len(handlers)
        )

        # Run all handlers concurrently, catching individual failures
        tasks = [self._safe_call(h, payload, event_type) for h in handlers]
        await asyncio.gather(*tasks)

    async def _safe_call(
        self, handler: EventHandler, payload: dict, event_type: str
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

    @property
    def handler_count(self) -> int:
        """Total number of registered handlers across all event types."""
        return sum(len(h) for h in self._handlers.values())

    @property
    def recent_events(self) -> list[dict]:
        """Return recent events for debugging."""
        return [e.to_dict() for e in self._event_log[-20:]]

    def clear_handlers(self) -> None:
        """Remove all handlers. Useful for testing."""
        self._handlers.clear()

    def stats(self) -> dict:
        """Return event bus statistics."""
        return {
            "registered_handlers": self.handler_count,
            "event_types": list(self._handlers.keys()),
            "recent_event_count": len(self._event_log),
        }

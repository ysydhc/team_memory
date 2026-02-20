"""Webhook notification service (P3-4).

Sends HTTP POST notifications to configured URLs when events occur.
Uses HMAC-SHA256 for signature verification and exponential backoff for retries.

Configuration in config.yaml:
    webhooks:
      - url: https://example.com/webhook
        events: [experience.created, experience.updated]
        secret: "your-hmac-secret"
        active: true
"""

from __future__ import annotations

import hashlib
import hmac
import json
import logging
import time
from typing import Any

import httpx

from team_memory.services.event_bus import EventBus

logger = logging.getLogger("team_memory.webhook")


class WebhookConfig:
    """Parsed webhook target configuration."""

    def __init__(self, url: str, events: list[str], secret: str = "", active: bool = True):
        self.url = url
        self.events = set(events)
        self.secret = secret
        self.active = active


class WebhookService:
    """Sends webhook notifications on event bus events.

    Subscribes to all configured event types and sends HTTP POST
    to the target URLs with HMAC signature and retry logic.
    """

    MAX_RETRIES = 3
    TIMEOUT = 10.0

    def __init__(self, event_bus: EventBus, webhook_configs: list[dict]):
        self._event_bus = event_bus
        self._targets: list[WebhookConfig] = []

        for cfg in webhook_configs:
            if cfg.get("active", True):
                target = WebhookConfig(
                    url=cfg["url"],
                    events=cfg.get("events", []),
                    secret=cfg.get("secret", ""),
                    active=True,
                )
                self._targets.append(target)

        # Register handlers for all unique event types
        event_types: set[str] = set()
        for t in self._targets:
            event_types.update(t.events)

        for evt in event_types:
            event_bus.on(evt, self._create_handler(evt))

        if self._targets:
            logger.info(
                "Webhook service: %d target(s), %d event type(s)",
                len(self._targets),
                len(event_types),
            )

    def _create_handler(self, event_type: str):
        """Create an async handler for a specific event type."""
        async def handler(payload: dict[str, Any]) -> None:
            for target in self._targets:
                if event_type in target.events:
                    await self._send(target, event_type, payload)
        handler.__name__ = f"webhook_{event_type}"
        return handler

    @staticmethod
    def _sign(payload_bytes: bytes, secret: str) -> str:
        """Generate HMAC-SHA256 signature."""
        return hmac.new(
            secret.encode(), payload_bytes, hashlib.sha256
        ).hexdigest()

    async def _send(
        self, target: WebhookConfig, event_type: str, payload: dict
    ) -> None:
        """Send webhook with exponential backoff retry."""
        body = json.dumps({
            "event": event_type,
            "payload": payload,
            "timestamp": time.time(),
        })
        body_bytes = body.encode()

        headers = {"Content-Type": "application/json"}
        if target.secret:
            headers["X-Signature-256"] = f"sha256={self._sign(body_bytes, target.secret)}"

        for attempt in range(self.MAX_RETRIES):
            try:
                async with httpx.AsyncClient(timeout=self.TIMEOUT) as client:
                    resp = await client.post(target.url, content=body_bytes, headers=headers)
                    if resp.status_code < 400:
                        logger.debug(
                            "Webhook sent: %s -> %s (status=%d)",
                            event_type,
                            target.url,
                            resp.status_code,
                        )
                        return
                    logger.warning(
                        "Webhook failed: %s -> %s (status=%d)",
                        event_type,
                        target.url,
                        resp.status_code,
                    )
            except Exception as e:
                logger.warning("Webhook error: %s -> %s: %s", event_type, target.url, e)

            # Exponential backoff: 1s, 2s, 4s
            if attempt < self.MAX_RETRIES - 1:
                import asyncio
                await asyncio.sleep(2 ** attempt)

        logger.error("Webhook exhausted retries: %s -> %s", event_type, target.url)

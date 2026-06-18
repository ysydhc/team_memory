"""Debug endpoints for the TM Daemon.

POST /alert/test
GET /debug/payloads
GET /debug/contracts
"""

from __future__ import annotations

import logging
from typing import Any

from fastapi import APIRouter, Request

from team_memory.daemon.runtime.app import send_daemon_alert
from team_memory.daemon.runtime.models import HookPayload

logger = logging.getLogger("daemon.app")

router = APIRouter()


@router.post("/alert/test")
async def alert_test(
    request: Request,
    title: str = "TM Daemon Alert Test",
    body: str = "This is a test alert from TM Daemon.",
    severity: str = "info",
) -> dict[str, Any]:
    """Send a test alert through all configured channels (feishu etc.).

    Useful for verifying feishu webhook integration at deploy time.
    Query params: title, body, severity (info|warning|critical).
    """
    app = request.app
    await send_daemon_alert(title=title, body=body, severity=severity)
    alert_mgr = getattr(app.state, "alert_manager", None)
    n_channels = len(alert_mgr.channels) if alert_mgr else 0
    return {"sent": True, "channels": n_channels, "title": title, "severity": severity}


@router.get("/debug/payloads")
async def debug_payloads(request: Request, endpoint: str = "", limit: int = 20) -> dict[str, Any]:
    """Return captured payloads for debugging."""
    app = request.app
    capture = getattr(app.state, "payload_capture", None)
    if capture is None:
        return {"error": "payload_capture not initialized"}
    if endpoint:
        payloads = await capture.get_payloads(endpoint=endpoint, limit=limit)
        summary = await capture.get_field_summary(endpoint)
        return {"payloads": payloads, "summary": summary}
    payloads = await capture.get_payloads(limit=limit)
    return {"payloads": payloads}


@router.get("/debug/contracts")
async def debug_contracts(request: Request) -> dict[str, Any]:
    """Return derived contract schema for all hook endpoints."""
    app = request.app
    capture = getattr(app.state, "payload_capture", None)
    if capture is None:
        return {"error": "payload_capture not initialized"}
    endpoints = [
        "/hooks/before_prompt",
        "/hooks/after_response",
        "/hooks/session_start",
        "/hooks/session_end",
        "/hooks/mcp_callback",
    ]
    result: dict[str, Any] = {}
    for ep in endpoints:
        result[ep] = await capture.get_field_summary(ep)
    result["_hook_payload_schema"] = {
        "fields": list(HookPayload.model_fields.keys()),
        "extra": "allow",
    }
    return result

"""ASGI middleware for MCP HTTP endpoint authentication.

Extracts Bearer token from the Authorization header, validates it
via the shared AuthProvider, and stores the resolved user name in
``scope["state"]["mcp_user"]`` so that ``server._get_current_user()``
can read it from the FastMCP request context.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from typing import Any

from starlette.requests import Request
from starlette.responses import JSONResponse
from starlette.types import ASGIApp, Receive, Scope, Send

logger = logging.getLogger("team_memory.web.mcp_auth")


def _default_get_auth() -> Any:
    """Lazy import: get the auth provider from the web app module."""
    from team_memory.web import app as app_module

    return app_module._auth


class MCPAuthMiddleware:
    """Authenticate HTTP MCP requests via Bearer token."""

    def __init__(
        self,
        app: ASGIApp,
        get_auth: Callable[[], Any] = _default_get_auth,
    ) -> None:
        self.app = app
        self._get_auth = get_auth

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        request = Request(scope)
        auth_header = request.headers.get("authorization", "")
        api_key = auth_header[7:] if auth_header.startswith("Bearer ") else ""

        if not api_key:
            response = JSONResponse(
                status_code=401,
                content={
                    "error": True,
                    "message": "Authorization header required for MCP access",
                    "code": "auth_required",
                },
            )
            await response(scope, receive, send)
            return

        _auth = self._get_auth()
        if not _auth:
            response = JSONResponse(
                status_code=503,
                content={
                    "error": True,
                    "message": "Auth provider not configured",
                    "code": "auth_not_configured",
                },
            )
            await response(scope, receive, send)
            return

        user = await _auth.authenticate({"api_key": api_key})
        if user is None:
            response = JSONResponse(
                status_code=401,
                content={
                    "error": True,
                    "message": "Invalid API key",
                    "code": "invalid_credentials",
                },
            )
            await response(scope, receive, send)
            return

        # Store on scope state — shared with FastMCP's RequestContextMiddleware
        scope.setdefault("state", {})["mcp_user"] = user.name
        logger.debug("MCP HTTP auth: user=%s", user.name)
        await self.app(scope, receive, send)

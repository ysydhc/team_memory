"""Tests for MCP HTTP auth middleware."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest
from starlette.applications import Starlette
from starlette.responses import JSONResponse
from starlette.routing import Route
from starlette.testclient import TestClient

from team_memory.auth.provider import User
from team_memory.web.mcp_auth_middleware import MCPAuthMiddleware


def _make_mock_auth():
    """Auth provider that recognises 'valid-key' -> alice."""
    auth = MagicMock()

    async def _authenticate(creds: dict) -> User | None:
        if creds.get("api_key") == "valid-key":
            return User(name="alice", role="admin")
        return None

    auth.authenticate = AsyncMock(side_effect=_authenticate)
    return auth


def _echo_app():
    """Tiny ASGI app that echoes mcp_user from request state."""

    async def _ok(request):
        mcp_user = getattr(request.state, "mcp_user", None)
        return JSONResponse({"user": mcp_user})

    return Starlette(routes=[Route("/", _ok, methods=["POST", "GET"])])


def _build_client(auth_value):
    inner = _echo_app()
    mw = MCPAuthMiddleware(inner, get_auth=lambda: auth_value)
    return TestClient(mw, raise_server_exceptions=False)


class TestMCPAuthMiddleware:
    def test_no_header_returns_401(self):
        client = _build_client(_make_mock_auth())
        resp = client.post("/")
        assert resp.status_code == 401
        assert resp.json()["code"] == "auth_required"

    def test_invalid_key_returns_401(self):
        client = _build_client(_make_mock_auth())
        resp = client.post("/", headers={"Authorization": "Bearer wrong-key"})
        assert resp.status_code == 401
        assert resp.json()["code"] == "invalid_credentials"

    def test_valid_key_passes_through(self):
        client = _build_client(_make_mock_auth())
        resp = client.post("/", headers={"Authorization": "Bearer valid-key"})
        assert resp.status_code == 200
        assert resp.json()["user"] == "alice"

    def test_no_auth_provider_returns_503(self):
        client = _build_client(None)
        resp = client.post("/", headers={"Authorization": "Bearer any-key"})
        assert resp.status_code == 503
        assert resp.json()["code"] == "auth_not_configured"

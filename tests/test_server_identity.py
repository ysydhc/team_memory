"""Tests for MCP server identity: _get_current_user resolution chain.

Verifies: HTTP context (remote) → env API Key (stdio) → RuntimeError.
"""

from __future__ import annotations

import os
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from team_memory.auth.provider import User
from team_memory.server import _get_current_user


class TestGetCurrentUserFromHttpContext:
    """When HTTP MCP request has mcp_user in state, return it."""

    @pytest.mark.asyncio
    async def test_http_context_returns_mcp_user(self):
        mock_request = MagicMock()
        mock_request.state.mcp_user = "alice"
        with patch(
            "team_memory.server._current_http_request",
            create=True,
        ) as mock_ctx_var:
            mock_ctx_var.get.return_value = mock_request
            # Also patch the import to use our mock
            with patch.dict("sys.modules", {}):
                from fastmcp.server.http import (
                    _current_http_request as real_ctx_var,
                )

                token = real_ctx_var.set(mock_request)
                try:
                    result = await _get_current_user()
                finally:
                    real_ctx_var.reset(token)
        assert result == "alice"

    @pytest.mark.asyncio
    async def test_http_context_takes_priority_over_env(self):
        """HTTP request user takes priority over TEAM_MEMORY_API_KEY."""
        mock_request = MagicMock()
        mock_request.state.mcp_user = "http-alice"

        mock_ctx = MagicMock()
        mock_ctx.auth.authenticate = AsyncMock(return_value=User(name="env-bob", role="admin"))

        from fastmcp.server.http import _current_http_request

        token = _current_http_request.set(mock_request)
        try:
            with (
                patch("team_memory.bootstrap.get_context", return_value=mock_ctx),
                patch.dict(os.environ, {"TEAM_MEMORY_API_KEY": "some-key"}, clear=False),
            ):
                result = await _get_current_user()
        finally:
            _current_http_request.reset(token)
        assert result == "http-alice"


class TestGetCurrentUserFromApiKey:
    """When TEAM_MEMORY_API_KEY is set and auth returns a user, return user.name."""

    @pytest.mark.asyncio
    async def test_valid_api_key_returns_user_name(self):
        mock_ctx = MagicMock()
        mock_ctx.auth.authenticate = AsyncMock(return_value=User(name="alice", role="editor"))
        with (
            patch("team_memory.bootstrap.get_context", return_value=mock_ctx),
            patch.dict(os.environ, {"TEAM_MEMORY_API_KEY": "test-key"}, clear=False),
        ):
            result = await _get_current_user()
        assert result == "alice"


class TestGetCurrentUserNoIdentity:
    """When no API key and no HTTP context, raise RuntimeError."""

    @pytest.mark.asyncio
    async def test_no_api_key_raises_runtime_error(self):
        env = {k: v for k, v in os.environ.items() if k not in ("TEAM_MEMORY_API_KEY",)}
        env["TEAM_MEMORY_API_KEY"] = ""
        with (
            patch.dict(os.environ, env, clear=False),
            pytest.raises(RuntimeError, match="No authenticated user"),
        ):
            await _get_current_user()

    @pytest.mark.asyncio
    async def test_invalid_api_key_raises_runtime_error(self):
        mock_ctx = MagicMock()
        mock_ctx.auth.authenticate = AsyncMock(return_value=None)
        with (
            patch("team_memory.bootstrap.get_context", return_value=mock_ctx),
            patch.dict(os.environ, {"TEAM_MEMORY_API_KEY": "invalid"}, clear=False),
            pytest.raises(RuntimeError, match="No authenticated user"),
        ):
            await _get_current_user()

    @pytest.mark.asyncio
    async def test_get_context_raises_then_runtime_error(self):
        with (
            patch(
                "team_memory.bootstrap.get_context",
                side_effect=RuntimeError("not bootstrapped"),
            ),
            patch.dict(os.environ, {"TEAM_MEMORY_API_KEY": "key"}, clear=False),
            pytest.raises(RuntimeError, match="No authenticated user"),
        ):
            await _get_current_user()

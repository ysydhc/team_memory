"""Tests for MCP server identity: _get_current_user from TEAM_MEMORY_API_KEY or fallback."""

from __future__ import annotations

import os
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from team_memory.auth.provider import User
from team_memory.server import _get_current_user


class TestGetCurrentUserFromApiKey:
    """When TEAM_MEMORY_API_KEY is set and auth returns a user, return user.name."""

    @pytest.mark.asyncio
    async def test_valid_api_key_returns_user_name(self):
        mock_ctx = MagicMock()
        mock_ctx.auth.authenticate = AsyncMock(
            return_value=User(name="alice", role="editor")
        )
        with (
            patch("team_memory.server.get_context", return_value=mock_ctx),
            patch.dict(os.environ, {"TEAM_MEMORY_API_KEY": "test-key"}, clear=False),
        ):
            result = await _get_current_user()
        assert result == "alice"


class TestGetCurrentUserFallbackNoKey:
    """When TEAM_MEMORY_API_KEY is not set, use TEAM_MEMORY_USER."""

    @pytest.mark.asyncio
    async def test_no_api_key_uses_team_memory_user(self):
        env = os.environ.copy()
        env.pop("TEAM_MEMORY_API_KEY", None)
        env["TEAM_MEMORY_API_KEY"] = ""  # Explicitly clear so patch removes it
        env["TEAM_MEMORY_USER"] = "fallback_user"
        with patch.dict(os.environ, env, clear=False):
            result = await _get_current_user()
        assert result == "fallback_user"

    @pytest.mark.asyncio
    async def test_no_api_key_no_team_memory_user_returns_anonymous(self):
        env = {
            k: v
            for k, v in os.environ.items()
            if k not in ("TEAM_MEMORY_API_KEY", "TEAM_MEMORY_USER")
        }
        with patch.dict(os.environ, env, clear=True):
            result = await _get_current_user()
        assert result == "anonymous"


class TestGetCurrentUserInvalidKey:
    """When API key is invalid (authenticate returns None), fall back."""

    @pytest.mark.asyncio
    async def test_invalid_api_key_returns_fallback(self):
        mock_ctx = MagicMock()
        mock_ctx.auth.authenticate = AsyncMock(return_value=None)
        with (
            patch("team_memory.server.get_context", return_value=mock_ctx),
            patch.dict(
                os.environ,
                {"TEAM_MEMORY_API_KEY": "invalid", "TEAM_MEMORY_USER": "fallback"},
                clear=False,
            ),
        ):
            result = await _get_current_user()
        assert result == "fallback"


class TestGetCurrentUserContextException:
    """When get_context() raises RuntimeError, fall back."""

    @pytest.mark.asyncio
    async def test_get_context_raises_returns_fallback(self):
        with (
            patch("team_memory.server.get_context", side_effect=RuntimeError("not bootstrapped")),
            patch.dict(
                os.environ,
                {"TEAM_MEMORY_API_KEY": "key", "TEAM_MEMORY_USER": "fallback"},
                clear=False,
            ),
        ):
            result = await _get_current_user()
        assert result == "fallback"

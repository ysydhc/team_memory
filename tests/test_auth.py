"""Tests for authentication providers."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from team_memory.auth.init_admin import ensure_default_admin, is_api_keys_empty
from team_memory.auth.provider import ApiKeyAuth, NoAuth, create_auth_provider


class TestNoAuth:
    """Test NoAuth provider."""

    @pytest.mark.asyncio
    async def test_always_returns_user(self):
        auth = NoAuth()
        user = await auth.authenticate({})
        assert user is not None
        assert user.name == "anonymous"
        assert user.role == "admin"

    @pytest.mark.asyncio
    async def test_ignores_credentials(self):
        auth = NoAuth()
        user = await auth.authenticate({"api_key": "anything"})
        assert user is not None


class TestApiKeyAuth:
    """Test ApiKeyAuth provider."""

    @pytest.mark.asyncio
    async def test_valid_key_returns_user(self):
        auth = ApiKeyAuth()
        auth.register_key("tm_test_key_123", "alice", "admin")

        user = await auth.authenticate({"api_key": "tm_test_key_123"})
        assert user is not None
        assert user.name == "alice"
        assert user.role == "admin"

    @pytest.mark.asyncio
    async def test_invalid_key_returns_none(self):
        auth = ApiKeyAuth()
        auth.register_key("tm_test_key_123", "alice", "admin")

        user = await auth.authenticate({"api_key": "wrong_key"})
        assert user is None

    @pytest.mark.asyncio
    async def test_empty_key_returns_none(self):
        auth = ApiKeyAuth()
        user = await auth.authenticate({"api_key": ""})
        assert user is None

    @pytest.mark.asyncio
    async def test_missing_key_returns_none(self):
        auth = ApiKeyAuth()
        user = await auth.authenticate({})
        assert user is None

    def test_hash_key_consistency(self):
        hash1 = ApiKeyAuth.hash_key("test_key")
        hash2 = ApiKeyAuth.hash_key("test_key")
        assert hash1 == hash2

    def test_hash_key_different_inputs(self):
        hash1 = ApiKeyAuth.hash_key("key_a")
        hash2 = ApiKeyAuth.hash_key("key_b")
        assert hash1 != hash2

    @pytest.mark.asyncio
    async def test_multiple_keys(self):
        auth = ApiKeyAuth()
        auth.register_key("key_alice", "alice", "admin")
        auth.register_key("key_bob", "bob", "member")

        user_alice = await auth.authenticate({"api_key": "key_alice"})
        user_bob = await auth.authenticate({"api_key": "key_bob"})

        assert user_alice is not None and user_alice.name == "alice"
        assert user_bob is not None and user_bob.name == "bob"


class TestCreateAuthProvider:
    """Test auth provider factory."""

    def test_create_none_auth(self):
        provider = create_auth_provider("none")
        assert isinstance(provider, NoAuth)

    def test_create_api_key_auth(self):
        provider = create_auth_provider("api_key")
        assert isinstance(provider, ApiKeyAuth)

    def test_create_unknown_raises(self):
        with pytest.raises(ValueError, match="Unknown auth type"):
            create_auth_provider("oauth")


class TestEnsureDefaultAdmin:
    """Test ensure_default_admin and is_api_keys_empty."""

    @pytest.mark.asyncio
    async def test_ensure_default_admin_creates_when_empty(self):
        """When api_keys is empty, creates admin and returns True."""
        mock_result = MagicMock()
        mock_result.scalar.return_value = 0
        mock_session = AsyncMock()
        mock_session.execute = AsyncMock(return_value=mock_result)
        mock_session.add = MagicMock()
        mock_session.flush = AsyncMock()

        def mock_get_session(_db_url):
            cm = AsyncMock()
            cm.__aenter__ = AsyncMock(return_value=mock_session)
            cm.__aexit__ = AsyncMock(return_value=False)
            return cm

        with patch("team_memory.storage.database.get_session", side_effect=mock_get_session):
            created = await ensure_default_admin("sqlite+aiosqlite:///:memory:", "secret123")
        assert created is True
        mock_session.add.assert_called_once()
        added = mock_session.add.call_args[0][0]
        assert added.user_name == "admin"
        assert added.role == "admin"
        assert added.is_active is True
        assert added.password_hash is not None

    @pytest.mark.asyncio
    async def test_ensure_default_admin_skips_when_not_empty(self):
        """When api_keys has rows, does nothing and returns False."""
        mock_result = MagicMock()
        mock_result.scalar.return_value = 1
        mock_session = AsyncMock()
        mock_session.execute = AsyncMock(return_value=mock_result)
        mock_session.add = MagicMock()

        def mock_get_session(_db_url):
            cm = AsyncMock()
            cm.__aenter__ = AsyncMock(return_value=mock_session)
            cm.__aexit__ = AsyncMock(return_value=False)
            return cm

        with patch("team_memory.storage.database.get_session", side_effect=mock_get_session):
            created = await ensure_default_admin("sqlite+aiosqlite:///:memory:", "secret123")
        assert created is False
        mock_session.add.assert_not_called()

    @pytest.mark.asyncio
    async def test_is_api_keys_empty_true(self):
        """Returns True when api_keys has no rows."""
        mock_result = MagicMock()
        mock_result.scalar.return_value = 0
        mock_session = AsyncMock()
        mock_session.execute = AsyncMock(return_value=mock_result)

        def mock_get_session(_db_url):
            cm = AsyncMock()
            cm.__aenter__ = AsyncMock(return_value=mock_session)
            cm.__aexit__ = AsyncMock(return_value=False)
            return cm

        with patch("team_memory.storage.database.get_session", side_effect=mock_get_session):
            empty = await is_api_keys_empty("sqlite+aiosqlite:///:memory:")
        assert empty is True

    @pytest.mark.asyncio
    async def test_is_api_keys_empty_false(self):
        """Returns False when api_keys has rows."""
        mock_result = MagicMock()
        mock_result.scalar.return_value = 3
        mock_session = AsyncMock()
        mock_session.execute = AsyncMock(return_value=mock_result)

        def mock_get_session(_db_url):
            cm = AsyncMock()
            cm.__aenter__ = AsyncMock(return_value=mock_session)
            cm.__aexit__ = AsyncMock(return_value=False)
            return cm

        with patch("team_memory.storage.database.get_session", side_effect=mock_get_session):
            empty = await is_api_keys_empty("sqlite+aiosqlite:///:memory:")
        assert empty is False

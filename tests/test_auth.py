"""Tests for authentication providers."""

import pytest

from team_doc.auth.provider import ApiKeyAuth, NoAuth, User, create_auth_provider


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
        auth.register_key("td_test_key_123", "alice", "admin")

        user = await auth.authenticate({"api_key": "td_test_key_123"})
        assert user is not None
        assert user.name == "alice"
        assert user.role == "admin"

    @pytest.mark.asyncio
    async def test_invalid_key_returns_none(self):
        auth = ApiKeyAuth()
        auth.register_key("td_test_key_123", "alice", "admin")

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

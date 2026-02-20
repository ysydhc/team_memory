"""Authentication providers for team_memory."""

from __future__ import annotations

import hashlib
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from sqlalchemy.ext.asyncio import AsyncSession

logger = logging.getLogger("team_memory.auth")


@dataclass
class User:
    """Authenticated user information."""

    name: str
    role: str  # "admin" or "member"


class AuthProvider(ABC):
    """Abstract authentication provider interface."""

    @abstractmethod
    async def authenticate(self, credentials: dict) -> User | None:
        """Authenticate a user from credentials.

        Args:
            credentials: Dictionary containing authentication data.
                         For API Key auth: {"api_key": "the-key"}

        Returns:
            User object if authenticated, None otherwise.
        """
        ...


class NoAuth(AuthProvider):
    """No authentication — always returns a default user.

    Useful for development/testing.
    """

    async def authenticate(self, credentials: dict) -> User | None:
        return User(name="anonymous", role="admin")


class ApiKeyAuth(AuthProvider):
    """API Key based authentication.

    Keys are stored as SHA256 hashes for security.
    Supports both in-memory keys (from config/env) and DB-persisted keys.
    """

    def __init__(self, keys: dict[str, User] | None = None):
        """Initialize with a mapping of key_hash -> User.

        Args:
            keys: Dict mapping SHA256(api_key) -> User.
                  If None, no keys are registered.
        """
        self._keys: dict[str, User] = keys or {}

    @staticmethod
    def hash_key(api_key: str) -> str:
        """Hash an API key using SHA256."""
        return hashlib.sha256(api_key.encode()).hexdigest()

    def register_key(self, api_key: str, user_name: str, role: str = "member") -> None:
        """Register a new API key in memory.

        Args:
            api_key: The raw API key string.
            user_name: The user name associated with this key.
            role: User role, defaults to "member".
        """
        key_hash = self.hash_key(api_key)
        self._keys[key_hash] = User(name=user_name, role=role)

    async def authenticate(self, credentials: dict) -> User | None:
        """Authenticate using an API key.

        Args:
            credentials: Dict with "api_key" field.

        Returns:
            User if the key is valid, None otherwise.
        """
        api_key = credentials.get("api_key", "")
        if not api_key:
            return None

        key_hash = self.hash_key(api_key)
        return self._keys.get(key_hash)


class DbApiKeyAuth(ApiKeyAuth):
    """API Key auth with database persistence.

    Extends in-memory ApiKeyAuth by also checking the api_keys table.
    In-memory keys (from config/env) are checked first for speed,
    then DB keys as fallback.
    """

    def __init__(
        self,
        db_url: str,
        keys: dict[str, User] | None = None,
    ):
        super().__init__(keys)
        self._db_url = db_url

    async def authenticate(self, credentials: dict) -> User | None:
        """Authenticate: check in-memory keys first, then DB."""
        # Try in-memory first
        user = await super().authenticate(credentials)
        if user is not None:
            return user

        # Fall back to database lookup
        api_key = credentials.get("api_key", "")
        if not api_key:
            return None

        key_hash = self.hash_key(api_key)
        try:
            from sqlalchemy import select

            from team_memory.storage.database import get_session
            from team_memory.storage.models import ApiKey

            async with get_session(self._db_url) as session:
                result = await session.execute(
                    select(ApiKey).where(
                        ApiKey.key_hash == key_hash,
                        ApiKey.is_active == True,  # noqa: E712
                    )
                )
                db_key = result.scalar_one_or_none()
                if db_key is not None:
                    user = User(name=db_key.user_name, role=db_key.role)
                    # Cache in memory for subsequent fast lookups
                    self._keys[key_hash] = user
                    return user
        except Exception:
            logger.warning("DB API key lookup failed, using memory only", exc_info=True)

        return None

    async def register_key_db(
        self,
        session: "AsyncSession",
        api_key: str,
        user_name: str,
        role: str = "member",
    ) -> dict:
        """Register a new API key in the database."""
        from team_memory.storage.models import ApiKey

        key_hash = self.hash_key(api_key)

        db_key = ApiKey(
            key_hash=key_hash,
            user_name=user_name,
            role=role,
        )
        session.add(db_key)
        await session.flush()

        # Also cache in memory
        self._keys[key_hash] = User(name=user_name, role=role)

        return {
            "id": db_key.id,
            "user_name": db_key.user_name,
            "role": db_key.role,
            "is_active": db_key.is_active,
            "created_at": db_key.created_at.isoformat() if db_key.created_at else None,
        }

    async def list_keys_db(self, session: "AsyncSession") -> list[dict]:
        """List all API keys from the database (without exposing hashes)."""
        from sqlalchemy import select

        from team_memory.storage.models import ApiKey

        result = await session.execute(
            select(ApiKey).order_by(ApiKey.created_at.desc())
        )
        keys = result.scalars().all()
        return [
            {
                "id": k.id,
                "user_name": k.user_name,
                "role": k.role,
                "is_active": k.is_active,
                "created_at": k.created_at.isoformat() if k.created_at else None,
            }
            for k in keys
        ]

    async def deactivate_key_db(
        self, session: "AsyncSession", key_id: int
    ) -> bool:
        """Deactivate a DB API key by its ID."""
        from sqlalchemy import update

        from team_memory.storage.models import ApiKey

        result = await session.execute(
            update(ApiKey)
            .where(ApiKey.id == key_id)
            .values(is_active=False)
            .returning(ApiKey.key_hash)
        )
        row = result.first()
        if row:
            # Remove from memory cache
            self._keys.pop(row[0], None)
            return True
        return False


def create_auth_provider(
    auth_type: str, db_url: str | None = None
) -> AuthProvider:
    """Factory function to create an auth provider.

    Args:
        auth_type: "api_key", "db_api_key", or "none".
        db_url: Database URL, required for "db_api_key" type.

    Returns:
        An AuthProvider instance.
    """
    if auth_type == "none":
        return NoAuth()
    elif auth_type == "db_api_key":
        if not db_url:
            raise ValueError("db_url required for db_api_key auth type")
        return DbApiKeyAuth(db_url=db_url)
    elif auth_type == "api_key":
        return ApiKeyAuth()
    else:
        raise ValueError(f"Unknown auth type: {auth_type}")

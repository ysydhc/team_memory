"""Authentication providers for team_memory."""

from __future__ import annotations

import hashlib
import logging
import secrets
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import TYPE_CHECKING

import bcrypt

if TYPE_CHECKING:
    from sqlalchemy.ext.asyncio import AsyncSession

logger = logging.getLogger("team_memory.auth")


@dataclass
class User:
    """Authenticated user information."""

    name: str
    role: str  # "admin", "editor", or "viewer"


class AuthProvider(ABC):
    """Abstract authentication provider interface."""

    @abstractmethod
    async def authenticate(self, credentials: dict) -> User | None:
        """Authenticate a user from credentials.

        Args:
            credentials: Dictionary containing authentication data.
                         For API Key auth: {"api_key": "the-key"}
                         For password auth: {"username": "x", "password": "y"}

        Returns:
            User object if authenticated, None otherwise.
        """
        ...


class NoAuth(AuthProvider):
    """No authentication — always returns a default user."""

    async def authenticate(self, credentials: dict) -> User | None:
        return User(name="anonymous", role="admin")


class ApiKeyAuth(AuthProvider):
    """API Key based authentication (in-memory only)."""

    def __init__(self, keys: dict[str, User] | None = None):
        self._keys: dict[str, User] = keys or {}

    @staticmethod
    def hash_key(api_key: str) -> str:
        return hashlib.sha256(api_key.encode()).hexdigest()

    def register_key(self, api_key: str, user_name: str, role: str = "member") -> None:
        key_hash = self.hash_key(api_key)
        self._keys[key_hash] = User(name=user_name, role=role)

    async def authenticate(self, credentials: dict) -> User | None:
        api_key = credentials.get("api_key", "")
        if not api_key:
            return None
        key_hash = self.hash_key(api_key)
        return self._keys.get(key_hash)


# ---------------------------------------------------------------------------
# Password hashing helpers
# ---------------------------------------------------------------------------

def _hash_password(password: str) -> str:
    return bcrypt.hashpw(password.encode(), bcrypt.gensalt()).decode()


def _verify_password(password: str, hashed: str) -> bool:
    return bcrypt.checkpw(password.encode(), hashed.encode())


class DbApiKeyAuth(ApiKeyAuth):
    """API Key auth with database persistence and password support.

    Supports two authentication paths:
      1. API Key (SHA256 hash) — in-memory first, then DB fallback
      2. Username + password (bcrypt) — DB lookup
    """

    def __init__(self, db_url: str, keys: dict[str, User] | None = None):
        super().__init__(keys)
        self._db_url = db_url

    # ------------------------------------------------------------------
    # authenticate: dual-path
    # ------------------------------------------------------------------
    async def authenticate(self, credentials: dict) -> User | None:
        if credentials.get("api_key"):
            user = await super().authenticate(credentials)
            if user is not None:
                return user
            return await self._db_key_lookup(credentials["api_key"])

        username = credentials.get("username", "")
        password = credentials.get("password", "")
        if username and password:
            return await self._db_password_lookup(username, password)

        return None

    # ------------------------------------------------------------------
    # DB lookup helpers
    # ------------------------------------------------------------------
    async def _db_key_lookup(self, api_key: str) -> User | None:
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
                    self._keys[key_hash] = user
                    return user
        except Exception:
            logger.warning("DB API key lookup failed", exc_info=True)
        return None

    async def _db_password_lookup(self, username: str, password: str) -> User | None:
        """Authenticate via username + password against DB records."""
        try:
            from sqlalchemy import select

            from team_memory.storage.database import get_session
            from team_memory.storage.models import ApiKey

            async with get_session(self._db_url) as session:
                result = await session.execute(
                    select(ApiKey).where(ApiKey.user_name == username)
                )
                db_key = result.scalar_one_or_none()
                if db_key is None:
                    return None
                if not db_key.is_active:
                    return None
                if not db_key.password_hash:
                    return None
                if _verify_password(password, db_key.password_hash):
                    return User(name=db_key.user_name, role=db_key.role)
        except Exception:
            logger.warning("DB password lookup failed", exc_info=True)
        return None

    # ------------------------------------------------------------------
    # Check login status (for detailed error messages)
    # ------------------------------------------------------------------
    async def check_user_status(self, username: str) -> str:
        """Return status string for a username: 'not_found', 'pending', 'active'."""
        try:
            from sqlalchemy import select

            from team_memory.storage.database import get_session
            from team_memory.storage.models import ApiKey

            async with get_session(self._db_url) as session:
                result = await session.execute(
                    select(ApiKey).where(ApiKey.user_name == username)
                )
                db_key = result.scalar_one_or_none()
                if db_key is None:
                    return "not_found"
                if not db_key.is_active:
                    return "pending"
                return "active"
        except Exception:
            logger.warning("check_user_status failed", exc_info=True)
            return "not_found"

    # ------------------------------------------------------------------
    # Registration (user self-register, pending approval)
    # ------------------------------------------------------------------
    async def register_user_db(self, username: str, password: str) -> dict:
        """Create a pending user record (is_active=False, no API key)."""
        from sqlalchemy import select

        from team_memory.storage.database import get_session
        from team_memory.storage.models import ApiKey

        async with get_session(self._db_url) as session:
            existing = await session.execute(
                select(ApiKey).where(ApiKey.user_name == username)
            )
            if existing.scalar_one_or_none() is not None:
                raise ValueError(f"用户名 '{username}' 已存在")

            db_key = ApiKey(
                key_hash=None,
                user_name=username,
                role="editor",
                is_active=False,
                password_hash=_hash_password(password),
            )
            session.add(db_key)
            await session.flush()
            return {
                "id": db_key.id,
                "user_name": db_key.user_name,
                "role": db_key.role,
                "is_active": db_key.is_active,
                "created_at": db_key.created_at.isoformat() if db_key.created_at else None,
            }

    # ------------------------------------------------------------------
    # Approve user (admin action — generates API key)
    # ------------------------------------------------------------------
    async def approve_user_db(self, session: "AsyncSession", key_id: int) -> dict:
        """Approve a pending user: set is_active=True, generate API key."""
        from sqlalchemy import select

        from team_memory.storage.models import ApiKey

        result = await session.execute(select(ApiKey).where(ApiKey.id == key_id))
        db_key = result.scalar_one_or_none()
        if db_key is None:
            raise ValueError("用户不存在")

        raw_key = secrets.token_hex(32)
        db_key.is_active = True
        db_key.key_hash = self.hash_key(raw_key)
        await session.flush()

        self._keys[db_key.key_hash] = User(name=db_key.user_name, role=db_key.role)
        return {
            "id": db_key.id,
            "user_name": db_key.user_name,
            "role": db_key.role,
            "is_active": True,
            "api_key": raw_key,
            "created_at": db_key.created_at.isoformat() if db_key.created_at else None,
        }

    # ------------------------------------------------------------------
    # Admin create user (immediate activation, auto-gen API key)
    # ------------------------------------------------------------------
    async def register_key_db(
        self,
        session: "AsyncSession",
        user_name: str,
        role: str = "editor",
        password: str | None = None,
        api_key: str | None = None,
    ) -> dict:
        """Admin creates a user: active immediately, API key auto-generated."""
        from sqlalchemy import select

        from team_memory.storage.models import ApiKey

        existing = await session.execute(
            select(ApiKey).where(ApiKey.user_name == user_name)
        )
        if existing.scalar_one_or_none() is not None:
            raise ValueError(f"用户名 '{user_name}' 已存在")

        raw_key = api_key or secrets.token_hex(32)
        key_hash = self.hash_key(raw_key)
        pwd_hash = _hash_password(password) if password else None

        db_key = ApiKey(
            key_hash=key_hash,
            user_name=user_name,
            role=role,
            is_active=True,
            password_hash=pwd_hash,
        )
        session.add(db_key)
        await session.flush()

        self._keys[key_hash] = User(name=user_name, role=role)
        return {
            "id": db_key.id,
            "user_name": db_key.user_name,
            "role": db_key.role,
            "is_active": True,
            "api_key": raw_key,
            "created_at": db_key.created_at.isoformat() if db_key.created_at else None,
        }

    # ------------------------------------------------------------------
    # List / deactivate / update password
    # ------------------------------------------------------------------
    async def list_keys_db(self, session: "AsyncSession") -> list[dict]:
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
                "has_api_key": k.key_hash is not None,
                "has_password": k.password_hash is not None,
                "created_at": k.created_at.isoformat() if k.created_at else None,
            }
            for k in keys
        ]

    async def deactivate_key_db(self, session: "AsyncSession", key_id: int) -> bool:
        from sqlalchemy import update

        from team_memory.storage.models import ApiKey

        result = await session.execute(
            update(ApiKey)
            .where(ApiKey.id == key_id)
            .values(is_active=False)
            .returning(ApiKey.key_hash)
        )
        row = result.first()
        if row and row[0]:
            self._keys.pop(row[0], None)
        return row is not None

    async def delete_key_db(self, session: "AsyncSession", key_id: int) -> bool:
        """Hard-delete a record (for rejecting pending registrations)."""
        from sqlalchemy import delete

        from team_memory.storage.models import ApiKey

        result = await session.execute(
            delete(ApiKey).where(ApiKey.id == key_id).returning(ApiKey.id)
        )
        return result.first() is not None

    async def update_password_db(
        self, username: str, old_password: str, new_password: str
    ) -> bool:
        """Update a user's password after verifying the old one."""
        from sqlalchemy import select

        from team_memory.storage.database import get_session
        from team_memory.storage.models import ApiKey

        async with get_session(self._db_url) as session:
            result = await session.execute(
                select(ApiKey).where(ApiKey.user_name == username)
            )
            db_key = result.scalar_one_or_none()
            if db_key is None:
                return False
            if db_key.password_hash and not _verify_password(old_password, db_key.password_hash):
                return False
            db_key.password_hash = _hash_password(new_password)
            await session.flush()
            return True


def create_auth_provider(
    auth_type: str, db_url: str | None = None
) -> AuthProvider:
    """Factory function to create an auth provider."""
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

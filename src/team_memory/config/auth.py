"""Authentication configuration."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel


class AuthConfig(BaseModel):
    """Authentication configuration."""

    type: Literal["api_key", "db_api_key", "none"] = "db_api_key"
    api_key: str | None = None
    user: str = "admin"
    default_admin_password: str | None = None
    default_admin_username: str = "admin"
    session_secret: str | None = None
    key_hash_secret: str | None = None

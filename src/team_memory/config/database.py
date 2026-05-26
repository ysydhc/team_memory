"""Database configuration."""

from __future__ import annotations

import os
from typing import Any

from pydantic import BaseModel, model_validator


class DatabaseConfig(BaseModel):
    """Database connection configuration."""

    url: str = ""

    @model_validator(mode="before")
    @classmethod
    def _url_from_legacy_env(cls, data: Any) -> Any:
        """Hydrate missing URL from TEAM_MEMORY_DB_URL (sync-style DSN)."""
        if not isinstance(data, dict):
            return data
        raw = (data.get("url") or "").strip()
        if raw:
            return data
        legacy = (os.environ.get("TEAM_MEMORY_DB_URL") or "").strip()
        if not legacy:
            return data
        u = legacy
        if u.startswith("postgresql://") and "+asyncpg" not in u:
            u = u.replace("postgresql://", "postgresql+asyncpg://", 1)
        return {**data, "url": u}

    @model_validator(mode="after")
    def _require_nonempty_url(self) -> DatabaseConfig:
        if not (self.url or "").strip():
            msg = (
                "database.url is required: set database.url in YAML, TEAM_MEMORY_DATABASE__URL, or TEAM_MEMORY_DB_URL."
            )
            raise ValueError(msg)
        return self

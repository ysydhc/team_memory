"""Database configuration."""

from __future__ import annotations

from pydantic import BaseModel


class DatabaseConfig(BaseModel):
    """Database connection configuration."""

    url: str = "postgresql+asyncpg://developer:devpass@localhost:5433/team_memory"

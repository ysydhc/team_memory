"""Database connection and session management."""

from __future__ import annotations

import logging
import uuid
from contextlib import asynccontextmanager
from typing import AsyncGenerator
from urllib.parse import urlparse

from sqlalchemy import text
from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)

from team_memory.storage.models import Base
from team_memory.storage.tenant_context import get_active_tenant_id, parse_tenant_uuid

logger = logging.getLogger("team_memory.database")

# Module-level engine and session factory (initialized lazily)
_engine: AsyncEngine | None = None
_session_factory: async_sessionmaker[AsyncSession] | None = None


def _normalize_local_pg_port(database_url: str) -> str:
    """Auto-fallback localhost:5433 <-> localhost:5432 when switching between Docker and Native PG."""
    import socket

    if "localhost:5433" not in database_url and "127.0.0.1:5433" not in database_url:
        return database_url

    def _is_open(port: int) -> bool:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.settimeout(0.3)
            return s.connect_ex(("127.0.0.1", port)) == 0

    if not _is_open(5433) and _is_open(5432):
        fallback = database_url.replace(":5433/", ":5432/")
        logger.info("Port 5433 (Docker PG) closed; auto-routing to native PostgreSQL on :5432")
        return fallback
    return database_url


def get_engine(database_url: str) -> AsyncEngine:
    """Create or return the async SQLAlchemy engine."""
    global _engine
    if _engine is None:
        database_url = _normalize_local_pg_port(database_url)
        kwargs: dict = {"echo": False}
        if "sqlite" not in database_url:
            kwargs["pool_size"] = 5
            kwargs["max_overflow"] = 10
            kwargs["pool_recycle"] = 1800
            kwargs["pool_pre_ping"] = True
        _engine = create_async_engine(database_url, **kwargs)
        parsed = urlparse(database_url)
        masked = f"{parsed.scheme}://{parsed.hostname}:{parsed.port}/{parsed.path.lstrip('/')}"
        logger.info("Database engine created: %s", masked)
    return _engine


def get_session_factory(database_url: str) -> async_sessionmaker[AsyncSession]:
    """Create or return the async session factory."""
    global _session_factory
    if _session_factory is None:
        engine = get_engine(database_url)
        _session_factory = async_sessionmaker(
            engine,
            class_=AsyncSession,
            expire_on_commit=False,
        )
    return _session_factory


@asynccontextmanager
async def get_session(
    database_url: str,
    *,
    tenant_id: uuid.UUID | str | None = None,
    bypass_rls: bool = False,
) -> AsyncGenerator[AsyncSession, None]:
    """Provide a transactional async session scope.

    When running against PostgreSQL with an active tenant context (either via
    ``tenant_id`` argument or ``tenant_context`` ContextVar) and ``bypass_rls``
    is False, switches the transaction to the non-superuser ``tm_app_user`` role
    and sets ``app.current_tenant_id`` so PostgreSQL Row-Level Security (RLS)
    is strictly enforced for the duration of the transaction.

    Usage:
        async with get_session(db_url) as session:
            result = await session.execute(...)
    """
    factory = get_session_factory(database_url)
    effective_tid = None if bypass_rls else (parse_tenant_uuid(tenant_id) or get_active_tenant_id())
    is_postgres = "sqlite" not in database_url

    async with factory() as session:
        try:
            if is_postgres and effective_tid is not None:
                await session.execute(text("SET LOCAL ROLE tm_app_user"))
                await session.execute(
                    text("SELECT set_config('app.current_tenant_id', :tid, true)"),
                    {"tid": str(effective_tid)},
                )
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise


async def init_db(database_url: str) -> None:
    """Create all tables. Used for development/testing only.

    In production, use Alembic migrations instead.
    """
    engine = get_engine(database_url)
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)


async def close_db() -> None:
    """Close the database engine and release all connections."""
    global _engine, _session_factory
    if _engine is not None:
        await _engine.dispose()
        _engine = None
        _session_factory = None
        logger.info("Database engine closed")

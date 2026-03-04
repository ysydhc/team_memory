"""Tests for per-user expansion config (Task 4)."""

from __future__ import annotations

import os

import pytest
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from team_memory.storage.models import Base
from team_memory.storage.repository import UserExpansionRepository

DB_URL = os.environ.get(
    "TEAM_MEMORY_TEST_DB_URL",
    "postgresql+asyncpg://developer:devpass@localhost:5432/team_memory_test",
)
if DB_URL.endswith("/team_memory"):
    pytest.skip(
        "Refusing to run on non-test database. Set TEAM_MEMORY_TEST_DB_URL.",
        allow_module_level=True,
    )

_db_available = None


def _check_db():
    global _db_available
    if _db_available is not None:
        return _db_available
    try:
        import asyncio

        async def _try():
            engine = create_async_engine(DB_URL, pool_pre_ping=True)
            async with engine.connect() as conn:
                await conn.execute(text("SELECT 1"))
            await engine.dispose()

        asyncio.run(_try())
        _db_available = True
    except Exception:
        _db_available = False
    return _db_available


pytestmark = pytest.mark.skipif(
    not _check_db(),
    reason="PostgreSQL not available",
)


@pytest.fixture
async def engine():
    eng = create_async_engine(DB_URL, echo=False)
    async with eng.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    yield eng
    async with eng.begin() as conn:
        await conn.execute(text("DELETE FROM user_expansion_configs"))
    await eng.dispose()


@pytest.fixture
async def session(engine) -> AsyncSession:
    factory = async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)
    async with factory() as sess:
        yield sess
        await sess.rollback()


@pytest.mark.asyncio
async def test_user_expansion_get_empty(session):
    """get_by_user returns {} when user has no config."""
    repo = UserExpansionRepository(session)
    out = await repo.get_by_user("user-unknown")
    assert out == {}


@pytest.mark.asyncio
async def test_user_expansion_upsert_and_get(session):
    """upsert creates/updates; get_by_user returns stored tag_synonyms."""
    repo = UserExpansionRepository(session)
    await repo.upsert("user-1", {"PG": "PostgreSQL", "JS": "JavaScript"})
    await session.commit()
    out = await repo.get_by_user("user-1")
    assert out == {"PG": "PostgreSQL", "JS": "JavaScript"}

    await repo.upsert("user-1", {"PG": "PostgreSQL", "TS": "TypeScript"})
    await session.commit()
    out = await repo.get_by_user("user-1")
    assert out == {"PG": "PostgreSQL", "TS": "TypeScript"}

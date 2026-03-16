"""Tests for ArchiveRepository.

Requires PostgreSQL. Run with:
  pytest tests/test_archive_repository.py -v
"""

from __future__ import annotations

import os
import uuid

import pytest
from sqlalchemy import select, text
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from team_memory.storage.archive_repository import ArchiveRepository
from team_memory.storage.models import (
    Archive,
    ArchiveAttachment,
    ArchiveExperienceLink,
    Base,
    Experience,
)

DB_URL = os.environ.get(
    "TEAM_MEMORY_TEST_DB_URL",
    "postgresql+asyncpg://developer:devpass@localhost:5432/team_memory_test",
)
if DB_URL.endswith("/team_memory"):
    raise RuntimeError(
        "Refusing to run on non-test database 'team_memory'. "
        "Set TEAM_MEMORY_TEST_DB_URL to a test DB."
    )

_db_available = None


def _check_db():
    global _db_available
    if _db_available is not None:
        return _db_available
    try:
        import asyncio

        async def _try_connect():
            engine = create_async_engine(DB_URL, pool_pre_ping=True)
            async with engine.connect() as conn:
                await conn.execute(text("SELECT 1"))
            await engine.dispose()

        asyncio.run(_try_connect())
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
        await conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
        await conn.run_sync(Base.metadata.create_all)
    yield eng
    async with eng.begin() as conn:
        await conn.execute(text("DELETE FROM archive_attachments"))
        await conn.execute(text("DELETE FROM archive_experience_links"))
        await conn.execute(text("DELETE FROM archives"))
    await eng.dispose()


@pytest.fixture
async def session(engine) -> AsyncSession:
    factory = async_sessionmaker(
        engine, class_=AsyncSession, expire_on_commit=False
    )
    async with factory() as sess:
        yield sess
        await sess.rollback()


@pytest.mark.asyncio
async def test_archive_repository_create_returns_id(session: AsyncSession):
    """Create archive returns UUID and persists row."""
    repo = ArchiveRepository(session)
    archive_id = await repo.create_archive(
        title="Test Archive",
        solution_doc="Doc body",
        created_by="user1",
        overview=None,
    )
    assert archive_id is not None
    assert isinstance(archive_id, uuid.UUID)

    result = await session.execute(select(Archive).where(Archive.id == archive_id))
    row = result.scalar_one_or_none()
    assert row is not None
    assert row.title == "Test Archive"
    assert row.solution_doc == "Doc body"
    assert row.created_by == "user1"
    assert row.status == "draft"


@pytest.mark.asyncio
async def test_create_archive_and_links(session: AsyncSession):
    """Create archive with 2 experience links; links count is 2."""
    e1 = Experience(
        title="E1",
        description="D1",
        solution="S1",
        created_by="u",
        exp_status="draft",
    )
    e2 = Experience(
        title="E2",
        description="D2",
        solution="S2",
        created_by="u",
        exp_status="draft",
    )
    session.add(e1)
    session.add(e2)
    await session.flush()

    repo = ArchiveRepository(session)
    archive_id = await repo.create_archive(
        title="With Links",
        solution_doc="Body",
        created_by="u",
        overview=None,
        linked_experience_ids=[e1.id, e2.id],
    )
    await session.commit()

    result = await session.execute(
        select(ArchiveExperienceLink).where(
            ArchiveExperienceLink.archive_id == archive_id
        )
    )
    links = result.scalars().all()
    assert len(links) == 2


@pytest.mark.asyncio
async def test_create_archive_with_attachments(session: AsyncSession):
    """Create archive with attachments; attachment count correct."""
    repo = ArchiveRepository(session)
    archive_id = await repo.create_archive(
        title="With Attachments",
        solution_doc="Body",
        created_by="u",
        overview=None,
        attachments=[
            {"kind": "code_snippet", "snippet": "print(1)"},
            {"kind": "file_ref", "path": "/a/b.py"},
        ],
    )
    await session.commit()

    result = await session.execute(
        select(ArchiveAttachment).where(ArchiveAttachment.archive_id == archive_id)
    )
    atts = result.scalars().all()
    assert len(atts) == 2

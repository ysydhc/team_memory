"""Tests for ExperienceRepository file location binding methods.

Requires PostgreSQL (same as test_integration). Run with:
  pytest tests/test_repository_file_locations.py -v
"""

from __future__ import annotations

import os
from datetime import datetime, timedelta, timezone

import pytest
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from team_memory.storage.models import Base
from team_memory.storage.repository import ExperienceRepository

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


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


@pytest.fixture
async def engine():
    eng = create_async_engine(DB_URL, echo=False)
    async with eng.begin() as conn:
        await conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
        await conn.run_sync(Base.metadata.create_all)
    yield eng
    async with eng.begin() as conn:
        await conn.execute(text("DELETE FROM experience_file_locations"))
        await conn.execute(text("DELETE FROM experience_feedbacks"))
        await conn.execute(text("DELETE FROM experiences"))
    await eng.dispose()


@pytest.fixture
async def session(engine) -> AsyncSession:
    factory = async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)
    async with factory() as sess:
        yield sess
        await sess.rollback()


# --- replace_file_location_bindings ---


@pytest.mark.asyncio
async def test_replace_file_location_bindings_inserts_and_replaces(session):
    """replace_file_location_bindings: delete all for experience then insert new list."""
    repo = ExperienceRepository(session)
    exp = await repo.create(
        title="Loc",
        description="D",
        solution="S",
        created_by="u",
    )
    await session.flush()
    now = _utcnow()
    expires = now + timedelta(days=30)
    bindings = [
        {
            "path": "src/foo.py",
            "start_line": 10,
            "end_line": 20,
            "content_fingerprint": "fp1",
            "expires_at": expires,
        },
        {
            "path": "src/bar.py",
            "start_line": 1,
            "end_line": 5,
            "expires_at": expires,
        },
    ]
    await repo.replace_file_location_bindings(exp.id, bindings)
    await session.commit()

    got = await repo.get_file_location_bindings(exp.id)
    assert len(got) == 2
    paths = {b["path"] for b in got}
    assert paths == {"src/foo.py", "src/bar.py"}
    one = next(b for b in got if b["path"] == "src/foo.py")
    assert one["start_line"] == 10 and one["end_line"] == 20
    assert one.get("content_fingerprint") == "fp1"

    # Replace with single binding: previous two removed
    await repo.replace_file_location_bindings(
        exp.id,
        [{"path": "src/baz.py", "start_line": 1, "end_line": 1, "expires_at": expires}],
    )
    await session.commit()
    got2 = await repo.get_file_location_bindings(exp.id)
    assert len(got2) == 1
    assert got2[0]["path"] == "src/baz.py"


@pytest.mark.asyncio
async def test_replace_file_location_bindings_accepts_optional_fields(session):
    """replace accepts snippet, file_mtime_at_bind, file_content_hash_at_bind, last_accessed_at."""
    repo = ExperienceRepository(session)
    exp = await repo.create(
        title="E",
        description="D",
        solution="S",
        created_by="u",
    )
    await session.flush()
    expires = _utcnow() + timedelta(days=30)
    bindings = [
        {
            "path": "p.py",
            "start_line": 1,
            "end_line": 2,
            "snippet": "def x(): pass",
            "file_mtime_at_bind": 12345.0,
            "file_content_hash_at_bind": "abc123",
            "expires_at": expires,
            "last_accessed_at": expires,
        },
    ]
    await repo.replace_file_location_bindings(exp.id, bindings)
    await session.commit()
    got = await repo.get_file_location_bindings(exp.id)
    assert len(got) == 1
    assert got[0]["snippet"] == "def x(): pass"
    assert got[0].get("file_mtime_at_bind") == 12345.0
    assert got[0].get("file_content_hash_at_bind") == "abc123"


# --- get_file_location_bindings ---


@pytest.mark.asyncio
async def test_get_file_location_bindings_empty(session):
    """get_file_location_bindings returns [] when no bindings."""
    repo = ExperienceRepository(session)
    exp = await repo.create(title="E", description="D", solution="S", created_by="u")
    await session.flush()
    got = await repo.get_file_location_bindings(exp.id)
    assert got == []


# --- list_bindings_by_paths ---


@pytest.mark.asyncio
async def test_list_bindings_by_paths_returns_unexpired_only(session):
    """list_bindings_by_paths: filter by path in list and expires_at > now."""
    repo = ExperienceRepository(session)
    exp = await repo.create(title="E", description="D", solution="S", created_by="u")
    await session.flush()
    now = _utcnow()
    future = now + timedelta(days=30)
    past = now - timedelta(days=1)
    await repo.replace_file_location_bindings(
        exp.id,
        [
            {"path": "a.py", "start_line": 1, "end_line": 2, "expires_at": future},
            {"path": "b.py", "start_line": 3, "end_line": 4, "expires_at": future},
            {"path": "c.py", "start_line": 5, "end_line": 6, "expires_at": past},
        ],
    )
    await session.commit()

    result = await repo.list_bindings_by_paths(["a.py", "c.py"], ttl_days=30)
    assert "a.py" in result
    assert len(result["a.py"]) == 1
    assert result["a.py"][0]["start_line"] == 1
    # c.py is expired so may be excluded (expires_at > now)
    assert "c.py" not in result or len(result["c.py"]) == 0


@pytest.mark.asyncio
async def test_list_bindings_by_paths_key_per_requested_path(session):
    """list_bindings_by_paths returns dict path -> list[binding] for requested paths."""
    repo = ExperienceRepository(session)
    exp = await repo.create(title="E", description="D", solution="S", created_by="u")
    await session.flush()
    future = _utcnow() + timedelta(days=30)
    await repo.replace_file_location_bindings(
        exp.id,
        [
            {"path": "x.py", "start_line": 1, "end_line": 2, "expires_at": future},
            {"path": "x.py", "start_line": 10, "end_line": 11, "expires_at": future},
        ],
    )
    await session.commit()
    result = await repo.list_bindings_by_paths(["x.py"], ttl_days=30)
    assert "x.py" in result
    assert len(result["x.py"]) == 2
    result_empty = await repo.list_bindings_by_paths(["nonexistent.py"], ttl_days=30)
    assert "nonexistent.py" in result_empty
    assert result_empty["nonexistent.py"] == []


# --- find_experience_ids_by_location ---


@pytest.mark.asyncio
async def test_find_experience_ids_by_location_exact_overlap(session):
    """find_experience_ids_by_location returns (exp_id, 1.0) when range overlaps."""
    repo = ExperienceRepository(session)
    exp = await repo.create(title="E", description="D", solution="S", created_by="u")
    await session.flush()
    future = _utcnow() + timedelta(days=30)
    await repo.replace_file_location_bindings(
        exp.id,
        [{"path": "f.py", "start_line": 10, "end_line": 25, "expires_at": future}],
    )
    await session.commit()

    pairs = await repo.find_experience_ids_by_location(
        path="f.py", start_line=12, end_line=18
    )
    assert len(pairs) == 1
    assert pairs[0][0] == exp.id
    assert pairs[0][1] == 1.0


@pytest.mark.asyncio
async def test_find_experience_ids_by_location_same_file_no_overlap(session):
    """Same path but no line overlap -> LOCATION_SCORE_SAME_FILE 0.7."""
    repo = ExperienceRepository(session)
    exp = await repo.create(title="E", description="D", solution="S", created_by="u")
    await session.flush()
    future = _utcnow() + timedelta(days=30)
    await repo.replace_file_location_bindings(
        exp.id,
        [{"path": "f.py", "start_line": 10, "end_line": 20, "expires_at": future}],
    )
    await session.commit()

    pairs = await repo.find_experience_ids_by_location(
        path="f.py", start_line=100, end_line=110
    )
    assert len(pairs) == 1
    assert pairs[0][0] == exp.id
    assert pairs[0][1] == 0.7


@pytest.mark.asyncio
async def test_find_experience_ids_by_location_no_match(session):
    """Different path -> no result."""
    repo = ExperienceRepository(session)
    exp = await repo.create(title="E", description="D", solution="S", created_by="u")
    await session.flush()
    future = _utcnow() + timedelta(days=30)
    await repo.replace_file_location_bindings(
        exp.id,
        [{"path": "f.py", "start_line": 10, "end_line": 20, "expires_at": future}],
    )
    await session.commit()
    pairs = await repo.find_experience_ids_by_location(
        path="other.py", start_line=10, end_line=20
    )
    assert pairs == []


# --- delete_expired_file_location_bindings ---


@pytest.mark.asyncio
async def test_delete_expired_file_location_bindings(session):
    """delete_expired_file_location_bindings removes expires_at < now, returns count."""
    repo = ExperienceRepository(session)
    exp = await repo.create(title="E", description="D", solution="S", created_by="u")
    await session.flush()
    past = _utcnow() - timedelta(days=1)
    future = _utcnow() + timedelta(days=30)
    await repo.replace_file_location_bindings(
        exp.id,
        [
            {"path": "a.py", "start_line": 1, "end_line": 2, "expires_at": past},
            {"path": "b.py", "start_line": 3, "end_line": 4, "expires_at": future},
        ],
    )
    await session.commit()

    n = await repo.delete_expired_file_location_bindings(batch_size=500)
    assert n >= 1
    await session.commit()
    got = await repo.get_file_location_bindings(exp.id)
    assert len(got) == 1
    assert got[0]["path"] == "b.py"

    n2 = await repo.delete_expired_file_location_bindings(batch_size=500)
    assert n2 == 0


@pytest.mark.asyncio
async def test_delete_expired_respects_batch_size(session):
    """delete_expired_file_location_bindings deletes at most batch_size."""
    repo = ExperienceRepository(session)
    past = _utcnow() - timedelta(days=1)
    for i in range(5):
        exp = await repo.create(
            title=f"E{i}", description="D", solution="S", created_by="u"
        )
        await session.flush()
        await repo.replace_file_location_bindings(
            exp.id,
            [{"path": f"p{i}.py", "start_line": 1, "end_line": 2, "expires_at": past}],
        )
    await session.commit()

    n = await repo.delete_expired_file_location_bindings(batch_size=2)
    assert n == 2
    await session.commit()
    n2 = await repo.delete_expired_file_location_bindings(batch_size=2)
    assert n2 == 2
    n3 = await repo.delete_expired_file_location_bindings(batch_size=2)
    assert n3 == 1
    n4 = await repo.delete_expired_file_location_bindings(batch_size=500)
    assert n4 == 0

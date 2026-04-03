"""Tests for personal memory storage and semantic overwrite (Task 1)."""

from __future__ import annotations

import os

import pytest
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from team_memory.services.personal_memory import PersonalMemoryService
from team_memory.storage.models import Base
from team_memory.storage.personal_memory_repository import (
    PERSONAL_MEMORY_OVERWRITE_THRESHOLD,
    PersonalMemoryRepository,
)
from tests.conftest import MockEmbeddingProvider

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

TEST_DIM = 768


@pytest.fixture
async def engine():
    eng = create_async_engine(DB_URL, echo=False)
    async with eng.begin() as conn:
        await conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
        await conn.run_sync(Base.metadata.create_all)
    yield eng
    async with eng.begin() as conn:
        await conn.execute(text("DELETE FROM personal_memories"))
    await eng.dispose()


@pytest.fixture
async def session(engine) -> AsyncSession:
    factory = async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)
    async with factory() as sess:
        yield sess
        await sess.rollback()


@pytest.fixture
def mock_embedding():
    return MockEmbeddingProvider(dimension=TEST_DIM)


# ---- Repository-level: insert, conflict overwrite, no-conflict new ----
@pytest.mark.asyncio
async def test_personal_memory_insert(session, mock_embedding):
    """Insert: new row when no similar exists."""
    repo = PersonalMemoryRepository(session)
    vec = await mock_embedding.encode_single("I like concise UI")
    mem = await repo.create(
        user_id="alice",
        content="I like concise UI",
        scope="generic",
        context_hint=None,
        embedding=vec,
    )
    assert mem.id is not None
    assert mem.user_id == "alice"
    assert mem.content == "I like concise UI"
    assert mem.scope == "generic"
    assert mem.profile_kind == "static"

    listed = await repo.list_by_user("alice")
    assert len(listed) == 1
    assert listed[0].content == "I like concise UI"


@pytest.mark.asyncio
async def test_personal_memory_conflict_overwrite(session, mock_embedding):
    """Conflict overwrite: when new content is semantically same, update existing row."""
    repo = PersonalMemoryRepository(session)
    # Same text => same embedding => find_most_similar will find it
    content = "User prefers short answers"
    vec = await mock_embedding.encode_single(content)
    mem1 = await repo.create(
        user_id="bob",
        content=content,
        scope="generic",
        embedding=vec,
    )
    # Second "write" with same content (same embedding) -> upsert should update
    mem2 = await repo.upsert_by_semantic(
        user_id="bob",
        content="User prefers short answers (updated)",
        embedding=vec,  # same embedding => will match mem1 (same profile_kind)
        scope="generic",
        context_hint="when asking for help",
        threshold=PERSONAL_MEMORY_OVERWRITE_THRESHOLD,
        profile_kind="static",
    )
    assert mem2.id == mem1.id
    assert mem2.content == "User prefers short answers (updated)"
    assert mem2.scope == "generic"
    assert mem2.context_hint == "when asking for help"

    listed = await repo.list_by_user("bob")
    assert len(listed) == 1


@pytest.mark.asyncio
async def test_personal_memory_no_conflict_new(session, mock_embedding):
    """No conflict: different content => different embedding => new row."""
    repo = PersonalMemoryRepository(session)
    vec1 = await mock_embedding.encode_single("Prefer dark theme")
    vec2 = await mock_embedding.encode_single("Always add unit tests")
    await repo.create(
        user_id="carol",
        content="Prefer dark theme",
        scope="generic",
        embedding=vec1,
    )
    mem2 = await repo.upsert_by_semantic(
        user_id="carol",
        content="Always add unit tests",
        embedding=vec2,
        scope="generic",
        threshold=PERSONAL_MEMORY_OVERWRITE_THRESHOLD,
    )
    assert mem2.content == "Always add unit tests"
    listed = await repo.list_by_user("carol")
    assert len(listed) == 2


@pytest.mark.asyncio
async def test_personal_memory_upsert_cross_kind_no_merge(session, mock_embedding):
    """High similarity across profile_kind must not overwrite — two rows."""
    repo = PersonalMemoryRepository(session)
    vec = await mock_embedding.encode_single("same embedding text")
    await repo.create(
        user_id="dana",
        content="static fact",
        scope="generic",
        embedding=vec,
        profile_kind="static",
    )
    mem2 = await repo.upsert_by_semantic(
        user_id="dana",
        content="dynamic fact",
        embedding=vec,
        scope="context",
        threshold=PERSONAL_MEMORY_OVERWRITE_THRESHOLD,
        profile_kind="dynamic",
    )
    assert mem2.content == "dynamic fact"
    listed = await repo.list_by_user("dana")
    assert len(listed) == 2


# ---- Task 2: Pull API — generic only; generic+context (DB required) ----
@pytest.mark.asyncio
async def test_pull_generic_only_and_generic_plus_context(engine, session, mock_embedding):
    """No current_context => only generic; with context => generic + matching context."""
    from team_memory.services.personal_memory import PersonalMemoryService

    repo = PersonalMemoryRepository(session)
    # Add one generic and one context memory
    v1 = await mock_embedding.encode_single("I like concise UI")
    v2 = await mock_embedding.encode_single("When editing web frontend keep UI minimal")
    await repo.create("pull_user", "I like concise UI", scope="generic", embedding=v1)
    await repo.create(
        "pull_user",
        "When editing web frontend keep UI minimal",
        scope="context",
        context_hint="web UI edits",
        embedding=v2,
    )
    await session.commit()

    db_url = engine.url.render_as_string(hide_password=False)
    svc = PersonalMemoryService(embedding_provider=mock_embedding, db_url=db_url)

    # No current_context: only generic
    out = await svc.pull(user_id="pull_user", current_context=None)
    assert len(out) == 1
    assert out[0]["scope"] == "generic"
    assert out[0].get("profile_kind") == "static"

    # With current_context similar to context item: generic + context
    out2 = await svc.pull(
        user_id="pull_user",
        current_context="editing web frontend",
        context_similarity_threshold=0.3,
    )
    assert len(out2) >= 1
    assert any(m["scope"] == "generic" for m in out2)
    assert any(m.get("profile_kind") == "dynamic" for m in out2)


# ---- Service-level: write (upsert) and list ----
@pytest.mark.asyncio
async def test_personal_memory_service_write_and_list(engine, mock_embedding):
    """Service write() computes embedding and upserts; list_by_user returns items."""
    db_url = engine.url.render_as_string(hide_password=False)
    svc = PersonalMemoryService(embedding_provider=mock_embedding, db_url=db_url)

    out = await svc.write(
        user_id="svc_user",
        content="I like plan with 收口",
        scope="generic",
    )
    assert "id" in out
    assert out["content"] == "I like plan with 收口"
    assert out["scope"] == "generic"

    listed = await svc.list_by_user("svc_user")
    assert len(listed) >= 1
    assert any(m["content"] == "I like plan with 收口" for m in listed)

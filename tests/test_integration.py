"""Integration tests requiring a real PostgreSQL database with pgvector.

These tests verify the full flow: embedding -> storage -> search.
Skip these when PostgreSQL is not available.

Run with: pytest tests/test_integration.py -v
Requires: PostgreSQL with pgvector running (docker compose up -d)
"""

from __future__ import annotations

import os

import pytest
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from team_memory.auth.provider import NoAuth
from team_memory.services.experience import ExperienceService
from team_memory.storage.models import Base
from team_memory.storage.repository import ExperienceRepository
from tests.conftest import MockEmbeddingProvider

# Skip all tests if DB is not available
DB_URL = os.environ.get(
    "TEAM_MEMORY_TEST_DB_URL",
    "postgresql+asyncpg://developer:devpass@localhost:5432/team_memory_test",
)
if DB_URL.endswith("/team_memory"):
    raise RuntimeError(
        "Refusing to run integration tests on non-test database 'team_memory'. "
        "Set TEAM_MEMORY_TEST_DB_URL to a dedicated test DB."
    )

_db_available = None


def _check_db():
    """Check if the database is reachable."""
    global _db_available
    if _db_available is not None:
        return _db_available
    try:
        import asyncio

        from sqlalchemy.ext.asyncio import create_async_engine

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


# Use a module-level skip marker
pytestmark = pytest.mark.skipif(
    not _check_db(),
    reason="PostgreSQL not available",
)

# Dimension must match the database column Vector(768) — Ollama nomic-embed-text
TEST_DIM = 768


@pytest.fixture
async def engine():
    """Create a test engine and set up tables."""
    eng = create_async_engine(DB_URL, echo=False)

    async with eng.begin() as conn:
        # Enable pgvector extension
        await conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
        await conn.run_sync(Base.metadata.create_all)

    yield eng

    # Cleanup: drop test data (but keep tables for next run)
    async with eng.begin() as conn:
        await conn.execute(text("DELETE FROM experience_feedbacks"))
        await conn.execute(text("DELETE FROM experiences"))

    await eng.dispose()


@pytest.fixture
async def session(engine) -> AsyncSession:
    """Create a test session."""
    factory = async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)
    async with factory() as sess:
        yield sess
        await sess.rollback()


@pytest.fixture
def mock_embed():
    return MockEmbeddingProvider(dimension=TEST_DIM)


class TestRepositoryIntegration:
    """Test Repository with real PostgreSQL + pgvector."""

    @pytest.mark.asyncio
    async def test_create_and_get(self, session, mock_embed):
        """Create an experience and retrieve it by ID."""
        repo = ExperienceRepository(session)
        embedding = await mock_embed.encode_single("test content")

        exp = await repo.create(
            title="Test Experience",
            description="A test problem",
            solution="A test solution",
            created_by="tester",
            tags=["test", "python"],
            embedding=embedding,
        )
        await session.commit()

        retrieved = await repo.get_by_id(exp.id)
        assert retrieved is not None
        assert retrieved.title == "Test Experience"
        assert retrieved.tags == ["test", "python"]

    @pytest.mark.asyncio
    async def test_vector_search(self, session, mock_embed):
        """Test vector similarity search."""
        repo = ExperienceRepository(session)

        # Create two experiences with different content
        emb1 = await mock_embed.encode_single("Docker container networking issue")
        await repo.create(
            title="Docker Networking Fix",
            description="Container can't reach host network",
            solution="Use --network host flag",
            created_by="alice",
            tags=["docker"],
            embedding=emb1,
        )

        emb2 = await mock_embed.encode_single("Python async programming tutorial")
        await repo.create(
            title="Python Async Guide",
            description="How to use async/await in Python",
            solution="Use asyncio.run() as entry point",
            created_by="bob",
            tags=["python"],
            embedding=emb2,
        )
        await session.commit()

        # Search for docker-related content
        query_emb = await mock_embed.encode_single("Docker network problem")
        results = await repo.search_by_vector(
            query_embedding=query_emb,
            max_results=5,
            min_similarity=0.0,  # Low threshold for test
        )

        assert len(results) >= 1
        # Results should be sorted by similarity
        if len(results) > 1:
            assert results[0]["similarity"] >= results[1]["similarity"]

    @pytest.mark.asyncio
    async def test_tag_filter_search(self, session, mock_embed):
        """Test search with tag filtering."""
        repo = ExperienceRepository(session)

        emb1 = await mock_embed.encode_single("Python web framework")
        await repo.create(
            title="FastAPI Setup",
            description="Setting up FastAPI",
            solution="pip install fastapi",
            created_by="alice",
            tags=["python", "fastapi"],
            embedding=emb1,
        )

        emb2 = await mock_embed.encode_single("JavaScript web framework")
        await repo.create(
            title="React Setup",
            description="Setting up React",
            solution="npx create-react-app",
            created_by="bob",
            tags=["javascript", "react"],
            embedding=emb2,
        )
        await session.commit()

        query_emb = await mock_embed.encode_single("web framework setup")
        results = await repo.search_by_vector(
            query_embedding=query_emb,
            max_results=5,
            min_similarity=0.0,
            tags=["python"],
        )

        # Only Python-tagged results
        for r in results:
            assert "python" in r.get("tags", [])

    @pytest.mark.asyncio
    async def test_feedback(self, session, mock_embed):
        """Test adding feedback and rating update."""
        repo = ExperienceRepository(session)
        embedding = await mock_embed.encode_single("test")

        exp = await repo.create(
            title="Feedback Test",
            description="Test",
            solution="Test",
            created_by="tester",
            embedding=embedding,
        )
        await session.commit()

        await repo.add_feedback(exp.id, rating=5, feedback_by="user1")
        await repo.add_feedback(exp.id, rating=1, feedback_by="user2")
        await session.commit()

        updated = await repo.get_by_id(exp.id)
        assert updated is not None
        # avg_rating should be 3.0 (average of 5 and 1)
        assert abs(updated.avg_rating - 3.0) < 0.01

    @pytest.mark.asyncio
    async def test_list_recent(self, session, mock_embed):
        """Test listing recent experiences."""
        repo = ExperienceRepository(session)
        embedding = await mock_embed.encode_single("test")

        for i in range(3):
            await repo.create(
                title=f"Experience {i}",
                description=f"Problem {i}",
                solution=f"Solution {i}",
                created_by="tester",
                embedding=embedding,
            )
        await session.commit()

        recent = await repo.list_recent(limit=2)
        assert len(recent) == 2

    @pytest.mark.asyncio
    async def test_count(self, session, mock_embed):
        """Test counting experiences."""
        repo = ExperienceRepository(session)
        embedding = await mock_embed.encode_single("test")

        initial_count = await repo.count()

        await repo.create(
            title="Count Test",
            description="Test",
            solution="Test",
            created_by="tester",
            embedding=embedding,
        )
        await session.commit()

        new_count = await repo.count()
        assert new_count == initial_count + 1

    @pytest.mark.asyncio
    async def test_delete(self, session, mock_embed):
        """Test deleting an experience."""
        repo = ExperienceRepository(session)
        embedding = await mock_embed.encode_single("test")

        exp = await repo.create(
            title="Delete Test",
            description="Test",
            solution="Test",
            created_by="tester",
            embedding=embedding,
        )
        await session.commit()

        deleted = await repo.delete(exp.id)
        assert deleted is True
        await session.commit()

        retrieved = await repo.get_by_id(exp.id)
        assert retrieved is None


class TestServiceIntegration:
    """Test ExperienceService with real database."""

    @pytest.fixture
    def service(self, mock_embed):
        return ExperienceService(
            embedding_provider=mock_embed,
            auth_provider=NoAuth(),
            db_url=DB_URL,
        )

    @pytest.mark.asyncio
    async def test_save_and_search_flow(self, service, session):
        """End-to-end: save an experience, then search for it."""
        # Save
        result = await service.save(
            session=session,
            title="Fix PostgreSQL Connection Timeout",
            problem="PostgreSQL connection times out after 30 seconds of idle",
            solution="Set idle_in_transaction_session_timeout in postgresql.conf",
            created_by="alice",
            tags=["postgresql", "database", "timeout"],
            language="sql",
        )
        await session.commit()
        assert "id" in result

        # Search
        results = await service.search(
            query="PostgreSQL connection timeout issue",
            min_similarity=0.0,
        )
        assert len(results) >= 1
        # The saved experience should be in results
        found = any(r["title"] == "Fix PostgreSQL Connection Timeout" for r in results)
        assert found, f"Expected to find saved experience in results: {results}"

    @pytest.mark.asyncio
    async def test_save_and_feedback_flow(self, service, session):
        """Save experience, then add feedback."""
        result = await service.save(
            session=session,
            title="Feedback Flow Test",
            problem="Test problem",
            solution="Test solution",
            created_by="tester",
        )
        await session.commit()

        success = await service.feedback(
            session=session,
            experience_id=result["id"],
            rating=5,
            feedback_by="reviewer",
            comment="Very helpful!",
        )
        assert success is True

    @pytest.mark.asyncio
    async def test_save_and_update_flow(self, service, session):
        """Save experience, then update it."""
        result = await service.save(
            session=session,
            title="Update Flow Test",
            problem="Original problem",
            solution="Original solution",
            created_by="tester",
            tags=["test"],
        )
        await session.commit()

        updated = await service.update(
            session=session,
            experience_id=result["id"],
            solution_addendum="Additional approach: try restarting the service",
            tags=["operations"],
        )
        assert updated is not None
        assert "Additional approach" in updated["solution"]
        # In-place update replaces tags (not merge)
        assert "operations" in updated["tags"]

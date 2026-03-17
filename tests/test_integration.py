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
from team_memory.config import (
    CacheConfig,
    FileLocationBindingConfig,
    PageIndexLiteConfig,
    RetrievalConfig,
    SearchConfig,
)
from team_memory.reranker.noop_provider import NoopRerankerProvider
from team_memory.services.experience import ExperienceService
from team_memory.services.search_pipeline import SearchPipeline
from team_memory.storage.models import Base
from team_memory.storage.repository import ExperienceRepository
from team_memory.utils.location_fingerprint import (
    LOCATION_SCORE_EXACT,
    LOCATION_SCORE_SAME_FILE,
)
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
    async def test_find_duplicates_excludes_groups_with_different_children(
        self, session, mock_embed
    ):
        """Regression: two groups, same parent embedding, different child titles
        must be excluded from duplicate list (Jaccard < 0.2)."""
        repo = ExperienceRepository(session)
        # Same embedding so vector similarity is 1.0 and they would appear as duplicate
        same_embedding = await mock_embed.encode_single("本组4条相关经验")
        parent_data = {
            "title": "经验组",
            "description": "本组4条相关经验",
            "solution": "",
            "embedding": same_embedding,
            "created_by": "tester",
            "exp_status": "published",
        }
        # Group A: children about "托底修复"
        await repo.create_group(
            parent_data={**parent_data, "title": "经验组A"},
            children_data=[
                {"title": "托底修复步骤1", "description": "d1", "solution": "s1", "embedding": None},  # noqa: E501
                {"title": "配置检查", "description": "d2", "solution": "s2", "embedding": None},
            ],
            created_by="tester",
        )
        # Group B: children about "架构决策" (no overlap with A)
        await repo.create_group(
            parent_data={**parent_data, "title": "经验组B"},
            children_data=[
                {"title": "架构决策", "description": "d1", "solution": "s1", "embedding": None},
                {"title": "技术选型", "description": "d2", "solution": "s2", "embedding": None},
            ],
            created_by="tester",
        )
        await session.commit()

        pairs = await repo.find_duplicates(threshold=0.5, limit=10)
        # Pair must be excluded by group-aware filter (Jaccard of child titles = 0)
        assert len(pairs) == 0, "Groups with different children must be excluded"

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

    @pytest.fixture
    def service_with_pipeline(self, mock_embed):
        """ExperienceService with SearchPipeline so search applies location_boost."""
        pipeline = SearchPipeline(
            embedding_provider=mock_embed,
            reranker_provider=NoopRerankerProvider(),
            search_config=SearchConfig(location_weight=0.15),
            retrieval_config=RetrievalConfig(),
            cache_config=CacheConfig(enabled=False),
            pageindex_lite_config=PageIndexLiteConfig(enabled=False),
            file_location_config=FileLocationBindingConfig(
                file_location_ttl_days=30,
                file_location_refresh_on_access=False,
            ),
            db_url=DB_URL,
        )
        return ExperienceService(
            embedding_provider=mock_embed,
            auth_provider=NoAuth(),
            search_pipeline=pipeline,
            file_location_config=FileLocationBindingConfig(
                file_location_ttl_days=30,
                file_location_refresh_on_access=False,
            ),
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

    @pytest.mark.asyncio
    async def test_file_locations_e2e_location_score_and_final_score(
        self, service_with_pipeline, session
    ):
        """E2E: save with file_locations; search with overlapping current_file_locations
        yields location_score >= LOCATION_SCORE_EXACT and higher final score than
        same query without current_file_locations. Optional: same path non-overlap
        yields LOCATION_SCORE_SAME_FILE.
        """
        # (1) Save experience with file_locations (path, start_line, end_line, optional snippet)
        file_path = "src/team_memory/file_locations_e2e.py"
        result = await service_with_pipeline.save(
            session=session,
            title="E2E file location binding helper pattern",
            problem="Need to bind experience to a file region for retrieval boost.",
            solution="Use file_locations with path, start_line, end_line when saving.",
            created_by="e2e",
            tags=["e2e", "file_locations"],
            file_locations=[
                {
                    "path": file_path,
                    "start_line": 10,
                    "end_line": 25,
                    "snippet": "def bind_location(): ...",
                }
            ],
        )
        await session.commit()
        assert "id" in result
        exp_id = result["id"]

        # (2) Search with same path and overlapping line range -> experience in results,
        #     location_score >= LOCATION_SCORE_EXACT (1.0)
        results_with_loc = await service_with_pipeline.search(
            query="file location binding retrieval boost",
            min_similarity=0.0,
            current_file_locations=[
                {"path": file_path, "start_line": 12, "end_line": 20}
            ],
        )
        found_with_loc = next(
            (r for r in results_with_loc if r.get("id") == exp_id), None
        )
        assert found_with_loc is not None, (
            "Expected experience in results when current_file_locations overlap"
        )
        loc_score = found_with_loc.get("location_score", 0.0)
        assert loc_score >= LOCATION_SCORE_EXACT, (
            f"location_score should be >= {LOCATION_SCORE_EXACT}, got {loc_score}"
        )
        score_with_loc = found_with_loc.get("score", 0.0)

        # (3) Same query without current_file_locations -> same experience has lower score
        results_no_loc = await service_with_pipeline.search(
            query="file location binding retrieval boost",
            min_similarity=0.0,
        )
        found_no_loc = next(
            (r for r in results_no_loc if r.get("id") == exp_id), None
        )
        assert found_no_loc is not None
        score_no_loc = found_no_loc.get("score", 0.0)
        assert score_with_loc > score_no_loc, (
            f"With location boost score ({score_with_loc}) should be > "
            f"without ({score_no_loc})"
        )

        # (4) Optional: same path, non-overlapping lines -> location_score == LOCATION_SCORE_SAME_FILE
        results_same_file = await service_with_pipeline.search(
            query="file location binding retrieval boost",
            min_similarity=0.0,
            current_file_locations=[
                {"path": file_path, "start_line": 100, "end_line": 110}
            ],
        )
        found_same_file = next(
            (r for r in results_same_file if r.get("id") == exp_id), None
        )
        assert found_same_file is not None
        assert found_same_file.get("location_score") == LOCATION_SCORE_SAME_FILE

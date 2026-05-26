"""Unit tests for LLMWiki — local SQLite knowledge store with FTS5 search."""

from __future__ import annotations

import pytest

from team_memory.daemon.wiki.llm_wiki import LLMWiki

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def wiki_db(tmp_path):
    """Return a temporary DB path for LLMWiki."""
    return str(tmp_path / "wiki.db")


# ---------------------------------------------------------------------------
# Tests: basic CRUD
# ---------------------------------------------------------------------------


class TestLLMWikiAdd:
    @pytest.mark.asyncio
    async def test_add_new_entry(self, wiki_db):
        async with LLMWiki(db_path=wiki_db) as wiki:
            entry_id = await wiki.add(
                title="Python GIL",
                content="The GIL prevents true parallelism in CPython.",
                project="team_doc",
                tags=["python", "concurrency"],
            )
            assert entry_id  # non-empty UUID
            entry = await wiki.get(entry_id)
            assert entry is not None
            assert entry["title"] == "Python GIL"
            assert entry["project"] == "team_doc"
            assert entry["tags"] == ["python", "concurrency"]

    @pytest.mark.asyncio
    async def test_add_upsert_by_title_project(self, wiki_db):
        """Same title + project → update, not duplicate."""
        async with LLMWiki(db_path=wiki_db) as wiki:
            id1 = await wiki.add("GIL", "version 1", project="proj")
            id2 = await wiki.add("GIL", "version 2", project="proj")
            assert id1 == id2  # same entry updated

            entry = await wiki.get(id1)
            assert entry["content"] == "version 2"

    @pytest.mark.asyncio
    async def test_add_different_project_creates_new(self, wiki_db):
        """Same title but different project → separate entries."""
        async with LLMWiki(db_path=wiki_db) as wiki:
            id1 = await wiki.add("GIL", "content a", project="proj_a")
            id2 = await wiki.add("GIL", "content b", project="proj_b")
            assert id1 != id2

    @pytest.mark.asyncio
    async def test_add_without_tags(self, wiki_db):
        async with LLMWiki(db_path=wiki_db) as wiki:
            entry_id = await wiki.add("Simple", "No tags here", project="default")
            entry = await wiki.get(entry_id)
            assert entry["tags"] == []


# ---------------------------------------------------------------------------
# Tests: search
# ---------------------------------------------------------------------------


class TestLLMWikiSearch:
    @pytest.mark.asyncio
    async def test_search_basic(self, wiki_db):
        async with LLMWiki(db_path=wiki_db) as wiki:
            await wiki.add("Python GIL", "The GIL prevents parallelism", project="team_doc")
            await wiki.add("Asyncio Event Loop", "Asyncio uses a single-threaded event loop", project="team_doc")
            await wiki.add("Docker Networking", "Docker containers use bridge networks", project="infra")

            results = await wiki.search("GIL parallelism")
            assert len(results) >= 1
            assert results[0]["title"] == "Python GIL"

    @pytest.mark.asyncio
    async def test_search_with_project_filter(self, wiki_db):
        async with LLMWiki(db_path=wiki_db) as wiki:
            await wiki.add("Python GIL", "The GIL prevents parallelism", project="team_doc")
            await wiki.add("Docker GIL issue", "GIL in Docker context", project="infra")

            results = await wiki.search("GIL", project="team_doc")
            assert len(results) == 1
            assert results[0]["project"] == "team_doc"

    @pytest.mark.asyncio
    async def test_search_empty_query(self, wiki_db):
        async with LLMWiki(db_path=wiki_db) as wiki:
            await wiki.add("Test", "Content", project="default")
            results = await wiki.search("")
            assert results == []

    @pytest.mark.asyncio
    async def test_search_no_results(self, wiki_db):
        async with LLMWiki(db_path=wiki_db) as wiki:
            await wiki.add("Python GIL", "About GIL", project="default")
            results = await wiki.search("kubernetes deployment")
            assert results == []

    @pytest.mark.asyncio
    async def test_search_limit(self, wiki_db):
        async with LLMWiki(db_path=wiki_db) as wiki:
            for i in range(10):
                await wiki.add(f"Entry {i}", f"Content about topic {i}", project="default")
            results = await wiki.search("topic", limit=3)
            assert len(results) <= 3


# ---------------------------------------------------------------------------
# Tests: delete
# ---------------------------------------------------------------------------


class TestLLMWikiDelete:
    @pytest.mark.asyncio
    async def test_delete_existing(self, wiki_db):
        async with LLMWiki(db_path=wiki_db) as wiki:
            entry_id = await wiki.add("To Delete", "Temporary content", project="default")
            deleted = await wiki.delete(entry_id)
            assert deleted is True
            assert await wiki.get(entry_id) is None

    @pytest.mark.asyncio
    async def test_delete_nonexistent(self, wiki_db):
        async with LLMWiki(db_path=wiki_db) as wiki:
            deleted = await wiki.delete("nonexistent-id")
            assert deleted is False


# ---------------------------------------------------------------------------
# Tests: stats
# ---------------------------------------------------------------------------


class TestLLMWikiStats:
    @pytest.mark.asyncio
    async def test_stats_empty(self, wiki_db):
        async with LLMWiki(db_path=wiki_db) as wiki:
            s = await wiki.stats()
            assert s["total"] == 0
            assert s["by_project"] == {}

    @pytest.mark.asyncio
    async def test_stats_with_entries(self, wiki_db):
        async with LLMWiki(db_path=wiki_db) as wiki:
            await wiki.add("A", "a", project="proj1")
            await wiki.add("B", "b", project="proj1")
            await wiki.add("C", "c", project="proj2")
            s = await wiki.stats()
            assert s["total"] == 3
            assert s["by_project"]["proj1"] == 2
            assert s["by_project"]["proj2"] == 1


# ---------------------------------------------------------------------------
# Tests: context manager / error handling
# ---------------------------------------------------------------------------


class TestLLMWikiLifecycle:
    @pytest.mark.asyncio
    async def test_not_connected_raises(self, wiki_db):
        wiki = LLMWiki(db_path=wiki_db)
        with pytest.raises(RuntimeError, match="not connected"):
            await wiki.add("X", "Y")

    @pytest.mark.asyncio
    async def test_search_after_delete_removes_from_fts(self, wiki_db):
        """Verify FTS triggers work: deleted entries are not searchable."""
        async with LLMWiki(db_path=wiki_db) as wiki:
            entry_id = await wiki.add("UniqueXYZ", "Unique content XYZ", project="default")
            # Should find it
            results = await wiki.search("UniqueXYZ")
            assert len(results) == 1

            # Delete and verify not searchable
            await wiki.delete(entry_id)
            results = await wiki.search("UniqueXYZ")
            assert len(results) == 0

    @pytest.mark.asyncio
    async def test_search_after_update_reflects_new_content(self, wiki_db):
        """Verify FTS triggers work: updated entries reflect new content."""
        async with LLMWiki(db_path=wiki_db) as wiki:
            await wiki.add("Topic", "Old content about alpha", project="default")
            await wiki.add("Topic", "New content about beta", project="default")

            # Should not find old content
            results = await wiki.search("alpha")
            assert len(results) == 0

            # Should find new content
            results = await wiki.search("beta")
            assert len(results) == 1
            assert results[0]["title"] == "Topic"

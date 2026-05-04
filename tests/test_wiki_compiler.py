"""Tests for WikiCompiler — compile PG experiences into markdown wiki files."""
from __future__ import annotations

import os

import pytest
from daemon.wiki_compiler import WikiCompiler

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def wiki_dir(tmp_path):
    """Create a temporary wiki directory."""
    wiki_root = tmp_path / "wiki"
    wiki_root.mkdir()
    return str(wiki_root)


@pytest.fixture
def db_path(tmp_path):
    """Create a temporary cache DB path."""
    return str(tmp_path / ".wiki" / "cache.db")


@pytest.fixture
def compiler(wiki_dir, db_path):
    """Create a WikiCompiler instance (entered)."""
    c = WikiCompiler(wiki_root=wiki_dir, db_path=db_path)
    return c


def _make_experience(
    id: str = "abc123",
    title: str = "测试经验",
    description: str = "这是一个测试描述",
    solution: str = "这是解决方案",
    tags: list | None = None,
    project: str = "test_project",
    experience_type: str = "general",
) -> dict:
    """Create a test experience dict."""
    return {
        "id": id,
        "title": title,
        "description": description,
        "solution": solution,
        "tags": tags or ["python", "testing"],
        "project": project,
        "experience_type": experience_type,
        "created_at": "2026-05-01T10:00:00",
        "updated_at": "2026-05-03T15:00:00",
        "confidence": "high",
        "recall_count": 10,
    }


# ---------------------------------------------------------------------------
# Lifecycle tests
# ---------------------------------------------------------------------------


class TestLifecycle:
    async def test_enter_creates_db_and_dirs(self, wiki_dir, db_path):
        async with WikiCompiler(wiki_root=wiki_dir, db_path=db_path):
            assert os.path.isdir(os.path.join(wiki_dir, "concepts"))
            assert os.path.isdir(os.path.join(wiki_dir, "queries"))
            assert os.path.isfile(db_path)

    async def test_exit_closes_db(self, wiki_dir, db_path):
        c = WikiCompiler(wiki_root=wiki_dir, db_path=db_path)
        await c.__aenter__()
        assert c._db is not None
        await c.__aexit__(None, None, None)
        assert c._db is None

    async def test_require_db_raises_when_not_initialized(self, wiki_dir, db_path):
        c = WikiCompiler(wiki_root=wiki_dir, db_path=db_path)
        with pytest.raises(RuntimeError, match="not initialized"):
            c._require_db()


# ---------------------------------------------------------------------------
# Slugify tests
# ---------------------------------------------------------------------------


class TestSlugify:
    def test_english_title(self):
        result = WikiCompiler._slugify("How to fix Docker networking")
        assert result == "how-to-fix-docker-networking"

    def test_chinese_title(self):
        result = WikiCompiler._slugify("Docker 网络问题解决方法")
        assert "docker" in result
        assert "网络" in result

    def test_special_chars_replaced(self):
        result = WikiCompiler._slugify("A/B:C|D")
        assert "/" not in result
        assert ":" not in result
        assert "|" not in result

    def test_plus_replaced(self):
        result = WikiCompiler._slugify("C++ memory leak")
        assert "加" in result

    def test_empty_title(self):
        assert WikiCompiler._slugify("") == "untitled"

    def test_consecutive_dashes_collapsed(self):
        result = WikiCompiler._slugify("A   B")
        assert "---" not in result

    def test_truncation(self):
        long_title = "a" * 200
        result = WikiCompiler._slugify(long_title)
        assert len(result) <= 100


# ---------------------------------------------------------------------------
# Hash tests
# ---------------------------------------------------------------------------


class TestHash:
    def test_same_experience_same_hash(self):
        exp = _make_experience()
        h1 = WikiCompiler._compute_hash(exp)
        h2 = WikiCompiler._compute_hash(exp)
        assert h1 == h2

    def test_different_experience_different_hash(self):
        exp1 = _make_experience(title="A")
        exp2 = _make_experience(title="B")
        assert WikiCompiler._compute_hash(exp1) != WikiCompiler._compute_hash(exp2)


# ---------------------------------------------------------------------------
# Find related tests
# ---------------------------------------------------------------------------


class TestFindRelated:
    def _make_compiler(self) -> WikiCompiler:
        """Create a WikiCompiler instance for testing _find_related."""
        return WikiCompiler(wiki_root="/tmp/test_wiki")

    def test_tag_overlap(self):
        compiler = self._make_compiler()
        exp1 = _make_experience(id="1", title="A", tags=["python", "docker"])
        exp2 = _make_experience(id="2", title="B", tags=["python", "flask"])
        exp3 = _make_experience(id="3", title="C", tags=["java"], project="other_project")
        related = compiler._find_related(exp1, [exp2, exp3])
        assert len(related) == 1
        assert related[0]["id"] == "2"

    def test_same_project_no_tag_overlap(self):
        compiler = self._make_compiler()
        exp1 = _make_experience(id="1", tags=["python"], project="tm")
        exp2 = _make_experience(id="2", tags=["java"], project="tm")
        related = compiler._find_related(exp1, [exp2])
        assert len(related) == 1  # Same project gives +1 score

    def test_no_relation(self):
        compiler = self._make_compiler()
        exp1 = _make_experience(id="1", tags=["python"], project="tm")
        exp2 = _make_experience(id="2", tags=["java"], project="other")
        related = compiler._find_related(exp1, [exp2])
        assert len(related) == 0

    def test_max_related(self):
        compiler = self._make_compiler()
        exp1 = _make_experience(id="1", tags=["python"])
        others = [_make_experience(id=str(i), tags=["python"]) for i in range(10)]
        related = compiler._find_related(exp1, others, max_related=3)
        assert len(related) == 3

    def test_excludes_self(self):
        compiler = self._make_compiler()
        exp1 = _make_experience(id="1", tags=["python"])
        related = compiler._find_related(exp1, [exp1])
        assert len(related) == 0

    def test_empty_list(self):
        compiler = self._make_compiler()
        exp = _make_experience()
        assert compiler._find_related(exp, []) == []

    def test_entity_overlap(self):
        """Entity-graph overlap adds score."""
        compiler = self._make_compiler()
        compiler._exp_entity_map = {
            "1": {"entity-a", "entity-b"},
            "2": {"entity-a", "entity-c"},
            "3": {"entity-d"},
        }
        exp1 = _make_experience(id="1", tags=[], project="x")
        exp2 = _make_experience(id="2", tags=[], project="y")
        exp3 = _make_experience(id="3", tags=[], project="x")
        related = compiler._find_related(exp1, [exp2, exp3])
        # exp2 shares entity-a (score=3), exp3 shares project only (score=1)
        assert len(related) == 2
        assert related[0]["id"] == "2"  # entity overlap > project only

    def test_embedding_similarity(self):
        """Embedding cosine similarity > 0.7 adds score."""
        compiler = self._make_compiler()
        try:
            import numpy as np
        except ImportError:
            pytest.skip("numpy not available")

        # Two similar vectors (cosine ~0.97)
        compiler._exp_embeddings = {
            "1": np.array([1.0, 0.0, 0.0], dtype=np.float32),
            "2": np.array([0.98, 0.2, 0.0], dtype=np.float32),
            "3": np.array([0.0, 0.0, 1.0], dtype=np.float32),  # orthogonal
        }
        exp1 = _make_experience(id="1", tags=[], project="x")
        exp2 = _make_experience(id="2", tags=[], project="y")
        exp3 = _make_experience(id="3", tags=[], project="x")
        related = compiler._find_related(exp1, [exp2, exp3])
        # exp2 is embedding-similar (score~1.94), exp3 only shares project (score=1)
        assert len(related) == 2
        assert related[0]["id"] == "2"


# ---------------------------------------------------------------------------
# Compile one tests
# ---------------------------------------------------------------------------


class TestCompileOne:
    async def test_creates_markdown_file(self, compiler):
        async with compiler:
            exp = _make_experience()
            path = await compiler.compile_one(exp)
            assert path == "concepts/测试经验.md"
            full_path = os.path.join(compiler._wiki_root, path)
            assert os.path.isfile(full_path)

    async def test_file_contains_frontmatter(self, compiler):
        async with compiler:
            exp = _make_experience()
            path = await compiler.compile_one(exp)
            full_path = os.path.join(compiler._wiki_root, path)
            with open(full_path, "r", encoding="utf-8") as f:
                content = f.read()
            assert "---" in content
            assert "title: 测试经验" in content
            assert "project: test_project" in content

    async def test_file_contains_problem_section(self, compiler):
        async with compiler:
            exp = _make_experience(description="这个问题很严重")
            path = await compiler.compile_one(exp)
            full_path = os.path.join(compiler._wiki_root, path)
            with open(full_path, "r", encoding="utf-8") as f:
                content = f.read()
            assert "问题描述" in content
            assert "这个问题很严重" in content

    async def test_file_contains_solution_section(self, compiler):
        async with compiler:
            exp = _make_experience(solution="用这个方法解决")
            path = await compiler.compile_one(exp)
            full_path = os.path.join(compiler._wiki_root, path)
            with open(full_path, "r", encoding="utf-8") as f:
                content = f.read()
            assert "解决方案" in content
            assert "用这个方法解决" in content

    async def test_slug_collision_handling(self, compiler):
        async with compiler:
            exp1 = _make_experience(id="id1", title="Same Title")
            exp2 = _make_experience(id="id2", title="Same Title")
            path1 = await compiler.compile_one(exp1)
            path2 = await compiler.compile_one(exp2)
            assert path1 != path2
            assert "id2" in path2 or "id2"[:8] in path2

    async def test_updates_cache(self, compiler):
        async with compiler:
            exp = _make_experience(id="cache-test-1")
            await compiler.compile_one(exp)
            cached = await compiler._get_all_cached()
            assert "cache-test-1" in cached
            assert cached["cache-test-1"]["status"] == "compiled"


# ---------------------------------------------------------------------------
# Compile batch tests
# ---------------------------------------------------------------------------


class TestCompileBatch:
    async def test_batch_creates_multiple_files(self, compiler):
        async with compiler:
            compiler._all_experiences = [
                _make_experience(id="b1", title="Batch 1"),
                _make_experience(id="b2", title="Batch 2"),
            ]
            paths = await compiler.compile_batch(compiler._all_experiences)
            assert len(paths) == 2

    async def test_batch_skips_errors(self, compiler):
        async with compiler:
            exps = [_make_experience(id="ok", title="OK")]
            paths = await compiler.compile_batch(exps)
            assert len(paths) == 1


# ---------------------------------------------------------------------------
# Incremental compile tests
# ---------------------------------------------------------------------------


class TestCompileIncremental:
    async def test_creates_new_experiences(self, compiler):
        async with compiler:
            exps = [_make_experience(id="inc-1"), _make_experience(id="inc-2")]
            result = await compiler.compile_incremental(exps)
            assert result.created == 2
            assert result.skipped == 0

    async def test_skips_unchanged(self, compiler):
        async with compiler:
            exps = [_make_experience(id="inc-3")]
            await compiler.compile_incremental(exps)
            result = await compiler.compile_incremental(exps)
            assert result.skipped == 1
            assert result.created == 0

    async def test_detects_updates(self, compiler):
        async with compiler:
            exps = [_make_experience(id="inc-4", title="Original")]
            await compiler.compile_incremental(exps)
            exps_updated = [_make_experience(id="inc-4", title="Updated")]
            result = await compiler.compile_incremental(exps_updated)
            assert result.updated == 1

    async def test_deletes_removed_experiences(self, compiler):
        async with compiler:
            exps = [_make_experience(id="inc-5"), _make_experience(id="inc-6")]
            await compiler.compile_incremental(exps)
            result = await compiler.compile_incremental([_make_experience(id="inc-5")])
            assert result.deleted == 1

    async def test_updates_index_and_log(self, compiler):
        async with compiler:
            exps = [_make_experience(id="inc-7")]
            await compiler.compile_incremental(exps)
            index_path = os.path.join(compiler._wiki_root, "index.md")
            log_path = os.path.join(compiler._wiki_root, "log.md")
            assert os.path.isfile(index_path)
            assert os.path.isfile(log_path)
            with open(log_path, "r") as f:
                log_content = f.read()
            assert "incremental" in log_content


# ---------------------------------------------------------------------------
# Full rebuild tests
# ---------------------------------------------------------------------------


class TestFullRebuild:
    async def test_rebuild_clears_and_recompiles(self, compiler):
        async with compiler:
            exps = [_make_experience(id="rb-1"), _make_experience(id="rb-2")]
            await compiler.compile_incremental(exps)
            result = await compiler.full_rebuild(exps)
            assert result.created == 2
            assert result.errors == 0

    async def test_rebuild_updates_index(self, compiler):
        async with compiler:
            exps = [_make_experience(id="rb-3")]
            await compiler.full_rebuild(exps)
            index_path = os.path.join(compiler._wiki_root, "index.md")
            with open(index_path, "r") as f:
                content = f.read()
            assert "rb-3" in content or "测试经验" in content


# ---------------------------------------------------------------------------
# Query API tests
# ---------------------------------------------------------------------------


class TestQueryAPI:
    async def test_get_stale(self, compiler):
        async with compiler:
            exps = [_make_experience(id="stale-1")]
            await compiler.compile_incremental(exps)
            db = compiler._require_db()
            await db.execute(
                "UPDATE wiki_cache SET status = 'stale' WHERE experience_id = 'stale-1'"
            )
            await db.commit()
            stale = await compiler.get_stale()
            assert "stale-1" in stale

    async def test_get_uncompiled(self, compiler):
        async with compiler:
            exps = [_make_experience(id="uncomp-1")]
            await compiler.compile_incremental(exps)
            uncompiled = await compiler.get_uncompiled(["uncomp-1", "uncomp-2"])
            assert "uncomp-2" in uncompiled
            assert "uncomp-1" not in uncompiled

    async def test_get_by_project(self, compiler):
        async with compiler:
            exps = [
                _make_experience(id="proj-1", project="alpha"),
                _make_experience(id="proj-2", project="beta"),
            ]
            await compiler.compile_incremental(exps)
            alpha = await compiler.get_by_project("alpha")
            assert len(alpha) == 1
            assert alpha[0]["experience_id"] == "proj-1"

    async def test_get_stats(self, compiler):
        async with compiler:
            exps = [_make_experience(id="stat-1", project="p1")]
            await compiler.compile_incremental(exps)
            stats = await compiler.get_stats()
            assert stats["total"] == 1
            assert "p1" in stats["by_project"]


# ---------------------------------------------------------------------------
# Page rendering tests
# ---------------------------------------------------------------------------


class TestPageRendering:
    async def test_related_links_rendered(self, compiler):
        async with compiler:
            exp1 = _make_experience(id="rel-1", title="Related A", tags=["python"])
            exp2 = _make_experience(id="rel-2", title="Related B", tags=["python"])
            compiler._all_experiences = [exp1, exp2]
            path = await compiler.compile_one(exp1)
            full_path = os.path.join(compiler._wiki_root, path)
            with open(full_path, "r", encoding="utf-8") as f:
                content = f.read()
            assert "相关经验" in content
            assert "Related B" in content

    async def test_no_solution_no_section(self, compiler):
        async with compiler:
            exp = _make_experience(solution="")
            path = await compiler.compile_one(exp)
            full_path = os.path.join(compiler._wiki_root, path)
            with open(full_path, "r", encoding="utf-8") as f:
                content = f.read()
            assert "解决方案" not in content

    async def test_date_formatting(self, compiler):
        async with compiler:
            from datetime import datetime, timezone
            dt = datetime(2026, 5, 1, 10, 0, 0, tzinfo=timezone.utc)
            exp = _make_experience()
            exp["created_at"] = dt
            path = await compiler.compile_one(exp)
            full_path = os.path.join(compiler._wiki_root, path)
            with open(full_path, "r", encoding="utf-8") as f:
                content = f.read()
            assert "2026-05-01" in content

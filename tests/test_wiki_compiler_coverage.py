"""Additional tests for WikiCompiler to increase coverage to 90%+.

Covers previously uncovered lines:
- Lines 246, 253-266: _load_pg_data embedding paths
- Lines 350-352: compile_batch exception handling
- Line 373: compile_incremental with pg_url
- Lines 388-389, 395-396: compile_incremental exception in new/updated
- Line 444: full_rebuild with pg_url
- Lines 463-464: full_rebuild exception
- Line 494: get_uncompiled empty list
- Lines 654-655: _find_related embedding exception
- Line 779: _append_log with title
- Line 817: _format_date(None)
- Lines 832-908: _update_topics
- Lines 912-917: _get_cache_by_experience_id
- Lines 921-938: _update_index_with_topics
- Lines 949-993: _generate_contradictions
"""
from __future__ import annotations

import os
import sys
import struct
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from daemon.wiki_compiler import WikiCompiler, CompileResult


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


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_experience(
    id: str = "cov-1",
    title: str = "测试经验",
    description: str = "这是一个测试描述",
    solution: str = "这是解决方案",
    tags: list | None = None,
    project: str = "test_project",
    experience_type: str = "general",
) -> dict:
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
# Lines 817: _format_date(None)
# ---------------------------------------------------------------------------

class TestFormatDate:
    def test_none_returns_empty(self):
        assert WikiCompiler._format_date(None) == ""

    def test_string_value(self):
        assert WikiCompiler._format_date("2026-05-01T10:00:00") == "2026-05-01"

    def test_datetime_value(self):
        dt = datetime(2026, 6, 15, tzinfo=timezone.utc)
        assert WikiCompiler._format_date(dt) == "2026-06-15"


# ---------------------------------------------------------------------------
# Line 494: get_uncompiled with empty list
# ---------------------------------------------------------------------------

class TestGetUncompiled:
    async def test_empty_list_returns_empty(self, compiler):
        async with compiler:
            result = await compiler.get_uncompiled([])
            assert result == []


# ---------------------------------------------------------------------------
# Lines 654-655: _find_related embedding exception
# ---------------------------------------------------------------------------

class TestFindRelatedEmbeddingException:
    def test_embedding_exception_does_not_crash(self):
        compiler = WikiCompiler(wiki_root="/tmp/test_wiki")
        # Create embeddings that will cause an exception in numpy
        compiler._exp_embeddings = {
            "1": "not-a-numpy-array",
            "2": "also-not-a-numpy-array",
        }
        exp1 = _make_experience(id="1", tags=[], project="x")
        exp2 = _make_experience(id="2", tags=[], project="x")
        # Should not crash, just skip embedding similarity
        related = compiler._find_related(exp1, [exp2])
        # Should still find via project signal
        assert len(related) == 1


# ---------------------------------------------------------------------------
# Lines 350-352: compile_batch exception handling
# ---------------------------------------------------------------------------

class TestCompileBatchException:
    async def test_batch_skips_individual_failures(self, compiler):
        async with compiler:
            exps = [
                _make_experience(id="batch-ok", title="OK Experience"),
            ]
            paths = await compiler.compile_batch(exps)
            assert len(paths) == 1

    async def test_batch_exception_skips_one(self, compiler):
        """Lines 350-352: exception in one experience skips it in batch."""
        async with compiler:
            exps = [
                _make_experience(id="batch-fail", title="Will Fail"),
                _make_experience(id="batch-ok2", title="Will Succeed"),
            ]
            original_compile_one = compiler.compile_one

            async def selective_fail(exp):
                if exp["id"] == "batch-fail":
                    raise RuntimeError("compile failed")
                return await original_compile_one(exp)

            with patch.object(compiler, "compile_one", side_effect=selective_fail):
                paths = await compiler.compile_batch(exps)
            assert len(paths) == 1
            assert "will-succeed" in paths[0]


# ---------------------------------------------------------------------------
# Lines 373, 388-389, 395-396: compile_incremental with pg_url and errors
# ---------------------------------------------------------------------------

class TestCompileIncrementalPgAndErrors:
    async def test_compile_incremental_with_pg_url(self, compiler):
        """Line 373: if pg_url: await self._load_pg_data(pg_url)"""
        async with compiler:
            exps = [_make_experience(id="pg-1")]
            with patch.object(compiler, "_load_pg_data", new_callable=AsyncMock) as mock_load:
                result = await compiler.compile_incremental(exps, pg_url="postgresql://test")
                mock_load.assert_called_once_with("postgresql://test")
                assert result.created == 1

    async def test_compile_incremental_new_experience_error(self, compiler):
        """Lines 388-389: exception in new experience compile increments errors."""
        async with compiler:
            exps = [_make_experience(id="err-new"), _make_experience(id="err-ok")]
            original_compile_one = compiler.compile_one

            async def failing_compile_one(exp):
                if exp["id"] == "err-new":
                    raise RuntimeError("compile failed")
                return await original_compile_one(exp)

            with patch.object(compiler, "compile_one", side_effect=failing_compile_one):
                result = await compiler.compile_incremental(exps)
            assert result.errors >= 1
            assert result.created >= 1

    async def test_compile_incremental_updated_experience_error(self, compiler):
        """Lines 395-396: exception in updated experience compile increments errors."""
        async with compiler:
            # First compile to create cache
            exps1 = [_make_experience(id="upd-err", title="Original")]
            await compiler.compile_incremental(exps1)

            # Now update with error
            exps2 = [_make_experience(id="upd-err", title="Updated")]
            with patch.object(compiler, "compile_one", side_effect=RuntimeError("fail")):
                result = await compiler.compile_incremental(exps2)
            assert result.errors >= 1


# ---------------------------------------------------------------------------
# Line 444, 463-464: full_rebuild with pg_url and errors
# ---------------------------------------------------------------------------

class TestFullRebuildPgAndErrors:
    async def test_full_rebuild_with_pg_url(self, compiler):
        """Line 444: if pg_url: await self._load_pg_data(pg_url)"""
        async with compiler:
            exps = [_make_experience(id="rb-pg")]
            with patch.object(compiler, "_load_pg_data", new_callable=AsyncMock) as mock_load:
                result = await compiler.full_rebuild(exps, pg_url="postgresql://test")
                mock_load.assert_called_once_with("postgresql://test")
                assert result.created == 1

    async def test_full_rebuild_with_error(self, compiler):
        """Lines 463-464: exception in full_rebuild compile increments errors."""
        async with compiler:
            exps = [_make_experience(id="rb-err")]
            with patch.object(compiler, "compile_one", side_effect=RuntimeError("fail")):
                result = await compiler.full_rebuild(exps)
            assert result.errors == 1
            assert result.created == 0


# ---------------------------------------------------------------------------
# Lines 832-908: _update_topics
# ---------------------------------------------------------------------------

class TestUpdateTopics:
    async def test_update_topics_no_pg_url(self, compiler):
        """Lines 829-830: early return when no pg_url."""
        async with compiler:
            compiler._pg_url = None
            compiler._exp_embeddings = {}
            await compiler._update_topics()

    async def test_update_topics_no_embeddings(self, compiler):
        """Lines 829-830: early return when no embeddings."""
        async with compiler:
            compiler._pg_url = "postgresql://test"
            compiler._exp_embeddings = {}
            await compiler._update_topics()

    async def test_update_topics_import_error(self, compiler):
        """Lines 833-835: ImportError from topic_discovery."""
        async with compiler:
            compiler._pg_url = "postgresql://test"
            compiler._exp_embeddings = {"1": "fake_emb"}
            # Remove topic_discovery from sys.modules to force ImportError
            saved = sys.modules.pop("team_memory.services.topic_discovery", None)
            sys.modules["team_memory.services.topic_discovery"] = None
            try:
                await compiler._update_topics()
            finally:
                if saved is not None:
                    sys.modules["team_memory.services.topic_discovery"] = saved
                else:
                    sys.modules.pop("team_memory.services.topic_discovery", None)

    async def test_update_topics_success(self, compiler):
        """Lines 832-908: full topic discovery path."""
        async with compiler:
            compiler._pg_url = "postgresql://test"
            try:
                import numpy as np
                compiler._exp_embeddings = {
                    "t1": np.array([1.0, 0.0], dtype=np.float32),
                    "t2": np.array([0.9, 0.1], dtype=np.float32),
                }
            except ImportError:
                pytest.skip("numpy not available")

            # Mock topic discovery
            mock_topic = MagicMock()
            mock_topic.name = "Test Topic"
            mock_topic.entities = ["entity1", "entity2", "entity3"]
            mock_topic.experience_ids = ["t1", "t2"]
            mock_topic.tags = ["tag1", "tag2"]

            mock_result = MagicMock()
            mock_result.topics = [mock_topic]

            mock_discover = AsyncMock(return_value=mock_result)

            # Mock load_settings
            mock_settings = MagicMock()
            mock_entity_config = MagicMock()
            mock_entity_config.enabled = True
            mock_settings.entity_extraction = mock_entity_config

            mock_td_module = MagicMock()
            mock_td_module.discover_topics = mock_discover

            mock_config_module = MagicMock()
            mock_config_module.load_settings = MagicMock(return_value=mock_settings)

            saved_td = sys.modules.pop("team_memory.services.topic_discovery", None)
            saved_cfg = sys.modules.pop("team_memory.config", None)
            sys.modules["team_memory.services.topic_discovery"] = mock_td_module
            sys.modules["team_memory.config"] = mock_config_module

            try:
                # Pre-populate cache so _get_cache_by_experience_id works
                await compiler.compile_one(_make_experience(id="t1", title="Exp T1"))
                await compiler.compile_one(_make_experience(id="t2", title="Exp T2"))

                await compiler._update_topics()
            finally:
                if saved_td is not None:
                    sys.modules["team_memory.services.topic_discovery"] = saved_td
                else:
                    sys.modules.pop("team_memory.services.topic_discovery", None)
                if saved_cfg is not None:
                    sys.modules["team_memory.config"] = saved_cfg
                else:
                    sys.modules.pop("team_memory.config", None)

            topics_dir = os.path.join(compiler._wiki_root, "topics")
            assert os.path.isdir(topics_dir)
            topic_files = [f for f in os.listdir(topics_dir) if f.endswith(".md")]
            assert len(topic_files) >= 1

    async def test_update_topics_no_topics_found(self, compiler):
        """Lines 849-850: early return when no topics found."""
        async with compiler:
            compiler._pg_url = "postgresql://test"
            try:
                import numpy as np
                compiler._exp_embeddings = {"t1": np.array([1.0], dtype=np.float32)}
            except ImportError:
                pytest.skip("numpy not available")

            mock_result = MagicMock()
            mock_result.topics = []

            mock_discover = AsyncMock(return_value=mock_result)

            mock_td_module = MagicMock()
            mock_td_module.discover_topics = mock_discover

            saved_td = sys.modules.pop("team_memory.services.topic_discovery", None)
            sys.modules["team_memory.services.topic_discovery"] = mock_td_module

            try:
                await compiler._update_topics()
            finally:
                if saved_td is not None:
                    sys.modules["team_memory.services.topic_discovery"] = saved_td
                else:
                    sys.modules.pop("team_memory.services.topic_discovery", None)

    async def test_update_topics_with_existing_files(self, compiler):
        """Lines 854-857: clear existing topic files before regenerating."""
        async with compiler:
            compiler._pg_url = "postgresql://test"
            try:
                import numpy as np
                compiler._exp_embeddings = {"t1": np.array([1.0, 0.0], dtype=np.float32)}
            except ImportError:
                pytest.skip("numpy not available")

            # Create existing topic file
            topics_dir = os.path.join(compiler._wiki_root, "topics")
            os.makedirs(topics_dir, exist_ok=True)
            old_file = os.path.join(topics_dir, "old-topic.md")
            with open(old_file, "w") as f:
                f.write("old content")

            mock_topic = MagicMock()
            mock_topic.name = "New Topic"
            mock_topic.entities = []
            mock_topic.experience_ids = []
            mock_topic.tags = ["new"]

            mock_result = MagicMock()
            mock_result.topics = [mock_topic]
            mock_discover = AsyncMock(return_value=mock_result)

            mock_td_module = MagicMock()
            mock_td_module.discover_topics = mock_discover

            saved_td = sys.modules.pop("team_memory.services.topic_discovery", None)
            sys.modules["team_memory.services.topic_discovery"] = mock_td_module

            try:
                await compiler._update_topics()
            finally:
                if saved_td is not None:
                    sys.modules["team_memory.services.topic_discovery"] = saved_td
                else:
                    sys.modules.pop("team_memory.services.topic_discovery", None)

            assert not os.path.exists(old_file)

    async def test_update_topics_llm_config_disabled(self, compiler):
        """Lines 843-844: llm_config set to None when enabled=False."""
        async with compiler:
            compiler._pg_url = "postgresql://test"
            try:
                import numpy as np
                compiler._exp_embeddings = {"t1": np.array([1.0, 0.0], dtype=np.float32)}
            except ImportError:
                pytest.skip("numpy not available")

            mock_topic = MagicMock()
            mock_topic.name = "Topic A"
            mock_topic.entities = ["ent1"]
            mock_topic.experience_ids = ["t1"]
            mock_topic.tags = ["tag1"]

            mock_result = MagicMock()
            mock_result.topics = [mock_topic]
            mock_discover = AsyncMock(return_value=mock_result)

            mock_settings = MagicMock()
            mock_entity_config = MagicMock()
            mock_entity_config.enabled = False  # disabled
            mock_settings.entity_extraction = mock_entity_config

            mock_td_module = MagicMock()
            mock_td_module.discover_topics = mock_discover
            mock_config_module = MagicMock()
            mock_config_module.load_settings = MagicMock(return_value=mock_settings)

            saved_td = sys.modules.pop("team_memory.services.topic_discovery", None)
            saved_cfg = sys.modules.pop("team_memory.config", None)
            sys.modules["team_memory.services.topic_discovery"] = mock_td_module
            sys.modules["team_memory.config"] = mock_config_module

            try:
                await compiler.compile_one(_make_experience(id="t1", title="Exp T1"))
                await compiler._update_topics()
            finally:
                if saved_td is not None:
                    sys.modules["team_memory.services.topic_discovery"] = saved_td
                else:
                    sys.modules.pop("team_memory.services.topic_discovery", None)
                if saved_cfg is not None:
                    sys.modules["team_memory.config"] = saved_cfg
                else:
                    sys.modules.pop("team_memory.config", None)

    async def test_update_topics_config_exception(self, compiler):
        """Lines 845-846: exception in config loading falls through to pass."""
        async with compiler:
            compiler._pg_url = "postgresql://test"
            try:
                import numpy as np
                compiler._exp_embeddings = {"t1": np.array([1.0, 0.0], dtype=np.float32)}
            except ImportError:
                pytest.skip("numpy not available")

            mock_topic = MagicMock()
            mock_topic.name = "Topic B"
            mock_topic.entities = ["ent1"]
            mock_topic.experience_ids = ["t1"]
            mock_topic.tags = ["tag1"]

            mock_result = MagicMock()
            mock_result.topics = [mock_topic]
            mock_discover = AsyncMock(return_value=mock_result)

            mock_td_module = MagicMock()
            mock_td_module.discover_topics = mock_discover
            mock_config_module = MagicMock()
            mock_config_module.load_settings = MagicMock(side_effect=Exception("config broken"))

            saved_td = sys.modules.pop("team_memory.services.topic_discovery", None)
            saved_cfg = sys.modules.pop("team_memory.config", None)
            sys.modules["team_memory.services.topic_discovery"] = mock_td_module
            sys.modules["team_memory.config"] = mock_config_module

            try:
                await compiler.compile_one(_make_experience(id="t1", title="Exp T1"))
                await compiler._update_topics()
            finally:
                if saved_td is not None:
                    sys.modules["team_memory.services.topic_discovery"] = saved_td
                else:
                    sys.modules.pop("team_memory.services.topic_discovery", None)
                if saved_cfg is not None:
                    sys.modules["team_memory.config"] = saved_cfg
                else:
                    sys.modules.pop("team_memory.config", None)

    async def test_update_topics_exp_not_in_cache(self, compiler):
        """Line 879: experience_id not found in cache uses fallback text."""
        async with compiler:
            compiler._pg_url = "postgresql://test"
            try:
                import numpy as np
                compiler._exp_embeddings = {"t1": np.array([1.0, 0.0], dtype=np.float32)}
            except ImportError:
                pytest.skip("numpy not available")

            mock_topic = MagicMock()
            mock_topic.name = "Topic C"
            mock_topic.entities = ["ent1"]
            mock_topic.experience_ids = ["nonexistent-id"]  # not in cache
            mock_topic.tags = ["tag1"]

            mock_result = MagicMock()
            mock_result.topics = [mock_topic]
            mock_discover = AsyncMock(return_value=mock_result)

            mock_td_module = MagicMock()
            mock_td_module.discover_topics = mock_discover

            saved_td = sys.modules.pop("team_memory.services.topic_discovery", None)
            sys.modules["team_memory.services.topic_discovery"] = mock_td_module

            try:
                await compiler._update_topics()
            finally:
                if saved_td is not None:
                    sys.modules["team_memory.services.topic_discovery"] = saved_td
                else:
                    sys.modules.pop("team_memory.services.topic_discovery", None)

            topics_dir = os.path.join(compiler._wiki_root, "topics")
            topic_files = [f for f in os.listdir(topics_dir) if f.endswith(".md")]
            assert len(topic_files) >= 1
            # Verify the fallback text is used
            with open(os.path.join(topics_dir, topic_files[0]), "r") as f:
                content = f.read()
            assert "nonexist" in content  # first 8 chars of "nonexistent-id"


# ---------------------------------------------------------------------------
# Lines 912-917: _get_cache_by_experience_id
# ---------------------------------------------------------------------------

class TestGetCacheByExperienceId:
    async def test_get_cache_by_experience_id_found(self, compiler):
        async with compiler:
            exp = _make_experience(id="cache-eid-1")
            await compiler.compile_one(exp)
            result = await compiler._get_cache_by_experience_id("cache-eid-1")
            assert result is not None
            assert result["experience_id"] == "cache-eid-1"

    async def test_get_cache_by_experience_id_not_found(self, compiler):
        async with compiler:
            result = await compiler._get_cache_by_experience_id("nonexistent")
            assert result is None


# ---------------------------------------------------------------------------
# Lines 921-938: _update_index_with_topics
# ---------------------------------------------------------------------------

class TestUpdateIndexWithTopics:
    async def test_update_index_with_topics_no_index(self, compiler):
        """Lines 922-923: early return when index.md doesn't exist."""
        async with compiler:
            await compiler._update_index_with_topics([])

    async def test_update_index_with_topics(self, compiler):
        """Lines 921-938: append topics section to index.md."""
        async with compiler:
            # Create a basic index.md first
            index_path = os.path.join(compiler._wiki_root, "index.md")
            with open(index_path, "w") as f:
                f.write("# Wiki Index\n")

            mock_topic = MagicMock()
            mock_topic.name = "My Topic"
            mock_topic.experience_ids = ["a", "b", "c"]

            await compiler._update_index_with_topics([mock_topic])

            with open(index_path, "r") as f:
                content = f.read()
            assert "主题聚类" in content
            assert "My Topic" in content
            assert "3 条经验" in content


# ---------------------------------------------------------------------------
# Lines 949-993: _generate_contradictions
# ---------------------------------------------------------------------------

class TestGenerateContradictions:
    async def test_no_pg_url(self, compiler):
        """Lines 946-947: early return when no pg_url."""
        async with compiler:
            compiler._pg_url = None
            await compiler._generate_contradictions()

    async def test_import_error(self, compiler):
        """Lines 953-954: ImportError from contradiction_detector."""
        async with compiler:
            compiler._pg_url = "postgresql://test"
            saved = sys.modules.pop("team_memory.services.contradiction_detector", None)
            sys.modules["team_memory.services.contradiction_detector"] = None
            try:
                await compiler._generate_contradictions()
            finally:
                if saved is not None:
                    sys.modules["team_memory.services.contradiction_detector"] = saved
                else:
                    sys.modules.pop("team_memory.services.contradiction_detector", None)

    async def test_no_pairs_found(self, compiler):
        """Lines 968-969: early return when no contradictions."""
        async with compiler:
            compiler._pg_url = "postgresql://test"
            mock_detect = AsyncMock(return_value=[])

            mock_cd_module = MagicMock()
            mock_cd_module.detect_contradictions = mock_detect

            saved_cd = sys.modules.pop("team_memory.services.contradiction_detector", None)
            sys.modules["team_memory.services.contradiction_detector"] = mock_cd_module

            try:
                await compiler._generate_contradictions()
            finally:
                if saved_cd is not None:
                    sys.modules["team_memory.services.contradiction_detector"] = saved_cd
                else:
                    sys.modules.pop("team_memory.services.contradiction_detector", None)

            # contradictions.md should not exist
            assert not os.path.exists(os.path.join(compiler._wiki_root, "contradictions.md"))

    async def test_pairs_found(self, compiler):
        """Lines 949-993: full contradiction report generation."""
        async with compiler:
            compiler._pg_url = "postgresql://test"

            mock_pair = MagicMock()
            mock_pair.exp_a_title = "Experience A Title"
            mock_pair.exp_a_id = "exp-a-id"
            mock_pair.exp_b_title = "Experience B Title"
            mock_pair.exp_b_id = "exp-b-id"
            mock_pair.shared_entities = ["entity1", "entity2"]
            mock_pair.reason = "Conflicting advice"

            mock_detect = AsyncMock(return_value=[mock_pair])

            mock_cd_module = MagicMock()
            mock_cd_module.detect_contradictions = mock_detect

            saved_cd = sys.modules.pop("team_memory.services.contradiction_detector", None)
            sys.modules["team_memory.services.contradiction_detector"] = mock_cd_module

            try:
                await compiler._generate_contradictions()
            finally:
                if saved_cd is not None:
                    sys.modules["team_memory.services.contradiction_detector"] = saved_cd
                else:
                    sys.modules.pop("team_memory.services.contradiction_detector", None)

            contradictions_path = os.path.join(compiler._wiki_root, "contradictions.md")
            assert os.path.exists(contradictions_path)
            with open(contradictions_path, "r") as f:
                content = f.read()
            assert "矛盾经验检测报告" in content
            assert "Experience A Title" in content
            assert "Conflicting advice" in content
            assert "entity1" in content

    async def test_llm_config_disabled(self, compiler):
        """Lines 962-963: llm_config set to None when enabled=False."""
        async with compiler:
            compiler._pg_url = "postgresql://test"

            mock_detect = AsyncMock(return_value=[])

            mock_cd_module = MagicMock()
            mock_cd_module.detect_contradictions = mock_detect

            mock_settings = MagicMock()
            mock_entity_config = MagicMock()
            mock_entity_config.enabled = False
            mock_settings.entity_extraction = mock_entity_config
            mock_config_module = MagicMock()
            mock_config_module.load_settings = MagicMock(return_value=mock_settings)

            saved_cd = sys.modules.pop("team_memory.services.contradiction_detector", None)
            saved_cfg = sys.modules.pop("team_memory.config", None)
            sys.modules["team_memory.services.contradiction_detector"] = mock_cd_module
            sys.modules["team_memory.config"] = mock_config_module

            try:
                await compiler._generate_contradictions()
            finally:
                if saved_cd is not None:
                    sys.modules["team_memory.services.contradiction_detector"] = saved_cd
                else:
                    sys.modules.pop("team_memory.services.contradiction_detector", None)
                if saved_cfg is not None:
                    sys.modules["team_memory.config"] = saved_cfg
                else:
                    sys.modules.pop("team_memory.config", None)

    async def test_llm_config_load_error(self, compiler):
        """Lines 958-965: exception loading settings falls through to pass."""
        async with compiler:
            compiler._pg_url = "postgresql://test"

            mock_detect = AsyncMock(return_value=[])
            mock_cd_module = MagicMock()
            mock_cd_module.detect_contradictions = mock_detect

            mock_config_module = MagicMock()
            mock_config_module.load_settings = MagicMock(side_effect=Exception("config broken"))

            saved_cd = sys.modules.pop("team_memory.services.contradiction_detector", None)
            saved_cfg = sys.modules.pop("team_memory.config", None)
            sys.modules["team_memory.services.contradiction_detector"] = mock_cd_module
            sys.modules["team_memory.config"] = mock_config_module

            try:
                await compiler._generate_contradictions()
            finally:
                if saved_cd is not None:
                    sys.modules["team_memory.services.contradiction_detector"] = saved_cd
                else:
                    sys.modules.pop("team_memory.services.contradiction_detector", None)
                if saved_cfg is not None:
                    sys.modules["team_memory.config"] = saved_cfg
                else:
                    sys.modules.pop("team_memory.config", None)


# ---------------------------------------------------------------------------
# Line 779: _append_log with title
# ---------------------------------------------------------------------------

class TestAppendLogWithTitle:
    async def test_append_log_with_title(self, compiler):
        """Line 779: entry += f\" | {title}\" when title is provided."""
        async with compiler:
            await compiler._append_log("test-action", "some detail", title="My Title")
            log_path = os.path.join(compiler._wiki_root, "log.md")
            with open(log_path, "r") as f:
                content = f.read()
            assert "My Title" in content
            assert "test-action" in content


# ---------------------------------------------------------------------------
# _load_pg_data embedding paths (lines 246, 253-266)
# ---------------------------------------------------------------------------

class TestLoadPgDataEmbeddings:
    async def test_load_pg_data_emb_none_skips(self, compiler):
        """Line 246: continue when emb is None."""
        async with compiler:
            compiler._pg_url = "postgresql://test"

            mock_entity_result = MagicMock()
            mock_entity_result.fetchall.return_value = []

            mock_emb_result = MagicMock()
            mock_row = MagicMock()
            mock_row.__getitem__ = lambda self, idx: "exp-1" if idx == 0 else None
            mock_emb_result.fetchall.return_value = [mock_row]

            call_count = 0

            async def mock_execute(sql_str):
                nonlocal call_count
                call_count += 1
                if call_count == 1:
                    return mock_entity_result
                return mock_emb_result

            mock_session = AsyncMock()
            mock_session.execute = mock_execute
            mock_session.__aenter__ = AsyncMock(return_value=mock_session)
            mock_session.__aexit__ = AsyncMock(return_value=False)

            mock_get_session = MagicMock()
            mock_get_session.return_value.__aenter__ = AsyncMock(return_value=mock_session)
            mock_get_session.return_value.__aexit__ = AsyncMock(return_value=False)

            saved_db = sys.modules.pop("team_memory.storage.database", None)
            mock_db_module = MagicMock()
            mock_db_module.get_session = mock_get_session
            sys.modules["team_memory.storage.database"] = mock_db_module

            try:
                await compiler._load_pg_data("postgresql://test")
            finally:
                if saved_db is not None:
                    sys.modules["team_memory.storage.database"] = saved_db
                else:
                    sys.modules.pop("team_memory.storage.database", None)

            assert "exp-1" not in compiler._exp_embeddings

    async def test_load_pg_data_emb_tolist(self, compiler):
        """Lines 253-254: emb.tolist() path."""
        async with compiler:
            compiler._pg_url = "postgresql://test"

            mock_entity_result = MagicMock()
            mock_entity_result.fetchall.return_value = []

            mock_emb = MagicMock()
            mock_emb.tolist.return_value = [0.1, 0.2, 0.3]

            mock_emb_result = MagicMock()
            mock_row = MagicMock()
            mock_row.__getitem__ = lambda self, idx: "exp-1" if idx == 0 else mock_emb
            mock_emb_result.fetchall.return_value = [mock_row]

            call_count = 0

            async def mock_execute(sql_str):
                nonlocal call_count
                call_count += 1
                if call_count == 1:
                    return mock_entity_result
                return mock_emb_result

            mock_session = AsyncMock()
            mock_session.execute = mock_execute
            mock_session.__aenter__ = AsyncMock(return_value=mock_session)
            mock_session.__aexit__ = AsyncMock(return_value=False)

            mock_get_session = MagicMock()
            mock_get_session.return_value.__aenter__ = AsyncMock(return_value=mock_session)
            mock_get_session.return_value.__aexit__ = AsyncMock(return_value=False)

            saved_db = sys.modules.pop("team_memory.storage.database", None)
            mock_db_module = MagicMock()
            mock_db_module.get_session = mock_get_session
            sys.modules["team_memory.storage.database"] = mock_db_module

            try:
                await compiler._load_pg_data("postgresql://test")
            finally:
                if saved_db is not None:
                    sys.modules["team_memory.storage.database"] = saved_db
                else:
                    sys.modules.pop("team_memory.storage.database", None)

            assert "exp-1" in compiler._exp_embeddings

    async def test_load_pg_data_emb_list(self, compiler):
        """Lines 255-256: emb is list path."""
        async with compiler:
            compiler._pg_url = "postgresql://test"

            mock_entity_result = MagicMock()
            mock_entity_result.fetchall.return_value = []

            mock_emb_result = MagicMock()
            mock_row = MagicMock()
            mock_row.__getitem__ = lambda self, idx: "exp-1" if idx == 0 else [0.1, 0.2, 0.3]
            mock_emb_result.fetchall.return_value = [mock_row]

            call_count = 0

            async def mock_execute(sql_str):
                nonlocal call_count
                call_count += 1
                if call_count == 1:
                    return mock_entity_result
                return mock_emb_result

            mock_session = AsyncMock()
            mock_session.execute = mock_execute
            mock_session.__aenter__ = AsyncMock(return_value=mock_session)
            mock_session.__aexit__ = AsyncMock(return_value=False)

            mock_get_session = MagicMock()
            mock_get_session.return_value.__aenter__ = AsyncMock(return_value=mock_session)
            mock_get_session.return_value.__aexit__ = AsyncMock(return_value=False)

            saved_db = sys.modules.pop("team_memory.storage.database", None)
            mock_db_module = MagicMock()
            mock_db_module.get_session = mock_get_session
            sys.modules["team_memory.storage.database"] = mock_db_module

            try:
                await compiler._load_pg_data("postgresql://test")
            finally:
                if saved_db is not None:
                    sys.modules["team_memory.storage.database"] = saved_db
                else:
                    sys.modules.pop("team_memory.storage.database", None)

            assert "exp-1" in compiler._exp_embeddings

    async def test_load_pg_data_emb_bytes(self, compiler):
        """Lines 257-258: emb is bytes path."""
        async with compiler:
            compiler._pg_url = "postgresql://test"

            mock_entity_result = MagicMock()
            mock_entity_result.fetchall.return_value = []

            emb_bytes = struct.pack("3f", 0.1, 0.2, 0.3)

            mock_emb_result = MagicMock()
            mock_row = MagicMock()
            mock_row.__getitem__ = lambda self, idx: "exp-1" if idx == 0 else emb_bytes
            mock_emb_result.fetchall.return_value = [mock_row]

            call_count = 0

            async def mock_execute(sql_str):
                nonlocal call_count
                call_count += 1
                if call_count == 1:
                    return mock_entity_result
                return mock_emb_result

            mock_session = AsyncMock()
            mock_session.execute = mock_execute
            mock_session.__aenter__ = AsyncMock(return_value=mock_session)
            mock_session.__aexit__ = AsyncMock(return_value=False)

            mock_get_session = MagicMock()
            mock_get_session.return_value.__aenter__ = AsyncMock(return_value=mock_session)
            mock_get_session.return_value.__aexit__ = AsyncMock(return_value=False)

            saved_db = sys.modules.pop("team_memory.storage.database", None)
            mock_db_module = MagicMock()
            mock_db_module.get_session = mock_get_session
            sys.modules["team_memory.storage.database"] = mock_db_module

            try:
                await compiler._load_pg_data("postgresql://test")
            finally:
                if saved_db is not None:
                    sys.modules["team_memory.storage.database"] = saved_db
                else:
                    sys.modules.pop("team_memory.storage.database", None)

            assert "exp-1" in compiler._exp_embeddings

    async def test_load_pg_data_emb_exception_continues(self, compiler):
        """Lines 259-260: exception during embedding parsing continues."""
        async with compiler:
            compiler._pg_url = "postgresql://test"

            mock_entity_result = MagicMock()
            mock_entity_result.fetchall.return_value = []

            mock_emb_result = MagicMock()
            mock_row = MagicMock()
            mock_row.__getitem__ = lambda self, idx: "exp-1" if idx == 0 else {"invalid": "type"}
            mock_emb_result.fetchall.return_value = [mock_row]

            call_count = 0

            async def mock_execute(sql_str):
                nonlocal call_count
                call_count += 1
                if call_count == 1:
                    return mock_entity_result
                return mock_emb_result

            mock_session = AsyncMock()
            mock_session.execute = mock_execute
            mock_session.__aenter__ = AsyncMock(return_value=mock_session)
            mock_session.__aexit__ = AsyncMock(return_value=False)

            mock_get_session = MagicMock()
            mock_get_session.return_value.__aenter__ = AsyncMock(return_value=mock_session)
            mock_get_session.return_value.__aexit__ = AsyncMock(return_value=False)

            saved_db = sys.modules.pop("team_memory.storage.database", None)
            mock_db_module = MagicMock()
            mock_db_module.get_session = mock_get_session
            sys.modules["team_memory.storage.database"] = mock_db_module

            try:
                await compiler._load_pg_data("postgresql://test")
            finally:
                if saved_db is not None:
                    sys.modules["team_memory.storage.database"] = saved_db
                else:
                    sys.modules.pop("team_memory.storage.database", None)

            assert "exp-1" not in compiler._exp_embeddings

    async def test_load_pg_data_numpy_not_available(self, compiler):
        """Lines 261-263: ImportError when numpy not available."""
        async with compiler:
            compiler._pg_url = "postgresql://test"

            mock_entity_result = MagicMock()
            mock_entity_result.fetchall.return_value = []

            mock_emb_result = MagicMock()
            mock_emb_result.fetchall.return_value = []

            call_count = 0

            async def mock_execute(sql_str):
                nonlocal call_count
                call_count += 1
                if call_count == 1:
                    return mock_entity_result
                return mock_emb_result

            mock_session = AsyncMock()
            mock_session.execute = mock_execute
            mock_session.__aenter__ = AsyncMock(return_value=mock_session)
            mock_session.__aexit__ = AsyncMock(return_value=False)

            mock_get_session = MagicMock()
            mock_get_session.return_value.__aenter__ = AsyncMock(return_value=mock_session)
            mock_get_session.return_value.__aexit__ = AsyncMock(return_value=False)

            saved_db = sys.modules.pop("team_memory.storage.database", None)
            mock_db_module = MagicMock()
            mock_db_module.get_session = mock_get_session
            sys.modules["team_memory.storage.database"] = mock_db_module

            try:
                # Force ImportError for numpy
                saved_numpy = sys.modules.pop("numpy", None)
                sys.modules["numpy"] = None
                try:
                    await compiler._load_pg_data("postgresql://test")
                finally:
                    if saved_numpy is not None:
                        sys.modules["numpy"] = saved_numpy
                    else:
                        sys.modules.pop("numpy", None)
            finally:
                if saved_db is not None:
                    sys.modules["team_memory.storage.database"] = saved_db
                else:
                    sys.modules.pop("team_memory.storage.database", None)

            assert compiler._exp_embeddings == {}

    async def test_load_pg_data_embedding_exception(self, compiler):
        """Lines 264-266: general exception during embedding loading."""
        async with compiler:
            compiler._pg_url = "postgresql://test"

            mock_entity_result = MagicMock()
            mock_entity_result.fetchall.return_value = []

            call_count = 0

            async def mock_execute(sql_str):
                nonlocal call_count
                call_count += 1
                if call_count == 1:
                    return mock_entity_result
                raise RuntimeError("DB connection lost")

            mock_session = AsyncMock()
            mock_session.execute = mock_execute
            mock_session.__aenter__ = AsyncMock(return_value=mock_session)
            mock_session.__aexit__ = AsyncMock(return_value=False)

            mock_get_session = MagicMock()
            mock_get_session.return_value.__aenter__ = AsyncMock(return_value=mock_session)
            mock_get_session.return_value.__aexit__ = AsyncMock(return_value=False)

            saved_db = sys.modules.pop("team_memory.storage.database", None)
            mock_db_module = MagicMock()
            mock_db_module.get_session = mock_get_session
            sys.modules["team_memory.storage.database"] = mock_db_module

            try:
                await compiler._load_pg_data("postgresql://test")
            finally:
                if saved_db is not None:
                    sys.modules["team_memory.storage.database"] = saved_db
                else:
                    sys.modules.pop("team_memory.storage.database", None)

            assert compiler._exp_embeddings == {}

    async def test_load_pg_data_entity_exception(self, compiler):
        """Lines 228-230: entity loading exception sets empty map."""
        async with compiler:
            compiler._pg_url = "postgresql://test"

            async def mock_execute(sql_str):
                raise RuntimeError("DB entity query failed")

            mock_session = AsyncMock()
            mock_session.execute = mock_execute
            mock_session.__aenter__ = AsyncMock(return_value=mock_session)
            mock_session.__aexit__ = AsyncMock(return_value=False)

            mock_get_session = MagicMock()
            mock_get_session.return_value.__aenter__ = AsyncMock(return_value=mock_session)
            mock_get_session.return_value.__aexit__ = AsyncMock(return_value=False)

            saved_db = sys.modules.pop("team_memory.storage.database", None)
            mock_db_module = MagicMock()
            mock_db_module.get_session = mock_get_session
            sys.modules["team_memory.storage.database"] = mock_db_module

            try:
                await compiler._load_pg_data("postgresql://test")
            finally:
                if saved_db is not None:
                    sys.modules["team_memory.storage.database"] = saved_db
                else:
                    sys.modules.pop("team_memory.storage.database", None)

            assert compiler._exp_entity_map == {}

    async def test_load_pg_data_string_embedding(self, compiler):
        """Lines 248-252: emb is string (JSON-encoded)."""
        async with compiler:
            compiler._pg_url = "postgresql://test"

            import json
            mock_entity_result = MagicMock()
            mock_entity_result.fetchall.return_value = []

            mock_emb_result = MagicMock()
            mock_row = MagicMock()
            mock_row.__getitem__ = lambda self, idx: "exp-1" if idx == 0 else json.dumps([0.1, 0.2, 0.3])
            mock_emb_result.fetchall.return_value = [mock_row]

            call_count = 0

            async def mock_execute(sql_str):
                nonlocal call_count
                call_count += 1
                if call_count == 1:
                    return mock_entity_result
                return mock_emb_result

            mock_session = AsyncMock()
            mock_session.execute = mock_execute
            mock_session.__aenter__ = AsyncMock(return_value=mock_session)
            mock_session.__aexit__ = AsyncMock(return_value=False)

            mock_get_session = MagicMock()
            mock_get_session.return_value.__aenter__ = AsyncMock(return_value=mock_session)
            mock_get_session.return_value.__aexit__ = AsyncMock(return_value=False)

            saved_db = sys.modules.pop("team_memory.storage.database", None)
            mock_db_module = MagicMock()
            mock_db_module.get_session = mock_get_session
            sys.modules["team_memory.storage.database"] = mock_db_module

            try:
                await compiler._load_pg_data("postgresql://test")
            finally:
                if saved_db is not None:
                    sys.modules["team_memory.storage.database"] = saved_db
                else:
                    sys.modules.pop("team_memory.storage.database", None)

            assert "exp-1" in compiler._exp_embeddings

    async def test_load_pg_data_emb_inner_exception(self, compiler):
        """Lines 259-260: inner exception during embedding type dispatch."""
        async with compiler:
            compiler._pg_url = "postgresql://test"

            mock_entity_result = MagicMock()
            mock_entity_result.fetchall.return_value = []

            # Create an object with tolist that raises on call
            mock_emb = MagicMock(spec=[])  # no tolist, not str, list, bytes
            mock_emb_result = MagicMock()
            mock_row = MagicMock()
            mock_row.__getitem__ = lambda self, idx: "exp-1" if idx == 0 else mock_emb
            mock_emb_result.fetchall.return_value = [mock_row]

            call_count = 0

            async def mock_execute(sql_str):
                nonlocal call_count
                call_count += 1
                if call_count == 1:
                    return mock_entity_result
                return mock_emb_result

            mock_session = AsyncMock()
            mock_session.execute = mock_execute
            mock_session.__aenter__ = AsyncMock(return_value=mock_session)
            mock_session.__aexit__ = AsyncMock(return_value=False)

            mock_get_session = MagicMock()
            mock_get_session.return_value.__aenter__ = AsyncMock(return_value=mock_session)
            mock_get_session.return_value.__aexit__ = AsyncMock(return_value=False)

            saved_db = sys.modules.pop("team_memory.storage.database", None)
            mock_db_module = MagicMock()
            mock_db_module.get_session = mock_get_session
            sys.modules["team_memory.storage.database"] = mock_db_module

            try:
                await compiler._load_pg_data("postgresql://test")
            finally:
                if saved_db is not None:
                    sys.modules["team_memory.storage.database"] = saved_db
                else:
                    sys.modules.pop("team_memory.storage.database", None)

            # The object falls through all isinstance checks and doesn't enter any branch
            assert "exp-1" not in compiler._exp_embeddings

    async def test_load_pg_data_emb_tolist_exception(self, compiler):
        """Lines 259-260: tolist() raises exception during embedding processing."""
        async with compiler:
            compiler._pg_url = "postgresql://test"

            mock_entity_result = MagicMock()
            mock_entity_result.fetchall.return_value = []

            # Create an object with tolist that raises
            mock_emb = MagicMock()
            mock_emb.tolist.side_effect = ValueError("tolist failed")

            mock_emb_result = MagicMock()
            mock_row = MagicMock()
            mock_row.__getitem__ = lambda self, idx: "exp-1" if idx == 0 else mock_emb
            mock_emb_result.fetchall.return_value = [mock_row]

            call_count = 0

            async def mock_execute(sql_str):
                nonlocal call_count
                call_count += 1
                if call_count == 1:
                    return mock_entity_result
                return mock_emb_result

            mock_session = AsyncMock()
            mock_session.execute = mock_execute
            mock_session.__aenter__ = AsyncMock(return_value=mock_session)
            mock_session.__aexit__ = AsyncMock(return_value=False)

            mock_get_session = MagicMock()
            mock_get_session.return_value.__aenter__ = AsyncMock(return_value=mock_session)
            mock_get_session.return_value.__aexit__ = AsyncMock(return_value=False)

            saved_db = sys.modules.pop("team_memory.storage.database", None)
            mock_db_module = MagicMock()
            mock_db_module.get_session = mock_get_session
            sys.modules["team_memory.storage.database"] = mock_db_module

            try:
                await compiler._load_pg_data("postgresql://test")
            finally:
                if saved_db is not None:
                    sys.modules["team_memory.storage.database"] = saved_db
                else:
                    sys.modules.pop("team_memory.storage.database", None)

            assert "exp-1" not in compiler._exp_embeddings

    async def test_load_pg_data_entity_success(self, compiler):
        """Lines 216-227: successful entity loading."""
        async with compiler:
            compiler._pg_url = "postgresql://test"

            mock_entity_result = MagicMock()
            mock_entity_row1 = MagicMock()
            mock_entity_row1.__getitem__ = lambda self, idx: "exp-1" if idx == 0 else "entity-a"
            mock_entity_row2 = MagicMock()
            mock_entity_row2.__getitem__ = lambda self, idx: "exp-1" if idx == 0 else "entity-b"
            mock_entity_result.fetchall.return_value = [mock_entity_row1, mock_entity_row2]

            mock_emb_result = MagicMock()
            mock_emb_result.fetchall.return_value = []

            call_count = 0

            async def mock_execute(sql_str):
                nonlocal call_count
                call_count += 1
                if call_count == 1:
                    return mock_entity_result
                return mock_emb_result

            mock_session = AsyncMock()
            mock_session.execute = mock_execute
            mock_session.__aenter__ = AsyncMock(return_value=mock_session)
            mock_session.__aexit__ = AsyncMock(return_value=False)

            mock_get_session = MagicMock()
            mock_get_session.return_value.__aenter__ = AsyncMock(return_value=mock_session)
            mock_get_session.return_value.__aexit__ = AsyncMock(return_value=False)

            saved_db = sys.modules.pop("team_memory.storage.database", None)
            mock_db_module = MagicMock()
            mock_db_module.get_session = mock_get_session
            sys.modules["team_memory.storage.database"] = mock_db_module

            try:
                await compiler._load_pg_data("postgresql://test")
            finally:
                if saved_db is not None:
                    sys.modules["team_memory.storage.database"] = saved_db
                else:
                    sys.modules.pop("team_memory.storage.database", None)

            assert "exp-1" in compiler._exp_entity_map
            assert compiler._exp_entity_map["exp-1"] == {"entity-a", "entity-b"}

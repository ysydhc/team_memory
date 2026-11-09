"""Tests for MemoryJanitor service.

Tests the automated cleanup and quality management functionality
with mocked database sessions.
"""

from __future__ import annotations

from contextlib import asynccontextmanager
from datetime import datetime, timedelta, timezone
from unittest.mock import AsyncMock, MagicMock

import pytest

pytestmark = pytest.mark.unit

from team_memory.config.janitor import JanitorConfig
from team_memory.services.maintenance.janitor import MemoryJanitor


def _utcnow() -> datetime:
    """Get current UTC datetime."""
    return datetime.now(timezone.utc)


class TestMemoryJanitor:
    """Tests for MemoryJanitor service."""

    @pytest.fixture
    def mock_config(self):
        """Mock configuration object."""
        return JanitorConfig(
            protection_period_days=10,
            auto_soft_delete_outdated=False,
            purge_soft_deleted_days=30,
            draft_expiry_days=7,
            personal_memory_retention_days=90,
        )

    @pytest.fixture
    def janitor(self, mock_config):
        """Create MemoryJanitor instance with mock config and mocked DB session."""
        j = MemoryJanitor(db_url="sqlite+aiosqlite://", config=mock_config)

        mock_scalars = MagicMock()
        mock_scalars.all.return_value = []

        mock_result = MagicMock()
        mock_result.scalars.return_value = mock_scalars

        mock_session = AsyncMock()
        mock_session.execute.return_value = mock_result
        mock_session.commit = AsyncMock()

        @asynccontextmanager
        async def _fake_session():
            yield mock_session

        j._session = _fake_session
        return j

    @pytest.mark.asyncio
    async def test_init(self, mock_config):
        """Test janitor initialization."""
        janitor = MemoryJanitor(db_url="test_url", config=mock_config)
        assert janitor._db_url == "test_url"
        assert janitor._config == mock_config

    def test_get_config(self, mock_config):
        """Test configuration value retrieval."""
        janitor = MemoryJanitor(db_url="test", config=mock_config)
        assert janitor._get_config("protection_period_days") == 10

        assert janitor._get_config("missing_key", "default_value") == "default_value"

        janitor_no_config = MemoryJanitor(db_url="test", config=None)
        assert janitor_no_config._get_config("any_key", "default") == "default"

    @pytest.mark.asyncio
    async def test_run_score_decay_no_experiences(self, janitor):
        """Test score decay with no experiences returns zero updates."""
        result = await janitor.run_score_decay()
        assert result["updated_count"] == 0

    @pytest.mark.asyncio
    async def test_sweep_outdated_no_experiences(self, janitor):
        """Test outdated sweep with no experiences returns empty result."""
        result = await janitor.sweep_outdated()
        assert result["found_count"] == 0
        assert result["deleted_count"] == 0

    @pytest.mark.asyncio
    async def test_purge_soft_deleted_no_experiences(self, janitor):
        """Test soft-deleted purge with no experiences returns empty result."""
        result = await janitor.purge_soft_deleted()
        assert result["purged_count"] == 0

    @pytest.mark.asyncio
    async def test_expire_drafts_no_experiences(self, janitor):
        """Test draft expiration with no experiences returns empty result."""
        result = await janitor.expire_drafts()
        assert result["expired_count"] == 0

    @pytest.mark.asyncio
    async def test_prune_personal_memory_no_memories(self, janitor):
        """Test personal memory pruning with no memories returns zero."""
        result = await janitor.prune_personal_memory()
        assert result["pruned_count"] == 0

    @pytest.mark.asyncio
    async def test_run_all_structure(self, janitor):
        """Test that run_all returns proper structure even on failure."""
        result = await janitor.run_all()

        # Should have basic structure even if operations fail
        assert "project" in result
        assert "completed_at" in result
        assert "operations" in result
        assert "config" in result

        # Config should have expected keys
        config = result["config"]
        assert "protection_days" in config
        assert "auto_soft_delete" in config
        assert "purge_days" in config
        assert "draft_expiry_days" in config
        assert "personal_memory_days" in config

    def test_default_config_auto_soft_delete_enabled(self):
        """JanitorConfig enables auto_soft_delete_outdated by default."""
        cfg = JanitorConfig()
        assert cfg.auto_soft_delete_outdated is True

    @pytest.mark.asyncio
    async def test_run_score_decay_continues_after_previous_scoring(self):
        """Experiences past protection period continue decaying on subsequent days."""
        now = _utcnow()
        exp = MagicMock()
        exp.id = "exp-decay-1"
        exp.created_at = now - timedelta(days=25)
        exp.last_scored_at = now - timedelta(days=2)
        exp.quality_score = 10.0
        exp.quality_tier = "Outdated"

        j = MemoryJanitor(db_url="sqlite+aiosqlite://", config=JanitorConfig())
        mock_scalars = MagicMock()
        mock_scalars.all.return_value = [exp]
        mock_result = MagicMock()
        mock_result.scalars.return_value = mock_scalars
        mock_session = AsyncMock()
        mock_session.execute.return_value = mock_result
        mock_session.commit = AsyncMock()

        @asynccontextmanager
        async def _fake_session():
            yield mock_session

        j._session = _fake_session
        result = await j.run_score_decay()

        assert result["updated_count"] == 1
        assert exp.quality_score == 9.0
        assert exp.quality_tier == "Outdated"

    @pytest.mark.asyncio
    async def test_sweep_outdated_auto_soft_delete_marks_deleted(self):
        """sweep_outdated with auto_soft_delete=True marks items as deleted."""
        exp = MagicMock()
        exp.id = "exp-outdated-1"
        exp.is_deleted = False
        exp.deleted_at = None

        j = MemoryJanitor(db_url="sqlite+aiosqlite://", config=JanitorConfig())
        mock_scalars = MagicMock()
        mock_scalars.all.return_value = [exp]
        mock_result = MagicMock()
        mock_result.scalars.return_value = mock_scalars
        mock_session = AsyncMock()
        mock_session.execute.return_value = mock_result
        mock_session.commit = AsyncMock()

        @asynccontextmanager
        async def _fake_session():
            yield mock_session

        j._session = _fake_session
        result = await j.sweep_outdated(auto_soft_delete=True)

        assert result["found_count"] == 1
        assert result["deleted_count"] == 1
        assert exp.is_deleted is True
        assert exp.deleted_at is not None

    @pytest.mark.asyncio
    async def test_prune_personal_memory_semantic_compaction(self):
        """prune_personal_memory compacts redundant static memories by vector similarity."""
        now = _utcnow()
        m1 = MagicMock()
        m1.id = "m1"
        m1.user_id = "user1"
        m1.persona = "default"
        m1.profile_kind = "static"
        m1.priority = 70
        m1.content = "回复使用简体中文，保持简洁直接"
        m1.embedding = [1.0, 0.0, 0.0]
        m1.updated_at = now

        m2 = MagicMock()
        m2.id = "m2"
        m2.user_id = "user1"
        m2.persona = "default"
        m2.profile_kind = "static"
        m2.priority = 60
        m2.content = "使用简体中文回复，简洁为主"
        m2.embedding = [0.99, 0.05, 0.0]
        m2.updated_at = now - timedelta(hours=1)

        j = MemoryJanitor(db_url="sqlite+aiosqlite://", config=JanitorConfig())

        res_dynamic = MagicMock()
        res_dynamic.scalars.return_value.all.return_value = []
        res_static = MagicMock()
        res_static.scalars.return_value.all.return_value = [m1, m2]

        mock_session = AsyncMock()
        mock_session.execute = AsyncMock(side_effect=[res_dynamic, res_static])
        mock_session.commit = AsyncMock()
        mock_session.delete = AsyncMock()

        @asynccontextmanager
        async def _fake_session():
            yield mock_session

        j._session = _fake_session
        result = await j.prune_personal_memory()

        assert result["expired_dynamic_count"] == 0
        assert result["deduped_static_count"] == 1
        assert result["pruned_count"] == 1
        assert m1.priority == 80
        mock_session.delete.assert_called_once_with(m2)

    @pytest.mark.asyncio
    async def test_maintain_archives_normalizes_and_deduplicates(self):
        """maintain_archives normalizes content_type/project, promotes complete drafts, and deduplicates."""
        now = _utcnow()
        a1 = MagicMock()
        a1.id = "arch-1"
        a1.title = "Android WebView Blank Fix"
        a1.project = "adx-sdk-android"
        a1.content_type = "architecture_refactor"
        a1.status = "draft"
        a1.solution_doc = "A" * 250
        a1.overview = "Detailed overview of the Android WebView blank screen fix."
        a1.embedding = [1.0, 0.0, 0.0]
        a1.created_at = now - timedelta(days=10)
        a1.updated_at = now - timedelta(days=10)

        a2 = MagicMock()
        a2.id = "arch-2"
        a2.title = "Android WebView Blank Fix"
        a2.project = "adx_sdk_android"
        a2.content_type = "session_archive"
        a2.status = "draft"
        a2.solution_doc = "A" * 220
        a2.overview = "Duplicate overview."
        a2.embedding = [0.99, 0.01, 0.0]
        a2.created_at = now - timedelta(days=11)
        a2.updated_at = now - timedelta(days=11)

        j = MemoryJanitor(db_url="sqlite+aiosqlite://", config=JanitorConfig())
        res_archives = MagicMock()
        res_archives.scalars.return_value.all.return_value = [a1, a2]
        res_links = MagicMock()
        res_links.scalars.return_value.all.return_value = []

        mock_session = AsyncMock()
        mock_session.execute = AsyncMock(side_effect=[res_archives, res_links])
        mock_session.commit = AsyncMock()
        mock_session.delete = AsyncMock()

        @asynccontextmanager
        async def _fake_session():
            yield mock_session

        j._session = _fake_session
        result = await j.maintain_archives()

        assert result["normalized_type_count"] == 1
        assert result["normalized_project_count"] == 1
        assert result["promoted_draft_count"] == 1
        assert result["deduped_archive_count"] == 1
        assert a1.content_type == "tech_design"
        assert a1.project == "adx_sdk_android"
        assert a1.status == "published"
        mock_session.delete.assert_called_once_with(a2)


class TestMemoryJanitorIntegration:
    """Integration tests for MemoryJanitor with real database operations.

    These tests would require a proper test database setup.
    For now, they serve as documentation of expected behavior.
    """

    @pytest.mark.skip(reason="Requires full database setup")
    @pytest.mark.asyncio
    async def test_score_decay_with_real_data(self):
        """Test score decay with actual database data."""
        # This would test:
        # 1. Create experiences with different ages and scores
        # 2. Run score decay
        # 3. Verify scores are updated according to rules
        # 4. Verify tiers are updated correctly
        pass

    @pytest.mark.skip(reason="Requires full database setup")
    @pytest.mark.asyncio
    async def test_sweep_outdated_with_auto_delete(self):
        """Test outdated sweep with auto-delete enabled."""
        # This would test:
        # 1. Create outdated experiences (score <= 0, tier = "Outdated")
        # 2. Run sweep with auto_soft_delete=True
        # 3. Verify experiences are soft-deleted
        pass

    @pytest.mark.skip(reason="Requires full database setup")
    @pytest.mark.asyncio
    async def test_purge_soft_deleted_old_records(self):
        """Test purging of old soft-deleted records."""
        # This would test:
        # 1. Create soft-deleted experiences with old deleted_at dates
        # 2. Run purge operation
        # 3. Verify records are permanently deleted
        pass

    @pytest.mark.skip(reason="Requires full database setup")
    @pytest.mark.asyncio
    async def test_expire_old_drafts(self):
        """Test expiration of old draft experiences."""
        # This would test:
        # 1. Create old draft experiences
        # 2. Run draft expiration
        # 3. Verify drafts are soft-deleted
        pass

    @pytest.mark.skip(reason="Requires full database setup")
    @pytest.mark.asyncio
    async def test_prune_old_personal_memories(self):
        """Test pruning of old personal memories."""
        # This would test:
        # 1. Create old dynamic personal memories
        # 2. Run personal memory pruning
        # 3. Verify old memories are deleted, static ones preserved
        pass

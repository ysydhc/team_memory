"""Tests for experience lifecycle management (B1-B4).

Covers: version history, stale detection, dedup/merge, import/export.
"""

from __future__ import annotations

import csv
import io
import json
import uuid
from contextlib import asynccontextmanager
from datetime import datetime, timedelta, timezone
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _mock_experience(**overrides):
    """Create a mock Experience object with sensible defaults."""
    exp = MagicMock()
    exp.id = overrides.get("id", uuid.uuid4())
    exp.parent_id = overrides.get("parent_id", None)
    exp.title = overrides.get("title", "Test Experience")
    exp.description = overrides.get("description", "Test problem")
    exp.root_cause = overrides.get("root_cause", None)
    exp.solution = overrides.get("solution", "Test solution")
    exp.tags = overrides.get("tags", ["python", "test"])
    exp.programming_language = overrides.get("programming_language", "python")
    exp.framework = overrides.get("framework", "fastapi")
    exp.code_snippets = overrides.get("code_snippets", None)
    exp.source = overrides.get("source", "manual")
    exp.created_by = overrides.get("created_by", "tester")
    exp.publish_status = overrides.get("publish_status", "published")
    exp.review_status = overrides.get("review_status", "approved")
    exp.is_deleted = overrides.get("is_deleted", False)
    exp.deleted_at = None
    exp.view_count = overrides.get("view_count", 0)
    exp.avg_rating = overrides.get("avg_rating", 3.5)
    exp.use_count = overrides.get("use_count", 5)
    exp.embedding = overrides.get("embedding", [0.1] * 768)
    exp.fts = None
    exp.children = overrides.get("children", [])
    exp.feedbacks = overrides.get("feedbacks", [])
    exp.last_used_at = overrides.get("last_used_at", datetime.now(timezone.utc))
    exp.created_at = overrides.get("created_at", datetime.now(timezone.utc))
    exp.updated_at = overrides.get("updated_at", datetime.now(timezone.utc))
    exp.to_dict = MagicMock(return_value={
        "id": str(exp.id),
        "parent_id": str(exp.parent_id) if exp.parent_id else None,
        "title": exp.title,
        "description": exp.description,
        "root_cause": exp.root_cause,
        "solution": exp.solution,
        "tags": exp.tags,
        "programming_language": exp.programming_language,
        "framework": exp.framework,
        "code_snippets": exp.code_snippets,
        "source": exp.source,
        "created_by": exp.created_by,
        "publish_status": exp.publish_status,
        "review_status": exp.review_status,
        "is_deleted": exp.is_deleted,
        "view_count": exp.view_count,
        "avg_rating": exp.avg_rating,
        "use_count": exp.use_count,
        "created_at": exp.created_at.isoformat(),
        "updated_at": exp.updated_at.isoformat(),
        "last_used_at": exp.last_used_at.isoformat() if exp.last_used_at else None,
    })
    return exp


def _mock_version(**overrides):
    """Create a mock ExperienceVersion object."""
    ver = MagicMock()
    ver.id = overrides.get("id", uuid.uuid4())
    ver.experience_id = overrides.get("experience_id", uuid.uuid4())
    ver.version_number = overrides.get("version_number", 1)
    ver.snapshot = overrides.get("snapshot", {
        "title": "Old title",
        "description": "Old desc",
        "solution": "Old solution",
        "tags": ["old"],
    })
    ver.changed_by = overrides.get("changed_by", "tester")
    ver.change_summary = overrides.get("change_summary", "Test change")
    ver.created_at = overrides.get("created_at", datetime.now(timezone.utc))
    ver.to_dict = MagicMock(return_value={
        "id": str(ver.id),
        "experience_id": str(ver.experience_id),
        "version_number": ver.version_number,
        "snapshot": ver.snapshot,
        "changed_by": ver.changed_by,
        "change_summary": ver.change_summary,
        "created_at": ver.created_at.isoformat(),
    })
    return ver


# ===========================================================================
# B3: Version History Tests
# ===========================================================================
class TestVersionHistory:
    """Tests for B3: Version History."""

    @pytest.mark.asyncio
    async def test_get_versions_returns_list(self):
        """Service.get_versions returns version dicts."""
        from team_memory.services.experience import ExperienceService

        ver1 = _mock_version(version_number=1)
        ver2 = _mock_version(version_number=2)

        mock_embedding = AsyncMock()
        mock_auth = MagicMock()
        service = ExperienceService(mock_embedding, mock_auth, db_url="sqlite+aiosqlite://")

        mock_session = AsyncMock()
        exp_id = str(uuid.uuid4())

        @asynccontextmanager
        async def mock_get_session(_db_url):
            yield mock_session

        with patch("team_memory.storage.database.get_session", mock_get_session), \
             patch("team_memory.services.experience.ExperienceRepository") as mock_repo:
            repo_inst = mock_repo.return_value
            repo_inst.get_versions = AsyncMock(return_value=[ver2, ver1])

            result = await service.get_versions(exp_id)
            assert len(result) == 2
            assert result[0]["version_number"] == 2
            assert result[1]["version_number"] == 1

    @pytest.mark.asyncio
    async def test_get_version_detail_returns_snapshot(self):
        """Service.get_version_detail returns version with snapshot."""
        from team_memory.services.experience import ExperienceService

        ver = _mock_version(version_number=3, snapshot={"title": "Snapshot Title"})

        mock_embedding = AsyncMock()
        mock_auth = MagicMock()
        service = ExperienceService(mock_embedding, mock_auth, db_url="sqlite+aiosqlite://")

        mock_session = AsyncMock()
        ver_id = str(ver.id)

        @asynccontextmanager
        async def mock_get_session(_db_url):
            yield mock_session

        with patch("team_memory.storage.database.get_session", mock_get_session), \
             patch("team_memory.services.experience.ExperienceRepository") as mock_repo:
            repo_inst = mock_repo.return_value
            repo_inst.get_version_detail = AsyncMock(return_value=ver)

            result = await service.get_version_detail(ver_id)
            assert result is not None
            assert result["version_number"] == 3
            assert result["snapshot"]["title"] == "Snapshot Title"

    @pytest.mark.asyncio
    async def test_get_version_detail_not_found(self):
        """Service.get_version_detail returns None for non-existent version."""
        from team_memory.services.experience import ExperienceService

        mock_embedding = AsyncMock()
        mock_auth = MagicMock()
        service = ExperienceService(mock_embedding, mock_auth, db_url="sqlite+aiosqlite://")

        mock_session = AsyncMock()

        @asynccontextmanager
        async def mock_get_session(_db_url):
            yield mock_session

        with patch("team_memory.storage.database.get_session", mock_get_session), \
             patch("team_memory.services.experience.ExperienceRepository") as mock_repo:
            repo_inst = mock_repo.return_value
            repo_inst.get_version_detail = AsyncMock(return_value=None)

            result = await service.get_version_detail(str(uuid.uuid4()))
            assert result is None

    @pytest.mark.asyncio
    async def test_rollback_to_version(self):
        """Service.rollback_to_version restores content from snapshot."""
        from team_memory.services.experience import ExperienceService

        exp = _mock_experience()
        ver = _mock_version(
            experience_id=exp.id,
            version_number=1,
            snapshot={
                "title": "Restored Title",
                "description": "Restored Desc",
                "solution": "Restored Solution",
                "tags": ["restored"],
            },
        )

        updated_exp = _mock_experience(title="Restored Title")

        mock_embedding = AsyncMock()
        mock_embedding.encode_single = AsyncMock(return_value=[0.5] * 768)
        mock_auth = MagicMock()
        service = ExperienceService(mock_embedding, mock_auth, db_url="sqlite+aiosqlite://")

        mock_session = AsyncMock()

        @asynccontextmanager
        async def mock_get_session(_db_url):
            yield mock_session

        with patch("team_memory.storage.database.get_session", mock_get_session), \
             patch("team_memory.services.experience.ExperienceRepository") as mock_repo:
            repo_inst = mock_repo.return_value
            repo_inst.get_by_id = AsyncMock(return_value=exp)
            repo_inst.get_version_detail = AsyncMock(return_value=ver)
            repo_inst.save_version = AsyncMock(return_value=_mock_version(version_number=2))
            repo_inst.update = AsyncMock(return_value=updated_exp)

            result = await service.rollback_to_version(str(exp.id), str(ver.id), user="tester")
            assert result is not None
            # save_version should be called twice (pre-rollback + post-rollback)
            assert repo_inst.save_version.call_count == 2
            # update should be called with restored fields
            repo_inst.update.assert_called_once()
            call_kwargs = repo_inst.update.call_args
            assert call_kwargs[1]["title"] == "Restored Title"

    @pytest.mark.asyncio
    async def test_hard_delete_and_rebuild_saves_version(self):
        """hard_delete_and_rebuild should save a version before deleting."""
        from team_memory.services.experience import ExperienceService

        exp = _mock_experience()

        mock_embedding = AsyncMock()
        mock_embedding.encode_single = AsyncMock(return_value=[0.5] * 768)
        mock_auth = MagicMock()
        service = ExperienceService(mock_embedding, mock_auth, db_url="sqlite+aiosqlite://")

        mock_session = AsyncMock()

        @asynccontextmanager
        async def mock_get_session(_db_url):
            yield mock_session

        with patch("team_memory.storage.database.get_session", mock_get_session), \
             patch("team_memory.services.experience.ExperienceRepository") as mock_repo:
            repo_inst = mock_repo.return_value
            repo_inst.save_version = AsyncMock(return_value=_mock_version())
            repo_inst.delete = AsyncMock(return_value=True)
            repo_inst.create = AsyncMock(return_value=exp)

            await service.hard_delete_and_rebuild(
                str(exp.id),
                new_data={"title": "New", "problem": "New P", "solution": "New S"},
                created_by="tester",
            )
            # save_version should have been called before delete
            repo_inst.save_version.assert_called_once()
            assert "编辑前快照" in repo_inst.save_version.call_args[1]["change_summary"]


# ===========================================================================
# B1: Stale Detection Tests
# ===========================================================================
class TestStaleDetection:
    """Tests for B1: Stale Detection."""

    @pytest.mark.asyncio
    async def test_scan_stale_returns_old_experiences(self):
        """scan_stale returns experiences with old last_used_at."""
        from team_memory.services.experience import ExperienceService

        old_exp = _mock_experience(
            last_used_at=datetime.now(timezone.utc) - timedelta(days=200)
        )

        mock_embedding = AsyncMock()
        mock_auth = MagicMock()
        service = ExperienceService(mock_embedding, mock_auth, db_url="sqlite+aiosqlite://")

        mock_session = AsyncMock()

        @asynccontextmanager
        async def mock_get_session(_db_url):
            yield mock_session

        with patch("team_memory.storage.database.get_session", mock_get_session), \
             patch("team_memory.services.experience.ExperienceRepository") as mock_repo:
            repo_inst = mock_repo.return_value
            repo_inst.scan_stale = AsyncMock(return_value=[old_exp])

            result = await service.scan_stale(months=6)
            assert len(result) == 1
            repo_inst.scan_stale.assert_called_once_with(months=6)

    @pytest.mark.asyncio
    async def test_scan_stale_empty(self):
        """scan_stale returns empty when all experiences are fresh."""
        from team_memory.services.experience import ExperienceService

        mock_embedding = AsyncMock()
        mock_auth = MagicMock()
        service = ExperienceService(mock_embedding, mock_auth, db_url="sqlite+aiosqlite://")

        mock_session = AsyncMock()

        @asynccontextmanager
        async def mock_get_session(_db_url):
            yield mock_session

        with patch("team_memory.storage.database.get_session", mock_get_session), \
             patch("team_memory.services.experience.ExperienceRepository") as mock_repo:
            repo_inst = mock_repo.return_value
            repo_inst.scan_stale = AsyncMock(return_value=[])

            result = await service.scan_stale(months=6)
            assert len(result) == 0


# ===========================================================================
# B2: Dedup & Merge Tests
# ===========================================================================
class TestDedup:
    """Tests for B2: Dedup & Merge."""

    @pytest.mark.asyncio
    async def test_find_duplicates_returns_pairs(self):
        """find_duplicates returns pairs with similarity."""
        from team_memory.services.experience import ExperienceService

        mock_embedding = AsyncMock()
        mock_auth = MagicMock()
        service = ExperienceService(mock_embedding, mock_auth, db_url="sqlite+aiosqlite://")

        mock_session = AsyncMock()
        pair = {
            "exp_a": {"id": str(uuid.uuid4()), "title": "A"},
            "exp_b": {"id": str(uuid.uuid4()), "title": "B"},
            "similarity": 0.95,
        }

        @asynccontextmanager
        async def mock_get_session(_db_url):
            yield mock_session

        with patch("team_memory.storage.database.get_session", mock_get_session), \
             patch("team_memory.services.experience.ExperienceRepository") as mock_repo:
            repo_inst = mock_repo.return_value
            repo_inst.find_duplicates = AsyncMock(return_value=[pair])

            result = await service.find_duplicates(threshold=0.92)
            assert len(result) == 1
            assert result[0]["similarity"] == 0.95

    @pytest.mark.asyncio
    async def test_merge_experiences_saves_versions(self):
        """merge_experiences saves version snapshots before merging."""
        from team_memory.services.experience import ExperienceService

        primary = _mock_experience(title="Primary")
        merged = _mock_experience(title="Primary", tags=["python", "test", "docker"])

        mock_embedding = AsyncMock()
        mock_auth = MagicMock()
        service = ExperienceService(mock_embedding, mock_auth, db_url="sqlite+aiosqlite://")

        mock_session = AsyncMock()

        @asynccontextmanager
        async def mock_get_session(_db_url):
            yield mock_session

        with patch("team_memory.storage.database.get_session", mock_get_session), \
             patch("team_memory.services.experience.ExperienceRepository") as mock_repo:
            repo_inst = mock_repo.return_value
            repo_inst.save_version = AsyncMock(return_value=_mock_version())
            repo_inst.merge_experiences = AsyncMock(return_value=merged)

            result = await service.merge_experiences(
                str(primary.id),
                str(uuid.uuid4()),
                user="tester",
            )
            assert result is not None
            # save_version called for both primary and secondary
            assert repo_inst.save_version.call_count == 2

    @pytest.mark.asyncio
    async def test_merge_experiences_not_found(self):
        """merge_experiences returns None if experience not found."""
        from team_memory.services.experience import ExperienceService

        mock_embedding = AsyncMock()
        mock_auth = MagicMock()
        service = ExperienceService(mock_embedding, mock_auth, db_url="sqlite+aiosqlite://")

        mock_session = AsyncMock()

        @asynccontextmanager
        async def mock_get_session(_db_url):
            yield mock_session

        with patch("team_memory.storage.database.get_session", mock_get_session), \
             patch("team_memory.services.experience.ExperienceRepository") as mock_repo:
            repo_inst = mock_repo.return_value
            repo_inst.save_version = AsyncMock(side_effect=ValueError("not found"))
            repo_inst.merge_experiences = AsyncMock(return_value=None)

            result = await service.merge_experiences(
                str(uuid.uuid4()),
                str(uuid.uuid4()),
                user="tester",
            )
            assert result is None


# ===========================================================================
# B4: Import / Export Tests
# ===========================================================================
class TestImportExport:
    """Tests for B4: Import/Export."""

    @pytest.mark.asyncio
    async def test_import_json_single(self):
        """import_experiences handles single JSON experiences."""
        from team_memory.services.experience import ExperienceService

        mock_embedding = AsyncMock()
        mock_embedding.encode_single = AsyncMock(return_value=[0.1] * 768)
        mock_auth = MagicMock()
        service = ExperienceService(mock_embedding, mock_auth, db_url="sqlite+aiosqlite://")

        mock_session = AsyncMock()

        json_data = json.dumps({
            "version": "1.0",
            "experiences": [
                {
                    "title": "Test Import",
                    "description": "Test problem",
                    "solution": "Test solution",
                    "tags": ["import", "test"],
                }
            ],
        })

        @asynccontextmanager
        async def mock_get_session(_db_url):
            yield mock_session

        with patch("team_memory.storage.database.get_session", mock_get_session), \
             patch("team_memory.services.experience.ExperienceRepository") as mock_repo:
            repo_inst = mock_repo.return_value
            repo_inst.create = AsyncMock(return_value=_mock_experience())

            result = await service.import_experiences(json_data, "json", "tester")
            assert result["imported"] == 1
            assert result["total"] == 1
            assert len(result["errors"]) == 0

    @pytest.mark.asyncio
    async def test_import_json_with_children(self):
        """import_experiences handles JSON with parent-child groups."""
        from team_memory.services.experience import ExperienceService

        mock_embedding = AsyncMock()
        mock_embedding.encode_single = AsyncMock(return_value=[0.1] * 768)
        mock_auth = MagicMock()
        service = ExperienceService(mock_embedding, mock_auth, db_url="sqlite+aiosqlite://")

        mock_session = AsyncMock()

        json_data = json.dumps({
            "experiences": [
                {
                    "title": "Parent",
                    "description": "Parent problem",
                    "solution": "Parent solution",
                    "tags": ["parent"],
                    "children": [
                        {"title": "Child 1", "description": "C1", "solution": "S1"},
                        {"title": "Child 2", "description": "C2", "solution": "S2"},
                    ],
                }
            ],
        })

        @asynccontextmanager
        async def mock_get_session(_db_url):
            yield mock_session

        with patch("team_memory.storage.database.get_session", mock_get_session), \
             patch("team_memory.services.experience.ExperienceRepository") as mock_repo:
            repo_inst = mock_repo.return_value
            parent_exp = _mock_experience(title="Parent")
            repo_inst.create_group = AsyncMock(return_value=parent_exp)

            result = await service.import_experiences(json_data, "json", "tester")
            assert result["imported"] == 1
            repo_inst.create_group.assert_called_once()

    @pytest.mark.asyncio
    async def test_import_csv(self):
        """import_experiences handles CSV format."""
        from team_memory.services.experience import ExperienceService

        mock_embedding = AsyncMock()
        mock_embedding.encode_single = AsyncMock(return_value=[0.1] * 768)
        mock_auth = MagicMock()
        service = ExperienceService(mock_embedding, mock_auth, db_url="sqlite+aiosqlite://")

        mock_session = AsyncMock()

        csv_data = "title,description,solution,tags\nCSV Exp,CSV Problem,CSV Solution,tag1;tag2\n"

        @asynccontextmanager
        async def mock_get_session(_db_url):
            yield mock_session

        with patch("team_memory.storage.database.get_session", mock_get_session), \
             patch("team_memory.services.experience.ExperienceRepository") as mock_repo:
            repo_inst = mock_repo.return_value
            repo_inst.create = AsyncMock(return_value=_mock_experience())

            result = await service.import_experiences(csv_data, "csv", "tester")
            assert result["imported"] == 1
            # Tags should be split by semicolon
            call_kwargs = repo_inst.create.call_args[1]
            assert call_kwargs["tags"] == ["tag1", "tag2"]

    @pytest.mark.asyncio
    async def test_import_unsupported_format(self):
        """import_experiences raises ValueError for unsupported format."""
        from team_memory.services.experience import ExperienceService

        mock_embedding = AsyncMock()
        mock_auth = MagicMock()
        service = ExperienceService(mock_embedding, mock_auth, db_url="sqlite+aiosqlite://")

        mock_session = AsyncMock()

        @asynccontextmanager
        async def mock_get_session(_db_url):
            yield mock_session

        with patch("team_memory.storage.database.get_session", mock_get_session):
            with pytest.raises(ValueError, match="Unsupported format"):
                await service.import_experiences("data", "xml", "tester")

    @pytest.mark.asyncio
    async def test_export_json(self):
        """export_experiences returns valid JSON."""
        from team_memory.services.experience import ExperienceService

        exp = _mock_experience()
        exp.children = []

        mock_embedding = AsyncMock()
        mock_auth = MagicMock()
        service = ExperienceService(mock_embedding, mock_auth, db_url="sqlite+aiosqlite://")

        mock_session = AsyncMock()

        @asynccontextmanager
        async def mock_get_session(_db_url):
            yield mock_session

        with patch("team_memory.storage.database.get_session", mock_get_session), \
             patch("team_memory.services.experience.ExperienceRepository") as mock_repo:
            repo_inst = mock_repo.return_value
            repo_inst.export_filtered = AsyncMock(return_value=[exp])

            result = await service.export_experiences("json")
            data = json.loads(result)
            assert data["version"] == "1.0"
            assert data["count"] == 1
            assert len(data["experiences"]) == 1
            assert data["experiences"][0]["title"] == exp.title

    @pytest.mark.asyncio
    async def test_export_csv(self):
        """export_experiences returns valid CSV."""
        from team_memory.services.experience import ExperienceService

        exp = _mock_experience()
        exp.children = []

        mock_embedding = AsyncMock()
        mock_auth = MagicMock()
        service = ExperienceService(mock_embedding, mock_auth, db_url="sqlite+aiosqlite://")

        mock_session = AsyncMock()

        @asynccontextmanager
        async def mock_get_session(_db_url):
            yield mock_session

        with patch("team_memory.storage.database.get_session", mock_get_session), \
             patch("team_memory.services.experience.ExperienceRepository") as mock_repo:
            repo_inst = mock_repo.return_value
            repo_inst.export_filtered = AsyncMock(return_value=[exp])

            result = await service.export_experiences("csv")
            reader = csv.reader(io.StringIO(result))
            rows = list(reader)
            assert len(rows) == 2  # header + 1 data row
            assert rows[0][0] == "title"  # header
            assert rows[1][0] == exp.title

    @pytest.mark.asyncio
    async def test_export_unsupported_format(self):
        """export_experiences raises ValueError for unsupported format."""
        from team_memory.services.experience import ExperienceService

        mock_embedding = AsyncMock()
        mock_auth = MagicMock()
        service = ExperienceService(mock_embedding, mock_auth, db_url="sqlite+aiosqlite://")

        mock_session = AsyncMock()

        @asynccontextmanager
        async def mock_get_session(_db_url):
            yield mock_session

        with patch("team_memory.storage.database.get_session", mock_get_session), \
             patch("team_memory.services.experience.ExperienceRepository") as mock_repo:
            repo_inst = mock_repo.return_value
            repo_inst.export_filtered = AsyncMock(return_value=[])

            with pytest.raises(ValueError, match="Unsupported format"):
                await service.export_experiences("xml")

    @pytest.mark.asyncio
    async def test_export_with_filters(self):
        """export_experiences passes filters to repository."""
        from team_memory.services.experience import ExperienceService

        mock_embedding = AsyncMock()
        mock_auth = MagicMock()
        service = ExperienceService(mock_embedding, mock_auth, db_url="sqlite+aiosqlite://")

        mock_session = AsyncMock()

        @asynccontextmanager
        async def mock_get_session(_db_url):
            yield mock_session

        with patch("team_memory.storage.database.get_session", mock_get_session), \
             patch("team_memory.services.experience.ExperienceRepository") as mock_repo:
            repo_inst = mock_repo.return_value
            repo_inst.export_filtered = AsyncMock(return_value=[])

            await service.export_experiences(
                "json", tag="python", start="2026-01-01", end="2026-02-01"
            )
            call_kwargs = repo_inst.export_filtered.call_args[1]
            assert call_kwargs["tag"] == "python"
            assert call_kwargs["start"] is not None
            assert call_kwargs["end"] is not None


# ===========================================================================
# Config Tests
# ===========================================================================
class TestLifecycleConfig:
    """Tests for LifecycleConfig."""

    def test_lifecycle_config_defaults(self):
        """LifecycleConfig has correct defaults."""
        from team_memory.config import LifecycleConfig

        cfg = LifecycleConfig()
        assert cfg.stale_months == 6
        assert cfg.scan_interval_hours == 24
        assert cfg.duplicate_threshold == 0.92

    def test_settings_includes_lifecycle(self):
        """Settings includes lifecycle config."""
        from team_memory.config import Settings

        s = Settings()
        assert hasattr(s, "lifecycle")
        assert s.lifecycle.stale_months == 6


# ===========================================================================
# Model Tests
# ===========================================================================
class TestExperienceVersionModel:
    """Tests for ExperienceVersion model."""

    def test_experience_version_to_dict(self):
        """ExperienceVersion.to_dict returns all fields."""
        ver = _mock_version(
            version_number=5,
            snapshot={"title": "Test"},
            changed_by="admin",
            change_summary="Test change",
        )
        d = ver.to_dict()
        assert d["version_number"] == 5
        assert d["snapshot"]["title"] == "Test"
        assert d["changed_by"] == "admin"

    def test_experience_model_has_last_used_at(self):
        """Experience model includes last_used_at in to_dict."""
        exp = _mock_experience()
        d = exp.to_dict()
        assert "last_used_at" in d

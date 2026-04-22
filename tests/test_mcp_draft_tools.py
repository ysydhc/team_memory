"""Tests for memory_draft_save and memory_draft_publish MCP tools."""

from __future__ import annotations

import json
import uuid
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from team_memory.server import mcp
from tests.conftest import _O, _patch_user

# ============================================================
# memory_draft_save
# ============================================================


class TestMemoryDraftSave:
    """Test memory_draft_save creates Experience with source=pipeline, exp_status=draft."""

    @pytest.mark.asyncio
    async def test_draft_save_success(self):
        """Pipeline write: save draft with forced source='pipeline', exp_status='draft'."""
        exp_id = str(uuid.uuid4())
        mock_result = {
            "id": exp_id,
            "title": "Draft fix",
            "exp_status": "draft",
            "source": "pipeline",
        }

        with (
            patch(f"{_O}._get_service") as mock_get_service,
            patch(f"{_O}._get_db_url", return_value="sqlite://"),
            _patch_user(),
            patch(f"{_O}._resolve_project", return_value="default"),
            patch(f"{_O}.get_session") as mock_get_session,
        ):
            mock_service = MagicMock()
            mock_service.save = AsyncMock(return_value=mock_result)
            mock_get_service.return_value = mock_service

            mock_session = AsyncMock()
            mock_get_session.return_value.__aenter__ = AsyncMock(return_value=mock_session)
            mock_get_session.return_value.__aexit__ = AsyncMock(return_value=False)

            tools = {t.name: t for t in await mcp.list_tools()}
            fn = tools["memory_draft_save"].fn
            result = await fn(
                title="Draft fix",
                content="Connection pool exhausted under load",
            )

        data = json.loads(result)
        assert data["id"] == exp_id
        assert data["status"] == "draft"

        # Verify save was called with source='pipeline' and exp_status='draft'
        call_kw = mock_service.save.await_args.kwargs
        assert call_kw["source"] == "pipeline"
        assert call_kw["exp_status"] == "draft"
        assert call_kw["title"] == "Draft fix"
        assert call_kw["problem"] == "Connection pool exhausted under load"
        assert call_kw["created_by"] == "admin"

    @pytest.mark.asyncio
    async def test_draft_save_with_optional_fields(self):
        """Draft save passes tags, project, group_key through."""
        exp_id = str(uuid.uuid4())
        mock_result = {
            "id": exp_id,
            "title": "Draft",
            "exp_status": "draft",
            "source": "pipeline",
        }

        with (
            patch(f"{_O}._get_service") as mock_get_service,
            patch(f"{_O}._get_db_url", return_value="sqlite://"),
            _patch_user(),
            patch(f"{_O}._resolve_project", return_value="myproj"),
            patch(f"{_O}.get_session") as mock_get_session,
        ):
            mock_service = MagicMock()
            mock_service.save = AsyncMock(return_value=mock_result)
            mock_get_service.return_value = mock_service

            mock_session = AsyncMock()
            mock_get_session.return_value.__aenter__ = AsyncMock(return_value=mock_session)
            mock_get_session.return_value.__aexit__ = AsyncMock(return_value=False)

            tools = {t.name: t for t in await mcp.list_tools()}
            fn = tools["memory_draft_save"].fn
            result = await fn(
                title="Draft",
                content="Some content",
                tags=["python", "bug"],
                project="myproj",
                group_key="auth-group",
            )

        data = json.loads(result)
        assert data["status"] == "draft"
        call_kw = mock_service.save.await_args.kwargs
        assert call_kw["tags"] == ["python", "bug"]
        assert call_kw["group_key"] == "auth-group"
        assert call_kw["project"] == "myproj"


# ============================================================
# memory_draft_publish
# ============================================================


class TestMemoryDraftPublish:
    """Test memory_draft_publish transitions draft -> published."""

    @pytest.mark.asyncio
    async def test_publish_success(self):
        """Promote pipeline draft to published."""
        draft_id = str(uuid.uuid4())
        mock_exp = {
            "id": draft_id,
            "source": "pipeline",
            "status": "draft",
        }

        mock_updated = {
            "id": draft_id,
            "title": "Draft fix",
            "exp_status": "published",
            "source": "pipeline",
        }

        with (
            patch(f"{_O}._get_service") as mock_get_service,
            _patch_user(),
        ):
            mock_service = MagicMock()
            mock_service.get_by_id = AsyncMock(return_value=mock_exp)
            mock_service.update = AsyncMock(return_value=mock_updated)
            mock_get_service.return_value = mock_service

            tools = {t.name: t for t in await mcp.list_tools()}
            fn = tools["memory_draft_publish"].fn
            result = await fn(draft_id=draft_id)

        data = json.loads(result)
        assert data["id"] == draft_id
        assert data["status"] == "published"

        # Verify update called with exp_status='published'
        update_kw = mock_service.update.await_args.kwargs
        assert update_kw["experience_id"] == draft_id
        assert update_kw["exp_status"] == "published"

    @pytest.mark.asyncio
    async def test_publish_with_refined_content(self):
        """Publish with refined_content updates description too."""
        draft_id = str(uuid.uuid4())
        mock_exp = {
            "id": draft_id,
            "source": "pipeline",
            "status": "draft",
        }

        mock_updated = {
            "id": draft_id,
            "exp_status": "published",
        }

        with (
            patch(f"{_O}._get_service") as mock_get_service,
            _patch_user(),
        ):
            mock_service = MagicMock()
            mock_service.get_by_id = AsyncMock(return_value=mock_exp)
            mock_service.update = AsyncMock(return_value=mock_updated)
            mock_get_service.return_value = mock_service

            tools = {t.name: t for t in await mcp.list_tools()}
            fn = tools["memory_draft_publish"].fn
            result = await fn(draft_id=draft_id, refined_content="Improved solution")

        data = json.loads(result)
        assert data["status"] == "published"
        update_kw = mock_service.update.await_args.kwargs
        assert update_kw["description"] == "Improved solution"
        assert update_kw["exp_status"] == "published"

    @pytest.mark.asyncio
    async def test_publish_fails_non_pipeline_source(self):
        """Cannot publish a draft with source != 'pipeline'."""
        draft_id = str(uuid.uuid4())
        mock_exp = {
            "id": draft_id,
            "source": "auto_extract",
            "status": "draft",
        }

        with (
            patch(f"{_O}._get_service") as mock_get_service,
            _patch_user(),
        ):
            mock_service = MagicMock()
            mock_service.get_by_id = AsyncMock(return_value=mock_exp)
            mock_get_service.return_value = mock_service

            tools = {t.name: t for t in await mcp.list_tools()}
            fn = tools["memory_draft_publish"].fn
            result = await fn(draft_id=draft_id)

        data = json.loads(result)
        assert data.get("error") is True
        assert "pipeline" in data.get("message", "").lower()

    @pytest.mark.asyncio
    async def test_publish_fails_non_draft_status(self):
        """Cannot publish an experience that is not in 'draft' status."""
        draft_id = str(uuid.uuid4())
        mock_exp = {
            "id": draft_id,
            "source": "pipeline",
            "status": "published",
        }

        with (
            patch(f"{_O}._get_service") as mock_get_service,
            _patch_user(),
        ):
            mock_service = MagicMock()
            mock_service.get_by_id = AsyncMock(return_value=mock_exp)
            mock_get_service.return_value = mock_service

            tools = {t.name: t for t in await mcp.list_tools()}
            fn = tools["memory_draft_publish"].fn
            result = await fn(draft_id=draft_id)

        data = json.loads(result)
        assert data.get("error") is True
        assert "draft" in data.get("message", "").lower()

    @pytest.mark.asyncio
    async def test_publish_fails_not_found(self):
        """Cannot publish a non-existent experience."""
        draft_id = str(uuid.uuid4())

        with (
            patch(f"{_O}._get_service") as mock_get_service,
            _patch_user(),
        ):
            mock_service = MagicMock()
            mock_service.get_by_id = AsyncMock(return_value=None)
            mock_get_service.return_value = mock_service

            tools = {t.name: t for t in await mcp.list_tools()}
            fn = tools["memory_draft_publish"].fn
            result = await fn(draft_id=draft_id)

        data = json.loads(result)
        assert data.get("error") is True
        assert "not found" in data.get("message", "").lower()


# ============================================================
# Regression: existing memory_save still works
# ============================================================


class TestMemorySaveRegression:
    """Ensure memory_save still works after adding draft tools."""

    @pytest.mark.asyncio
    async def test_direct_save_still_works(self):
        """Existing memory_save direct save path is not broken."""
        mock_result = {
            "id": str(uuid.uuid4()),
            "title": "Fix DB timeout",
            "exp_status": "published",
        }

        with (
            patch(f"{_O}._get_service") as mock_get_service,
            patch(f"{_O}._get_db_url", return_value="sqlite://"),
            _patch_user(),
            patch(f"{_O}._resolve_project", return_value="default"),
            patch(f"{_O}.get_session") as mock_get_session,
        ):
            mock_service = MagicMock()
            mock_service.save = AsyncMock(return_value=mock_result)
            mock_get_service.return_value = mock_service

            mock_session = AsyncMock()
            mock_get_session.return_value.__aenter__ = AsyncMock(return_value=mock_session)
            mock_get_session.return_value.__aexit__ = AsyncMock(return_value=False)

            tools = {t.name: t for t in await mcp.list_tools()}
            fn = tools["memory_save"].fn
            result = await fn(
                title="Fix DB timeout",
                problem="Connection pool exhausted under load",
                solution="Increase pool size to 20",
            )

        data = json.loads(result)
        assert "data" in data
        assert data["data"]["title"] == "Fix DB timeout"

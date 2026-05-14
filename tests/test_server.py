"""Tests for MCP Server (server.py).

Validates tool registration, routing logic, and delegation to services.
"""

from __future__ import annotations

import json
import uuid
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

pytestmark = pytest.mark.integration

from team_memory.server import mcp
from team_memory.services.search.search_orchestrator import OrchestratedSearchResult
from tests.conftest import (
    _O,
    _P,
    _patch_app_context,
    _patch_expansion,
    _patch_personal,
    _patch_user,
    make_mcp_tool_settings_stub,
    make_mock_app_context,
)

# ============================================================
# Tool registration
# ============================================================


@pytest.mark.unit
class TestLiteToolRegistration:
    """Verify lite server registers exactly 9 memory_* tools."""

    @pytest.mark.asyncio
    async def test_exactly_six_tools(self):
        tools = {t.name: t for t in await mcp.list_tools()}
        assert (
            len(tools) == 9
        ), f"Expected 9 tools, got {len(tools)}: {list(tools.keys())}"

    @pytest.mark.asyncio
    async def test_tool_names(self):
        tools = {t.name: t for t in await mcp.list_tools()}
        expected = {
            "memory_save",
            "memory_recall",
            "memory_get_archive",
            "memory_archive_upsert",
            "memory_context",
            "memory_feedback",
            "memory_draft_save",
            "memory_draft_publish",
            "memory_submit_response",
        }
        assert set(tools.keys()) == expected

    @pytest.mark.asyncio
    async def test_all_tools_have_descriptions(self):
        tools = {t.name: t for t in await mcp.list_tools()}
        for name, tool in tools.items():
            assert tool.description, f"Tool '{name}' has no description"

    @pytest.mark.asyncio
    async def test_no_tm_prefix_tools(self):
        """Lite server should NOT register any tm_* tools."""
        tools = {t.name: t for t in await mcp.list_tools()}
        for name in tools:
            assert not name.startswith("tm_"), f"Unexpected tm_ tool: {name}"


# ============================================================
# memory_get_archive
# ============================================================


class TestMemoryGetArchive:
    @pytest.mark.asyncio
    async def test_returns_l2_json(self):
        aid = str(uuid.uuid4())
        l2 = {
            "id": aid,
            "title": "T",
            "solution_doc": "body",
            "overview": "o",
            "attachments": [],
        }
        mock_svc = MagicMock()
        mock_svc.get_archive = AsyncMock(return_value=l2)
        mock_app = make_mock_app_context(archive_service=mock_svc)
        with (
            _patch_app_context(mock_app),
            patch(
                f"{_P}._get_current_user", new_callable=AsyncMock, return_value="alice"
            ),
            patch(f"{_P}._resolve_project", return_value="default"),
        ):
            tools = {t.name: t for t in await mcp.list_tools()}
            fn = tools["memory_get_archive"].fn
            out = await fn(archive_id=aid)
        data = json.loads(out)
        assert data["id"] == aid
        assert data["solution_doc"] == "body"
        mock_svc.get_archive.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_invalid_uuid(self):
        tools = {t.name: t for t in await mcp.list_tools()}
        fn = tools["memory_get_archive"].fn
        mock_app = make_mock_app_context()
        with _patch_app_context(mock_app), _patch_user("alice"):
            out = await fn(archive_id="bad")
        data = json.loads(out)
        assert data.get("code") == "not_found"


# ============================================================
# memory_archive_upsert
# ============================================================


class TestMemoryArchiveUpsert:
    @pytest.mark.asyncio
    async def test_success_created(self):
        aid = uuid.uuid4()
        st = MagicMock()
        st.mcp.max_archive_solution_doc_chars = 64_000
        mock_svc = MagicMock()
        mock_svc.archive_upsert = AsyncMock(
            return_value={
                "action": "created",
                "archive_id": aid,
                "previous_updated_at": None,
            }
        )
        mock_app = make_mock_app_context(settings=st, archive_service=mock_svc)
        with (
            _patch_app_context(mock_app),
            patch(
                f"{_P}._get_current_user", new_callable=AsyncMock, return_value="alice"
            ),
            patch(f"{_P}._resolve_project", return_value="myproj"),
        ):
            tools = {t.name: t for t in await mcp.list_tools()}
            fn = tools["memory_archive_upsert"].fn
            out = await fn(title="Doc", solution_doc="# Body\n", project="p")

        data = json.loads(out)
        assert data["archive_id"] == str(aid)
        assert data["action"] == "created"
        assert data["message"] == "Created successfully"
        mock_svc.archive_upsert.assert_awaited_once()
        call_kw = mock_svc.archive_upsert.await_args.kwargs
        assert call_kw["created_by"] == "alice"
        assert call_kw["project"] == "myproj"

    @pytest.mark.asyncio
    async def test_validation_error_from_schema(self):
        mock_app = make_mock_app_context(settings=make_mcp_tool_settings_stub())
        with _patch_app_context(mock_app), _patch_user("alice"):
            tools = {t.name: t for t in await mcp.list_tools()}
            fn = tools["memory_archive_upsert"].fn
            out = await fn(
                title="T",
                solution_doc="x",
                tags=[f"t{i}" for i in range(21)],
            )

        data = json.loads(out)
        assert data.get("error") is True
        assert data.get("code") == "validation_error"

    @pytest.mark.asyncio
    async def test_solution_doc_exceeds_mcp_limit(self):
        st = make_mcp_tool_settings_stub()
        st.mcp.max_archive_solution_doc_chars = 10
        mock_app = make_mock_app_context(settings=st)
        with _patch_app_context(mock_app), _patch_user("alice"):
            tools = {t.name: t for t in await mcp.list_tools()}
            fn = tools["memory_archive_upsert"].fn
            out = await fn(title="T", solution_doc="x" * 11)

        data = json.loads(out)
        assert data.get("code") == "validation_error"
        assert "exceeds MCP limit" in data.get("message", "")

    @pytest.mark.asyncio
    async def test_embedding_failed(self):
        from team_memory.services.archive import ArchiveUploadError

        st = MagicMock()
        st.mcp.max_archive_solution_doc_chars = 64_000
        mock_svc = MagicMock()
        mock_svc.archive_upsert = AsyncMock(
            side_effect=ArchiveUploadError(
                "embedding_failed",
                "Embedding generation failed.",
                http_status=500,
            )
        )
        mock_app = make_mock_app_context(settings=st, archive_service=mock_svc)
        with (
            _patch_app_context(mock_app),
            patch(
                f"{_P}._get_current_user", new_callable=AsyncMock, return_value="alice"
            ),
            patch(f"{_P}._resolve_project", return_value=None),
        ):
            tools = {t.name: t for t in await mcp.list_tools()}
            fn = tools["memory_archive_upsert"].fn
            out = await fn(title="T", solution_doc="body")

        data = json.loads(out)
        assert data.get("error") is True
        assert data.get("code") == "embedding_failed"


# ============================================================
# memory_save
# ============================================================


class TestMemorySave:
    """Test memory_save routing logic."""

    @pytest.mark.asyncio
    async def test_direct_save(self):
        """title + problem → service.save()."""
        mock_result = {
            "id": str(uuid.uuid4()),
            "title": "Fix DB timeout",
            "exp_status": "published",
        }

        mock_service = MagicMock()
        mock_service.save = AsyncMock(return_value=mock_result)
        mock_app = make_mock_app_context(service=mock_service, db_url="sqlite://")
        with (
            _patch_app_context(mock_app),
            _patch_user(),
            patch(f"{_P}._resolve_project", return_value="default"),
            patch(f"{_O}.get_session") as mock_get_session,
        ):
            mock_session = AsyncMock()
            mock_get_session.return_value.__aenter__ = AsyncMock(
                return_value=mock_session
            )
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
        mock_service.save.assert_called_once()

    @pytest.mark.asyncio
    async def test_content_parse_mode(self):
        """content → LLM parse → service.save()."""
        mock_parsed = {
            "title": "Parsed Title",
            "problem": "Parsed problem",
            "solution": "Parsed solution",
            "tags": ["python"],
        }
        mock_result = {
            "id": str(uuid.uuid4()),
            "title": "Parsed Title",
            "tags": ["python"],
            "exp_status": "draft",
        }

        mock_settings = MagicMock()
        mock_settings.extraction = None
        mock_settings.llm = MagicMock()
        mock_settings.mcp.max_content_chars = 200_000
        mock_settings.mcp.max_tags = 20
        mock_settings.mcp.max_tag_length = 50

        mock_service = MagicMock()
        mock_service.save = AsyncMock(return_value=mock_result)
        mock_app = make_mock_app_context(
            service=mock_service,
            settings=mock_settings,
            db_url="sqlite://",
        )

        with (
            _patch_app_context(mock_app),
            _patch_user(),
            patch(f"{_P}._resolve_project", return_value="default"),
            patch(f"{_O}.get_session") as mock_get_session,
            patch(
                "team_memory.services.llm.llm_parser.parse_content",
                new_callable=AsyncMock,
                return_value=mock_parsed,
            ),
            _patch_personal(),
        ):
            mock_session = AsyncMock()
            mock_get_session.return_value.__aenter__ = AsyncMock(
                return_value=mock_session
            )
            mock_get_session.return_value.__aexit__ = AsyncMock(return_value=False)

            tools = {t.name: t for t in await mcp.list_tools()}
            fn = tools["memory_save"].fn
            result = await fn(content="Long conversation about fixing a bug...")

        data = json.loads(result)
        assert "data" in data
        assert data["data"]["title"] == "Parsed Title"

    @pytest.mark.asyncio
    async def test_archive_scope_rejected(self):
        """scope='archive' → hard error, no longer supported."""
        dummy = make_mock_app_context()
        with (
            _patch_app_context(dummy),
            _patch_user(),
            patch(f"{_P}._resolve_project", return_value="default"),
        ):
            tools = {t.name: t for t in await mcp.list_tools()}
            fn = tools["memory_save"].fn
            result = await fn(
                title="Sprint 42 Summary",
                solution="Completed auth migration...",
                scope="archive",
            )

        data = json.loads(result)
        assert data["error"] is True
        assert "no longer supported" in data["message"]

    @pytest.mark.asyncio
    async def test_missing_params_error(self):
        """No title/problem/content → error."""
        mock_settings = MagicMock()
        mock_settings.mcp.max_content_chars = 200_000
        mock_settings.mcp.max_tags = 20
        mock_settings.mcp.max_tag_length = 50

        mock_app = make_mock_app_context(settings=mock_settings, db_url="sqlite://")
        with (
            _patch_app_context(mock_app),
            _patch_user(),
            patch(f"{_P}._resolve_project", return_value="default"),
        ):
            tools = {t.name: t for t in await mcp.list_tools()}
            fn = tools["memory_save"].fn
            result = await fn(tags=["python"])

        data = json.loads(result)
        assert data.get("error") is True


# ============================================================
# memory_recall
# ============================================================


class TestMemoryRecall:
    """Test memory_recall routing logic."""

    @pytest.mark.asyncio
    async def test_solve_mode(self):
        """problem → solve mode with enhanced query."""
        mock_results = [
            {
                "id": str(uuid.uuid4()),
                "group_id": str(uuid.uuid4()),
                "title": "DB Timeout Fix",
                "solution": "Increase pool size",
                "score": 0.9,
                "confidence": "high",
                "type": "experience",
            }
        ]

        mock_orchestrator = MagicMock()
        mock_orchestrator.search = AsyncMock(
            return_value=OrchestratedSearchResult(results=mock_results)
        )
        mock_app = make_mock_app_context(
            db_url="sqlite://",
            search_orchestrator=mock_orchestrator,
            embedding=MagicMock(),
            settings=make_mcp_tool_settings_stub(),
        )
        with (
            _patch_app_context(mock_app),
            _patch_user(),
            patch(f"{_P}._resolve_project", return_value="default"),
            patch(
                f"{_O}.write_search_log",
                new_callable=AsyncMock,
                return_value="00000000-0000-0000-0000-000000000001",
            ),
            _patch_expansion(),
        ):
            tools = {t.name: t for t in await mcp.list_tools()}
            fn = tools["memory_recall"].fn
            result = await fn(problem="Database connection timeout", language="python")

        data = json.loads(result)
        assert "results" in data
        assert len(data["results"]) == 1
        call_args = mock_orchestrator.search.call_args
        assert "python" in call_args.kwargs.get("query", "").lower()
        assert call_args.kwargs["min_similarity"] == 0.5

    @pytest.mark.asyncio
    async def test_search_mode(self):
        """query → search mode."""
        mock_results = [
            {
                "id": str(uuid.uuid4()),
                "group_id": str(uuid.uuid4()),
                "title": "Caching patterns",
                "score": 0.8,
            }
        ]

        mock_orchestrator = MagicMock()
        mock_orchestrator.search = AsyncMock(
            return_value=OrchestratedSearchResult(results=mock_results)
        )
        mock_app = make_mock_app_context(
            db_url="sqlite://",
            search_orchestrator=mock_orchestrator,
            embedding=MagicMock(),
            settings=make_mcp_tool_settings_stub(),
        )
        with (
            _patch_app_context(mock_app),
            _patch_user(),
            patch(f"{_P}._resolve_project", return_value="default"),
            patch(
                f"{_O}.write_search_log",
                new_callable=AsyncMock,
                return_value="00000000-0000-0000-0000-000000000001",
            ),
            _patch_expansion(),
        ):
            tools = {t.name: t for t in await mcp.list_tools()}
            fn = tools["memory_recall"].fn
            result = await fn(query="redis caching best practices")

        data = json.loads(result)
        assert "results" in data
        call_args = mock_orchestrator.search.call_args
        assert call_args.kwargs["min_similarity"] == 0.6

    @pytest.mark.asyncio
    async def test_search_mode_include_user_profile(self):
        """include_user_profile=True adds profile.static / profile.dynamic."""
        mock_results = [
            {
                "id": str(uuid.uuid4()),
                "group_id": str(uuid.uuid4()),
                "title": "Caching patterns",
                "score": 0.8,
            }
        ]

        mock_settings = MagicMock()
        mock_settings.mcp.profile_max_strings_per_side = 20
        mock_settings.mcp.max_output_tokens = 4000
        mock_settings.mcp.truncate_solution_at = 2000

        mock_orchestrator = MagicMock()
        mock_orchestrator.search = AsyncMock(
            return_value=OrchestratedSearchResult(results=mock_results)
        )
        mock_embed = MagicMock()
        mock_app = make_mock_app_context(
            db_url="sqlite://",
            search_orchestrator=mock_orchestrator,
            settings=mock_settings,
            embedding=mock_embed,
        )
        with (
            _patch_app_context(mock_app),
            _patch_user(),
            patch(f"{_P}._resolve_project", return_value="default"),
            patch(
                f"{_O}.write_search_log",
                new_callable=AsyncMock,
                return_value="00000000-0000-0000-0000-000000000001",
            ),
            patch(
                "team_memory.services.memory.personal_memory.PersonalMemoryService",
            ) as mock_pm_cls,
            _patch_expansion(),
        ):
            mock_pm = MagicMock()
            mock_pm.build_profile_for_user = AsyncMock(
                return_value={"static": ["prefers tests"], "dynamic": []}
            )
            mock_pm_cls.return_value = mock_pm

            tools = {t.name: t for t in await mcp.list_tools()}
            fn = tools["memory_recall"].fn
            result = await fn(query="redis", include_user_profile=True)

        data = json.loads(result)
        assert data["profile"]["static"] == ["prefers tests"]
        assert data["profile"]["dynamic"] == []
        mock_pm.build_profile_for_user.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_suggest_mode(self):
        """file_path only → suggest mode."""
        mock_results = [
            {
                "id": str(uuid.uuid4()),
                "parent": {
                    "id": str(uuid.uuid4()),
                    "title": "Auth middleware",
                    "tags": ["auth"],
                },
                "score": 0.7,
                "confidence": "medium",
                "total_children": 0,
            }
        ]

        mock_orchestrator = MagicMock()
        mock_orchestrator.search = AsyncMock(
            return_value=OrchestratedSearchResult(results=mock_results)
        )
        mock_app = make_mock_app_context(
            db_url="sqlite://",
            search_orchestrator=mock_orchestrator,
            embedding=MagicMock(),
            settings=make_mcp_tool_settings_stub(),
        )
        with (
            _patch_app_context(mock_app),
            _patch_user(),
            patch(f"{_P}._resolve_project", return_value="default"),
            patch(
                f"{_O}.write_search_log",
                new_callable=AsyncMock,
                return_value="00000000-0000-0000-0000-000000000001",
            ),
        ):
            tools = {t.name: t for t in await mcp.list_tools()}
            fn = tools["memory_recall"].fn
            result = await fn(file_path="src/auth/middleware.py")

        data = json.loads(result)
        assert "results" in data
        call_args = mock_orchestrator.search.call_args
        assert call_args.kwargs["min_similarity"] == 0.4
        assert "middleware.py" in call_args.kwargs["query"]

    @pytest.mark.asyncio
    async def test_empty_params_error(self):
        """No params → error."""
        mock_orchestrator = MagicMock()
        mock_app = make_mock_app_context(
            db_url="sqlite://",
            search_orchestrator=mock_orchestrator,
            embedding=MagicMock(),
            settings=make_mcp_tool_settings_stub(),
        )
        with (
            _patch_app_context(mock_app),
            _patch_user(),
            patch(f"{_P}._resolve_project", return_value="default"),
            patch(
                f"{_O}.write_search_log",
                new_callable=AsyncMock,
                return_value="00000000-0000-0000-0000-000000000001",
            ),
        ):
            tools = {t.name: t for t in await mcp.list_tools()}
            fn = tools["memory_recall"].fn
            result = await fn()

        data = json.loads(result)
        assert data.get("error") is True

    @pytest.mark.asyncio
    async def test_no_results_suggests_save(self):
        """No results in solve mode → suggest memory_save."""
        mock_orchestrator = MagicMock()
        mock_orchestrator.search = AsyncMock(
            return_value=OrchestratedSearchResult(results=[])
        )
        mock_app = make_mock_app_context(
            db_url="sqlite://",
            search_orchestrator=mock_orchestrator,
            embedding=MagicMock(),
            settings=make_mcp_tool_settings_stub(),
        )
        with (
            _patch_app_context(mock_app),
            _patch_user(),
            patch(f"{_P}._resolve_project", return_value="default"),
            patch(
                f"{_O}.write_search_log",
                new_callable=AsyncMock,
                return_value="00000000-0000-0000-0000-000000000001",
            ),
        ):
            tools = {t.name: t for t in await mcp.list_tools()}
            fn = tools["memory_recall"].fn
            result = await fn(problem="Some unique problem")

        data = json.loads(result)
        assert "memory_save" in data["message"]

    @pytest.mark.asyncio
    async def test_recall_include_archives_true(self):
        """memory_recall with include_archives=True passes it to search_orchestrator.search."""
        mock_results = [
            {
                "id": str(uuid.uuid4()),
                "group_id": str(uuid.uuid4()),
                "title": "Archived solution",
                "score": 0.85,
                "type": "archive",
            }
        ]

        mock_orchestrator = MagicMock()
        mock_orchestrator.search = AsyncMock(
            return_value=OrchestratedSearchResult(results=mock_results)
        )
        mock_app = make_mock_app_context(
            db_url="sqlite://",
            search_orchestrator=mock_orchestrator,
            embedding=MagicMock(),
            settings=make_mcp_tool_settings_stub(),
        )
        with (
            _patch_app_context(mock_app),
            _patch_user(),
            patch(f"{_P}._resolve_project", return_value="default"),
            patch(
                f"{_O}.write_search_log",
                new_callable=AsyncMock,
                return_value="00000000-0000-0000-0000-000000000001",
            ),
            _patch_expansion(),
        ):
            tools = {t.name: t for t in await mcp.list_tools()}
            fn = tools["memory_recall"].fn
            result = await fn(query="redis patterns", include_archives=True)

        data = json.loads(result)
        assert "results" in data
        assert len(data["results"]) == 1
        call_kwargs = mock_orchestrator.search.call_args.kwargs
        assert call_kwargs["include_archives"] is True

    @pytest.mark.asyncio
    async def test_recall_solve_mode_include_archives_true(self):
        """memory_recall solve mode with include_archives=True passes it to search."""
        mock_results = [
            {
                "id": str(uuid.uuid4()),
                "group_id": str(uuid.uuid4()),
                "title": "Archived fix",
                "score": 0.9,
                "confidence": "high",
                "type": "archive",
            }
        ]

        mock_orchestrator = MagicMock()
        mock_orchestrator.search = AsyncMock(
            return_value=OrchestratedSearchResult(results=mock_results)
        )
        mock_app = make_mock_app_context(
            db_url="sqlite://",
            search_orchestrator=mock_orchestrator,
            embedding=MagicMock(),
            settings=make_mcp_tool_settings_stub(),
        )
        with (
            _patch_app_context(mock_app),
            _patch_user(),
            patch(f"{_P}._resolve_project", return_value="default"),
            patch(
                f"{_O}.write_search_log",
                new_callable=AsyncMock,
                return_value="00000000-0000-0000-0000-000000000001",
            ),
            _patch_expansion(),
        ):
            tools = {t.name: t for t in await mcp.list_tools()}
            fn = tools["memory_recall"].fn
            result = await fn(
                problem="Connection pool exhaustion",
                include_archives=True,
            )

        data = json.loads(result)
        assert "results" in data
        call_kwargs = mock_orchestrator.search.call_args.kwargs
        assert call_kwargs["include_archives"] is True

    @pytest.mark.asyncio
    async def test_recall_default_include_archives_dev(self, monkeypatch):
        """In development env, include_archives defaults to True."""
        monkeypatch.setenv("TEAM_MEMORY_ENV", "development")

        mock_orchestrator = MagicMock()
        mock_orchestrator.search = AsyncMock(
            return_value=OrchestratedSearchResult(results=[])
        )
        mock_app = make_mock_app_context(
            db_url="sqlite://",
            search_orchestrator=mock_orchestrator,
            embedding=MagicMock(),
            settings=make_mcp_tool_settings_stub(),
        )
        with (
            _patch_app_context(mock_app),
            _patch_user(),
            patch(f"{_P}._resolve_project", return_value="default"),
            patch(
                f"{_O}.write_search_log",
                new_callable=AsyncMock,
                return_value="00000000-0000-0000-0000-000000000001",
            ),
            _patch_expansion(),
        ):
            tools = {t.name: t for t in await mcp.list_tools()}
            fn = tools["memory_recall"].fn
            await fn(query="test query")

        call_kwargs = mock_orchestrator.search.call_args.kwargs
        assert call_kwargs["include_archives"] is True

    @pytest.mark.asyncio
    async def test_recall_default_include_archives_prod(self, monkeypatch):
        """In production env, include_archives defaults to False."""
        monkeypatch.setenv("TEAM_MEMORY_ENV", "production")

        mock_orchestrator = MagicMock()
        mock_orchestrator.search = AsyncMock(
            return_value=OrchestratedSearchResult(results=[])
        )
        mock_app = make_mock_app_context(
            db_url="sqlite://",
            search_orchestrator=mock_orchestrator,
            embedding=MagicMock(),
            settings=make_mcp_tool_settings_stub(),
        )
        with (
            _patch_app_context(mock_app),
            _patch_user(),
            patch(f"{_P}._resolve_project", return_value="default"),
            patch(
                f"{_O}.write_search_log",
                new_callable=AsyncMock,
                return_value="00000000-0000-0000-0000-000000000001",
            ),
            _patch_expansion(),
        ):
            tools = {t.name: t for t in await mcp.list_tools()}
            fn = tools["memory_recall"].fn
            await fn(query="test query")

        call_kwargs = mock_orchestrator.search.call_args.kwargs
        assert call_kwargs["include_archives"] is False


# ============================================================
# memory_context
# ============================================================


class TestMemoryContext:
    """Test memory_context profile + context retrieval."""

    @pytest.mark.asyncio
    async def test_returns_profile_and_experiences(self):
        mock_results = [
            {
                "group_id": str(uuid.uuid4()),
                "parent": {
                    "id": str(uuid.uuid4()),
                    "title": "Auth patterns",
                    "tags": ["auth"],
                },
                "confidence": "high",
            }
        ]

        mock_settings = MagicMock()
        mock_settings.mcp.profile_max_strings_per_side = 20

        mock_orchestrator = MagicMock()
        mock_orchestrator.search = AsyncMock(
            return_value=OrchestratedSearchResult(results=mock_results, reranked=True)
        )
        mock_embed = MagicMock()
        mock_app = make_mock_app_context(
            db_url="sqlite://",
            search_orchestrator=mock_orchestrator,
            settings=mock_settings,
            embedding=mock_embed,
        )

        with (
            _patch_user(),
            patch(f"{_P}._resolve_project", return_value="default"),
            _patch_app_context(mock_app),
            patch(
                "team_memory.services.memory.personal_memory.PersonalMemoryService",
            ) as mock_pm_cls,
        ):
            mock_pm = MagicMock()
            mock_pm.build_profile_for_user = AsyncMock(
                return_value={
                    "static": ["Prefers dark mode"],
                    "dynamic": [],
                }
            )
            mock_pm_cls.return_value = mock_pm

            tools = {t.name: t for t in await mcp.list_tools()}
            fn = tools["memory_context"].fn
            result = await fn(
                file_paths=["src/auth/middleware.py"],
                task_description="Add rate limiting",
            )

        data = json.loads(result)
        assert data["user"] == "admin"
        assert data["profile"]["static"] == ["Prefers dark mode"]
        assert data["profile"]["dynamic"] == []
        assert len(data["relevant_experiences"]) == 1
        assert data.get("search_reranked") is True


# ============================================================
# memory_feedback
# ============================================================


class TestMemoryFeedback:
    """Test memory_feedback rating flow."""

    @pytest.mark.asyncio
    async def test_valid_rating(self):
        mock_service = MagicMock()
        mock_service.feedback = AsyncMock(return_value=True)
        mock_app = make_mock_app_context(service=mock_service)
        with (
            _patch_app_context(mock_app),
            _patch_user(),
        ):
            tools = {t.name: t for t in await mcp.list_tools()}
            fn = tools["memory_feedback"].fn
            result = await fn(experience_id=str(uuid.uuid4()), rating=5)

        data = json.loads(result)
        assert "Feedback recorded" in data["message"]

    @pytest.mark.asyncio
    async def test_invalid_rating(self):
        mock_service = MagicMock()
        mock_app = make_mock_app_context(service=mock_service)
        with (
            _patch_app_context(mock_app),
            _patch_user(),
        ):
            tools = {t.name: t for t in await mcp.list_tools()}
            fn = tools["memory_feedback"].fn
            result = await fn(experience_id=str(uuid.uuid4()), rating=10)

        data = json.loads(result)
        assert data.get("error") is True


# ============================================================
# Token budget guard tests
# ============================================================


@pytest.mark.unit
class TestGuardOutput:
    """Test the _guard_output token budget enforcement."""

    def test_small_output_unchanged(self):
        from team_memory.server import _guard_output

        data = {"message": "ok", "results": [{"title": "A"}]}
        raw = json.dumps(data)
        result = _guard_output(raw, max_tokens=5000)
        assert json.loads(result) == data

    def test_large_output_truncates_results(self):
        from team_memory.server import _guard_output

        results = [
            {"title": f"Experience {i}", "solution": "x" * 3000} for i in range(10)
        ]
        data = {"message": "Found 10", "results": results}
        raw = json.dumps(data, ensure_ascii=False)

        result = _guard_output(raw, max_tokens=500)
        parsed = json.loads(result)
        assert len(parsed["results"]) < 10
        assert parsed.get("truncated") is True

    def test_removes_low_confidence_first(self):
        from team_memory.server import _guard_output

        results = [
            {"title": "A", "confidence": "high", "solution": "good answer"},
            {"title": "B", "confidence": "medium", "solution": "ok answer"},
            {"title": "C", "confidence": "low", "solution": "x" * 5000},
        ]
        data = {"message": "Found 3", "results": results}
        raw = json.dumps(data)

        result = _guard_output(raw, max_tokens=200)
        parsed = json.loads(result)
        titles = [r["title"] for r in parsed["results"]]
        assert "C" not in titles

    def test_no_results_passthrough(self):
        from team_memory.server import _guard_output

        data = {"message": "No matches", "results": []}
        raw = json.dumps(data)
        result = _guard_output(raw, max_tokens=100)
        assert json.loads(result) == data


# ============================================================
# Input validation: content length (H5)
# ============================================================


@pytest.mark.unit
class TestMemorySaveContentLimit:
    """Test memory_save rejects content that exceeds max_content_chars."""

    @pytest.mark.asyncio
    async def test_content_too_long_rejected(self):
        """Content exceeding max_content_chars returns error."""
        mock_settings = MagicMock()
        mock_settings.mcp.max_content_chars = 100
        mock_settings.mcp.max_tags = 20
        mock_settings.mcp.max_tag_length = 50

        mock_app = make_mock_app_context(settings=mock_settings, db_url="sqlite://")
        with (
            _patch_app_context(mock_app),
            _patch_user(),
            patch(f"{_P}._resolve_project", return_value="default"),
        ):
            tools = {t.name: t for t in await mcp.list_tools()}
            fn = tools["memory_save"].fn
            result = await fn(content="x" * 101)

        data = json.loads(result)
        assert data["error"] is True
        assert "too long" in data["message"].lower()

    @pytest.mark.asyncio
    async def test_content_within_limit_passes(self):
        """Content within max_content_chars proceeds to LLM parse."""
        mock_parsed = {
            "title": "Parsed",
            "problem": "P",
            "solution": "S",
            "tags": [],
        }
        mock_result = {
            "id": str(uuid.uuid4()),
            "title": "Parsed",
            "tags": [],
            "exp_status": "draft",
        }

        mock_settings = MagicMock()
        mock_settings.mcp.max_content_chars = 200_000
        mock_settings.mcp.max_tags = 20
        mock_settings.mcp.max_tag_length = 50
        mock_settings.extraction = None
        mock_settings.llm = MagicMock()

        mock_service = MagicMock()
        mock_service.save = AsyncMock(return_value=mock_result)
        mock_app = make_mock_app_context(
            service=mock_service,
            settings=mock_settings,
            db_url="sqlite://",
        )

        with (
            _patch_app_context(mock_app),
            _patch_user(),
            patch(f"{_P}._resolve_project", return_value="default"),
            patch(f"{_O}.get_session") as mock_get_session,
            patch(
                "team_memory.services.llm.llm_parser.parse_content",
                new_callable=AsyncMock,
                return_value=mock_parsed,
            ),
            _patch_personal(),
        ):
            mock_session = AsyncMock()
            mock_get_session.return_value.__aenter__ = AsyncMock(
                return_value=mock_session,
            )
            mock_get_session.return_value.__aexit__ = AsyncMock(return_value=False)

            tools = {t.name: t for t in await mcp.list_tools()}
            fn = tools["memory_save"].fn
            result = await fn(content="Valid content under limit")

        data = json.loads(result)
        assert "data" in data


# ============================================================
# Input validation: tags in MCP (M6)
# ============================================================


@pytest.mark.unit
class TestMemorySaveTagsValidation:
    """Test memory_save rejects invalid tags."""

    @pytest.mark.asyncio
    async def test_too_many_tags_rejected(self):
        """More than max_tags returns error."""
        mock_settings = MagicMock()
        mock_settings.mcp.max_content_chars = 200_000
        mock_settings.mcp.max_tags = 5
        mock_settings.mcp.max_tag_length = 50

        mock_app = make_mock_app_context(settings=mock_settings, db_url="sqlite://")
        with (
            _patch_app_context(mock_app),
            _patch_user(),
            patch(f"{_P}._resolve_project", return_value="default"),
        ):
            tools = {t.name: t for t in await mcp.list_tools()}
            fn = tools["memory_save"].fn
            result = await fn(
                title="T",
                problem="P",
                tags=["a", "b", "c", "d", "e", "f"],
            )

        data = json.loads(result)
        assert data["error"] is True
        assert "too many tags" in data["message"].lower()

    @pytest.mark.asyncio
    async def test_tag_too_long_rejected(self):
        """Tag exceeding max_tag_length returns error."""
        mock_settings = MagicMock()
        mock_settings.mcp.max_content_chars = 200_000
        mock_settings.mcp.max_tags = 20
        mock_settings.mcp.max_tag_length = 10

        mock_app = make_mock_app_context(settings=mock_settings, db_url="sqlite://")
        with (
            _patch_app_context(mock_app),
            _patch_user(),
            patch(f"{_P}._resolve_project", return_value="default"),
        ):
            tools = {t.name: t for t in await mcp.list_tools()}
            fn = tools["memory_save"].fn
            result = await fn(
                title="T",
                problem="P",
                tags=["this_tag_is_way_too_long"],
            )

        data = json.loads(result)
        assert data["error"] is True
        assert "tag too long" in data["message"].lower()

    @pytest.mark.asyncio
    async def test_valid_tags_pass(self):
        """Tags within limits proceed to save."""
        mock_result = {
            "id": str(uuid.uuid4()),
            "title": "T",
            "exp_status": "published",
        }
        mock_settings = MagicMock()
        mock_settings.mcp.max_content_chars = 200_000
        mock_settings.mcp.max_tags = 20
        mock_settings.mcp.max_tag_length = 50

        mock_service = MagicMock()
        mock_service.save = AsyncMock(return_value=mock_result)
        mock_app = make_mock_app_context(
            service=mock_service,
            settings=mock_settings,
            db_url="sqlite://",
        )

        with (
            _patch_app_context(mock_app),
            _patch_user(),
            patch(f"{_P}._resolve_project", return_value="default"),
            patch(f"{_O}.get_session") as mock_get_session,
        ):
            mock_session = AsyncMock()
            mock_get_session.return_value.__aenter__ = AsyncMock(
                return_value=mock_session,
            )
            mock_get_session.return_value.__aexit__ = AsyncMock(return_value=False)

            tools = {t.name: t for t in await mcp.list_tools()}
            fn = tools["memory_save"].fn
            result = await fn(
                title="T",
                problem="P",
                tags=["python", "testing"],
            )

        data = json.loads(result)
        assert "data" in data
        mock_service.save.assert_called_once()


# ============================================================
# memory_submit_response
# ============================================================


@pytest.mark.unit
class TestMemorySubmitResponse:
    """memory_submit_response delegates to op_submit_response (local) or RemoteTMClient."""

    @pytest.mark.asyncio
    async def test_local_delegates_to_op_submit_response(self):
        buffer_id = str(uuid.uuid4())
        exp_id = str(uuid.uuid4())
        mock_op = AsyncMock(
            return_value={
                "status": "buffered",
                "buffer_id": buffer_id,
            }
        )
        mock_submit_ctx = make_mock_app_context(db_url="sqlite:///submit-response")
        with (
            patch(f"{_P}._remote_client", None),
            _patch_app_context(mock_submit_ctx),
            _patch_user(),
            patch(f"{_P}._resolve_project", return_value="proj-a"),
            patch(f"{_O}.op_submit_response", mock_op),
        ):
            tools = {t.name: t for t in await mcp.list_tools()}
            fn = tools["memory_submit_response"].fn
            out = await fn(
                query="why tests fail", response="because mocks", result_ids=[exp_id]
            )

        mock_op.assert_awaited_once_with(
            "admin",
            db_url="sqlite:///submit-response",
            query="why tests fail",
            response="because mocks",
            result_ids=[exp_id],
            project="proj-a",
        )
        data = json.loads(out)
        assert data["status"] == "buffered"
        assert data["buffer_id"] == buffer_id
        assert "faithfulness" in data["message"].lower()

    @pytest.mark.asyncio
    async def test_remote_delegates_to_client(self):
        buffer_id = str(uuid.uuid4())
        mock_client = MagicMock()
        mock_client.op_submit_response = AsyncMock(
            return_value={"status": "buffered", "buffer_id": buffer_id}
        )
        rid = str(uuid.uuid4())
        with (
            patch(f"{_P}._remote_client", mock_client),
            _patch_user(),
            patch(f"{_P}._resolve_project", return_value="remote-proj"),
        ):
            tools = {t.name: t for t in await mcp.list_tools()}
            fn = tools["memory_submit_response"].fn
            out = await fn(query="q1", response="r1", result_ids=[rid])

        mock_client.op_submit_response.assert_awaited_once_with(
            "admin",
            query="q1",
            response="r1",
            result_ids=[rid],
            project="remote-proj",
        )
        assert json.loads(out)["buffer_id"] == buffer_id

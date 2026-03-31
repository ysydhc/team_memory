"""Tests for Lite MCP Server (server_lite.py).

Validates tool registration, routing logic, and delegation to services.
"""

from __future__ import annotations

import json
import uuid
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from team_memory.server_lite import mcp

# ============================================================
# Tool registration
# ============================================================


class TestLiteToolRegistration:
    """Verify lite server registers exactly 5 memory_* tools."""

    @pytest.mark.asyncio
    async def test_exactly_five_tools(self):
        tools = {t.name: t for t in await mcp.list_tools()}
        assert len(tools) == 5, f"Expected 5 tools, got {len(tools)}: {list(tools.keys())}"

    @pytest.mark.asyncio
    async def test_tool_names(self):
        tools = {t.name: t for t in await mcp.list_tools()}
        expected = {
            "memory_save",
            "memory_recall",
            "memory_get_archive",
            "memory_context",
            "memory_feedback",
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
        with (
            patch(f"{_P}._get_archive_service") as mock_get,
            patch(f"{_P}._get_current_user", new_callable=AsyncMock, return_value="alice"),
            patch(f"{_P}._resolve_project", return_value="default"),
        ):
            mock_svc = MagicMock()
            mock_svc.get_archive = AsyncMock(return_value=l2)
            mock_get.return_value = mock_svc
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
        out = await fn(archive_id="bad")
        data = json.loads(out)
        assert data.get("code") == 404


# ============================================================
# memory_save
# ============================================================


LITE_PATCH_BASE = "team_memory.server_lite"
_P = LITE_PATCH_BASE  # short alias for line-length


def _patch_user():
    """Patch _get_current_user → 'admin'."""
    return patch(f"{_P}._get_current_user", new_callable=AsyncMock, return_value="admin")


def _patch_expansion():
    """No-op: UserExpansion removed in MVP simplification."""
    from contextlib import nullcontext

    return nullcontext()


def _patch_personal():
    """Patch _try_extract_and_save_personal_memory → no-op."""
    return patch(
        f"{_P}._try_extract_and_save_personal_memory",
        new_callable=AsyncMock,
    )


def _setup_session_cm(mock_get_session):
    """Configure async context manager for get_session mock."""
    mock_session = AsyncMock()
    mock_get_session.return_value.__aenter__ = AsyncMock(
        return_value=mock_session,
    )
    mock_get_session.return_value.__aexit__ = AsyncMock(return_value=False)
    return mock_session


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

        with (
            patch(f"{_P}._get_service") as mock_get_service,
            patch(f"{_P}._get_db_url", return_value="sqlite://"),
            _patch_user(),
            patch(f"{_P}._resolve_project", return_value="default"),
            patch(f"{_P}.get_session") as mock_get_session,
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
        assert "experience" in data
        assert data["experience"]["title"] == "Fix DB timeout"
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

        with (
            patch(f"{_P}._get_service") as mock_get_service,
            patch(f"{_P}._get_settings") as mock_get_settings,
            patch(f"{_P}._get_db_url", return_value="sqlite://"),
            _patch_user(),
            patch(f"{_P}._resolve_project", return_value="default"),
            patch(f"{_P}.get_session") as mock_get_session,
            patch(
                "team_memory.services.llm_parser.parse_content",
                new_callable=AsyncMock,
                return_value=mock_parsed,
            ),
            _patch_personal(),
        ):
            mock_service = MagicMock()
            mock_service.save = AsyncMock(return_value=mock_result)
            mock_get_service.return_value = mock_service

            mock_settings = MagicMock()
            mock_settings.extraction = None
            mock_settings.llm = MagicMock()
            mock_get_settings.return_value = mock_settings

            mock_session = AsyncMock()
            mock_get_session.return_value.__aenter__ = AsyncMock(return_value=mock_session)
            mock_get_session.return_value.__aexit__ = AsyncMock(return_value=False)

            tools = {t.name: t for t in await mcp.list_tools()}
            fn = tools["memory_save"].fn
            result = await fn(content="Long conversation about fixing a bug...")

        data = json.loads(result)
        assert "experience" in data
        assert data["experience"]["title"] == "Parsed Title"

    @pytest.mark.asyncio
    async def test_archive_mode(self):
        """scope='archive' → archive_svc.archive_save()."""
        archive_id = uuid.uuid4()

        with (
            patch(f"{_P}._get_archive_service") as mock_get_archive,
            patch(f"{_P}._get_settings"),
            _patch_user(),
            patch(f"{_P}._resolve_project", return_value="default"),
            patch(f"{_P}._get_service"),
            patch(f"{_P}._get_db_url", return_value="sqlite://"),
        ):
            mock_archive_svc = MagicMock()
            mock_archive_svc.archive_save = AsyncMock(return_value=archive_id)
            mock_get_archive.return_value = mock_archive_svc

            tools = {t.name: t for t in await mcp.list_tools()}
            fn = tools["memory_save"].fn
            result = await fn(
                title="Sprint 42 Summary",
                solution="Completed auth migration...",
                scope="archive",
            )

        data = json.loads(result)
        assert data["archive_id"] == str(archive_id)
        mock_archive_svc.archive_save.assert_called_once()

    @pytest.mark.asyncio
    async def test_missing_params_error(self):
        """No title/problem/content → error."""
        with (
            patch(f"{_P}._get_service"),
            patch(f"{_P}._get_settings"),
            patch(f"{_P}._get_db_url", return_value="sqlite://"),
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

        with (
            patch(f"{_P}._get_service") as mock_get_service,
            patch(f"{_P}._get_db_url", return_value="sqlite://"),
            _patch_user(),
            patch(f"{_P}._resolve_project", return_value="default"),
            patch(f"{_P}.get_session") as mock_get_session,
            _patch_expansion(),
        ):
            mock_service = MagicMock()
            mock_service.search = AsyncMock(return_value=mock_results)
            mock_get_service.return_value = mock_service

            mock_session = AsyncMock()
            mock_get_session.return_value.__aenter__ = AsyncMock(return_value=mock_session)
            mock_get_session.return_value.__aexit__ = AsyncMock(return_value=False)

            tools = {t.name: t for t in await mcp.list_tools()}
            fn = tools["memory_recall"].fn
            result = await fn(problem="Database connection timeout", language="python")

        data = json.loads(result)
        assert "results" in data
        assert len(data["results"]) == 1
        # Verify enhanced query was built
        call_args = mock_service.search.call_args
        assert "python" in call_args.kwargs.get("query", "").lower()
        assert call_args.kwargs["min_similarity"] == 0.5  # solve mode threshold

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

        with (
            patch(f"{_P}._get_service") as mock_get_service,
            patch(f"{_P}._get_db_url", return_value="sqlite://"),
            _patch_user(),
            patch(f"{_P}._resolve_project", return_value="default"),
            _patch_expansion(),
        ):
            mock_service = MagicMock()
            mock_service.search = AsyncMock(return_value=mock_results)
            mock_get_service.return_value = mock_service

            tools = {t.name: t for t in await mcp.list_tools()}
            fn = tools["memory_recall"].fn
            result = await fn(query="redis caching best practices")

        data = json.loads(result)
        assert "results" in data
        call_args = mock_service.search.call_args
        assert call_args.kwargs["min_similarity"] == 0.6  # search mode threshold

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

        with (
            patch(f"{_P}._get_service") as mock_get_service,
            patch(f"{_P}._get_db_url", return_value="sqlite://"),
            _patch_user(),
            patch(f"{_P}._resolve_project", return_value="default"),
            patch(f"{_P}.get_context") as mock_get_ctx,
            patch(f"{_P}._get_settings", return_value=mock_settings),
            patch(
                "team_memory.services.personal_memory.PersonalMemoryService",
            ) as mock_pm_cls,
            _patch_expansion(),
        ):
            mock_ctx = MagicMock()
            mock_ctx.embedding = MagicMock()
            mock_ctx.db_url = "sqlite://"
            mock_get_ctx.return_value = mock_ctx
            mock_pm = MagicMock()
            mock_pm.build_profile_for_user = AsyncMock(
                return_value={"static": ["prefers tests"], "dynamic": []}
            )
            mock_pm_cls.return_value = mock_pm

            mock_service = MagicMock()
            mock_service.search = AsyncMock(return_value=mock_results)
            mock_get_service.return_value = mock_service

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
                "parent": {"id": str(uuid.uuid4()), "title": "Auth middleware", "tags": ["auth"]},
                "score": 0.7,
                "confidence": "medium",
                "total_children": 0,
            }
        ]

        with (
            patch(f"{_P}._get_service") as mock_get_service,
            patch(f"{_P}._get_db_url", return_value="sqlite://"),
            _patch_user(),
            patch(f"{_P}._resolve_project", return_value="default"),
        ):
            mock_service = MagicMock()
            mock_service.search = AsyncMock(return_value=mock_results)
            mock_get_service.return_value = mock_service

            tools = {t.name: t for t in await mcp.list_tools()}
            fn = tools["memory_recall"].fn
            result = await fn(file_path="src/auth/middleware.py")

        data = json.loads(result)
        assert "results" in data
        call_args = mock_service.search.call_args
        assert call_args.kwargs["min_similarity"] == 0.4  # suggest mode threshold
        assert "middleware.py" in call_args.kwargs["query"]

    @pytest.mark.asyncio
    async def test_empty_params_error(self):
        """No params → error."""
        with (
            patch(f"{_P}._get_service"),
            patch(f"{_P}._get_db_url", return_value="sqlite://"),
            _patch_user(),
            patch(f"{_P}._resolve_project", return_value="default"),
        ):
            tools = {t.name: t for t in await mcp.list_tools()}
            fn = tools["memory_recall"].fn
            result = await fn()

        data = json.loads(result)
        assert data.get("error") is True

    @pytest.mark.asyncio
    async def test_no_results_suggests_save(self):
        """No results in solve mode → suggest memory_save."""
        with (
            patch(f"{_P}._get_service") as mock_get_service,
            patch(f"{_P}._get_db_url", return_value="sqlite://"),
            _patch_user(),
            patch(f"{_P}._resolve_project", return_value="default"),
        ):
            mock_service = MagicMock()
            mock_service.search = AsyncMock(return_value=[])
            mock_get_service.return_value = mock_service

            tools = {t.name: t for t in await mcp.list_tools()}
            fn = tools["memory_recall"].fn
            result = await fn(problem="Some unique problem")

        data = json.loads(result)
        assert "memory_save" in data["message"]


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
                "parent": {"id": str(uuid.uuid4()), "title": "Auth patterns", "tags": ["auth"]},
                "confidence": "high",
            }
        ]

        with (
            _patch_user(),
            patch(f"{_P}._resolve_project", return_value="default"),
            patch(f"{_P}.get_context") as mock_get_ctx,
            patch(f"{_P}._get_service") as mock_get_service,
            patch(
                "team_memory.services.personal_memory.PersonalMemoryService",
            ) as mock_pm_cls,
        ):
            mock_ctx = MagicMock()
            mock_ctx.embedding = MagicMock()
            mock_ctx.db_url = "sqlite://"
            mock_get_ctx.return_value = mock_ctx

            mock_pm = MagicMock()
            mock_pm.build_profile_for_user = AsyncMock(
                return_value={
                    "static": ["Prefers dark mode"],
                    "dynamic": [],
                }
            )
            mock_pm_cls.return_value = mock_pm

            mock_service = MagicMock()
            mock_service.search = AsyncMock(return_value=mock_results)
            mock_get_service.return_value = mock_service

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


# ============================================================
# memory_feedback
# ============================================================


class TestMemoryFeedback:
    """Test memory_feedback rating flow."""

    @pytest.mark.asyncio
    async def test_valid_rating(self):
        with (
            patch(f"{_P}._get_service") as mock_get_service,
            patch(f"{_P}._get_db_url", return_value="sqlite://"),
            _patch_user(),
            patch(f"{_P}.get_session") as mock_get_session,
        ):
            mock_service = MagicMock()
            mock_service.feedback = AsyncMock(return_value=True)
            mock_get_service.return_value = mock_service

            mock_session = AsyncMock()
            mock_execute = AsyncMock()
            mock_execute.scalar_one_or_none.return_value = None
            mock_session.execute = AsyncMock(return_value=mock_execute)
            mock_get_session.return_value.__aenter__ = AsyncMock(return_value=mock_session)
            mock_get_session.return_value.__aexit__ = AsyncMock(return_value=False)

            tools = {t.name: t for t in await mcp.list_tools()}
            fn = tools["memory_feedback"].fn
            result = await fn(experience_id=str(uuid.uuid4()), rating=5)

        data = json.loads(result)
        assert "Feedback recorded" in data["message"]

    @pytest.mark.asyncio
    async def test_invalid_rating(self):
        with (
            patch(f"{_P}._get_service"),
            patch(f"{_P}._get_db_url", return_value="sqlite://"),
            _patch_user(),
        ):
            tools = {t.name: t for t in await mcp.list_tools()}
            fn = tools["memory_feedback"].fn
            result = await fn(experience_id=str(uuid.uuid4()), rating=10)

        data = json.loads(result)
        assert data.get("error") is True

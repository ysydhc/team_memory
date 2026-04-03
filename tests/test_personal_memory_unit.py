"""Unit tests for PersonalMemoryService (services/personal_memory.py).

Validates write, list_by_user, and build_profile_for_user by mocking
the PersonalMemoryRepository and EmbeddingProvider.
"""

from __future__ import annotations

import uuid
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from team_memory.services.personal_memory import PersonalMemoryService

# ============================================================
# Helpers
# ============================================================


def _make_embedding_provider(dimension: int = 768) -> MagicMock:
    """Build a mock EmbeddingProvider."""
    provider = MagicMock()
    provider.dimension = dimension
    provider.encode_single = AsyncMock(return_value=[0.1] * dimension)
    provider.encode = AsyncMock(return_value=[[0.1] * dimension])
    return provider


def _make_personal_memory(
    content: str = "test content",
    scope: str = "generic",
    profile_kind: str = "static",
    context_hint: str | None = None,
    user_id: str = "user1",
) -> MagicMock:
    """Build a mock PersonalMemory ORM object."""
    mem = MagicMock()
    mem.id = uuid.uuid4()
    mem.user_id = user_id
    mem.content = content
    mem.scope = scope
    mem.profile_kind = profile_kind
    mem.context_hint = context_hint
    mem.embedding = [0.1] * 768
    now = datetime.now(timezone.utc)
    mem.created_at = now
    mem.updated_at = now
    mem.to_dict.return_value = {
        "id": str(mem.id),
        "user_id": user_id,
        "content": content,
        "scope": scope,
        "profile_kind": profile_kind,
        "context_hint": context_hint,
        "created_at": now.isoformat(),
        "updated_at": now.isoformat(),
    }
    return mem


def _make_mock_session() -> MagicMock:
    """Build a mock AsyncSession."""
    return MagicMock()


def _patch_session_and_repo(
    svc: PersonalMemoryService,
    mock_repo: MagicMock,
) -> None:
    """Patch the service's _session to yield a mock, and PersonalMemoryRepository
    constructor to return *mock_repo*.

    After calling this, `svc.write()` etc. will use the mock repo instead of
    hitting the real database.
    """
    mock_session = _make_mock_session()

    @asynccontextmanager
    async def _fake_session():
        yield mock_session

    svc._session = _fake_session  # type: ignore[assignment]


# ============================================================
# write
# ============================================================


class TestPersonalMemoryWrite:
    """Tests for PersonalMemoryService.write()."""

    @pytest.mark.asyncio
    async def test_write_calls_repo_upsert(self):
        """write() delegates to repository upsert_by_semantic."""
        mock_mem = _make_personal_memory(content="likes vim")
        embedding_provider = _make_embedding_provider()

        mock_repo = MagicMock()
        mock_repo.upsert_by_semantic = AsyncMock(return_value=mock_mem)

        svc = PersonalMemoryService(embedding_provider=embedding_provider, db_url="test://")
        _patch_session_and_repo(svc, mock_repo)

        with patch(
            "team_memory.services.personal_memory.PersonalMemoryRepository",
            return_value=mock_repo,
        ):
            result = await svc.write(user_id="user1", content="likes vim")

        mock_repo.upsert_by_semantic.assert_awaited_once()
        call_kwargs = mock_repo.upsert_by_semantic.call_args
        assert call_kwargs.kwargs["user_id"] == "user1"
        assert call_kwargs.kwargs["content"] == "likes vim"
        assert result["content"] == "likes vim"

    @pytest.mark.asyncio
    async def test_write_generates_embedding(self):
        """write() calls embedding provider's encode_single."""
        mock_mem = _make_personal_memory()
        embedding_provider = _make_embedding_provider()

        mock_repo = MagicMock()
        mock_repo.upsert_by_semantic = AsyncMock(return_value=mock_mem)

        svc = PersonalMemoryService(embedding_provider=embedding_provider, db_url="test://")
        _patch_session_and_repo(svc, mock_repo)

        with patch(
            "team_memory.services.personal_memory.PersonalMemoryRepository",
            return_value=mock_repo,
        ):
            await svc.write(user_id="user1", content="test content")

        embedding_provider.encode_single.assert_awaited_once_with("test content")

    @pytest.mark.asyncio
    async def test_write_dynamic_scope_normalisation(self):
        """When profile_kind='dynamic', scope is normalised to 'context'."""
        mock_mem = _make_personal_memory(scope="context", profile_kind="dynamic")
        embedding_provider = _make_embedding_provider()

        mock_repo = MagicMock()
        mock_repo.upsert_by_semantic = AsyncMock(return_value=mock_mem)

        svc = PersonalMemoryService(embedding_provider=embedding_provider, db_url="test://")
        _patch_session_and_repo(svc, mock_repo)

        with patch(
            "team_memory.services.personal_memory.PersonalMemoryRepository",
            return_value=mock_repo,
        ):
            await svc.write(
                user_id="user1",
                content="working on MCP",
                profile_kind="dynamic",
            )

        call_kwargs = mock_repo.upsert_by_semantic.call_args.kwargs
        assert call_kwargs["scope"] == "context"
        assert call_kwargs["profile_kind"] == "dynamic"


# ============================================================
# list_by_user
# ============================================================


class TestPersonalMemoryListByUser:
    """Tests for PersonalMemoryService.list_by_user()."""

    @pytest.mark.asyncio
    async def test_list_by_user_delegates_to_repo(self):
        """list_by_user() calls repo.list_by_user correctly."""
        mock_mem = _make_personal_memory()
        embedding_provider = _make_embedding_provider()

        mock_repo = MagicMock()
        mock_repo.list_by_user = AsyncMock(return_value=[mock_mem])

        svc = PersonalMemoryService(embedding_provider=embedding_provider, db_url="test://")
        _patch_session_and_repo(svc, mock_repo)

        with patch(
            "team_memory.services.personal_memory.PersonalMemoryRepository",
            return_value=mock_repo,
        ):
            result = await svc.list_by_user("user1")

        mock_repo.list_by_user.assert_awaited_once_with("user1", scope=None, profile_kind=None)
        assert len(result) == 1
        assert result[0]["content"] == "test content"

    @pytest.mark.asyncio
    async def test_list_by_user_with_scope_filter(self):
        """scope parameter is passed through to repository."""
        embedding_provider = _make_embedding_provider()

        mock_repo = MagicMock()
        mock_repo.list_by_user = AsyncMock(return_value=[])

        svc = PersonalMemoryService(embedding_provider=embedding_provider, db_url="test://")
        _patch_session_and_repo(svc, mock_repo)

        with patch(
            "team_memory.services.personal_memory.PersonalMemoryRepository",
            return_value=mock_repo,
        ):
            await svc.list_by_user("user1", scope="context", profile_kind="dynamic")

        mock_repo.list_by_user.assert_awaited_once_with(
            "user1", scope="context", profile_kind="dynamic"
        )


# ============================================================
# build_profile_for_user
# ============================================================


class TestBuildProfileForUser:
    """Tests for PersonalMemoryService.build_profile_for_user()."""

    @pytest.mark.asyncio
    async def test_build_profile_anonymous_returns_empty(self):
        """Anonymous user returns empty profile."""
        embedding_provider = _make_embedding_provider()
        svc = PersonalMemoryService(embedding_provider=embedding_provider, db_url="test://")

        result = await svc.build_profile_for_user(None)
        assert result == {"static": [], "dynamic": []}

        result2 = await svc.build_profile_for_user("anonymous")
        assert result2 == {"static": [], "dynamic": []}

        result3 = await svc.build_profile_for_user("  Anonymous  ")
        assert result3 == {"static": [], "dynamic": []}

    @pytest.mark.asyncio
    async def test_build_profile_builds_static_and_dynamic(self):
        """Returns correct profile shape with static and dynamic lists."""
        static_mem = _make_personal_memory(content="likes vim", profile_kind="static")
        dynamic_mem = _make_personal_memory(content="working on MCP", profile_kind="dynamic")
        embedding_provider = _make_embedding_provider()

        mock_repo = MagicMock()
        mock_repo.list_for_pull = AsyncMock(return_value=[static_mem, dynamic_mem])

        svc = PersonalMemoryService(embedding_provider=embedding_provider, db_url="test://")
        _patch_session_and_repo(svc, mock_repo)

        with patch(
            "team_memory.services.personal_memory.PersonalMemoryRepository",
            return_value=mock_repo,
        ):
            result = await svc.build_profile_for_user("user1")

        assert "static" in result
        assert "dynamic" in result
        assert "likes vim" in result["static"]
        assert "working on MCP" in result["dynamic"]

    @pytest.mark.asyncio
    async def test_build_profile_with_context(self):
        """Context embedding is generated and passed to repo when current_context provided."""
        mock_mem = _make_personal_memory(content="context pref", profile_kind="static")
        embedding_provider = _make_embedding_provider()

        mock_repo = MagicMock()
        mock_repo.list_for_pull = AsyncMock(return_value=[mock_mem])

        svc = PersonalMemoryService(embedding_provider=embedding_provider, db_url="test://")
        _patch_session_and_repo(svc, mock_repo)

        with patch(
            "team_memory.services.personal_memory.PersonalMemoryRepository",
            return_value=mock_repo,
        ):
            await svc.build_profile_for_user("user1", current_context="working on MCP tools")

        embedding_provider.encode_single.assert_awaited_once_with("working on MCP tools")
        call_kwargs = mock_repo.list_for_pull.call_args.kwargs
        assert call_kwargs["context_embedding"] is not None

    @pytest.mark.asyncio
    async def test_build_profile_dedup(self):
        """Deduplicate entries in profile (case-insensitive)."""
        mem1 = _make_personal_memory(content="likes vim", profile_kind="static")
        mem2 = _make_personal_memory(content="Likes Vim", profile_kind="static")
        mem3 = _make_personal_memory(content="likes vim", profile_kind="static")
        embedding_provider = _make_embedding_provider()

        mock_repo = MagicMock()
        mock_repo.list_for_pull = AsyncMock(return_value=[mem1, mem2, mem3])

        svc = PersonalMemoryService(embedding_provider=embedding_provider, db_url="test://")
        _patch_session_and_repo(svc, mock_repo)

        with patch(
            "team_memory.services.personal_memory.PersonalMemoryRepository",
            return_value=mock_repo,
        ):
            result = await svc.build_profile_for_user("user1")

        # "likes vim" and "Likes Vim" are same key (lowercased), so only first is kept
        assert len(result["static"]) == 1
        assert result["static"][0] == "likes vim"

    @pytest.mark.asyncio
    async def test_build_profile_caps_at_max_per_side(self):
        """Respects max_per_side limit."""
        mems = [
            _make_personal_memory(content=f"preference {i}", profile_kind="static")
            for i in range(30)
        ]
        embedding_provider = _make_embedding_provider()

        mock_repo = MagicMock()
        mock_repo.list_for_pull = AsyncMock(return_value=mems)

        svc = PersonalMemoryService(embedding_provider=embedding_provider, db_url="test://")
        _patch_session_and_repo(svc, mock_repo)

        with patch(
            "team_memory.services.personal_memory.PersonalMemoryRepository",
            return_value=mock_repo,
        ):
            result = await svc.build_profile_for_user("user1", max_per_side=5)

        assert len(result["static"]) == 5


# ============================================================
# _dedupe_cap (static helper)
# ============================================================


class TestDedupeCap:
    """Tests for PersonalMemoryService._dedupe_cap() static method."""

    def test_dedup_case_insensitive(self):
        result = PersonalMemoryService._dedupe_cap(["Hello", "hello", "HELLO", "World"], max_n=10)
        assert result == ["Hello", "World"]

    def test_cap_at_max_n(self):
        result = PersonalMemoryService._dedupe_cap(["a", "b", "c", "d", "e"], max_n=3)
        assert result == ["a", "b", "c"]

    def test_empty_list(self):
        result = PersonalMemoryService._dedupe_cap([], max_n=10)
        assert result == []

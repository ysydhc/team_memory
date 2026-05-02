"""Phase 1 integration tests — end-to-end flows.

Covers:
1. draft_save → draft_publish → recall flow with [mem:xxx] markers
2. published → use_count++ → promotion flow via Janitor
3. Regression: memory_save and memory_recall still work normally
"""

from __future__ import annotations

import json
import uuid
from contextlib import asynccontextmanager
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from team_memory.config.janitor import JanitorConfig
from team_memory.server import mcp
from team_memory.services.janitor import MemoryJanitor
from team_memory.services.search_orchestrator import OrchestratedSearchResult
from tests.conftest import _O, _patch_expansion, _patch_user


# ============================================================
# Helpers
# ============================================================

_DRAFT_ID = str(uuid.uuid4())


def _make_draft_save_result(exp_id: str = _DRAFT_ID) -> dict:
    """Simulate service.save() return for a draft."""
    return {
        "id": exp_id,
        "title": "Draft fix",
        "exp_status": "draft",
        "source": "pipeline",
    }


def _make_draft_get_by_id(exp_id: str = _DRAFT_ID) -> dict:
    """Simulate service.get_by_id() return for a pipeline draft."""
    return {
        "id": exp_id,
        "source": "pipeline",
        "status": "draft",
        "title": "Draft fix",
    }


def _make_published_get_by_id(exp_id: str = _DRAFT_ID) -> dict:
    """Simulate service.update() return after publishing."""
    return {
        "id": exp_id,
        "title": "Draft fix",
        "exp_status": "published",
        "source": "pipeline",
    }


def _make_recall_result(exp_id: str = _DRAFT_ID) -> list[dict]:
    """Simulate search result for recalling the published experience."""
    return [
        {
            "id": exp_id,
            "group_id": exp_id,
            "title": "Draft fix",
            "solution": "Increase pool size",
            "score": 0.92,
            "confidence": "high",
            "type": "experience",
        }
    ]


class _FakeScalarResult:
    """Mimics the result of scalars().all()."""

    def __init__(self, items):
        self._items = items

    def all(self):
        return self._items

    def scalars(self):
        return self


@asynccontextmanager
async def _mock_session_ctx(mock_session):
    yield mock_session


def _make_experience(
    *,
    exp_status: str = "published",
    recall_count: int = 0,
    group_key: str | None = None,
    is_deleted: bool = False,
    exp_id: uuid.UUID | None = None,
    project: str = "default",
    title: str = "Test Experience",
    description: str = "Test description",
    solution: str = "Test solution",
    tags: list[str] | None = None,
    created_at: str = "2026-04-23T00:00:00",
) -> MagicMock:
    """Create a mock Experience object."""
    exp = MagicMock()
    exp.id = exp_id or uuid.uuid4()
    exp.exp_status = exp_status
    exp.recall_count = recall_count
    exp.group_key = group_key
    exp.is_deleted = is_deleted
    exp.project = project
    exp.title = title
    exp.description = description
    exp.solution = solution
    exp.tags = tags or []
    exp.created_at = created_at
    exp.to_dict = lambda: {
        "id": str(exp.id),
        "title": title,
        "description": description,
        "solution": solution,
        "tags": tags or [],
        "project": project,
        "group_key": group_key,
        "created_at": created_at,
        "exp_status": exp_status,
    }
    return exp


# ============================================================
# 1. draft_save → draft_publish → recall flow
# ============================================================


class TestDraftPublishRecallFlow:
    """End-to-end: draft_save → draft_publish → recall with [mem:xxx] markers."""

    @pytest.mark.asyncio
    async def test_draft_save_creates_draft(self):
        """Step 1: op_draft_save creates a draft with source=pipeline, exp_status=draft."""
        exp_id = str(uuid.uuid4())
        mock_result = _make_draft_save_result(exp_id)

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

        # Verify the service was called with correct forced fields
        call_kw = mock_service.save.await_args.kwargs
        assert call_kw["source"] == "pipeline"
        assert call_kw["exp_status"] == "draft"

    @pytest.mark.asyncio
    async def test_draft_publish_transitions_to_published(self):
        """Step 2: op_draft_publish promotes draft → published."""
        exp_id = str(uuid.uuid4())
        mock_exp = _make_draft_get_by_id(exp_id)
        mock_updated = _make_published_get_by_id(exp_id)

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
            result = await fn(draft_id=exp_id)

        data = json.loads(result)
        assert data["id"] == exp_id
        assert data["status"] == "published"

        # Verify update was called with exp_status='published'
        update_kw = mock_service.update.await_args.kwargs
        assert update_kw["experience_id"] == exp_id
        assert update_kw["exp_status"] == "published"

    @pytest.mark.asyncio
    async def test_recall_returns_published_experience(self):
        """Step 3: op_recall finds the published experience with [mem:xxx] marker."""
        exp_id = str(uuid.uuid4())
        mock_results = _make_recall_result(exp_id)

        with (
            patch(f"{_O}._get_search_orchestrator") as mock_get_orch,
            patch(f"{_O}._get_db_url", return_value="sqlite://"),
            _patch_user(),
            patch(f"{_O}._resolve_project", return_value="default"),
            patch(f"{_O}.get_session") as mock_get_session,
            _patch_expansion(),
        ):
            mock_orchestrator = MagicMock()
            mock_orchestrator.search = AsyncMock(
                return_value=OrchestratedSearchResult(results=mock_results)
            )
            mock_get_orch.return_value = mock_orchestrator

            mock_session = AsyncMock()
            mock_get_session.return_value.__aenter__ = AsyncMock(return_value=mock_session)
            mock_get_session.return_value.__aexit__ = AsyncMock(return_value=False)

            tools = {t.name: t for t in await mcp.list_tools()}
            fn = tools["memory_recall"].fn
            result = await fn(problem="Connection pool exhausted")

        data = json.loads(result)
        assert "results" in data
        assert len(data["results"]) == 1
        assert data["results"][0]["id"] == exp_id

        # Verify [mem:xxx] marker in the result (id acts as the marker)
        result_id = data["results"][0].get("id") or data["results"][0].get("group_id")
        assert result_id == exp_id

    @pytest.mark.asyncio
    async def test_full_draft_publish_recall_flow(self):
        """Full end-to-end: draft_save → draft_publish → recall in sequence.

        Simulates the complete lifecycle of a pipeline draft going through
        save, publish, and then being discoverable via recall.
        """
        exp_id = str(uuid.uuid4())

        # ---- Step 1: draft_save ----
        draft_result = _make_draft_save_result(exp_id)
        with (
            patch(f"{_O}._get_service") as mock_get_service,
            patch(f"{_O}._get_db_url", return_value="sqlite://"),
            _patch_user("pipeline_user"),
            patch(f"{_O}._resolve_project", return_value="default"),
            patch(f"{_O}.get_session") as mock_gs,
        ):
            svc = MagicMock()
            svc.save = AsyncMock(return_value=draft_result)
            mock_get_service.return_value = svc

            s = AsyncMock()
            mock_gs.return_value.__aenter__ = AsyncMock(return_value=s)
            mock_gs.return_value.__aexit__ = AsyncMock(return_value=False)

            tools = {t.name: t for t in await mcp.list_tools()}
            save_result = await tools["memory_draft_save"].fn(
                title="Fix DB pool exhaustion",
                content="Connection pool exhausted under high concurrency",
                tags=["python", "database"],
            )

        save_data = json.loads(save_result)
        assert save_data["status"] == "draft"
        assert save_data["id"] == exp_id

        # ---- Step 2: draft_publish ----
        mock_exp_dict = _make_draft_get_by_id(exp_id)
        mock_published_dict = _make_published_get_by_id(exp_id)
        with (
            patch(f"{_O}._get_service") as mock_get_service,
            _patch_user("pipeline_user"),
        ):
            svc = MagicMock()
            svc.get_by_id = AsyncMock(return_value=mock_exp_dict)
            svc.update = AsyncMock(return_value=mock_published_dict)
            mock_get_service.return_value = svc

            publish_result = await tools["memory_draft_publish"].fn(
                draft_id=exp_id,
                refined_content="Increased pool size and added retry logic",
            )

        pub_data = json.loads(publish_result)
        assert pub_data["status"] == "published"
        assert pub_data["id"] == exp_id

        # Verify refined_content was passed through as description
        update_kw = svc.update.await_args.kwargs
        assert update_kw["description"] == "Increased pool size and added retry logic"

        # ---- Step 3: recall ----
        recall_results = [
            {
                "id": exp_id,
                "group_id": exp_id,
                "title": "Fix DB pool exhaustion",
                "solution": "Increased pool size and added retry logic",
                "score": 0.92,
                "confidence": "high",
                "type": "experience",
            }
        ]
        with (
            patch(f"{_O}._get_search_orchestrator") as mock_get_orch,
            patch(f"{_O}._get_db_url", return_value="sqlite://"),
            _patch_user("pipeline_user"),
            patch(f"{_O}._resolve_project", return_value="default"),
            patch(f"{_O}.get_session") as mock_gs,
            _patch_expansion(),
        ):
            orch = MagicMock()
            orch.search = AsyncMock(
                return_value=OrchestratedSearchResult(results=recall_results)
            )
            mock_get_orch.return_value = orch

            s = AsyncMock()
            mock_gs.return_value.__aenter__ = AsyncMock(return_value=s)
            mock_gs.return_value.__aexit__ = AsyncMock(return_value=False)

            recall_result = await tools["memory_recall"].fn(
                problem="DB connection pool exhaustion under high concurrency",
            )

        recall_data = json.loads(recall_result)
        assert len(recall_data["results"]) == 1
        assert recall_data["results"][0]["id"] == exp_id
        # [mem:xxx] marker: the experience id serves as the marker reference
        assert recall_data["results"][0]["title"] == "Fix DB pool exhaustion"

        # Verify the feedback_hint contains the [mem:xxx] style reference
        feedback_hint = recall_data.get("feedback_hint", "")
        assert exp_id in feedback_hint


# ============================================================
# 2. published → use_count++ → promotion flow
# ============================================================


class TestPublishedUseCountPromotionFlow:
    """End-to-end: published experience with incremented use_count gets promoted."""

    @pytest.mark.asyncio
    async def test_use_count_triggers_promotion(self):
        """Published experience with use_count >= threshold gets promoted."""
        exp = _make_experience(recall_count=3, exp_status="published")
        janitor = MemoryJanitor(
            db_url="sqlite+aiosqlite://",
            config=JanitorConfig(promotion_use_count_threshold=3),
        )

        mock_session = AsyncMock()
        mock_session.execute = AsyncMock(
            side_effect=[
                # 1. use_count query: returns 1 experience meeting threshold
                MagicMock(scalars=MagicMock(return_value=_FakeScalarResult([exp]))),
                # 2. group_key count query: no qualifying groups
                MagicMock(all=MagicMock(return_value=[])),
            ]
        )
        mock_session.commit = AsyncMock()

        with patch.object(janitor, "_session", lambda: _mock_session_ctx(mock_session)):
            result = await janitor.run_promotion()

        assert exp.exp_status == "promoted"
        assert result["promoted_by_recall_count"] == 1
        assert result["promoted_by_group"] == 0
        assert result["total"] == 1
        assert result["recall_count_threshold"] == 3

    @pytest.mark.asyncio
    async def test_below_threshold_not_promoted(self):
        """Published experience with use_count < threshold stays published."""
        exp = _make_experience(recall_count=2, exp_status="published")
        janitor = MemoryJanitor(
            db_url="sqlite+aiosqlite://",
            config=JanitorConfig(promotion_use_count_threshold=3),
        )

        mock_session = AsyncMock()
        mock_session.execute = AsyncMock(
            side_effect=[
                # 1. use_count query: no one meets threshold
                MagicMock(scalars=MagicMock(return_value=_FakeScalarResult([]))),
                # 2. group_key count query: no qualifying groups
                MagicMock(all=MagicMock(return_value=[])),
            ]
        )
        mock_session.commit = AsyncMock()

        with patch.object(janitor, "_session", lambda: _mock_session_ctx(mock_session)):
            result = await janitor.run_promotion()

        assert exp.exp_status == "published"
        assert result["promoted_by_recall_count"] == 0
        assert result["total"] == 0

    @pytest.mark.asyncio
    async def test_full_publish_increment_promote_flow(self):
        """Full flow: create published → increment use_count → run_promotion → promoted.

        Simulates:
        1. Save a published experience via memory_save
        2. Simulate use_count reaching threshold (via feedback / recall hits)
        3. Janitor run_promotion promotes it
        """
        exp_id = str(uuid.uuid4())

        # Step 1: Save a published experience via memory_save
        mock_save_result = {
            "id": exp_id,
            "title": "Redis caching pattern",
            "exp_status": "published",
        }
        with (
            patch(f"{_O}._get_service") as mock_get_service,
            patch(f"{_O}._get_db_url", return_value="sqlite://"),
            _patch_user(),
            patch(f"{_O}._resolve_project", return_value="default"),
            patch(f"{_O}.get_session") as mock_gs,
        ):
            svc = MagicMock()
            svc.save = AsyncMock(return_value=mock_save_result)
            mock_get_service.return_value = svc

            s = AsyncMock()
            mock_gs.return_value.__aenter__ = AsyncMock(return_value=s)
            mock_gs.return_value.__aexit__ = AsyncMock(return_value=False)

            tools = {t.name: t for t in await mcp.list_tools()}
            save_result = await tools["memory_save"].fn(
                title="Redis caching pattern",
                problem="Slow DB queries on repeated reads",
                solution="Implement Redis cache with 5min TTL",
            )

        save_data = json.loads(save_result)
        assert "data" in save_data
        assert save_data["data"]["title"] == "Redis caching pattern"

        # Step 2: Simulate use_count increment (3 uses via recall hits)
        # In production this happens when search_orchestrator calls increment_recall_count.
        # Here we directly create a mock experience with recall_count=3.
        promoted_exp = _make_experience(
            exp_id=uuid.UUID(exp_id),
            recall_count=3,
            exp_status="published",
        )

        # Step 3: Run Janitor promotion
        janitor = MemoryJanitor(
            db_url="sqlite+aiosqlite://",
            config=JanitorConfig(promotion_use_count_threshold=3),
        )

        mock_session = AsyncMock()
        mock_session.execute = AsyncMock(
            side_effect=[
                MagicMock(
                    scalars=MagicMock(return_value=_FakeScalarResult([promoted_exp]))
                ),
                MagicMock(all=MagicMock(return_value=[])),
            ]
        )
        mock_session.commit = AsyncMock()

        with patch.object(janitor, "_session", lambda: _mock_session_ctx(mock_session)):
            result = await janitor.run_promotion()

        # Verify promotion happened
        assert promoted_exp.exp_status == "promoted"
        assert result["promoted_by_recall_count"] == 1
        assert result["total"] == 1

    @pytest.mark.asyncio
    async def test_group_key_promotion_flow(self):
        """Multiple published experiences with same group_key get promoted together."""
        group_key = "auth-patterns"
        exps = [
            _make_experience(exp_status="published", recall_count=0, group_key=group_key)
            for _ in range(5)
        ]
        janitor = MemoryJanitor(
            db_url="sqlite+aiosqlite://",
            config=JanitorConfig(promotion_group_key_threshold=5),
        )

        mock_session = AsyncMock()
        group_row_0 = (group_key,)
        mock_session.execute = AsyncMock(
            side_effect=[
                # 1. use_count query: none
                MagicMock(scalars=MagicMock(return_value=_FakeScalarResult([]))),
                # 2. group_key count query: returns qualifying group
                MagicMock(all=MagicMock(return_value=[group_row_0])),
                # 3. group experiences query: returns 5 experiences
                MagicMock(scalars=MagicMock(return_value=_FakeScalarResult(exps))),
            ]
        )
        mock_session.commit = AsyncMock()

        with patch.object(janitor, "_session", lambda: _mock_session_ctx(mock_session)):
            result = await janitor.run_promotion()

        for exp in exps:
            assert exp.exp_status == "promoted"
        assert result["promoted_by_group"] == 5
        assert result["total"] == 5


# ============================================================
# 3. Regression: memory_save and memory_recall still work
# ============================================================


class TestMemorySaveRecallRegression:
    """Regression: existing memory_save and memory_recall paths remain functional."""

    @pytest.mark.asyncio
    async def test_memory_save_direct_still_works(self):
        """memory_save with title+problem still saves as published."""
        exp_id = str(uuid.uuid4())
        mock_result = {
            "id": exp_id,
            "title": "Fix DB timeout",
            "exp_status": "published",
        }

        with (
            patch(f"{_O}._get_service") as mock_get_service,
            patch(f"{_O}._get_db_url", return_value="sqlite://"),
            _patch_user(),
            patch(f"{_O}._resolve_project", return_value="default"),
            patch(f"{_O}.get_session") as mock_gs,
        ):
            svc = MagicMock()
            svc.save = AsyncMock(return_value=mock_result)
            mock_get_service.return_value = svc

            s = AsyncMock()
            mock_gs.return_value.__aenter__ = AsyncMock(return_value=s)
            mock_gs.return_value.__aexit__ = AsyncMock(return_value=False)

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
        svc.save.assert_called_once()

    @pytest.mark.asyncio
    async def test_memory_recall_solve_mode_still_works(self):
        """memory_recall with problem still routes to solve mode."""
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
            patch(f"{_O}._get_search_orchestrator") as mock_get_orch,
            patch(f"{_O}._get_db_url", return_value="sqlite://"),
            _patch_user(),
            patch(f"{_O}._resolve_project", return_value="default"),
            patch(f"{_O}.get_session") as mock_gs,
            _patch_expansion(),
        ):
            orch = MagicMock()
            orch.search = AsyncMock(
                return_value=OrchestratedSearchResult(results=mock_results)
            )
            mock_get_orch.return_value = orch

            s = AsyncMock()
            mock_gs.return_value.__aenter__ = AsyncMock(return_value=s)
            mock_gs.return_value.__aexit__ = AsyncMock(return_value=False)

            tools = {t.name: t for t in await mcp.list_tools()}
            fn = tools["memory_recall"].fn
            result = await fn(problem="Database connection timeout", language="python")

        data = json.loads(result)
        assert "results" in data
        assert len(data["results"]) == 1
        # Verify solve mode threshold
        call_args = orch.search.call_args
        assert call_args.kwargs["min_similarity"] == 0.5

    @pytest.mark.asyncio
    async def test_memory_recall_search_mode_still_works(self):
        """memory_recall with query still routes to search mode."""
        mock_results = [
            {
                "id": str(uuid.uuid4()),
                "group_id": str(uuid.uuid4()),
                "title": "Caching patterns",
                "score": 0.8,
            }
        ]

        with (
            patch(f"{_O}._get_search_orchestrator") as mock_get_orch,
            patch(f"{_O}._get_db_url", return_value="sqlite://"),
            _patch_user(),
            patch(f"{_O}._resolve_project", return_value="default"),
            _patch_expansion(),
        ):
            orch = MagicMock()
            orch.search = AsyncMock(
                return_value=OrchestratedSearchResult(results=mock_results)
            )
            mock_get_orch.return_value = orch

            tools = {t.name: t for t in await mcp.list_tools()}
            fn = tools["memory_recall"].fn
            result = await fn(query="redis caching best practices")

        data = json.loads(result)
        assert "results" in data
        call_args = orch.search.call_args
        assert call_args.kwargs["min_similarity"] == 0.6

    @pytest.mark.asyncio
    async def test_memory_save_with_content_still_works(self):
        """memory_save with content still routes to LLM parse."""
        exp_id = str(uuid.uuid4())
        mock_parsed = {
            "title": "Parsed Title",
            "problem": "Parsed problem",
            "solution": "Parsed solution",
            "tags": ["python"],
        }
        mock_result = {
            "id": exp_id,
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

        with (
            patch(f"{_O}._get_service") as mock_get_service,
            patch(f"{_O}._get_settings", return_value=mock_settings),
            patch(f"{_O}._get_db_url", return_value="sqlite://"),
            _patch_user(),
            patch(f"{_O}._resolve_project", return_value="default"),
            patch(f"{_O}.get_session") as mock_gs,
            patch(
                "team_memory.services.llm_parser.parse_content",
                new_callable=AsyncMock,
                return_value=mock_parsed,
            ),
            patch(f"{_O}._try_extract_and_save_personal_memory", new_callable=AsyncMock),
        ):
            svc = MagicMock()
            svc.save = AsyncMock(return_value=mock_result)
            mock_get_service.return_value = svc

            s = AsyncMock()
            mock_gs.return_value.__aenter__ = AsyncMock(return_value=s)
            mock_gs.return_value.__aexit__ = AsyncMock(return_value=False)

            tools = {t.name: t for t in await mcp.list_tools()}
            fn = tools["memory_save"].fn
            result = await fn(content="Long conversation about fixing a bug...")

        data = json.loads(result)
        assert "data" in data
        assert data["data"]["title"] == "Parsed Title"

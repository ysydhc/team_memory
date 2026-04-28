"""Tests for daemon pipeline logic."""
from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock

import pytest

from daemon.config import DaemonConfig, ProjectMapping, RetrievalSettings
from daemon.pipeline import (
    _resolve_project,
    process_after_response,
    process_before_prompt,
    process_session_end,
    process_session_start,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _make_config() -> DaemonConfig:
    return DaemonConfig(
        projects=[
            ProjectMapping(name="team_doc", path_patterns=["team_doc"]),
            ProjectMapping(name="ad_learning", path_patterns=["ad_learning"]),
        ],
        retrieval=RetrievalSettings(
            session_start_top_k=3,
            keyword_triggers=["之前", "上次", "经验", "踩坑"],
        ),
    )


def _make_buf(pending: list[dict] | None = None) -> AsyncMock:
    buf = AsyncMock()
    buf.get_pending = AsyncMock(return_value=pending or [])
    buf.update_draft = AsyncMock()
    return buf


def _make_detector(result: bool = False) -> MagicMock:
    det = MagicMock()
    det.detect_convergence = MagicMock(return_value=result)
    return det


def _make_refiner() -> AsyncMock:
    ref = AsyncMock()
    ref.save_draft = AsyncMock(return_value={"id": "draft-1"})
    ref.mark_for_refinement = AsyncMock(return_value=None)
    return ref


def _make_sink() -> AsyncMock:
    sink = AsyncMock()
    sink.context = AsyncMock(return_value={"experiences": []})
    sink.recall = AsyncMock(return_value=[])
    return sink


# ---------------------------------------------------------------------------
# _resolve_project
# ---------------------------------------------------------------------------

class TestResolveProject:
    def test_match_team_doc(self):
        config = _make_config()
        assert _resolve_project(["/Users/x/team_doc"], config) == "team_doc"

    def test_match_ad_learning(self):
        config = _make_config()
        assert _resolve_project(["/Users/x/ad_learning/src"], config) == "ad_learning"

    def test_no_match(self):
        config = _make_config()
        assert _resolve_project(["/Users/x/other_project"], config) is None

    def test_empty_roots(self):
        config = _make_config()
        assert _resolve_project([], config) is None


# ---------------------------------------------------------------------------
# process_after_response
# ---------------------------------------------------------------------------

class TestProcessAfterResponse:
    @pytest.mark.asyncio
    async def test_no_project_returns_ok(self):
        config = _make_config()
        result = await process_after_response(
            {"conversation_id": "123", "prompt": "hello", "workspace_roots": ["/unknown"]},
            config, _make_sink(), _make_buf(), _make_detector(), _make_refiner(),
        )
        assert result["action"] == "ok"
        assert result["convergence"] is False

    @pytest.mark.asyncio
    async def test_not_converged_saves_draft(self):
        config = _make_config()
        buf = _make_buf()
        refiner = _make_refiner()
        result = await process_after_response(
            {"conversation_id": "123", "prompt": "hello", "workspace_roots": ["/x/team_doc"]},
            config, _make_sink(), buf, _make_detector(result=False), refiner,
        )
        assert result["action"] == "draft_saved"
        assert result["convergence"] is False
        refiner.save_draft.assert_called_once()

    @pytest.mark.asyncio
    async def test_converged_with_existing_publishes(self):
        config = _make_config()
        buf = _make_buf(pending=[{"id": "d1", "content": "old text"}])
        refiner = _make_refiner()
        refiner.mark_for_refinement = AsyncMock(return_value={"draft_id": "d1"})
        result = await process_after_response(
            {"conversation_id": "123", "prompt": "问题解决了", "workspace_roots": ["/x/team_doc"]},
            config, _make_sink(), buf, _make_detector(result=True), refiner,
        )
        assert result["action"] == "needs_refinement"
        assert result["convergence"] is True
        buf.update_draft.assert_called_once()

    @pytest.mark.asyncio
    async def test_converged_without_existing_saves_and_publishes(self):
        config = _make_config()
        buf = _make_buf()
        refiner = _make_refiner()
        refiner.mark_for_refinement = AsyncMock(return_value={"draft_id": "draft-1"})
        result = await process_after_response(
            {"conversation_id": "123", "prompt": "总结完毕", "workspace_roots": ["/x/team_doc"]},
            config, _make_sink(), buf, _make_detector(result=True), refiner,
        )
        assert result["action"] == "needs_refinement"
        refiner.save_draft.assert_called_once()
        refiner.mark_for_refinement.assert_called_once()


# ---------------------------------------------------------------------------
# process_session_start
# ---------------------------------------------------------------------------

class TestProcessSessionStart:
    @pytest.mark.asyncio
    async def test_no_project_returns_empty(self):
        config = _make_config()
        result = await process_session_start(
            {"workspace_roots": ["/unknown"]}, config, _make_sink(),
        )
        assert result["additional_context"] == ""
        assert result["project"] is None

    @pytest.mark.asyncio
    async def test_with_project_returns_context(self):
        config = _make_config()
        sink = _make_sink()
        sink.context = AsyncMock(return_value={"experiences": [{"title": "test"}]})
        result = await process_session_start(
            {"workspace_roots": ["/x/team_doc"]}, config, sink,
        )
        assert result["project"] == "team_doc"
        sink.context.assert_called_once_with(project="team_doc")

    @pytest.mark.asyncio
    async def test_context_error_returns_empty(self):
        config = _make_config()
        sink = _make_sink()
        sink.context = AsyncMock(side_effect=Exception("boom"))
        result = await process_session_start(
            {"workspace_roots": ["/x/team_doc"]}, config, sink,
        )
        assert result["additional_context"] == ""
        assert result["project"] == "team_doc"


# ---------------------------------------------------------------------------
# process_before_prompt
# ---------------------------------------------------------------------------

class TestProcessBeforePrompt:
    @pytest.mark.asyncio
    async def test_keyword_trigger_recalls(self):
        config = _make_config()
        sink = _make_sink()
        sink.recall = AsyncMock(return_value=[{"title": "match"}])
        result = await process_before_prompt(
            {"prompt": "之前怎么解决的", "workspace_roots": ["/x/team_doc"]},
            config, sink,
        )
        assert len(result["results"]) == 1
        sink.recall.assert_called_once()

    @pytest.mark.asyncio
    async def test_no_trigger_no_query_returns_empty(self):
        config = _make_config()
        sink = _make_sink()
        result = await process_before_prompt(
            {"prompt": "", "workspace_roots": ["/x/team_doc"]},
            config, sink,
        )
        assert result["results"] == []
        sink.recall.assert_not_called()


# ---------------------------------------------------------------------------
# process_session_end
# ---------------------------------------------------------------------------

class TestProcessSessionEnd:
    @pytest.mark.asyncio
    async def test_no_pending_returns_ok(self):
        config = _make_config()
        buf = _make_buf()
        result = await process_session_end(
            {"conversation_id": "123"}, config, _make_sink(), buf, _make_refiner(),
        )
        assert result["action"] == "ok"
        assert result["flushed"] is False

    @pytest.mark.asyncio
    async def test_pending_draft_gets_published(self):
        config = _make_config()
        buf = _make_buf(pending=[{"id": "d1", "content": "text"}])
        refiner = _make_refiner()
        refiner.refine_and_publish = AsyncMock(return_value={"draft_id": "d1"})
        result = await process_session_end(
            {"conversation_id": "123"}, config, _make_sink(), buf, refiner,
        )
        assert result["action"] == "published"
        assert result["flushed"] is True

    @pytest.mark.asyncio
    async def test_refiner_failure_returns_ok(self):
        config = _make_config()
        buf = _make_buf(pending=[{"id": "d1", "content": "text"}])
        refiner = _make_refiner()
        refiner.refine_and_publish = AsyncMock(side_effect=Exception("boom"))
        result = await process_session_end(
            {"conversation_id": "123"}, config, _make_sink(), buf, refiner,
        )
        assert result["action"] == "ok"
        assert result["flushed"] is False

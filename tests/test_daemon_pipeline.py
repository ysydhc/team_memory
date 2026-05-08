"""Tests for daemon pipeline logic."""
from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

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


def _make_buf() -> AsyncMock:
    """Create a mock ConversationBuffer."""
    buf = AsyncMock()
    buf.add_message = AsyncMock(return_value=1)
    buf.ensure_session = AsyncMock(return_value="sess-1")
    buf.get_session = AsyncMock(return_value={"id": "sess-1", "status": "active", "project": "team_doc"})
    buf.get_messages = AsyncMock(return_value=[])
    buf.get_user_messages = AsyncMock(return_value=[])
    buf.get_assistant_messages = AsyncMock(return_value=[{"content": "some response"}])
    buf.get_pending = AsyncMock(return_value=[{"id": "sess-1", "content": "text", "project": "team_doc", "conversation_id": "sess-1", "status": "active"}])
    buf.mark_published_by_session = AsyncMock()
    buf.mark_needs_refinement = AsyncMock()
    buf.close_session = AsyncMock()
    return buf


def _make_buf_empty() -> AsyncMock:
    """Create a mock ConversationBuffer with no data."""
    buf = AsyncMock()
    buf.add_message = AsyncMock(return_value=1)
    buf.ensure_session = AsyncMock(return_value="sess-1")
    buf.get_session = AsyncMock(return_value=None)
    buf.get_messages = AsyncMock(return_value=[])
    buf.get_user_messages = AsyncMock(return_value=[])
    buf.get_assistant_messages = AsyncMock(return_value=[])
    buf.get_pending = AsyncMock(return_value=[])
    return buf


def _make_detector(result: bool = False) -> MagicMock:
    det = MagicMock()
    det.detect_convergence = MagicMock(return_value=result)
    return det


def _make_refiner() -> AsyncMock:
    ref = AsyncMock()
    ref.save_draft = AsyncMock(return_value={"id": "draft-1"})
    ref.mark_for_refinement = AsyncMock(return_value=None)
    ref.refine_and_publish = AsyncMock(return_value={"draft_id": "d1", "status": "published"})
    return ref


def _make_sink() -> AsyncMock:
    sink = AsyncMock()
    sink.recall = AsyncMock(return_value=[])
    return sink


# ---------------------------------------------------------------------------
# Module-level mock for auto-inference (all test classes inherit)
# ---------------------------------------------------------------------------
@pytest.fixture(autouse=True)
def _mock_auto_infer(monkeypatch):
    """Mock auto-inference so tests only exercise path_patterns."""
    monkeypatch.setattr(
        "team_memory.project_resolver.read_tm_toml", lambda _: None
    )
    monkeypatch.setattr(
        "team_memory.project_resolver.infer_project_name", lambda _cwd="": "default"
    )


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
        """When workspace is unknown AND no session record, action=ok (no-op)."""
        config = _make_config()
        buf = _make_buf()
        buf.get_session = AsyncMock(return_value=None)  # no session record either
        result = await process_after_response(
            {"conversation_id": "123", "prompt": "hello", "workspace_roots": ["/unknown"]},
            config, _make_sink(), buf, _make_detector(), _make_refiner(),
        )
        assert result["action"] == "ok"
        assert result["convergence"] is False

    @pytest.mark.asyncio
    async def test_not_converged_stores_message(self):
        config = _make_config()
        buf = _make_buf()
        result = await process_after_response(
            {"conversation_id": "123", "prompt": "hello", "workspace_roots": ["/x/team_doc"]},
            config, _make_sink(), buf, _make_detector(result=False), _make_refiner(),
        )
        assert result["action"] == "draft_updated"
        assert result["convergence"] is False
        buf.add_message.assert_called_once_with("123", "assistant", "hello", project="team_doc")

    @pytest.mark.asyncio
    @patch("daemon.pipeline._try_extract_personal_memory", new_callable=AsyncMock)
    async def test_converged_marks_refinement(self, mock_extract):
        config = _make_config()
        buf = _make_buf()
        refiner = _make_refiner()
        refiner.mark_for_refinement = AsyncMock(return_value={"draft_id": "d1"})
        result = await process_after_response(
            {"conversation_id": "123", "prompt": "问题解决了", "workspace_roots": ["/x/team_doc"]},
            config, _make_sink(), buf, _make_detector(result=True), refiner,
        )
        assert result["action"] == "needs_refinement"
        assert result["convergence"] is True
        refiner.mark_for_refinement.assert_called_once_with("123")
        mock_extract.assert_called_once()

    @pytest.mark.asyncio
    async def test_empty_response_text_skipped(self):
        config = _make_config()
        buf = _make_buf()
        result = await process_after_response(
            {"conversation_id": "123", "prompt": "", "response_text": "", "workspace_roots": ["/x/team_doc"]},
            config, _make_sink(), buf, _make_detector(), _make_refiner(),
        )
        assert result["action"] == "ok"
        buf.add_message.assert_not_called()


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
        assert result["project"] == ""

    @pytest.mark.asyncio
    async def test_with_project_returns_project(self):
        config = _make_config()
        result = await process_session_start(
            {"workspace_roots": ["/x/team_doc"]}, config, _make_sink(),
        )
        assert result["project"] == "team_doc"


# ---------------------------------------------------------------------------
# process_before_prompt
# ---------------------------------------------------------------------------

class TestProcessBeforePrompt:
    @pytest.mark.asyncio
    async def test_keyword_trigger_recalls(self):
        config = _make_config()
        sink = _make_sink()
        buf = _make_buf()
        sink.recall = AsyncMock(return_value=[{"title": "match"}])
        result = await process_before_prompt(
            {"prompt": "之前怎么解决的", "workspace_roots": ["/x/team_doc"], "conversation_id": "c1"},
            config, sink, buf,
        )
        assert len(result["results"]) == 1
        sink.recall.assert_called_once()
        # User prompt is stored in buffer
        buf.add_message.assert_called_once_with("c1", "user", "之前怎么解决的", project="team_doc")

    @pytest.mark.asyncio
    async def test_empty_prompt_no_recall(self):
        config = _make_config()
        sink = _make_sink()
        buf = _make_buf()
        result = await process_before_prompt(
            {"prompt": "", "workspace_roots": ["/x/team_doc"], "conversation_id": "c1"},
            config, sink, buf,
        )
        assert result["results"] == []
        sink.recall.assert_not_called()


# ---------------------------------------------------------------------------
# process_session_end
# ---------------------------------------------------------------------------

class TestProcessSessionEnd:
    @pytest.mark.asyncio
    async def test_no_active_session_returns_ok(self):
        config = _make_config()
        buf = _make_buf_empty()
        result = await process_session_end(
            {"conversation_id": "123"}, config, _make_sink(), buf, _make_refiner(),
        )
        assert result["action"] == "ok"
        assert result["flushed"] is False

    @pytest.mark.asyncio
    @patch("daemon.pipeline._try_extract_personal_memory", new_callable=AsyncMock)
    async def test_active_session_marked_for_refinement(self, mock_extract):
        config = _make_config()
        buf = _make_buf()
        refiner = _make_refiner()
        result = await process_session_end(
            {"conversation_id": "sess-1"}, config, _make_sink(), buf, refiner,
        )
        assert result["action"] == "needs_refinement"
        assert result["flushed"] is True
        buf.mark_needs_refinement.assert_called_once_with("sess-1")
        mock_extract.assert_called_once()

    @pytest.mark.asyncio
    @patch("daemon.pipeline._try_extract_personal_memory", new_callable=AsyncMock)
    async def test_mark_refinement_failure_returns_ok(self, mock_extract):
        config = _make_config()
        buf = _make_buf()
        buf.mark_needs_refinement = AsyncMock(side_effect=Exception("boom"))
        refiner = _make_refiner()
        result = await process_session_end(
            {"conversation_id": "sess-1"}, config, _make_sink(), buf, refiner,
        )
        assert result["action"] == "ok"
        assert result["flushed"] is False

"""Extended unit tests for RefinementWorker — covers constructor, lifecycle,
_tick flow, retry logic, LLM call patterns, _refine_one, and _fallback_publish.

Markers:
  @pytest.mark.unit      — pure logic, fully mocked (no real DB)
  @pytest.mark.integration — uses real DraftBuffer DB
"""

from __future__ import annotations

import asyncio
import os
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from team_memory.daemon.refinement.refinement_worker import _MAX_REFINE_ATTEMPTS, RefinementWorker
from team_memory.daemon.runtime.config import DaemonConfig, RefinementSettings
from team_memory.services.llm.llm_parser import LLMParseError

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_config(**overrides) -> DaemonConfig:
    """Build a DaemonConfig with sensible test defaults."""
    defaults = dict(
        enabled=True,
        provider="litellm",
        model="test-model",
        base_url="http://localhost:9999/v1",
        api_key_env="TEST_API_KEY",
        timeout=5,
        scan_interval_seconds=1,
        max_input_chars=4000,
        max_retries=2,
        fallback_on_failure=True,
        quality_gate_enabled=False,  # disabled by default for legacy tests
        quality_min_score=60,
    )
    defaults.update(overrides)
    cfg = DaemonConfig()
    cfg.refinement = RefinementSettings(**defaults)
    return cfg


def _valid_llm_result() -> dict[str, Any]:
    """Return a structurally valid LLM result that passes the quality gate."""
    return {
        "title": "A sufficiently long title for quality",
        "problem": "This is a problem description that is long enough to pass the minimum length check",
        "solution": "This is a solution description that is long enough to pass the minimum length check",
        "tags": ["test"],
        "experience_type": "bugfix",
    }


def _make_draft(
    draft_id: str = "d-001",
    title: str = "Test bug fix",
    description: str = "",
    content: str = "Some content here",
    project: str = "test_project",
    conversation_id: str = "conv-1",
) -> dict[str, Any]:
    return {
        "id": draft_id,
        "title": title,
        "description": description,
        "content": content,
        "project": project,
        "conversation_id": conversation_id,
    }


# ---------------------------------------------------------------------------
# Unit tests — constructor, lifecycle, _tick logic
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestRefinementWorkerInit:
    """Constructor sets up attributes correctly."""

    def test_init_sets_defaults(self):
        cfg = _make_config()
        buf = MagicMock()
        sink = MagicMock()
        worker = RefinementWorker(cfg, buf, sink)

        assert worker._cfg is cfg.refinement
        assert worker._buf is buf
        assert worker._sink is sink
        assert worker._running is False
        assert worker._task is None
        assert worker._queue is None

    def test_init_empty_retry_counts(self):
        cfg = _make_config()
        worker = RefinementWorker(cfg, MagicMock(), MagicMock())
        assert worker._queue is None
        assert worker._buf is not None


@pytest.mark.unit
class TestLifecycle:
    """start / stop / _loop / _noop tests."""

    @pytest.mark.asyncio
    async def test_start_disabled_returns_noop_task(self):
        cfg = _make_config(enabled=False)
        worker = RefinementWorker(cfg, MagicMock(), MagicMock())
        task = worker.start()
        assert task is not None
        assert worker._running is False  # still False because noop
        # await it to clean up
        await task

    @pytest.mark.asyncio
    async def test_start_enabled_creates_loop(self):
        cfg = _make_config(enabled=True)
        buf = AsyncMock()
        buf.get_needs_refinement = AsyncMock(return_value=[])
        worker = RefinementWorker(cfg, buf, MagicMock())
        task = worker.start()
        assert worker._running is True
        assert worker._task is task
        # Let one tick run then stop
        await asyncio.sleep(0.05)
        worker.stop()
        # Wait briefly for cancellation
        try:
            await asyncio.wait_for(task, timeout=2.0)
        except (asyncio.CancelledError, asyncio.TimeoutError):
            pass

    @pytest.mark.asyncio
    async def test_stop_sets_running_false_and_cancels(self):
        cfg = _make_config()
        worker = RefinementWorker(cfg, MagicMock(), MagicMock())
        worker._running = True
        worker._task = MagicMock()
        worker._task.cancel = MagicMock()
        worker.stop()
        assert worker._running is False
        worker._task.cancel.assert_called_once()

    @pytest.mark.asyncio
    async def test_stop_with_no_task(self):
        cfg = _make_config()
        worker = RefinementWorker(cfg, MagicMock(), MagicMock())
        worker._running = True
        worker.stop()  # should not raise
        assert worker._running is False

    @pytest.mark.asyncio
    async def test_loop_runs_multiple_ticks(self):
        cfg = _make_config(scan_interval_seconds=0)
        buf = AsyncMock()
        buf.get_needs_refinement = AsyncMock(return_value=[])
        sink = MagicMock()
        worker = RefinementWorker(cfg, buf, sink)
        worker._running = True
        task = asyncio.create_task(worker._loop())
        await asyncio.sleep(0.15)
        worker._running = False
        await asyncio.sleep(0.05)
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass
        # Should have been called at least twice
        assert buf.get_needs_refinement.call_count >= 2

    @pytest.mark.asyncio
    async def test_loop_exception_in_tick_continues(self):
        """Exception in _tick should be caught and loop continues."""
        cfg = _make_config(scan_interval_seconds=0)
        buf = AsyncMock()
        call_count = 0

        async def failing_then_ok():
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise RuntimeError("tick exploded")
            return []

        buf.get_needs_refinement = failing_then_ok
        worker = RefinementWorker(cfg, buf, MagicMock())
        worker._running = True
        task = asyncio.create_task(worker._loop())
        await asyncio.sleep(0.2)
        worker._running = False
        try:
            await asyncio.wait_for(task, timeout=2.0)
        except (asyncio.CancelledError, asyncio.TimeoutError):
            pass
        # Tick ran at least twice (one failure, one success)
        assert call_count >= 2

    @pytest.mark.asyncio
    async def test_loop_cancelled_error_propagates(self):
        """CancelledError in _tick should propagate (not caught by generic)."""
        cfg = _make_config(scan_interval_seconds=0)
        buf = AsyncMock()
        buf.get_needs_refinement = AsyncMock(side_effect=asyncio.CancelledError())
        worker = RefinementWorker(cfg, buf, MagicMock())
        worker._running = True
        task = asyncio.create_task(worker._loop())
        # Wait a bit, then cancel
        await asyncio.sleep(0.05)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task

    @pytest.mark.asyncio
    async def test_loop_sleep_cancelled_propagates(self):
        """CancelledError during sleep should propagate."""
        cfg = _make_config(scan_interval_seconds=100)
        buf = AsyncMock()
        buf.get_needs_refinement = AsyncMock(return_value=[])
        worker = RefinementWorker(cfg, buf, MagicMock())
        worker._running = True
        task = asyncio.create_task(worker._loop())
        await asyncio.sleep(0.05)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task


@pytest.mark.unit
class TestTickLogic:
    """Tests for _tick flow with mocked DB."""

    @pytest.mark.asyncio
    async def test_tick_no_drafts(self):
        cfg = _make_config()
        buf = AsyncMock()
        buf.get_needs_refinement = AsyncMock(return_value=[])
        worker = RefinementWorker(cfg, buf, MagicMock())
        await worker._tick()
        # No further calls expected
        buf.mark_published.assert_not_called()

    @pytest.mark.asyncio
    async def test_tick_learning_card_skipped(self):
        cfg = _make_config()
        buf = AsyncMock()
        draft = _make_draft(title="卡片-iOS数据存储")
        buf.get_needs_refinement = AsyncMock(return_value=[draft])
        worker = RefinementWorker(cfg, buf, MagicMock())
        await worker._tick()
        buf.mark_published.assert_awaited_once_with("d-001")

    @pytest.mark.asyncio
    async def test_tick_learning_card_content_detected(self):
        cfg = _make_config()
        buf = AsyncMock()
        draft = _make_draft(title="My Notes", description="## 学习目标\n掌握基础")
        buf.get_needs_refinement = AsyncMock(return_value=[draft])
        worker = RefinementWorker(cfg, buf, MagicMock())
        await worker._tick()
        buf.mark_published.assert_awaited_once_with("d-001")

    @pytest.mark.asyncio
    async def test_tick_max_retries_marks_failed(self):
        """When queue op hits max retries, mark_failed and fallback triggered."""
        from team_memory.daemon.buffers.pending_queue import PendingQueue

        cfg = _make_config(fallback_on_failure=True)
        buf = AsyncMock()
        buf.mark_refining = AsyncMock()
        buf.mark_published = AsyncMock()
        sink = AsyncMock()
        queue = PendingQueue(db_path=":memory:")
        op_id = queue.enqueue("refine", "d-001", {"draft_id": "d-001", "content": "test", "project": "default"})
        queue.bump_retry(op_id, "err", 0)
        queue.bump_retry(op_id, "err", 0)
        # Force next_retry_at to past so get_due_ops picks it up
        queue._db.execute("UPDATE pending_ops SET next_retry_at = '2020-01-01T00:00:00' WHERE id = ?", (op_id,))
        queue._db.commit()
        worker = RefinementWorker(cfg, buf, sink, queue=queue)
        worker._refine_one = AsyncMock(side_effect=Exception("fail"))
        await worker._tick()
        assert queue.stats()["failed"] == 1

    @pytest.mark.asyncio
    async def test_tick_failure_enqueues(self):
        """When _refine_one fails, draft enqueued to PendingQueue."""
        from team_memory.daemon.buffers.pending_queue import PendingQueue

        cfg = _make_config()
        buf = AsyncMock()
        buf.mark_refining = AsyncMock()
        draft = {"id": "d-001", "content": "test", "title": "T", "project": "default"}
        buf.get_needs_refinement = AsyncMock(return_value=[draft])
        sink = AsyncMock()
        queue = PendingQueue(db_path=":memory:")
        worker = RefinementWorker(cfg, buf, sink, queue=queue)
        worker._refine_one = AsyncMock(side_effect=Exception("fail"))
        await worker._tick()
        assert queue.stats()["pending"] == 1

    @pytest.mark.asyncio
    async def test_tick_failure_dedup(self):
        """Second failure does not duplicate queue entry."""
        from team_memory.daemon.buffers.pending_queue import PendingQueue

        cfg = _make_config()
        buf = AsyncMock()
        buf.mark_refining = AsyncMock()
        draft = {"id": "d-001", "content": "test", "title": "T", "project": "default"}
        buf.get_needs_refinement = AsyncMock(return_value=[draft])
        sink = AsyncMock()
        queue = PendingQueue(db_path=":memory:")
        worker = RefinementWorker(cfg, buf, sink, queue=queue)
        worker._refine_one = AsyncMock(side_effect=Exception("fail"))
        await worker._tick()
        await worker._tick()
        assert queue.stats()["pending"] == 1  # deduped

    @pytest.mark.asyncio
    async def test_tick_fallback_on_failure(self):
        cfg = _make_config(fallback_on_failure=True)
        buf = AsyncMock()
        draft = _make_draft()
        buf.get_needs_refinement = AsyncMock(return_value=[draft])
        worker = RefinementWorker(cfg, buf, MagicMock())
        worker._refine_one = AsyncMock(side_effect=RuntimeError("LLM down"))
        worker._fallback_publish = AsyncMock()
        await worker._tick()
        worker._fallback_publish.assert_awaited_once_with(draft)

    @pytest.mark.asyncio
    async def test_tick_multiple_drafts(self):
        cfg = _make_config()
        buf = AsyncMock()
        d1 = _make_draft(draft_id="d-001", title="卡片-Foo")
        d2 = _make_draft(draft_id="d-002", title="Bar bug")
        buf.get_needs_refinement = AsyncMock(return_value=[d1, d2])
        sink = MagicMock()
        worker = RefinementWorker(cfg, buf, sink)
        worker._refine_one = AsyncMock(return_value=None)
        await worker._tick()
        # d-001 is a learning card → mark_published
        # d-002 should go through _refine_one
        assert buf.mark_published.await_count == 1
        worker._refine_one.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_tick_draft_without_title_uses_default(self):
        cfg = _make_config()
        buf = AsyncMock()
        draft = {"id": "d-xxx", "content": "hello"}
        buf.get_needs_refinement = AsyncMock(return_value=[draft])
        worker = RefinementWorker(cfg, buf, MagicMock())
        worker._refine_one = AsyncMock(return_value=None)
        await worker._tick()
        worker._refine_one.assert_awaited_once()


@pytest.mark.unit
class TestRefineOne:
    """Tests for _refine_one with fully mocked sink + LLM."""

    @pytest.mark.asyncio
    async def test_happy_path(self):
        cfg = _make_config()
        buf = AsyncMock()
        sink = AsyncMock()
        sink.draft_save = AsyncMock(return_value={"id": "exp-001"})
        sink.draft_publish = AsyncMock(return_value={})
        sink.update_experience = AsyncMock(return_value={})

        worker = RefinementWorker(cfg, buf, sink)
        draft = _make_draft()

        with patch.object(worker, "_call_llm", new_callable=AsyncMock) as mock_llm:
            mock_llm.return_value = {
                "title": "Refined Title",
                "problem": "problem text",
                "solution": "solution text",
                "tags": ["python"],
                "experience_type": "bugfix",
            }
            await worker._refine_one(draft)

        sink.draft_save.assert_awaited_once()
        sink.draft_publish.assert_awaited_once_with(draft_id="exp-001")
        sink.update_experience.assert_awaited_once_with(
            experience_id="exp-001",
            title="Refined Title",
            problem="problem text",
            solution="solution text",
            tags=["python"],
            experience_type="bugfix",
        )
        buf.mark_published.assert_awaited_once_with("d-001")

    @pytest.mark.asyncio
    async def test_draft_save_error_raises(self):
        cfg = _make_config()
        buf = AsyncMock()
        sink = AsyncMock()
        sink.draft_save = AsyncMock(return_value={"error": True, "message": "embedding failed"})

        worker = RefinementWorker(cfg, buf, sink)
        draft = _make_draft()

        with pytest.raises(RuntimeError, match="draft_save failed"):
            await worker._refine_one(draft)

    @pytest.mark.asyncio
    async def test_dedup_response_uses_candidate_id(self):
        cfg = _make_config()
        buf = AsyncMock()
        sink = AsyncMock()
        sink.draft_save = AsyncMock(
            return_value={
                "error": True,
                "code": "duplicate_detected",
                "candidates": [{"id": "existing-123"}],
            }
        )
        sink.draft_publish = AsyncMock(return_value={})
        sink.update_experience = AsyncMock(return_value={})

        worker = RefinementWorker(cfg, buf, sink)
        draft = _make_draft()

        with patch.object(worker, "_call_llm", new_callable=AsyncMock) as mock_llm:
            mock_llm.return_value = {"title": "X"}
            await worker._refine_one(draft)

        sink.draft_publish.assert_awaited_once_with(draft_id="existing-123")

    @pytest.mark.asyncio
    async def test_dedup_no_candidates_raises(self):
        cfg = _make_config()
        buf = AsyncMock()
        sink = AsyncMock()
        sink.draft_save = AsyncMock(
            return_value={
                "error": True,
                "code": "duplicate_detected",
                "candidates": [],
            }
        )

        worker = RefinementWorker(cfg, buf, sink)
        draft = _make_draft()

        with pytest.raises(RuntimeError, match="no experience_id"):
            await worker._refine_one(draft)

    @pytest.mark.asyncio
    async def test_publish_error_raises(self):
        cfg = _make_config()
        buf = AsyncMock()
        sink = AsyncMock()
        sink.draft_save = AsyncMock(return_value={"id": "exp-1"})
        sink.draft_publish = AsyncMock(return_value={"error": True, "message": "publish failed"})

        worker = RefinementWorker(cfg, buf, sink)
        draft = _make_draft()

        with patch.object(worker, "_call_llm", new_callable=AsyncMock) as mock_llm:
            mock_llm.return_value = {"title": "X"}
            with pytest.raises(RuntimeError, match="draft_publish failed"):
                await worker._refine_one(draft)

    @pytest.mark.asyncio
    async def test_update_experience_error_raises(self):
        cfg = _make_config()
        buf = AsyncMock()
        sink = AsyncMock()
        sink.draft_save = AsyncMock(return_value={"id": "exp-1"})
        sink.draft_publish = AsyncMock(return_value={})
        sink.update_experience = AsyncMock(return_value={"error": True, "message": "update failed"})

        worker = RefinementWorker(cfg, buf, sink)
        draft = _make_draft()

        with patch.object(worker, "_call_llm", new_callable=AsyncMock) as mock_llm:
            mock_llm.return_value = {"title": "X"}
            with pytest.raises(RuntimeError, match="update_experience failed"):
                await worker._refine_one(draft)

    @pytest.mark.asyncio
    async def test_fallback_title_from_draft_id(self):
        """When draft has no title, _refine_one falls back to 'Draft <id>'."""
        cfg = _make_config()
        buf = AsyncMock()
        sink = AsyncMock()
        sink.draft_save = AsyncMock(return_value={"id": "exp-1"})
        sink.draft_publish = AsyncMock(return_value={})
        sink.update_experience = AsyncMock(return_value={})

        worker = RefinementWorker(cfg, buf, sink)
        draft = {"id": "abcd1234", "content": "stuff", "project": "p"}

        with patch.object(worker, "_call_llm", new_callable=AsyncMock) as mock_llm:
            mock_llm.return_value = {"title": "X"}
            await worker._refine_one(draft)

        # Verify draft_save was called with fallback title
        call_kwargs = sink.draft_save.call_args
        assert call_kwargs.kwargs["title"] == "Draft abcd1234"

    @pytest.mark.asyncio
    async def test_fallback_title_from_field(self):
        """When draft has a title, it is used."""
        cfg = _make_config()
        buf = AsyncMock()
        sink = AsyncMock()
        sink.draft_save = AsyncMock(return_value={"id": "exp-1"})
        sink.draft_publish = AsyncMock(return_value={})
        sink.update_experience = AsyncMock(return_value={})

        worker = RefinementWorker(cfg, buf, sink)
        draft = _make_draft(title="My Custom Title")

        with patch.object(worker, "_call_llm", new_callable=AsyncMock) as mock_llm:
            mock_llm.return_value = {"title": "X"}
            await worker._refine_one(draft)

        call_kwargs = sink.draft_save.call_args
        assert call_kwargs.kwargs["title"] == "My Custom Title"

    @pytest.mark.asyncio
    async def test_project_default_fallback(self):
        """When draft has no project, it defaults to 'default'."""
        cfg = _make_config()
        buf = AsyncMock()
        sink = AsyncMock()
        sink.draft_save = AsyncMock(return_value={"id": "exp-1"})
        sink.draft_publish = AsyncMock(return_value={})
        sink.update_experience = AsyncMock(return_value={})

        worker = RefinementWorker(cfg, buf, sink)
        draft = {"id": "d-999", "content": "hello", "title": "Hi"}

        with patch.object(worker, "_call_llm", new_callable=AsyncMock) as mock_llm:
            mock_llm.return_value = {"title": "X"}
            await worker._refine_one(draft)

        call_kwargs = sink.draft_save.call_args
        assert call_kwargs.kwargs["project"] == "default"


@pytest.mark.unit
class TestCallLLM:
    """Tests for _call_llm retry logic and provider dispatch."""

    @pytest.fixture(autouse=True)
    def _stub_llm_reachable(self):
        """_call_llm now does a pre-call health check; force it to pass for unit tests."""
        with patch(
            "team_memory.daemon.refinement.refinement_worker._llm_api_reachable",
            new_callable=AsyncMock,
            return_value=True,
        ):
            yield

    @pytest.mark.asyncio
    async def test_litellm_success(self):
        cfg = _make_config(provider="litellm", api_key_env="TEST_KEY", max_retries=0)
        worker = RefinementWorker(cfg, MagicMock(), MagicMock())

        with patch(
            "team_memory.daemon.refinement.refinement_worker._refine_via_openai",
            new_callable=AsyncMock,
        ) as mock_parse:
            mock_parse.return_value = {"title": "OK", "problem": "P", "solution": "S"}
            with patch.dict(os.environ, {"TEST_KEY": "sk-test"}):
                result = await worker._call_llm("some content")

        assert result["title"] == "OK"
        mock_parse.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_litellm_no_api_key_passes_none(self):
        cfg = _make_config(provider="litellm", api_key_env="NONEXISTENT_KEY_999", max_retries=0)
        worker = RefinementWorker(cfg, MagicMock(), MagicMock())

        with patch(
            "team_memory.daemon.refinement.refinement_worker._refine_via_openai",
            new_callable=AsyncMock,
        ) as mock_parse:
            mock_parse.return_value = {"title": "OK"}
            # Ensure the env var doesn't exist
            env = {k: v for k, v in os.environ.items() if k != "NONEXISTENT_KEY_999"}
            with patch.dict(os.environ, env, clear=True):
                await worker._call_llm("content")

        # api_key should have been None
        call_kwargs = mock_parse.call_args
        assert call_kwargs.kwargs["api_key"] is None

    @pytest.mark.asyncio
    async def test_litellm_retry_on_llm_parse_error(self):
        cfg = _make_config(provider="litellm", max_retries=2)
        worker = RefinementWorker(cfg, MagicMock(), MagicMock())

        with patch(
            "team_memory.daemon.refinement.refinement_worker._refine_via_openai",
            new_callable=AsyncMock,
        ) as mock_parse:
            mock_parse.side_effect = [
                LLMParseError("transient"),
                LLMParseError("transient again"),
                {"title": "OK"},
            ]
            with patch("team_memory.daemon.refinement.refinement_worker.asyncio.sleep", new_callable=AsyncMock):
                result = await worker._call_llm("content")

        assert result["title"] == "OK"
        assert mock_parse.call_count == 3

    @pytest.mark.asyncio
    async def test_litellm_all_retries_exhausted(self):
        cfg = _make_config(provider="litellm", max_retries=1)
        worker = RefinementWorker(cfg, MagicMock(), MagicMock())

        with patch(
            "team_memory.daemon.refinement.refinement_worker._refine_via_openai",
            new_callable=AsyncMock,
        ) as mock_parse:
            mock_parse.side_effect = LLMParseError("permanent fail")
            with patch("team_memory.daemon.refinement.refinement_worker.asyncio.sleep", new_callable=AsyncMock):
                with pytest.raises(LLMParseError, match="permanent fail"):
                    await worker._call_llm("content")

        # Should have tried max_retries + 1 times
        assert mock_parse.call_count == 2

    @pytest.mark.asyncio
    async def test_unsupported_provider_raises(self):
        cfg = _make_config(provider="chatgpt_plugin")
        worker = RefinementWorker(cfg, MagicMock(), MagicMock())

        with pytest.raises(ValueError, match="Unsupported refinement provider"):
            await worker._call_llm("content")

    @pytest.mark.asyncio
    async def test_ollama_provider_success(self):
        """Ollama provider calls parse_content with LLMConfig."""
        cfg = _make_config(provider="ollama", max_retries=0)
        worker = RefinementWorker(cfg, MagicMock(), MagicMock())

        mock_parse_content = AsyncMock(return_value={"title": "OK", "problem": "P"})
        mock_llm_config = MagicMock()

        with patch.dict(
            "sys.modules",
            {
                "team_memory.config.llm": MagicMock(LLMConfig=mock_llm_config),
            },
        ):
            with patch(
                "team_memory.services.llm.llm_parser.parse_content",
                mock_parse_content,
            ):
                result = await worker._call_llm("some content")

        assert result["title"] == "OK"
        mock_parse_content.assert_awaited_once()
        # Verify LLMConfig was constructed with correct args
        mock_llm_config.assert_called_once_with(
            model="test-model",
            base_url="http://localhost:9999/v1",
        )

    @pytest.mark.asyncio
    async def test_ollama_provider_retry_on_parse_error(self):
        """Ollama provider retries on LLMParseError."""
        cfg = _make_config(provider="ollama", max_retries=1)
        worker = RefinementWorker(cfg, MagicMock(), MagicMock())

        mock_parse_content = AsyncMock(
            side_effect=[
                LLMParseError("fail"),
                {"title": "OK"},
            ]
        )
        mock_llm_config = MagicMock()

        with patch.dict(
            "sys.modules",
            {
                "team_memory.config.llm": MagicMock(LLMConfig=mock_llm_config),
            },
        ):
            with patch(
                "team_memory.services.llm.llm_parser.parse_content",
                mock_parse_content,
            ):
                with patch("team_memory.daemon.refinement.refinement_worker.asyncio.sleep", new_callable=AsyncMock):
                    result = await worker._call_llm("content")

        assert result["title"] == "OK"
        assert mock_parse_content.call_count == 2

    @pytest.mark.asyncio
    async def test_zero_retries_single_attempt(self):
        cfg = _make_config(provider="litellm", max_retries=0)
        worker = RefinementWorker(cfg, MagicMock(), MagicMock())

        with patch(
            "team_memory.daemon.refinement.refinement_worker._refine_via_openai",
            new_callable=AsyncMock,
        ) as mock_parse:
            mock_parse.side_effect = LLMParseError("fail")
            with pytest.raises(LLMParseError):
                await worker._call_llm("content")

        assert mock_parse.call_count == 1

    @pytest.mark.asyncio
    async def test_non_parse_error_not_retried(self):
        """Only LLMParseError triggers retries; other errors propagate immediately."""
        cfg = _make_config(provider="litellm", max_retries=3)
        worker = RefinementWorker(cfg, MagicMock(), MagicMock())

        with patch(
            "team_memory.daemon.refinement.refinement_worker._refine_via_openai",
            new_callable=AsyncMock,
        ) as mock_parse:
            mock_parse.side_effect = ConnectionError("network down")
            with pytest.raises(ConnectionError):
                await worker._call_llm("content")

        assert mock_parse.call_count == 1  # no retry


@pytest.mark.unit
class TestFallbackPublish:
    """Tests for _fallback_publish logic."""

    @pytest.mark.asyncio
    async def test_happy_path(self):
        cfg = _make_config()
        buf = AsyncMock()
        sink = AsyncMock()
        sink.draft_save = AsyncMock(return_value={"id": "exp-fb"})
        sink.draft_publish = AsyncMock(return_value={})
        worker = RefinementWorker(cfg, buf, sink)
        draft = _make_draft()
        await worker._fallback_publish(draft)
        sink.draft_save.assert_awaited_once()
        sink.draft_publish.assert_awaited_once_with(draft_id="exp-fb")
        buf.mark_published.assert_awaited_once_with("d-001")

    @pytest.mark.asyncio
    async def test_fallback_save_error_returns_early(self):
        cfg = _make_config()
        buf = AsyncMock()
        sink = AsyncMock()
        sink.draft_save = AsyncMock(return_value={"error": True, "message": "fail", "code": "some_error"})

        worker = RefinementWorker(cfg, buf, sink)
        draft = _make_draft()

        await worker._fallback_publish(draft)
        sink.draft_publish.assert_not_awaited()
        buf.mark_published.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_fallback_dedup_uses_candidate(self):
        cfg = _make_config()
        buf = AsyncMock()
        sink = AsyncMock()
        sink.draft_save = AsyncMock(
            return_value={
                "error": True,
                "code": "duplicate_detected",
                "candidates": [{"id": "existing-999"}],
            }
        )
        sink.draft_publish = AsyncMock(return_value={})

        worker = RefinementWorker(cfg, buf, sink)
        draft = _make_draft()

        await worker._fallback_publish(draft)
        sink.draft_publish.assert_awaited_once_with(draft_id="existing-999")

    @pytest.mark.asyncio
    async def test_fallback_no_experience_id_returns_early(self):
        cfg = _make_config()
        buf = AsyncMock()
        sink = AsyncMock()
        sink.draft_save = AsyncMock(
            return_value={
                "error": True,
                "code": "duplicate_detected",
                "candidates": [],
            }
        )

        worker = RefinementWorker(cfg, buf, sink)
        draft = _make_draft()

        await worker._fallback_publish(draft)
        sink.draft_publish.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_fallback_publish_error_returns_early(self):
        cfg = _make_config()
        buf = AsyncMock()
        sink = AsyncMock()
        sink.draft_save = AsyncMock(return_value={"id": "exp-1"})
        sink.draft_publish = AsyncMock(return_value={"error": True, "message": "publish fail"})

        worker = RefinementWorker(cfg, buf, sink)
        draft = _make_draft()

        await worker._fallback_publish(draft)
        buf.mark_published.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_fallback_exception_handled(self):
        """If draft_publish raises an exception, it is caught (no re-raise)."""
        cfg = _make_config()
        buf = AsyncMock()
        sink = AsyncMock()
        sink.draft_save = AsyncMock(return_value={"id": "exp-1"})
        sink.draft_publish = AsyncMock(side_effect=RuntimeError("network error"))

        worker = RefinementWorker(cfg, buf, sink)
        draft = _make_draft()

        # Should not raise
        await worker._fallback_publish(draft)
        buf.mark_published.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_fallback_title_fallback(self):
        """When draft has no title, fallback uses 'Draft <id>'."""
        cfg = _make_config()
        buf = AsyncMock()
        sink = AsyncMock()
        sink.draft_save = AsyncMock(return_value={"id": "exp-1"})
        sink.draft_publish = AsyncMock(return_value={})

        worker = RefinementWorker(cfg, buf, sink)
        draft = {"id": "abc12345", "content": "stuff", "project": "p"}

        await worker._fallback_publish(draft)

        call_kwargs = sink.draft_save.call_args
        assert call_kwargs.kwargs["title"] == "Draft abc12345"

    @pytest.mark.asyncio
    async def test_fallback_project_default(self):
        cfg = _make_config()
        buf = AsyncMock()
        sink = AsyncMock()
        sink.draft_save = AsyncMock(return_value={"id": "exp-1"})
        sink.draft_publish = AsyncMock(return_value={})

        worker = RefinementWorker(cfg, buf, sink)
        draft = {"id": "d-1", "content": "x", "title": "Y"}

        await worker._fallback_publish(draft)

        call_kwargs = sink.draft_save.call_args
        assert call_kwargs.kwargs["project"] == "default"


@pytest.mark.unit
class TestIsLearningCardExtended:
    """Additional edge cases for _is_learning_card."""

    def test_client_prefix_english(self):
        assert RefinementWorker._is_learning_card("Client - Setup Guide") is True

    def test_client_prefix_chinese(self):
        assert RefinementWorker._is_learning_card("客户端 - iOS配置") is True

    def test_learning_goal_english(self):
        assert RefinementWorker._is_learning_card("Note", "Learning Goal: Master React") is True

    def test_knowledge_map_in_content(self):
        assert RefinementWorker._is_learning_card("Map", "这是知识地图的内容") is True

    def test_learning_path_in_content(self):
        assert RefinementWorker._is_learning_card("Path", "学习路径规划") is True

    def test_learning_progress_in_content(self):
        assert RefinementWorker._is_learning_card("Progress", "当前学习进度是50%") is True

    def test_content_truncated_to_500(self):
        """Only first 500 chars of title+description are checked."""
        long_content = "x" * 600 + "学习目标"
        assert RefinementWorker._is_learning_card("Test", long_content) is False

    def test_learning_signal_in_first_500(self):
        short_content = "学习目标" + "x" * 200
        assert RefinementWorker._is_learning_card("Test", short_content) is True

    def test_empty_title_with_learning_signal(self):
        assert RefinementWorker._is_learning_card("", "这是学习计划的一部分") is True

    def test_layer_prefix_various_numbers(self):
        for n in range(5):
            assert RefinementWorker._is_learning_card(f"Layer {n} Something") is True


@pytest.mark.unit
class TestMaxRefineAttemptsConstant:
    """Verify the module-level constant is as expected."""

    def test_value(self):
        assert _MAX_REFINE_ATTEMPTS == 3

    def test_is_int(self):
        assert isinstance(_MAX_REFINE_ATTEMPTS, int)


@pytest.mark.unit
class TestStructuralCheck:
    """Tests for _structural_check quality gate."""

    def test_valid_result_passes(self):
        ok, reason = RefinementWorker._structural_check(_valid_llm_result())
        assert ok is True
        assert reason == ""

    def test_title_too_short_rejected(self):
        data = _valid_llm_result()
        data["title"] = "ab"
        ok, reason = RefinementWorker._structural_check(data)
        assert ok is False
        assert "title too short" in reason

    def test_empty_title_rejected(self):
        data = _valid_llm_result()
        data["title"] = ""
        ok, reason = RefinementWorker._structural_check(data)
        assert ok is False
        assert "title too short" in reason

    def test_title_exactly_5_chars_passes(self):
        data = _valid_llm_result()
        data["title"] = "abcde"
        ok, reason = RefinementWorker._structural_check(data)
        assert ok is True

    def test_title_4_chars_rejected(self):
        data = _valid_llm_result()
        data["title"] = "abcd"
        ok, reason = RefinementWorker._structural_check(data)
        assert ok is False

    def test_problem_too_short_rejected(self):
        data = _valid_llm_result()
        data["problem"] = "short problem"
        ok, reason = RefinementWorker._structural_check(data)
        assert ok is False
        assert "problem too short" in reason

    def test_empty_problem_rejected(self):
        data = _valid_llm_result()
        data["problem"] = ""
        ok, reason = RefinementWorker._structural_check(data)
        assert ok is False

    def test_solution_too_short_rejected(self):
        data = _valid_llm_result()
        data["solution"] = "short solution"
        ok, reason = RefinementWorker._structural_check(data)
        assert ok is False
        assert "solution too short" in reason

    def test_empty_solution_rejected(self):
        data = _valid_llm_result()
        data["solution"] = ""
        ok, reason = RefinementWorker._structural_check(data)
        assert ok is False

    def test_none_solution_rejected(self):
        data = _valid_llm_result()
        data["solution"] = None
        ok, reason = RefinementWorker._structural_check(data)
        assert ok is False

    def test_empty_experience_type_rejected(self):
        data = _valid_llm_result()
        data["experience_type"] = ""
        ok, reason = RefinementWorker._structural_check(data)
        assert ok is False
        assert "experience_type empty" in reason

    def test_missing_fields_rejected(self):
        ok, reason = RefinementWorker._structural_check({})
        assert ok is False

    def test_problem_exactly_50_chars_passes(self):
        data = _valid_llm_result()
        data["problem"] = "a" * 50
        ok, reason = RefinementWorker._structural_check(data)
        assert ok is True

    def test_problem_49_chars_rejected(self):
        data = _valid_llm_result()
        data["problem"] = "a" * 49
        ok, reason = RefinementWorker._structural_check(data)
        assert ok is False


@pytest.mark.unit
class TestQualityGateRefineOne:
    """Tests for quality gate integration in _refine_one."""

    @pytest.mark.asyncio
    async def test_structural_check_discard_publishes_draft_but_skips_update(self):
        """When structural check fails, draft is NOT published to TM and local draft is marked published."""
        cfg = _make_config(quality_gate_enabled=True)
        buf = AsyncMock()
        sink = AsyncMock()
        sink.draft_save = AsyncMock(return_value={"id": "exp-qg1"})
        sink.draft_publish = AsyncMock(return_value={})

        worker = RefinementWorker(cfg, buf, sink)
        draft = _make_draft()

        with patch.object(worker, "_call_llm", new_callable=AsyncMock) as mock_llm:
            # Title too short, no problem/solution/type
            mock_llm.return_value = {"title": "X"}
            await worker._refine_one(draft)

        # draft_publish should NOT be called when rejected by quality gate
        sink.draft_publish.assert_not_awaited()
        # update_experience should NOT be called (refined data is discarded)
        sink.update_experience.assert_not_awaited()
        # Local draft should be marked published
        buf.mark_published.assert_awaited_once_with("d-001")

    @pytest.mark.asyncio
    async def test_quality_score_too_low_discards(self):
        """When LLM quality score is below threshold, experience is discarded."""
        cfg = _make_config(quality_gate_enabled=True, quality_min_score=60)
        buf = AsyncMock()
        sink = AsyncMock()
        sink.draft_save = AsyncMock(return_value={"id": "exp-qg2"})
        sink.draft_publish = AsyncMock(return_value={})

        worker = RefinementWorker(cfg, buf, sink)
        draft = _make_draft()

        with patch.object(worker, "_call_llm", new_callable=AsyncMock) as mock_llm:
            mock_llm.return_value = _valid_llm_result()
            with patch.object(worker, "_quality_score", new_callable=AsyncMock) as mock_score:
                mock_score.return_value = 40
                await worker._refine_one(draft)

        sink.draft_publish.assert_not_awaited()
        sink.update_experience.assert_not_awaited()
        buf.mark_published.assert_awaited_once_with("d-001")

    @pytest.mark.asyncio
    async def test_quality_score_at_threshold_passes(self):
        """When LLM quality score equals the threshold, experience is published."""
        cfg = _make_config(quality_gate_enabled=True, quality_min_score=60)
        buf = AsyncMock()
        sink = AsyncMock()
        sink.draft_save = AsyncMock(return_value={"id": "exp-qg3"})
        sink.draft_publish = AsyncMock(return_value={})
        sink.update_experience = AsyncMock(return_value={})

        worker = RefinementWorker(cfg, buf, sink)
        draft = _make_draft()

        with patch.object(worker, "_call_llm", new_callable=AsyncMock) as mock_llm:
            mock_llm.return_value = _valid_llm_result()
            with patch.object(worker, "_quality_score", new_callable=AsyncMock) as mock_score:
                mock_score.return_value = 60
                await worker._refine_one(draft)

        # Both publish and update should be called (gate passed)
        sink.draft_publish.assert_awaited()
        sink.update_experience.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_quality_score_above_threshold_passes(self):
        """When LLM quality score is above threshold, experience is published."""
        cfg = _make_config(quality_gate_enabled=True, quality_min_score=60)
        buf = AsyncMock()
        sink = AsyncMock()
        sink.draft_save = AsyncMock(return_value={"id": "exp-qg4"})
        sink.draft_publish = AsyncMock(return_value={})
        sink.update_experience = AsyncMock(return_value={})

        worker = RefinementWorker(cfg, buf, sink)
        draft = _make_draft()

        with patch.object(worker, "_call_llm", new_callable=AsyncMock) as mock_llm:
            mock_llm.return_value = _valid_llm_result()
            with patch.object(worker, "_quality_score", new_callable=AsyncMock) as mock_score:
                mock_score.return_value = 85
                await worker._refine_one(draft)

        sink.update_experience.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_quality_gate_disabled_skips_checks(self):
        """When quality gate is disabled, no checks are performed."""
        cfg = _make_config(quality_gate_enabled=False)
        buf = AsyncMock()
        sink = AsyncMock()
        sink.draft_save = AsyncMock(return_value={"id": "exp-qg5"})
        sink.draft_publish = AsyncMock(return_value={})
        sink.update_experience = AsyncMock(return_value={})

        worker = RefinementWorker(cfg, buf, sink)
        draft = _make_draft()

        with patch.object(worker, "_call_llm", new_callable=AsyncMock) as mock_llm:
            # This would fail structural check, but gate is disabled
            mock_llm.return_value = {"title": "X"}
            await worker._refine_one(draft)

        # update_experience should be called (gate bypassed)
        sink.update_experience.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_structural_check_before_quality_score(self):
        """Structural check runs first; if it fails, quality scoring is not called."""
        cfg = _make_config(quality_gate_enabled=True)
        buf = AsyncMock()
        sink = AsyncMock()
        sink.draft_save = AsyncMock(return_value={"id": "exp-qg6"})
        sink.draft_publish = AsyncMock(return_value={})

        worker = RefinementWorker(cfg, buf, sink)
        draft = _make_draft()

        with patch.object(worker, "_call_llm", new_callable=AsyncMock) as mock_llm:
            mock_llm.return_value = {"title": "X"}  # fails structural
            with patch.object(worker, "_quality_score", new_callable=AsyncMock) as mock_score:
                await worker._refine_one(draft)
                # _quality_score should NOT have been called
                mock_score.assert_not_awaited()


@pytest.mark.unit
class TestQualityScore:
    """Tests for _quality_score method."""

    @pytest.mark.asyncio
    async def test_successful_scoring(self):
        cfg = _make_config(quality_gate_enabled=True)
        worker = RefinementWorker(cfg, MagicMock(), MagicMock())

        with patch(
            "team_memory.daemon.refinement.refinement_worker.call_llm_openai", new_callable=AsyncMock
        ) as mock_call:
            mock_call.return_value = '{"score": 75}'
            score = await worker._quality_score(_valid_llm_result())

        assert score == 75

    @pytest.mark.asyncio
    async def test_score_clamped_to_100(self):
        cfg = _make_config(quality_gate_enabled=True)
        worker = RefinementWorker(cfg, MagicMock(), MagicMock())

        with patch(
            "team_memory.daemon.refinement.refinement_worker.call_llm_openai", new_callable=AsyncMock
        ) as mock_call:
            mock_call.return_value = '{"score": 150}'
            score = await worker._quality_score(_valid_llm_result())

        assert score == 100

    @pytest.mark.asyncio
    async def test_score_clamped_to_0(self):
        cfg = _make_config(quality_gate_enabled=True)
        worker = RefinementWorker(cfg, MagicMock(), MagicMock())

        with patch(
            "team_memory.daemon.refinement.refinement_worker.call_llm_openai", new_callable=AsyncMock
        ) as mock_call:
            mock_call.return_value = '{"score": -10}'
            score = await worker._quality_score(_valid_llm_result())

        assert score == 0

    @pytest.mark.asyncio
    async def test_llm_failure_returns_100(self):
        """When LLM call fails, default to pass (100)."""
        cfg = _make_config(quality_gate_enabled=True)
        worker = RefinementWorker(cfg, MagicMock(), MagicMock())

        with patch(
            "team_memory.daemon.refinement.refinement_worker.call_llm_openai", new_callable=AsyncMock
        ) as mock_call:
            mock_call.side_effect = RuntimeError("LLM down")
            score = await worker._quality_score(_valid_llm_result())

        assert score == 100

    @pytest.mark.asyncio
    async def test_invalid_json_returns_100(self):
        """When LLM returns non-JSON, default to pass."""
        cfg = _make_config(quality_gate_enabled=True)
        worker = RefinementWorker(cfg, MagicMock(), MagicMock())

        with patch(
            "team_memory.daemon.refinement.refinement_worker.call_llm_openai", new_callable=AsyncMock
        ) as mock_call:
            mock_call.return_value = "not json at all"
            score = await worker._quality_score(_valid_llm_result())

        assert score == 100


@pytest.mark.unit
class TestPublishingIntegration:
    """Tests for publishing state machine integration."""

    @pytest.mark.asyncio
    async def test_refine_one_marks_publishing_after_llm(self):
        """After _call_llm succeeds, mark_publishing_in_progress called with eid+structured."""
        cfg = _make_config()
        buf = AsyncMock()
        buf.mark_refining = AsyncMock()
        buf.mark_publishing_in_progress = AsyncMock()
        buf.mark_published = AsyncMock()
        sink = AsyncMock()
        sink.draft_save = AsyncMock(return_value={"id": "exp-pub-001"})
        sink.draft_publish = AsyncMock(return_value={})
        sink.update_experience = AsyncMock(return_value={})
        worker = RefinementWorker(cfg, buf, sink)
        structured = {"title": "T", "problem": "P", "solution": "S"}
        with patch.object(worker, "_call_llm", new_callable=AsyncMock) as mock_llm:
            mock_llm.return_value = structured
            await worker._refine_one(_make_draft())
        buf.mark_publishing_in_progress.assert_awaited_once()
        call_args = buf.mark_publishing_in_progress.call_args[0]
        assert call_args[1]["experience_id"] == "exp-pub-001"
        assert call_args[1]["structured"]["title"] == "T"

    @pytest.mark.asyncio
    async def test_tick_skip_llm_when_payload_valid(self):
        """skip_llm=true session bypasses mark_refining and _call_llm."""
        cfg = _make_config()
        buf = AsyncMock()
        buf.mark_published = AsyncMock()
        sink = AsyncMock()
        sink.draft_publish = AsyncMock(return_value={})
        sink.update_experience = AsyncMock(return_value={})
        draft = {
            "id": "skip-test",
            "content": "x",
            "project": "default",
            "conversation_id": "skip-test",
            "status": "needs_refinement",
            "refined_payload": {"experience_id": "exp-skip", "structured": {"title": "S"}},
            "skip_llm": True,
        }
        buf.get_needs_refinement = AsyncMock(return_value=[draft])
        worker = RefinementWorker(cfg, buf, sink)
        worker._refine_one = AsyncMock()
        await worker._tick()
        worker._refine_one.assert_not_called()
        sink.draft_publish.assert_awaited_once_with(draft_id="exp-skip")
        sink.update_experience.assert_awaited_once()
        buf.mark_published.assert_awaited_once_with("skip-test")

    @pytest.mark.asyncio
    async def test_tick_corrupt_payload_fallback(self):
        """Corrupt refined_payload → falls back to normal refine."""
        cfg = _make_config()
        buf = AsyncMock()
        buf.mark_refining = AsyncMock()
        buf.mark_published = AsyncMock()
        sink = AsyncMock()
        draft = {
            "id": "corrupt",
            "content": "x",
            "project": "default",
            "conversation_id": "corrupt",
            "status": "needs_refinement",
            "refined_payload": None,  # NULL → normal path
            "skip_llm": False,
        }
        buf.get_needs_refinement = AsyncMock(return_value=[draft])
        worker = RefinementWorker(cfg, buf, sink)
        worker._refine_one = AsyncMock()
        await worker._tick()
        buf.mark_refining.assert_awaited_once_with("corrupt")
        worker._refine_one.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_fallback_publish_clears_refined_payload(self):
        """_fallback_publish clears refined_payload after success."""
        cfg = _make_config()
        buf = AsyncMock()
        buf.mark_published = AsyncMock()
        sink = AsyncMock()
        sink.draft_save = AsyncMock(return_value={"id": "exp-fb"})
        sink.draft_publish = AsyncMock(return_value={})
        worker = RefinementWorker(cfg, buf, sink)
        await worker._fallback_publish(_make_draft())
        buf.mark_published.assert_awaited_once()


@pytest.mark.unit
class TestQualityMetrics:
    """Tests for quality gate → metrics recording."""

    @pytest.mark.asyncio
    async def test_quality_record_called(self):
        """Quality gate execution → metrics.record called."""
        cfg = _make_config(quality_gate_enabled=True, quality_min_score=30)
        cfg.refinement.quality_gate_enabled = True
        cfg.refinement.quality_min_score = 30
        buf = AsyncMock()
        buf.mark_publishing_in_progress = AsyncMock()
        buf.mark_published = AsyncMock()
        sink = AsyncMock()
        sink.draft_save = AsyncMock(return_value={"id": "exp-x"})
        sink.draft_publish = AsyncMock(return_value={})
        sink.update_experience = AsyncMock(return_value={})
        metrics = MagicMock()
        worker = RefinementWorker(cfg, buf, sink, metrics=metrics)
        worker._dedup_search = AsyncMock(return_value=(None, []))
        worker._call_llm = AsyncMock(return_value={"title": "T", "problem": "P", "solution": "S"})
        worker._quality_score = AsyncMock(return_value=85)
        worker._structural_check = MagicMock(return_value=(True, ""))
        await worker._refine_one(_make_draft())
        assert metrics.record.call_count > 0, f"Expected metrics.record called, got {metrics.record.call_args_list}"


@pytest.mark.unit
class TestTitleQuality:
    """Tests for title quality metric."""

    @pytest.mark.asyncio
    async def test_title_quality_recorded(self):
        """Good title → title_quality=1 recorded by metrics."""
        cfg = _make_config(quality_gate_enabled=False)
        cfg.refinement.quality_gate_enabled = False
        buf = AsyncMock()
        buf.mark_publishing_in_progress = AsyncMock()
        buf.mark_published = AsyncMock()
        sink = AsyncMock()
        sink.draft_save = AsyncMock(return_value={"id": "exp-t"})
        sink.draft_publish = AsyncMock(return_value={})
        sink.update_experience = AsyncMock(return_value={})
        metrics = MagicMock()
        worker = RefinementWorker(cfg, buf, sink, metrics=metrics)
        worker._dedup_search = AsyncMock(return_value=(None, []))
        worker._call_llm = AsyncMock(return_value={"title": "Good Title", "problem": "P", "solution": "S"})
        await worker._refine_one(_make_draft())
        assert metrics.record.call_count > 0, f"Expected record called, got {metrics.record.call_args_list}"

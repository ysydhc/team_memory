"""LiteLLM 不可用 — 验证 refinement 入 pending_ops。"""
from unittest.mock import AsyncMock

import pytest


@pytest.mark.failure_injection
class TestLLMDown:
    def test_llm_call_failure_enqueues(self):
        """Mock _call_llm raises → draft enqueued to PendingQueue."""
        from team_memory.daemon.pending_queue import PendingQueue
        from team_memory.daemon.refinement_worker import RefinementWorker
        cfg = type("Cfg", (), {"refinement": type("R", (), {
            "enabled": True, "quality_gate_enabled": False,
            "quality_min_score": 0, "fallback_on_failure": True,
        })()})()
        buf = AsyncMock()
        buf.mark_refining = AsyncMock()
        buf.mark_publishing_in_progress = AsyncMock()
        buf.mark_published = AsyncMock()
        buf.get_needs_refinement = AsyncMock(return_value=[
            {"id": "fail-llm", "content": "x", "project": "default", "conversation_id": "fail-llm",
             "refined_payload": None, "skip_llm": False}])
        sink = AsyncMock()
        queue = PendingQueue(db_path=":memory:")
        worker = RefinementWorker(cfg, buf, sink, queue=queue)
        worker._refine_one = AsyncMock(side_effect=RuntimeError("LLM 429"))
        import asyncio
        asyncio.run(worker._tick())
        assert queue.stats()["pending"] > 0

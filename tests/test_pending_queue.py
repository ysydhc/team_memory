"""Tests for PendingQueue — SQLite-backed retry queue for failed operations.

测试 PendingQueue：SQLite 重试队列。
"""

from __future__ import annotations

from pathlib import Path

import pytest

from team_memory.daemon.buffers.pending_queue import PendingQueue


class TestPendingQueue:
    """Unit tests for PendingQueue / 单元测试。"""

    @pytest.fixture
    def queue(self, tmp_path: Path) -> PendingQueue:
        """Create a temporary queue / 创建临时队列。"""
        db_path = str(tmp_path / "test_queue.db")
        return PendingQueue(db_path=db_path)

    def test_enqueue_dequeue(self, queue: PendingQueue) -> None:
        """Basic enqueue and dequeue cycle / 基本入队出队。"""
        op_id = queue.enqueue(
            op_type="refine",
            session_id="sess-001",
            payload={"content": "test content", "project": "test"},
        )
        assert op_id is not None
        assert len(op_id) > 0

        queue.dequeue(op_id)

    def test_enqueue_duplicate_session_rejected(self, queue: PendingQueue) -> None:
        """Same session_id should not be enqueued twice / 同 session 不应重复入队。"""
        first = queue.enqueue(
            op_type="refine",
            session_id="sess-dup",
            payload={"content": "first"},
        )
        assert first is not None

        second = queue.enqueue(
            op_type="refine",
            session_id="sess-dup",
            payload={"content": "second"},
        )
        assert second is None  # Dedup rejected / 去重拒绝

    def test_get_due_ops_only_expired(self, queue: PendingQueue) -> None:
        """get_due_ops() only returns ops past their next_retry_at / 只返回过期任务。"""

        # Enqueue one with immediate retry
        op1 = queue.enqueue(
            op_type="refine",
            session_id="due-now",
            payload={"content": "due"},
        )
        assert op1

        # Manually set next_retry_at to the past
        queue._db.execute(
            "UPDATE pending_ops SET next_retry_at = ? WHERE id = ?",
            ("2020-01-01T00:00:00", op1),
        )
        queue._db.commit()

        due = queue.get_due_ops()
        assert len(due) == 1
        assert due[0]["id"] == op1

    def test_backoff_sequence(self, queue: PendingQueue) -> None:
        """Backoff increases exponentially: 30s, 60s, 120s / 退避时间递增。"""

        op_id = queue.enqueue(
            op_type="refine",
            session_id="backoff-test",
            payload={"content": "test"},
        )
        assert op_id

        # Record initial next_retry_at
        row = queue._db.execute(
            "SELECT next_retry_at FROM pending_ops WHERE id = ?",
            (op_id,),
        ).fetchone()
        initial_retry = row[0]

        # Bump retry
        queue.bump_retry(op_id, "test error", backoff_seconds=60)

        row = queue._db.execute(
            "SELECT retries, next_retry_at FROM pending_ops WHERE id = ?",
            (op_id,),
        ).fetchone()
        assert row[0] == 1  # retries increased
        assert row[1] != initial_retry  # next_retry_at changed

    def test_queue_full_rejected(self, queue: PendingQueue) -> None:
        """When queue reaches MAX_QUEUE_SIZE, new entries rejected / 队列满时拒绝。"""
        # Fill queue to limit
        for i in range(PendingQueue.MAX_QUEUE_SIZE):
            queue.enqueue(
                op_type="refine",
                session_id=f"fill-{i}",
                payload={"content": f"test {i}"},
            )

        # Next should be rejected
        rejected = queue.enqueue(
            op_type="refine",
            session_id="overflow",
            payload={"content": "overflow"},
        )
        assert rejected is None

    def test_mark_failed(self, queue: PendingQueue) -> None:
        """mark_failed sets status to 'failed' / 标记失败。"""
        op_id = queue.enqueue(
            op_type="refine",
            session_id="fail-test",
            payload={"content": "test"},
        )
        assert op_id

        queue.mark_failed(op_id)

        # Failed ops should NOT appear in get_due_ops
        due = queue.get_due_ops()
        ids = [op["id"] for op in due]
        assert op_id not in ids

    def test_stats(self, queue: PendingQueue) -> None:
        """stats() returns correct counts / 统计正确。"""
        # Add 3 pending ops
        queue.enqueue("refine", "s1", {"content": "a"})
        queue.enqueue("refine", "s2", {"content": "b"})
        queue.enqueue("refine", "s3", {"content": "c"})

        s = queue.stats()
        assert s["pending"] == 3
        assert s["retrying"] == 0
        assert s["failed"] == 0

    def test_list_ops(self, queue: PendingQueue) -> None:
        """list_ops returns all queued ops / 列出所有队列操作。"""
        queue.enqueue("refine", "l1", {"content": "a"})
        queue.enqueue("publish", "l2", {"content": "b"})

        all_ops = queue.list_ops()
        assert len(all_ops) == 2

        refine_ops = queue.list_ops(op_type="refine")
        assert len(refine_ops) == 1
        assert refine_ops[0]["op_type"] == "refine"

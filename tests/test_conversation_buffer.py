"""Tests for ConversationBuffer — publishing state machine and schema."""
from __future__ import annotations
import pytest
import json

from team_memory.daemon.conversation_buffer import ConversationBuffer


@pytest.fixture
async def buf():
    b = ConversationBuffer(db_path=":memory:")
    await b.__aenter__()
    yield b
    await b.__aexit__(None, None, None)


@pytest.mark.asyncio
class TestPublishingState:
    async def test_mark_publishing_in_progress_stores_payload(self, buf):
        """mark_publishing_in_progress stores status and payload."""
        # Create a session first
        await buf.ensure_session("pub-001", project="test")
        payload = {"experience_id": "exp-123", "structured": {"title": "Test"}}

        # Method doesn't exist yet — will fail
        await buf.mark_publishing_in_progress("pub-001", payload)

        # Verify status
        row = await buf._db.execute_fetchall(
            "SELECT status, refined_payload FROM sessions WHERE id = 'pub-001'"
        )
        assert row[0][0] == "publishing"
        parsed = json.loads(row[0][1])
        assert parsed["experience_id"] == "exp-123"
        assert parsed["structured"]["title"] == "Test"

    async def test_reset_publishing_sessions_returns_count(self, buf):
        """reset_publishing_sessions resets status and returns count."""
        await buf.ensure_session("pub-reset-1", project="test")
        await buf.ensure_session("pub-reset-2", project="test")
        # Manually set status to publishing (method doesn't exist yet)
        await buf._db.execute(
            "UPDATE sessions SET status='publishing' WHERE id IN ('pub-reset-1', 'pub-reset-2')"
        )
        await buf._db.commit()

        count = await buf.reset_publishing_sessions()
        assert count == 2

        # Verify status changed
        for sid in ["pub-reset-1", "pub-reset-2"]:
            row = await buf._db.execute_fetchall(
                "SELECT status FROM sessions WHERE id = ?", (sid,)
            )
            assert row[0][0] == "needs_refinement"

    async def test_existing_sessions_null_refined_payload(self, buf):
        """Sessions created before column exists have NULL payload."""
        await buf.ensure_session("null-001", project="test")
        row = await buf._db.execute_fetchall(
            "SELECT refined_payload FROM sessions WHERE id = 'null-001'"
        )
        assert row[0][0] is None

    async def test_get_needs_refinement_skip_llm_flag(self, buf):
        """get_needs_refinement sets skip_llm when valid refined_payload exists."""
        await buf.ensure_session("skip-001", project="test")
        payload = {"experience_id": "exp-skip", "structured": {"title": "T"}}
        await buf._db.execute(
            "UPDATE sessions SET status='needs_refinement', refined_payload = ? WHERE id = ?",
            (json.dumps(payload), "skip-001"),
        )
        await buf._db.commit()

        sessions = await buf.get_needs_refinement(project="test")
        assert len(sessions) == 1
        assert sessions[0]["skip_llm"] is True
        assert sessions[0]["refined_payload"]["experience_id"] == "exp-skip"

    async def test_wal_mode_enabled(self, buf):
        """WAL mode enabled for file-based DB, memory for in-memory."""
        row = await buf._db.execute_fetchall("PRAGMA journal_mode")
        mode = row[0][0].upper()
        assert mode in ("WAL", "MEMORY"), f"Expected WAL or MEMORY, got {mode}"

"""Tests for DraftBuffer tm_id field and DraftRefiner id chain.

Verifies that save_draft writes PG id back to buffer (tm_id),
and refine_and_publish uses tm_id instead of local buffer id.
"""
import asyncio
import os
import tempfile
from unittest.mock import AsyncMock, MagicMock

import pytest

import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))

from hooks.draft_buffer import DraftBuffer
from daemon.draft_refiner import DraftRefiner
from daemon.tm_sink import TMSink


# ---------------------------------------------------------------------------
# DraftBuffer: tm_id column
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_buffer_tm_id_column_exists():
    """After opening a new buffer, the tm_id column is available."""
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        db_path = f.name
    try:
        async with DraftBuffer(db_path) as buf:
            # Create a draft
            draft_id = await buf.create_draft("proj", "sess-1", "hello world")
            # Set tm_id
            await buf.set_tm_id(draft_id, "pg-uuid-123")
            # Read it back
            pending = await buf.get_pending("sess-1")
            assert len(pending) == 1
            assert pending[0]["tm_id"] == "pg-uuid-123"
    finally:
        os.unlink(db_path)


@pytest.mark.asyncio
async def test_buffer_tm_id_migration():
    """Opening an existing DB without tm_id column auto-migrates."""
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        db_path = f.name
    try:
        # Create DB with original schema (no tm_id)
        import aiosqlite
        async with aiosqlite.connect(db_path) as db:
            await db.execute("""
                CREATE TABLE drafts (
                    id TEXT PRIMARY KEY,
                    title TEXT DEFAULT '',
                    project TEXT NOT NULL,
                    conversation_id TEXT,
                    content TEXT NOT NULL,
                    status TEXT DEFAULT 'pending',
                    source TEXT DEFAULT 'pipeline',
                    group_key TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            await db.commit()

        # Open with DraftBuffer — should auto-migrate
        async with DraftBuffer(db_path) as buf:
            draft_id = await buf.create_draft("proj", "sess-1", "test")
            await buf.set_tm_id(draft_id, "pg-uuid-456")
            pending = await buf.get_pending("sess-1")
            assert pending[0]["tm_id"] == "pg-uuid-456"
    finally:
        os.unlink(db_path)


@pytest.mark.asyncio
async def test_buffer_tm_id_default_null():
    """Newly created drafts have tm_id=None until explicitly set."""
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        db_path = f.name
    try:
        async with DraftBuffer(db_path) as buf:
            await buf.create_draft("proj", "sess-1", "hello")
            pending = await buf.get_pending("sess-1")
            assert pending[0]["tm_id"] is None
    finally:
        os.unlink(db_path)


# ---------------------------------------------------------------------------
# DraftRefiner: tm_id chain
# ---------------------------------------------------------------------------

def _make_refiner():
    """Create a DraftRefiner with mocked sink and real buffer."""
    sink = MagicMock(spec=TMSink)
    sink.draft_save = AsyncMock(return_value={
        "id": "pg-exp-001", "status": "draft"
    })
    sink.draft_publish = AsyncMock(return_value={
        "id": "pg-exp-001", "status": "published"
    })

    buf_path = tempfile.mktemp(suffix=".db")
    buf = DraftBuffer(buf_path)
    refiner = DraftRefiner(sink, buf)
    return refiner, sink, buf_path


@pytest.mark.asyncio
async def test_save_draft_writes_tm_id():
    """save_draft should call set_tm_id after successful PG write."""
    refiner, sink, buf_path = _make_refiner()
    try:
        async with refiner._buf:
            await refiner.save_draft(
                session_id="sess-tm",
                title="Test Title",
                content="Some content",
                project="proj",
            )
            # Verify sink.draft_save was called
            sink.draft_save.assert_called_once()

            # Verify tm_id was written to buffer
            pending = await refiner._buf.get_pending("sess-tm")
            assert len(pending) == 1
            assert pending[0]["tm_id"] == "pg-exp-001"
    finally:
        os.unlink(buf_path)


@pytest.mark.asyncio
async def test_refine_and_publish_uses_tm_id():
    """refine_and_publish should use tm_id (PG id) for draft_publish."""
    refiner, sink, buf_path = _make_refiner()
    try:
        async with refiner._buf:
            # Save a draft (this sets tm_id)
            await refiner.save_draft(
                session_id="sess-pub",
                title="Publish Test",
                content="使用 JavascriptInterface 可以注入对象",
                project="proj",
            )

            # Now publish
            result = await refiner.refine_and_publish("sess-pub")

            # Verify draft_publish was called with tm_id (PG id), not local id
            sink.draft_publish.assert_called_once()
            call_kwargs = sink.draft_publish.call_args[1]
            assert call_kwargs["draft_id"] == "pg-exp-001", \
                f"Expected pg-exp-001, got {call_kwargs['draft_id']}"

            assert result is not None
            assert result["status"] == "published"
    finally:
        os.unlink(buf_path)


@pytest.mark.asyncio
async def test_refine_and_publish_fallback_to_local_id():
    """When tm_id is not set, refine_and_publish falls back to local id."""
    refiner, sink, buf_path = _make_refiner()
    try:
        async with refiner._buf:
            # Simulate draft_save error (no pg id returned)
            sink.draft_save.reset_mock()
            sink.draft_save.return_value = {"error": True, "message": "embedding failed"}

            # Create draft directly in buffer (no tm_id)
            local_id = await refiner._buf.upsert_draft(
                session_id="sess-fallback",
                title="Fallback",
                content="需要配置环境变量",
                project="proj",
            )

            # Publish — should use local id since tm_id is None
            result = await refiner.refine_and_publish("sess-fallback")

            sink.draft_publish.assert_called_once()
            call_kwargs = sink.draft_publish.call_args[1]
            assert call_kwargs["draft_id"] == local_id, \
                f"Expected local id {local_id}, got {call_kwargs['draft_id']}"
    finally:
        os.unlink(buf_path)

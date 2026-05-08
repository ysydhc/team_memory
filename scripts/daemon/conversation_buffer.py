"""Structured conversation buffer for the memory pipeline.

Stores conversations as individual messages (user/assistant) in SQLite.
Replaces the old DraftBuffer which concatenated all text into one field.

Tables:
  sessions — one row per conversation session
  messages — one row per message (role=user/assistant)
"""
from __future__ import annotations

import uuid
from datetime import datetime, timedelta, timezone
from typing import Any

import aiosqlite

# ---------------------------------------------------------------------------
# SQL
# ---------------------------------------------------------------------------

_CREATE_SESSIONS = """
CREATE TABLE IF NOT EXISTS sessions (
    id TEXT PRIMARY KEY,
    project TEXT NOT NULL DEFAULT 'default',
    status TEXT DEFAULT 'active',
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
)
"""

_CREATE_MESSAGES = """
CREATE TABLE IF NOT EXISTS messages (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id TEXT NOT NULL,
    role TEXT NOT NULL,
    content TEXT NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (session_id) REFERENCES sessions(id)
)
"""

_CREATE_INDEX = """
CREATE INDEX IF NOT EXISTS idx_messages_session_role
ON messages(session_id, role)
"""


class ConversationBuffer:
    """Async SQLite-backed conversation buffer.

    Stores conversations as structured messages instead of concatenated text.

    Usage::

        buf = ConversationBuffer("/path/to/conversations.db")
        async with buf:
            await buf.add_message("sess-1", "user", "how to fix X?", project="team_doc")
            await buf.add_message("sess-1", "assistant", "you can do Y...")
            messages = await buf.get_messages("sess-1")
    """

    def __init__(self, db_path: str) -> None:
        self._db_path = db_path
        self._db: aiosqlite.Connection | None = None

    # ------------------------------------------------------------------
    # Context manager
    # ------------------------------------------------------------------

    async def __aenter__(self) -> "ConversationBuffer":
        self._db = await aiosqlite.connect(self._db_path)
        self._db.row_factory = aiosqlite.Row
        await self._db.execute(_CREATE_SESSIONS)
        await self._db.execute(_CREATE_MESSAGES)
        await self._db.execute(_CREATE_INDEX)
        await self._db.commit()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        if self._db is not None:
            await self._db.close()
            self._db = None

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _require_db(self) -> aiosqlite.Connection:
        if self._db is None:
            raise RuntimeError("ConversationBuffer not connected. Use 'async with buf:'")
        return self._db

    @staticmethod
    def _row_to_dict(row: aiosqlite.Row) -> dict[str, Any]:
        return dict(row)

    # ------------------------------------------------------------------
    # Session management
    # ------------------------------------------------------------------

    async def ensure_session(self, session_id: str, project: str = "default") -> str:
        """Ensure a session exists; create if not.

        Returns the session_id.
        """
        db = self._require_db()
        row = await db.execute(
            "SELECT id FROM sessions WHERE id = ?", (session_id,)
        )
        existing = await row.fetchone()
        if existing:
            return session_id

        now = datetime.now(timezone.utc).isoformat()
        await db.execute(
            "INSERT INTO sessions (id, project, status, created_at, updated_at) VALUES (?, ?, 'active', ?, ?)",
            (session_id, project, now, now),
        )
        await db.commit()
        return session_id

    async def get_session(self, session_id: str) -> dict[str, Any] | None:
        """Get session metadata. Returns None if not found."""
        db = self._require_db()
        row = await db.execute("SELECT * FROM sessions WHERE id = ?", (session_id,))
        result = await row.fetchone()
        return self._row_to_dict(result) if result else None

    async def get_active_sessions(self) -> list[dict[str, Any]]:
        """Return all sessions with status='active'."""
        db = self._require_db()
        cursor = await db.execute(
            "SELECT * FROM sessions WHERE status = 'active' ORDER BY updated_at DESC"
        )
        rows = await cursor.fetchall()
        return [self._row_to_dict(r) for r in rows]

    async def get_sessions_by_status(self, status: str) -> list[dict[str, Any]]:
        """Return all sessions with the given status."""
        db = self._require_db()
        cursor = await db.execute(
            "SELECT * FROM sessions WHERE status = ? ORDER BY updated_at DESC",
            (status,),
        )
        rows = await cursor.fetchall()
        return [self._row_to_dict(r) for r in rows]

    async def close_session(self, session_id: str, status: str = "published") -> None:
        """Close a session (mark as published/expired)."""
        db = self._require_db()
        now = datetime.now(timezone.utc).isoformat()
        await db.execute(
            "UPDATE sessions SET status = ?, updated_at = ? WHERE id = ?",
            (status, now, session_id),
        )
        await db.commit()

    async def mark_session(self, session_id: str, status: str) -> None:
        """Set session status (active/needs_refinement/published/expired)."""
        await self.close_session(session_id, status)

    # ------------------------------------------------------------------
    # Message operations
    # ------------------------------------------------------------------

    async def add_message(
        self,
        session_id: str,
        role: str,
        content: str,
        project: str = "default",
    ) -> int:
        """Add a message to a session. Auto-creates session if needed.

        Args:
            session_id: Conversation identifier.
            role: "user" or "assistant".
            content: Message text.
            project: Project name (used when creating session).

        Returns:
            The message row id.
        """
        db = self._require_db()
        await self.ensure_session(session_id, project)

        # Dedup: skip if identical message already exists in this session
        import hashlib
        content_hash = hashlib.sha256(f"{role}:{content}".encode()).hexdigest()[:16]
        existing = await db.execute_fetchall(
            "SELECT id FROM messages WHERE session_id = ? AND role = ? AND content = ? LIMIT 1",
            (session_id, role, content),
        )
        if existing:
            return existing[0][0]  # Return existing message id

        now = datetime.now(timezone.utc).isoformat()
        cursor = await db.execute(
            "INSERT INTO messages (session_id, role, content, created_at) VALUES (?, ?, ?, ?)",
            (session_id, role, content, now),
        )
        # Update session timestamp
        await db.execute(
            "UPDATE sessions SET updated_at = ? WHERE id = ?",
            (now, session_id),
        )
        await db.commit()
        return cursor.lastrowid or 0

    async def get_messages(
        self, session_id: str, role: str | None = None
    ) -> list[dict[str, Any]]:
        """Get messages for a session, optionally filtered by role.

        Args:
            session_id: Conversation identifier.
            role: Optional filter ("user" or "assistant").

        Returns:
            List of message dicts ordered by created_at.
        """
        db = self._require_db()
        if role:
            cursor = await db.execute(
                "SELECT * FROM messages WHERE session_id = ? AND role = ? ORDER BY created_at",
                (session_id, role),
            )
        else:
            cursor = await db.execute(
                "SELECT * FROM messages WHERE session_id = ? ORDER BY created_at",
                (session_id,),
            )
        rows = await cursor.fetchall()
        return [self._row_to_dict(r) for r in rows]

    async def get_user_messages(self, session_id: str) -> list[dict[str, Any]]:
        """Get only user messages for a session."""
        return await self.get_messages(session_id, role="user")

    async def get_assistant_messages(self, session_id: str) -> list[dict[str, Any]]:
        """Get only assistant messages for a session."""
        return await self.get_messages(session_id, role="assistant")

    async def get_accumulated_content(self, session_id: str, role: str | None = None) -> str:
        """Get concatenated content for a session (backward compat).

        This mimics old DraftBuffer behavior for code that needs a single string.
        """
        messages = await self.get_messages(session_id, role=role)
        return "\n".join(m["content"] for m in messages if m.get("content"))

    # ------------------------------------------------------------------
    # Backward-compatible API (eases migration)
    # ------------------------------------------------------------------

    async def get_pending(self, session_id: str) -> list[dict[str, Any]]:
        """Backward compat: return active session as a list with one 'draft' dict.

        The old DraftBuffer returned a list of draft dicts with 'id', 'content', etc.
        This adapts the new structure to that interface.
        """
        session = await self.get_session(session_id)
        if not session or session.get("status") != "active":
            return []

        content = await self.get_accumulated_content(session_id, role="assistant")
        if not content:
            return []

        return [{
            "id": session_id,
            "content": content,
            "project": session.get("project", "default"),
            "conversation_id": session_id,
            "status": "active",
        }]

    async def update_draft(self, draft_id: str, content: str) -> None:
        """Backward compat: no-op (messages are already stored individually)."""
        pass

    async def mark_published(self, draft_id: str) -> None:
        """Backward compat: close session."""
        await self.close_session(draft_id, "published")

    async def mark_needs_refinement(self, draft_id: str) -> None:
        """Backward compat: mark session for refinement."""
        await self.mark_session(draft_id, "needs_refinement")

    async def mark_published_by_session(self, session_id: str) -> None:
        """Backward compat: close session."""
        await self.close_session(session_id, "published")

    async def mark_idle_sessions_for_refinement(self, idle_minutes: int = 30) -> int:
        """Mark active sessions with no new messages for idle_minutes as needs_refinement.

        Returns the number of sessions marked.
        """
        cutoff = datetime.utcnow() - timedelta(minutes=idle_minutes)
        cutoff_str = cutoff.isoformat()
        rows = await self._db.execute_fetchall(
            """
            SELECT s.id FROM sessions s
            WHERE s.status = 'active'
              AND s.updated_at < ?
              AND EXISTS (
                SELECT 1 FROM messages m
                WHERE m.session_id = s.id AND m.role = 'assistant'
              )
            """,
            (cutoff_str,),
        )
        if not rows:
            return 0
        now = datetime.utcnow().isoformat()
        for row in rows:
            await self._db.execute(
                "UPDATE sessions SET status='needs_refinement', updated_at=? WHERE id=?",
                (now, row[0]),
            )
        await self._db.commit()
        return len(rows)

    async def get_needs_refinement(self) -> list[dict[str, Any]]:
        """Backward compat: return sessions needing refinement as draft-like dicts."""
        sessions = await self.get_sessions_by_status("needs_refinement")
        result = []
        for s in sessions:
            content = await self.get_accumulated_content(s["id"], role="assistant")
            result.append({
                "id": s["id"],
                "content": content,
                "project": s.get("project", "default"),
                "conversation_id": s["id"],
                "status": "needs_refinement",
            })
        return result

    async def get_all_pending(self) -> list[dict[str, Any]]:
        """Backward compat: return all active sessions."""
        sessions = await self.get_active_sessions()
        result = []
        for s in sessions:
            content = await self.get_accumulated_content(s["id"], role="assistant")
            if content:
                result.append({
                    "id": s["id"],
                    "content": content,
                    "project": s.get("project", "default"),
                    "conversation_id": s["id"],
                    "status": "active",
                })
        return result

    async def get_older_than(self, minutes: int) -> list[dict[str, Any]]:
        """Backward compat: return active sessions older than N minutes."""
        db = self._require_db()
        cursor = await db.execute(
            """
            SELECT * FROM sessions
            WHERE status = 'active'
              AND created_at < datetime('now', ? || ' minutes')
            ORDER BY created_at
            """,
            (f"-{minutes}",),
        )
        rows = await cursor.fetchall()
        result = []
        for s in [self._row_to_dict(r) for r in rows]:
            content = await self.get_accumulated_content(s["id"], role="assistant")
            if content:
                result.append({
                    "id": s["id"],
                    "content": content,
                    "project": s.get("project", "default"),
                    "conversation_id": s["id"],
                    "status": "active",
                })
        return result

    # ------------------------------------------------------------------
    # DraftRefiner compatibility
    # ------------------------------------------------------------------

    async def upsert_draft(
        self,
        session_id: str,
        title: str,
        content: str,
        project: str | None = None,
        group_key: str | None = None,
    ) -> str:
        """Backward compat: ensure session exists. Content is already stored as messages.

        This is called by DraftRefiner.save_draft() — in the new model,
        we just ensure the session exists (messages are already written by pipeline).
        """
        await self.ensure_session(session_id, project or "default")
        return session_id

    async def set_tm_id(self, local_id: str, tm_id: str) -> None:
        """Backward compat: store PG experience ID on session (for publish)."""
        db = self._require_db()
        # Use a simple metadata approach: store tm_id in a column or skip
        # For now we track it in-memory via the refiner
        pass

    async def find_pending_by_title(self, title: str, project: str) -> str | None:
        """Backward compat: not needed in new model."""
        return None

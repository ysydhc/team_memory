"""Local SQLite draft buffer for the memory pipeline.

Stores conversation drafts locally before they are promoted to TeamMemory.
Prevents data loss if hook scripts crash during execution.
"""
from __future__ import annotations

import uuid
from datetime import datetime, timezone
from typing import Any

import aiosqlite

# ---------------------------------------------------------------------------
# SQL
# ---------------------------------------------------------------------------

_CREATE_TABLE = """
CREATE TABLE IF NOT EXISTS drafts (
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
"""


class DraftBuffer:
    """Async SQLite-backed draft buffer for the memory pipeline.

    Usage::

        buf = DraftBuffer("/path/to/drafts.db")
        async with buf:
            draft_id = await buf.create_draft("team_doc", "conv-1", "facts...")
            ...
            await buf.mark_published(draft_id)

    The async context manager opens/closes the underlying aiosqlite connection.
    """

    def __init__(self, db_path: str) -> None:
        self._db_path = db_path
        self._db: aiosqlite.Connection | None = None

    # ------------------------------------------------------------------
    # Context manager
    # ------------------------------------------------------------------

    async def __aenter__(self) -> "DraftBuffer":
        self._db = await aiosqlite.connect(self._db_path)
        self._db.row_factory = aiosqlite.Row
        await self._db.execute(_CREATE_TABLE)
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
            raise RuntimeError("DraftBuffer is not connected. Use 'async with buf:'")
        return self._db

    @staticmethod
    def _row_to_dict(row: aiosqlite.Row) -> dict[str, Any]:
        """Convert an aiosqlite.Row to a plain dict."""
        return dict(row)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def create_draft(
        self,
        project: str,
        conversation_id: str | None,
        content: str,
        title: str = "",
        source: str = "pipeline",
    ) -> str:
        """Create a new draft and return its UUID.

        Args:
            project: Project name (e.g. "team_doc").
            conversation_id: Optional conversation identifier.
            content: Draft content (extracted facts).
            title: Optional title for the draft.
            source: Source tag, e.g. "pipeline" or "obsidian".

        Returns:
            The UUID string of the newly created draft.
        """
        db = self._require_db()
        draft_id = str(uuid.uuid4())
        now = datetime.now(timezone.utc).isoformat()
        await db.execute(
            """
            INSERT INTO drafts (id, title, project, conversation_id, content, status, source, created_at, updated_at)
            VALUES (?, ?, ?, ?, ?, 'pending', ?, ?, ?)
            """,
            (draft_id, title, project, conversation_id, content, source, now, now),
        )
        await db.commit()
        return draft_id

    async def update_draft(self, draft_id: str, content: str) -> None:
        """Update the content of an existing draft.

        Args:
            draft_id: The UUID of the draft to update.
            content: New content (overrides the old value).

        Raises:
            ValueError: If no draft with the given id exists.
        """
        db = self._require_db()
        now = datetime.now(timezone.utc).isoformat()
        cursor = await db.execute(
            """
            UPDATE drafts SET content = ?, updated_at = ?
            WHERE id = ?
            """,
            (content, now, draft_id),
        )
        await db.commit()
        if cursor.rowcount == 0:
            raise ValueError(f"Draft '{draft_id}' not found")

    async def find_pending_by_title(self, title: str, project: str) -> str | None:
        """Find a pending draft by title and project. Returns draft_id or None.

        Used for deduplication: before creating a new draft, check if one
        with the same title/project already exists and is still pending.
        """
        db = self._require_db()
        row = await db.execute(
            "SELECT id FROM drafts WHERE title = ? AND project = ? AND status = 'pending' LIMIT 1",
            (title, project),
        )
        result = await row.fetchone()
        return result[0] if result else None

    async def get_pending_drafts(self, project: str | None = None) -> list[dict[str, Any]]:
        """Return all drafts with status='pending'.

        Args:
            project: If provided, filter to drafts belonging to this project.

        Returns:
            List of draft dicts.
        """
        db = self._require_db()
        if project is not None:
            cursor = await db.execute(
                "SELECT * FROM drafts WHERE status = 'pending' AND project = ? ORDER BY created_at",
                (project,),
            )
        else:
            cursor = await db.execute(
                "SELECT * FROM drafts WHERE status = 'pending' ORDER BY created_at",
            )
        rows = await cursor.fetchall()
        return [self._row_to_dict(r) for r in rows]

    async def get_needs_refinement(self) -> list[dict[str, Any]]:
        """Return all drafts with status='needs_refinement'.

        Returns:
            List of draft dicts waiting for LLM refinement.
        """
        db = self._require_db()
        cursor = await db.execute(
            "SELECT * FROM drafts WHERE status = 'needs_refinement' ORDER BY created_at",
        )
        rows = await cursor.fetchall()
        return [self._row_to_dict(r) for r in rows]

    async def get_older_than(self, minutes: int) -> list[dict[str, Any]]:
        """Return pending drafts older than *minutes* minutes.

        Uses SQLite's datetime comparison against ``created_at``.
        Drafts exactly N minutes old are **not** included (strict <).

        Args:
            minutes: Age threshold in minutes.

        Returns:
            List of draft dicts whose ``created_at`` is more than *minutes*
            minutes in the past.
        """
        db = self._require_db()
        cursor = await db.execute(
            """
            SELECT * FROM drafts
            WHERE status = 'pending'
              AND created_at < datetime('now', ? || ' minutes')
            ORDER BY created_at
            """,
            (f"-{minutes}",),
        )
        rows = await cursor.fetchall()
        return [self._row_to_dict(r) for r in rows]

    async def find_pending_by_conversation(
        self,
        project: str,
        conversation_id: str,
    ) -> list[dict[str, Any]]:
        """Return pending drafts matching project AND conversation_id.

        Args:
            project: Project name to filter by.
            conversation_id: Conversation identifier to filter by.

        Returns:
            List of matching draft dicts (may be empty).
        """
        db = self._require_db()
        cursor = await db.execute(
            """
            SELECT * FROM drafts
            WHERE status = 'pending' AND project = ? AND conversation_id = ?
            ORDER BY created_at
            """,
            (project, conversation_id),
        )
        rows = await cursor.fetchall()
        return [self._row_to_dict(r) for r in rows]

    async def mark_for_publishing(self, draft_id: str) -> None:
        """Mark a draft as ready for publishing (status='ready_to_publish').

        This is a transitional state between 'pending' and 'published'.
        The actual MCP publish call happens in phase 2.

        Args:
            draft_id: The UUID of the draft to mark.

        Raises:
            ValueError: If no draft with the given id exists.
        """
        db = self._require_db()
        now = datetime.now(timezone.utc).isoformat()
        cursor = await db.execute(
            """
            UPDATE drafts SET status = 'ready_to_publish', updated_at = ?
            WHERE id = ?
            """,
            (now, draft_id),
        )
        await db.commit()
        if cursor.rowcount == 0:
            raise ValueError(f"Draft '{draft_id}' not found")

    async def mark_needs_refinement(self, draft_id: str) -> None:
        """Mark a draft as waiting for LLM refinement (status='needs_refinement').

        Args:
            draft_id: The UUID of the draft to mark.

        Raises:
            ValueError: If no draft with the given id exists.
        """
        db = self._require_db()
        now = datetime.now(timezone.utc).isoformat()
        cursor = await db.execute(
            """
            UPDATE drafts SET status = 'needs_refinement', updated_at = ?
            WHERE id = ?
            """,
            (now, draft_id),
        )
        await db.commit()
        if cursor.rowcount == 0:
            raise ValueError(f"Draft '{draft_id}' not found")

    async def mark_published(self, draft_id: str) -> None:
        """Mark a draft as published (status='published').

        Idempotent — calling this on an already-published draft is safe.

        Args:
            draft_id: The UUID of the draft to mark.

        Raises:
            ValueError: If no draft with the given id exists.
        """
        db = self._require_db()
        now = datetime.now(timezone.utc).isoformat()
        cursor = await db.execute(
            """
            UPDATE drafts SET status = 'published', updated_at = ?
            WHERE id = ?
            """,
            (now, draft_id),
        )
        await db.commit()
        if cursor.rowcount == 0:
            raise ValueError(f"Draft '{draft_id}' not found")

    # ------------------------------------------------------------------
    # Convenience methods used by DraftRefiner
    # ------------------------------------------------------------------

    async def upsert_draft(
        self,
        session_id: str,
        title: str,
        content: str,
        project: str | None = None,
        group_key: str | None = None,
    ) -> str:
        """Create or update a draft keyed by *session_id* (conversation_id).

        If a pending draft for this session already exists, its content is
        appended.  Otherwise a new draft row is inserted.

        Args:
            session_id: Conversation / session identifier (stored as conversation_id).
            title: Draft title.
            content: Draft content.
            project: Optional project name (defaults to "default").
            group_key: Optional grouping key.

        Returns:
            The draft UUID (existing or newly created).
        """
        db = self._require_db()
        project = project or "default"
        existing = await self.find_pending_by_conversation(project, session_id)
        if existing:
            draft_id: str = existing[0]["id"]
            old_content: str = existing[0].get("content", "")
            new_content = f"{old_content}\n{content}" if old_content else content
            now = datetime.now(timezone.utc).isoformat()
            await db.execute(
                "UPDATE drafts SET content = ?, updated_at = ? WHERE id = ?",
                (new_content, now, draft_id),
            )
            await db.commit()
            return draft_id

        # No existing draft — create one.
        return await self.create_draft(project, session_id, content)

    async def get_pending(self, session_id: str) -> list[dict[str, Any]]:
        """Return pending drafts for a given session (conversation_id).

        Args:
            session_id: Conversation / session identifier.

        Returns:
            List of matching draft dicts with status='pending'.
        """
        db = self._require_db()
        cursor = await db.execute(
            """
            SELECT * FROM drafts
            WHERE status = 'pending' AND conversation_id = ?
            ORDER BY created_at
            """,
            (session_id,),
        )
        rows = await cursor.fetchall()
        return [self._row_to_dict(r) for r in rows]

    async def get_all_pending(self) -> list[dict[str, Any]]:
        """Return all pending drafts across all sessions.

        Unlike ``get_pending(session_id)`` which filters by conversation_id,
        this method returns every draft with status='pending' regardless of
        which session it belongs to.

        Returns:
            List of all pending draft dicts.
        """
        return await self.get_pending_drafts(project=None)

    async def mark_published_by_session(self, session_id: str) -> None:
        """Mark all pending drafts for *session_id* as published.

        Args:
            session_id: Conversation / session identifier.
        """
        db = self._require_db()
        now = datetime.now(timezone.utc).isoformat()
        await db.execute(
            """
            UPDATE drafts SET status = 'published', updated_at = ?
            WHERE conversation_id = ? AND status = 'pending'
            """,
            (now, session_id),
        )
        await db.commit()

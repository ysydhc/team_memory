"""LLMWiki — local SQLite knowledge store with FTS5 full-text search.

A lightweight, standalone knowledge base for storing and retrieving
wiki entries (knowledge cards). Uses aiosqlite for async access and
FTS5 for fast keyword/title search.

Independent of PostgreSQL or TMSink — purely local SQLite.
"""
from __future__ import annotations

import json
import os
import uuid
from datetime import datetime, timezone
from typing import Any

import aiosqlite

# ---------------------------------------------------------------------------
# SQL DDL
# ---------------------------------------------------------------------------

_CREATE_ENTRIES = """
CREATE TABLE IF NOT EXISTS wiki_entries (
    id TEXT PRIMARY KEY,
    title TEXT NOT NULL,
    content TEXT NOT NULL,
    project TEXT NOT NULL DEFAULT 'default',
    tags TEXT,
    source_url TEXT,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
)
"""

_CREATE_INDEX_TITLE_PROJECT = """
CREATE UNIQUE INDEX IF NOT EXISTS idx_wiki_entries_title_project
ON wiki_entries(title, project)
"""

_CREATE_FTS = """
CREATE VIRTUAL TABLE IF NOT EXISTS wiki_fts USING fts5(
    title, content, content=wiki_entries, content_rowid=rowid
)
"""

# Triggers to keep FTS in sync with wiki_entries
_TRIGGER_INSERT = """
CREATE TRIGGER IF NOT EXISTS wiki_fts_insert AFTER INSERT ON wiki_entries BEGIN
    INSERT INTO wiki_fts(rowid, title, content)
    VALUES (new.rowid, new.title, new.content);
END
"""

_TRIGGER_DELETE = """
CREATE TRIGGER IF NOT EXISTS wiki_fts_delete AFTER DELETE ON wiki_entries BEGIN
    INSERT INTO wiki_fts(wiki_fts, rowid, title, content)
    VALUES ('delete', old.rowid, old.title, old.content);
END
"""

_TRIGGER_UPDATE = """
CREATE TRIGGER IF NOT EXISTS wiki_fts_update AFTER UPDATE ON wiki_entries BEGIN
    INSERT INTO wiki_fts(wiki_fts, rowid, title, content)
    VALUES ('delete', old.rowid, old.title, old.content);
    INSERT INTO wiki_fts(rowid, title, content)
    VALUES (new.rowid, new.title, new.content);
END
"""


# ---------------------------------------------------------------------------
# LLMWiki class
# ---------------------------------------------------------------------------


class LLMWiki:
    """Local knowledge base — stores knowledge cards separate from experiences.

    Usage::

        wiki = LLMWiki("~/.cache/tm-pipeline/wiki.db")
        async with wiki:
            await wiki.add("Python GIL", "The GIL is...", project="team_doc")
            results = await wiki.search("GIL concurrency")
    """

    def __init__(self, db_path: str = "~/.cache/tm-pipeline/wiki.db") -> None:
        self._db_path = os.path.expanduser(db_path)
        self._db: aiosqlite.Connection | None = None

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def __aenter__(self) -> "LLMWiki":
        os.makedirs(os.path.dirname(self._db_path), exist_ok=True)
        self._db = await aiosqlite.connect(self._db_path)
        self._db.row_factory = aiosqlite.Row
        await self._db.execute(_CREATE_ENTRIES)
        await self._db.execute(_CREATE_INDEX_TITLE_PROJECT)
        await self._db.execute(_CREATE_FTS)
        await self._db.execute(_TRIGGER_INSERT)
        await self._db.execute(_TRIGGER_DELETE)
        await self._db.execute(_TRIGGER_UPDATE)
        await self._db.commit()
        return self

    async def __aexit__(self, *exc: Any) -> None:
        if self._db is not None:
            await self._db.close()
            self._db = None

    def _require_db(self) -> aiosqlite.Connection:
        if self._db is None:
            raise RuntimeError("LLMWiki not connected. Use 'async with wiki:'")
        return self._db

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def add(
        self,
        title: str,
        content: str,
        project: str = "default",
        tags: list[str] | None = None,
        source_url: str | None = None,
    ) -> str:
        """Add or update a wiki entry. Returns the entry id.

        Upsert by title+project: same title in same project = update.
        """
        db = self._require_db()
        now = datetime.now(timezone.utc).isoformat()
        tags_json = json.dumps(tags) if tags else None

        # Check if entry exists with same title+project
        cursor = await db.execute(
            "SELECT id FROM wiki_entries WHERE title = ? AND project = ?",
            (title, project),
        )
        existing = await cursor.fetchone()

        if existing:
            entry_id = existing["id"]
            await db.execute(
                """
                UPDATE wiki_entries
                SET content = ?, tags = ?, source_url = ?, updated_at = ?
                WHERE id = ?
                """,
                (content, tags_json, source_url, now, entry_id),
            )
        else:
            entry_id = str(uuid.uuid4())
            await db.execute(
                """
                INSERT INTO wiki_entries (id, title, content, project, tags, source_url, created_at, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (entry_id, title, content, project, tags_json, source_url, now, now),
            )

        await db.commit()
        return entry_id

    async def search(
        self, query: str, project: str | None = None, limit: int = 5
    ) -> list[dict[str, Any]]:
        """FTS5 full-text search. Returns [{id, title, content, project, tags, score}].

        Uses BM25 ranking for relevance scoring.
        """
        db = self._require_db()

        if not query.strip():
            return []

        # Escape FTS5 special characters and build query
        fts_query = self._prepare_fts_query(query)
        if not fts_query:
            return []

        if project:
            sql = """
                SELECT e.id, e.title, e.content, e.project, e.tags, e.source_url,
                       rank AS score
                FROM wiki_fts f
                JOIN wiki_entries e ON e.rowid = f.rowid
                WHERE wiki_fts MATCH ? AND e.project = ?
                ORDER BY rank
                LIMIT ?
            """
            cursor = await db.execute(sql, (fts_query, project, limit))
        else:
            sql = """
                SELECT e.id, e.title, e.content, e.project, e.tags, e.source_url,
                       rank AS score
                FROM wiki_fts f
                JOIN wiki_entries e ON e.rowid = f.rowid
                WHERE wiki_fts MATCH ?
                ORDER BY rank
                LIMIT ?
            """
            cursor = await db.execute(sql, (fts_query, limit))

        rows = await cursor.fetchall()
        results = []
        for row in rows:
            tags_raw = row["tags"]
            tags = json.loads(tags_raw) if tags_raw else []
            results.append({
                "id": row["id"],
                "title": row["title"],
                "content": row["content"],
                "project": row["project"],
                "tags": tags,
                "source_url": row["source_url"],
                "score": row["score"],
            })
        return results

    async def get(self, entry_id: str) -> dict[str, Any] | None:
        """Get a wiki entry by id."""
        db = self._require_db()
        cursor = await db.execute(
            "SELECT * FROM wiki_entries WHERE id = ?", (entry_id,)
        )
        row = await cursor.fetchone()
        if row is None:
            return None
        tags_raw = row["tags"]
        return {
            "id": row["id"],
            "title": row["title"],
            "content": row["content"],
            "project": row["project"],
            "tags": json.loads(tags_raw) if tags_raw else [],
            "source_url": row["source_url"],
            "created_at": row["created_at"],
            "updated_at": row["updated_at"],
        }

    async def delete(self, entry_id: str) -> bool:
        """Delete a wiki entry by id. Returns True if deleted."""
        db = self._require_db()
        cursor = await db.execute(
            "DELETE FROM wiki_entries WHERE id = ?", (entry_id,)
        )
        await db.commit()
        return cursor.rowcount > 0

    async def stats(self) -> dict[str, Any]:
        """Return {total, by_project: {project: count}}."""
        db = self._require_db()
        cursor = await db.execute("SELECT COUNT(*) FROM wiki_entries")
        row = await cursor.fetchone()
        total = row[0] if row else 0

        cursor = await db.execute(
            "SELECT project, COUNT(*) as cnt FROM wiki_entries GROUP BY project"
        )
        rows = await cursor.fetchall()
        by_project = {row["project"]: row["cnt"] for row in rows}

        return {"total": total, "by_project": by_project}

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _prepare_fts_query(query: str) -> str:
        """Prepare a user query for FTS5 MATCH.

        Splits into tokens and joins with OR for fuzzy matching.
        Strips FTS5 special characters to prevent syntax errors.
        """
        # Remove FTS5 special chars
        special = set('"*(){}[]^~:')
        cleaned = "".join(c if c not in special else " " for c in query)
        tokens = cleaned.split()
        if not tokens:
            return ""
        # Join tokens with OR for broader matching; quote each token
        parts = [f'"{t}"' for t in tokens if t.strip()]
        if not parts:
            return ""
        return " OR ".join(parts)

"""WikiCompiler — compile PG experiences into structured markdown wiki files.

Reads experience dicts from PostgreSQL and generates a browsable wiki
directory of interlinked markdown files, inspired by Karpathy's LLM Wiki pattern.

Uses SQLite (aiosqlite) for incremental compilation caching, same pattern
as DraftBuffer.
"""

from __future__ import annotations

import hashlib
import os
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

import aiofiles
import aiosqlite

# ---------------------------------------------------------------------------
# SQL
# ---------------------------------------------------------------------------

_CREATE_TABLE = """
CREATE TABLE IF NOT EXISTS wiki_cache (
    experience_id TEXT PRIMARY KEY,
    content_hash TEXT NOT NULL,
    wiki_path TEXT NOT NULL,
    status TEXT NOT NULL DEFAULT 'compiled',
    project TEXT NOT NULL DEFAULT '',
    title TEXT NOT NULL DEFAULT '',
    compiled_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
)
"""

_CREATE_INDEX_STATUS = """
CREATE INDEX IF NOT EXISTS idx_wiki_cache_status ON wiki_cache(status)
"""

_CREATE_INDEX_PROJECT = """
CREATE INDEX IF NOT EXISTS idx_wiki_cache_project ON wiki_cache(project)
"""

# ---------------------------------------------------------------------------
# Page template
# ---------------------------------------------------------------------------

_PAGE_TEMPLATE = """\
---
title: {title}
tags: [{tags}]
source_ids: [{source_ids}]
project: {project}
experience_type: {experience_type}
created_at: {created_at}
updated_at: {updated_at}
confidence: {confidence}
recall_count: {recall_count}
---

## 摘要

{summary}

{problem_section}{solution_section}{related_section}
## 来源

- experience_id: {experience_id}
"""

_PROBLEM_SECTION = """\
## 问题描述

{problem}

"""

_SOLUTION_SECTION = """\
## 解决方案

{solution}

"""

_RELATED_SECTION = """\
## 相关经验

{related_links}

"""

# ---------------------------------------------------------------------------
# Index template
# ---------------------------------------------------------------------------

_INDEX_HEADER = """\
# TM Wiki Index

> 自动生成，勿手动编辑。最后更新: {timestamp}

"""

# ---------------------------------------------------------------------------
# Log template
# ---------------------------------------------------------------------------

_LOG_HEADER = """\
# TM Wiki Changelog

"""

# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass
class CompileResult:
    """Result of a compilation run."""

    created: int = 0
    updated: int = 0
    skipped: int = 0
    deleted: int = 0
    errors: int = 0


# ---------------------------------------------------------------------------
# WikiCompiler
# ---------------------------------------------------------------------------


class WikiCompiler:
    """Compile PG experiences into a browsable markdown wiki.

    Usage::

        compiler = WikiCompiler(wiki_root="/path/to/wiki")
        async with compiler:
            result = await compiler.compile_incremental(experiences)

    Args:
        wiki_root: Absolute path to the wiki/ directory.
        db_path: Path to the SQLite cache database. Defaults to
            ``<wiki_root/../.wiki/cache.db``.
    """

    def __init__(
        self,
        wiki_root: str,
        db_path: str | None = None,
    ) -> None:
        self._wiki_root = os.path.abspath(wiki_root)
        if db_path is None:
            meta_dir = os.path.join(os.path.dirname(self._wiki_root), ".wiki")
            db_path = os.path.join(meta_dir, "cache.db")
        self._db_path = db_path
        self._db: aiosqlite.Connection | None = None
        # Cache of all experiences for cross-referencing (populated before compile)
        self._all_experiences: list[dict] = []
        # Entity graph: experience_id → set of entity_ids
        self._exp_entity_map: dict[str, set[str]] = {}
        # Embedding matrix: experience_id → numpy array (or None)
        self._exp_embeddings: dict[str, Any] = {}
        # PG connection string for entity/embedding queries
        self._pg_url: str | None = None

    # ------------------------------------------------------------------
    # Lifecycle (same pattern as DraftBuffer)
    # ------------------------------------------------------------------

    async def __aenter__(self) -> WikiCompiler:
        os.makedirs(os.path.dirname(self._db_path), exist_ok=True)
        self._db = await aiosqlite.connect(self._db_path)
        self._db.row_factory = aiosqlite.Row
        await self._db.execute(_CREATE_TABLE)
        await self._db.execute(_CREATE_INDEX_STATUS)
        await self._db.execute(_CREATE_INDEX_PROJECT)
        await self._db.commit()
        # Ensure wiki directories exist
        os.makedirs(os.path.join(self._wiki_root, "concepts"), exist_ok=True)
        os.makedirs(os.path.join(self._wiki_root, "queries"), exist_ok=True)
        os.makedirs(os.path.join(self._wiki_root, "topics"), exist_ok=True)
        return self

    async def __aexit__(self, *exc: Any) -> None:
        if self._db is not None:
            await self._db.close()
            self._db = None

    def _require_db(self) -> aiosqlite.Connection:
        if self._db is None:
            raise RuntimeError("WikiCompiler is not initialized. Use 'async with compiler:'")
        return self._db

    # ------------------------------------------------------------------
    # Internal: PG data loading for enhanced cross-referencing
    # ------------------------------------------------------------------

    async def _load_pg_data(self, pg_url: str) -> None:
        """Load entity-graph and embedding data from PostgreSQL.

        Called once at the start of compile_incremental / full_rebuild.
        Results are cached in self._exp_entity_map and self._exp_embeddings.
        """
        from sqlalchemy import text

        from team_memory.storage.database import get_session

        self._pg_url = pg_url

        # 1. Load experience → entity mapping
        self._exp_entity_map = {}
        try:
            async with get_session(pg_url) as session:
                result = await session.execute(text("""
                    SELECT ee.experience_id, ee.entity_id
                    FROM experience_entities ee
                    JOIN entities e ON e.id = ee.entity_id
                """))
                for row in result.fetchall():
                    exp_id = str(row[0])
                    if exp_id not in self._exp_entity_map:
                        self._exp_entity_map[exp_id] = set()
                    self._exp_entity_map[exp_id].add(str(row[1]))
        except Exception:
            # Non-fatal: entity data is optional enhancement
            self._exp_entity_map = {}

        # 2. Load embeddings
        self._exp_embeddings = {}
        try:
            import numpy as np  # noqa: F811

            async with get_session(pg_url) as session:
                result = await session.execute(text("""
                    SELECT id, embedding FROM experiences
                    WHERE embedding IS NOT NULL AND exp_status = 'published'
                """))
                for row in result.fetchall():
                    exp_id = str(row[0])
                    emb = row[1]
                    if emb is None:
                        continue
                    try:
                        if isinstance(emb, str):
                            # asyncpg returns pgvector as string: "[0.1,0.2,...]"
                            import json
                            vec = json.loads(emb)
                            self._exp_embeddings[exp_id] = np.array(vec, dtype=np.float32)
                        elif hasattr(emb, "tolist"):
                            self._exp_embeddings[exp_id] = np.array(emb.tolist(), dtype=np.float32)
                        elif isinstance(emb, list):
                            self._exp_embeddings[exp_id] = np.array(emb, dtype=np.float32)
                        elif isinstance(emb, bytes):
                            self._exp_embeddings[exp_id] = np.frombuffer(emb, dtype=np.float32)
                    except Exception:
                        continue
        except ImportError:
            # numpy not available — skip embedding similarity
            self._exp_embeddings = {}
        except Exception:
            # Non-fatal: embedding data is optional enhancement
            self._exp_embeddings = {}

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def compile_one(self, experience: dict) -> str:
        """Compile a single experience into a wiki page.

        Args:
            experience: Experience dict with at least id, title, description.

        Returns:
            The relative wiki_path, e.g. "concepts/slug.md".
        """
        db = self._require_db()

        # 1. Compute slug → wiki_path
        slug = self._slugify(experience.get("title", "untitled"))
        wiki_path = f"concepts/{slug}.md"

        # Handle slug collision: append short hash if file already exists
        # for a *different* experience
        existing = await self._get_cache_by_path(wiki_path)
        existing_exp_id = existing.get("experience_id") if existing else None
        exp_id = str(experience.get("id", ""))
        if existing is not None and existing_exp_id != exp_id:
            slug = f"{slug}-{exp_id[:8]}"
            wiki_path = f"concepts/{slug}.md"

        # 2. Find related experiences
        related = self._find_related(experience, self._all_experiences)

        # 3. Render markdown
        content = self._render_page(experience, related)

        # 4. Write file
        full_path = os.path.join(self._wiki_root, wiki_path)
        os.makedirs(os.path.dirname(full_path), exist_ok=True)
        async with aiofiles.open(full_path, "w", encoding="utf-8") as f:
            await f.write(content)

        # 5. Update cache
        content_hash = self._compute_hash(experience)
        await db.execute(
            """
            INSERT INTO wiki_cache
                (experience_id, content_hash, wiki_path, status, project, title, compiled_at)
            VALUES (?, ?, ?, 'compiled', ?, ?, ?)
            ON CONFLICT(experience_id) DO UPDATE SET
                content_hash = excluded.content_hash,
                wiki_path = excluded.wiki_path,
                status = 'compiled',
                project = excluded.project,
                title = excluded.title,
                compiled_at = excluded.compiled_at
            """,
            (
                exp_id,
                content_hash,
                wiki_path,
                experience.get("project", ""),
                experience.get("title", ""),
                datetime.now(timezone.utc).isoformat(),
            ),
        )
        await db.commit()

        return wiki_path

    async def compile_batch(self, experiences: list[dict]) -> list[str]:
        """Compile multiple experiences.

        Args:
            experiences: List of experience dicts.

        Returns:
            List of wiki_path strings.
        """
        paths: list[str] = []
        for exp in experiences:
            try:
                path = await self.compile_one(exp)
                paths.append(path)
            except Exception:
                # Skip failures in batch mode
                continue
        return paths

    async def compile_incremental(
        self, all_experiences: list[dict], pg_url: str | None = None
    ) -> CompileResult:
        """Incremental compile: only process new/changed experiences.

        Args:
            all_experiences: All published experiences from PG.
            pg_url: PostgreSQL URL for entity/embedding data loading.

        Returns:
            CompileResult with statistics.
        """
        db = self._require_db()
        self._all_experiences = all_experiences
        result = CompileResult()

        # 0. Load entity-graph + embedding data from PG (for enhanced cross-ref)
        if pg_url:
            await self._load_pg_data(pg_url)

        # 1. Load all cached records
        cached = await self._get_all_cached()

        # 2. Process each experience
        for exp in all_experiences:
            exp_id = str(exp.get("id", ""))
            new_hash = self._compute_hash(exp)

            if exp_id not in cached:
                # New experience
                try:
                    await self.compile_one(exp)
                    result.created += 1
                except Exception:
                    result.errors += 1
            elif cached[exp_id]["content_hash"] != new_hash:
                # Changed experience
                try:
                    await self.compile_one(exp)
                    result.updated += 1
                except Exception:
                    result.errors += 1
            else:
                result.skipped += 1

        # 3. Delete wiki pages for experiences no longer in PG
        current_ids = {str(exp.get("id", "")) for exp in all_experiences}
        for exp_id, record in cached.items():
            if exp_id not in current_ids:
                wiki_path = record["wiki_path"]
                full_path = os.path.join(self._wiki_root, wiki_path)
                if os.path.exists(full_path):
                    os.remove(full_path)
                await db.execute(
                    "DELETE FROM wiki_cache WHERE experience_id = ?", (exp_id,)
                )
                result.deleted += 1
        await db.commit()

        # 4. Update index.md + log.md + topics if anything changed
        if result.created + result.updated + result.deleted > 0:
            await self._update_index()
            await self._update_topics()
            await self._generate_contradictions()
            await self._append_log(
                "incremental",
                f"created={result.created} updated={result.updated} "
                f"deleted={result.deleted} skipped={result.skipped}",
            )

        return result

    async def full_rebuild(
        self, all_experiences: list[dict], pg_url: str | None = None
    ) -> CompileResult:
        """Full rebuild: clear cache and recompile everything.

        Args:
            all_experiences: All published experiences from PG.
            pg_url: PostgreSQL URL for entity/embedding data loading.

        Returns:
            CompileResult with statistics.
        """
        db = self._require_db()
        self._all_experiences = all_experiences

        # 0. Load entity-graph + embedding data from PG (for enhanced cross-ref)
        if pg_url:
            await self._load_pg_data(pg_url)

        # 1. Clear cache
        await db.execute("DELETE FROM wiki_cache")
        await db.commit()

        # 2. Remove all existing wiki files
        concepts_dir = os.path.join(self._wiki_root, "concepts")
        if os.path.isdir(concepts_dir):
            for fname in os.listdir(concepts_dir):
                if fname.endswith(".md"):
                    os.remove(os.path.join(concepts_dir, fname))

        # 3. Compile all
        result = CompileResult()
        for exp in all_experiences:
            try:
                await self.compile_one(exp)
                result.created += 1
            except Exception:
                result.errors += 1

        # 4. Update index + topics + contradictions + log
        await self._update_index()
        await self._update_topics()
        await self._generate_contradictions()
        await self._append_log(
            "full-rebuild",
            f"created={result.created} errors={result.errors}",
        )

        return result

    # ------------------------------------------------------------------
    # Query API
    # ------------------------------------------------------------------

    async def get_stale(self) -> list[str]:
        """Return experience_ids with status='stale'."""
        db = self._require_db()
        cursor = await db.execute(
            "SELECT experience_id FROM wiki_cache WHERE status = 'stale'"
        )
        rows = await cursor.fetchall()
        return [row["experience_id"] for row in rows]

    async def get_uncompiled(self, experience_ids: list[str]) -> list[str]:
        """Return experience_ids not present in cache."""
        db = self._require_db()
        if not experience_ids:
            return []
        placeholders = ",".join("?" for _ in experience_ids)
        cursor = await db.execute(
            f"SELECT experience_id FROM wiki_cache WHERE experience_id IN ({placeholders})",
            experience_ids,
        )
        rows = await cursor.fetchall()
        compiled = {row["experience_id"] for row in rows}
        return [eid for eid in experience_ids if eid not in compiled]

    async def get_by_project(self, project: str) -> list[dict]:
        """Return all cache records for a project."""
        db = self._require_db()
        cursor = await db.execute(
            "SELECT * FROM wiki_cache WHERE project = ? ORDER BY compiled_at DESC",
            (project,),
        )
        rows = await cursor.fetchall()
        return [dict(row) for row in rows]

    async def get_stats(self) -> dict[str, Any]:
        """Return compilation statistics."""
        db = self._require_db()
        cursor = await db.execute("SELECT COUNT(*) as total FROM wiki_cache")
        row = await cursor.fetchone()
        total = row["total"] if row else 0

        cursor = await db.execute(
            "SELECT project, COUNT(*) as cnt FROM wiki_cache GROUP BY project ORDER BY cnt DESC"
        )
        rows = await cursor.fetchall()
        by_project = {row["project"]: row["cnt"] for row in rows}

        cursor = await db.execute(
            "SELECT status, COUNT(*) as cnt FROM wiki_cache GROUP BY status"
        )
        rows = await cursor.fetchall()
        by_status = {row["status"]: row["cnt"] for row in rows}

        return {
            "total": total,
            "by_project": by_project,
            "by_status": by_status,
        }

    # ------------------------------------------------------------------
    # Internal: slugify
    # ------------------------------------------------------------------

    @staticmethod
    def _slugify(title: str) -> str:
        """Convert a title to a filesystem-safe slug.

        Rules:
        - Chinese characters preserved
        - / \\ ： : → -
        - Spaces → -
        - Consecutive dashes collapsed
        - Lowercase ASCII
        - + → 加 (readability in Chinese context)
        - Strip leading/trailing dashes
        - Max 100 chars
        """
        if not title:
            return "untitled"

        s = title
        # Replace problematic characters
        s = re.sub(r"[/\\：:|]", "-", s)
        # + → 加 for readability
        s = s.replace("+", "加")
        # Spaces → dashes
        s = re.sub(r"\s+", "-", s)
        # Collapse consecutive dashes
        s = re.sub(r"-+", "-", s)
        # Lowercase ASCII letters only (preserve CJK etc.)
        s = s.lower()
        # Strip leading/trailing dashes
        s = s.strip("-")
        # Truncate to 100 chars
        if len(s) > 100:
            s = s[:100].rstrip("-")
        return s or "untitled"

    # ------------------------------------------------------------------
    # Internal: hash
    # ------------------------------------------------------------------

    @staticmethod
    def _compute_hash(experience: dict) -> str:
        """SHA-256 of title + description + solution (first 16 hex chars)."""
        parts = [
            experience.get("title", ""),
            experience.get("description", ""),
            experience.get("solution", "") or "",
        ]
        raw = "|".join(parts)
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]

    # ------------------------------------------------------------------
    # Internal: cross-referencing
    # ------------------------------------------------------------------

    def _find_related(
        self,
        experience: dict,
        all_experiences: list[dict],
        max_related: int = 5,
    ) -> list[dict]:
        """Find related experiences using multi-signal scoring.

        Signals:
        - tag_overlap * 2
        - same_project * 1
        - entity_overlap * 3 (shared entities via entity-graph)
        - embedding_similarity * 2 (cosine sim > 0.7, only when embeddings available)

        Returns up to max_related experiences sorted by score descending.
        """
        if not all_experiences:
            return []

        exp_id = str(experience.get("id", ""))
        exp_tags = set(experience.get("tags") or [])
        exp_project = experience.get("project", "")
        exp_entities = self._exp_entity_map.get(exp_id, set())
        exp_emb = self._exp_embeddings.get(exp_id)

        scored: list[tuple[float, dict]] = []
        for other in all_experiences:
            other_id = str(other.get("id", ""))
            if other_id == exp_id:
                continue

            other_tags = set(other.get("tags") or [])
            other_project = other.get("project", "")

            # Signal 1: tag overlap
            tag_overlap = len(exp_tags & other_tags)
            score = float(tag_overlap * 2)

            # Signal 2: same project
            if exp_project and exp_project == other_project:
                score += 1.0

            # Signal 3: entity-graph overlap (shared entities)
            other_entities = self._exp_entity_map.get(other_id, set())
            entity_overlap = len(exp_entities & other_entities)
            score += float(entity_overlap * 3)

            # Signal 4: embedding cosine similarity
            other_emb = self._exp_embeddings.get(other_id)
            if exp_emb is not None and other_emb is not None:
                try:
                    import numpy as np
                    sim = float(np.dot(exp_emb, other_emb) / (
                        np.linalg.norm(exp_emb) * np.linalg.norm(other_emb) + 1e-8
                    ))
                    if sim > 0.7:
                        score += sim * 2
                except Exception:
                    pass

            # Skip if no signal at all (no tags, no project, no entities, no embedding)
            if score <= 0 and not exp_entities and exp_emb is None:
                if tag_overlap == 0 and exp_project != other_project:
                    continue

            if score > 0:
                scored.append((score, other))

        scored.sort(key=lambda x: x[0], reverse=True)
        return [item[1] for item in scored[:max_related]]

    # ------------------------------------------------------------------
    # Internal: page rendering
    # ------------------------------------------------------------------

    def _render_page(self, experience: dict, related: list[dict]) -> str:
        """Render an experience dict into a markdown wiki page."""
        title = experience.get("title", "Untitled")
        tags = experience.get("tags") or []
        tags_str = ", ".join(str(t) for t in tags)
        source_id = str(experience.get("id", ""))
        project = experience.get("project", "")
        exp_type = experience.get("experience_type", "general")
        created_at = self._format_date(experience.get("created_at"))
        updated_at = self._format_date(experience.get("updated_at"))
        confidence = experience.get("confidence", "medium")
        recall_count = experience.get("recall_count", 0)

        # Summary: first 200 chars of description
        description = experience.get("description", "")
        summary = description[:200].strip() if description else "（无摘要）"

        # Problem section
        problem_section = ""
        if description:
            problem_section = _PROBLEM_SECTION.format(problem=description)

        # Solution section
        solution_section = ""
        solution = experience.get("solution") or ""
        if solution:
            solution_section = _SOLUTION_SECTION.format(solution=solution)

        # Related section
        related_section = ""
        if related:
            links = []
            for r in related:
                r_slug = self._slugify(r.get("title", ""))
                r_title = r.get("title", "Untitled")
                links.append(f"- [[{r_slug}|{r_title}]]")
            related_section = _RELATED_SECTION.format(
                related_links="\n".join(links)
            )

        return _PAGE_TEMPLATE.format(
            title=title,
            tags=tags_str,
            source_ids=source_id,
            project=project,
            experience_type=exp_type,
            created_at=created_at,
            updated_at=updated_at,
            confidence=confidence,
            recall_count=recall_count,
            summary=summary,
            problem_section=problem_section,
            solution_section=solution_section,
            related_section=related_section,
            experience_id=source_id,
        )

    # ------------------------------------------------------------------
    # Internal: index.md
    # ------------------------------------------------------------------

    async def _update_index(self) -> None:
        """Regenerate index.md from all cached wiki pages."""
        db = self._require_db()

        cursor = await db.execute(
            "SELECT title, wiki_path, project FROM wiki_cache ORDER BY project, title"
        )
        rows = await cursor.fetchall()

        now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M")
        lines = [_INDEX_HEADER.format(timestamp=now)]

        current_project = ""
        for row in rows:
            project = row["project"] or "default"
            title = row["title"]
            wiki_path = row["wiki_path"]
            slug = os.path.splitext(os.path.basename(wiki_path))[0]

            if project != current_project:
                # Count experiences in this project
                cnt_cursor = await db.execute(
                    "SELECT COUNT(*) as cnt FROM wiki_cache WHERE project = ?",
                    (project,),
                )
                cnt_row = await cnt_cursor.fetchone()
                count = cnt_row["cnt"] if cnt_row else 0
                lines.append(f"## {project} ({count})\n")
                current_project = project

            lines.append(f"- [[{slug}|{title}]]")

        content = "\n".join(lines)
        index_path = os.path.join(self._wiki_root, "index.md")
        async with aiofiles.open(index_path, "w", encoding="utf-8") as f:
            await f.write(content)

    # ------------------------------------------------------------------
    # Internal: log.md
    # ------------------------------------------------------------------

    async def _append_log(self, action: str, detail: str, title: str = "") -> None:
        """Append an entry to log.md."""
        now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M")
        entry = f"## [{now}] {action}"
        if title:
            entry += f" | {title}"
        entry += f" | {detail}\n"

        log_path = os.path.join(self._wiki_root, "log.md")
        if not os.path.exists(log_path):
            async with aiofiles.open(log_path, "w", encoding="utf-8") as f:
                await f.write(_LOG_HEADER)
        async with aiofiles.open(log_path, "a", encoding="utf-8") as f:
            await f.write(entry)

    # ------------------------------------------------------------------
    # Internal: cache helpers
    # ------------------------------------------------------------------

    async def _get_all_cached(self) -> dict[str, dict]:
        """Return all cache records keyed by experience_id."""
        db = self._require_db()
        cursor = await db.execute("SELECT * FROM wiki_cache")
        rows = await cursor.fetchall()
        return {row["experience_id"]: dict(row) for row in rows}

    async def _get_cache_by_path(self, wiki_path: str) -> dict | None:
        """Return cache record by wiki_path, or None."""
        db = self._require_db()
        cursor = await db.execute(
            "SELECT * FROM wiki_cache WHERE wiki_path = ?", (wiki_path,)
        )
        row = await cursor.fetchone()
        return dict(row) if row else None

    # ------------------------------------------------------------------
    # Internal: date formatting
    # ------------------------------------------------------------------

    @staticmethod
    def _format_date(value: Any) -> str:
        """Format a datetime value to ISO date string."""
        if value is None:
            return ""
        if isinstance(value, datetime):
            return value.strftime("%Y-%m-%d")
        s = str(value)[:10]  # Take first 10 chars (YYYY-MM-DD)
        return s

    # ------------------------------------------------------------------
    # Internal: topic pages (high-level retrieval)
    # ------------------------------------------------------------------

    async def _update_topics(self) -> None:
        """Generate wiki/topics/ pages from discovered topic clusters."""
        if not self._pg_url or not self._exp_embeddings:
            return

        try:
            from team_memory.services.topic_discovery import discover_topics
        except ImportError:
            return

        result = await discover_topics(self._pg_url)
        if not result.topics:
            return

        # Clear existing topic files
        topics_dir = os.path.join(self._wiki_root, "topics")
        if os.path.isdir(topics_dir):
            for fname in os.listdir(topics_dir):
                if fname.endswith(".md"):
                    os.remove(os.path.join(topics_dir, fname))

        # Generate a page for each topic
        for topic in result.topics:
            slug = self._slugify(topic.name)
            topic_path = os.path.join(topics_dir, f"{slug}.md")

            # Build entity list
            entity_lines = []
            for ent in topic.entities[:10]:
                entity_lines.append(f"- {ent}")

            # Build experience list (link to wiki pages)
            exp_lines = []
            for exp_id in topic.experience_ids[:20]:
                # Find wiki_path from cache
                exp_cache = await self._get_cache_by_experience_id(exp_id)
                if exp_cache:
                    r_slug = os.path.splitext(os.path.basename(exp_cache["wiki_path"]))[0]
                    r_title = exp_cache.get("title", exp_id[:8])
                    exp_lines.append(f"- [[{r_slug}|{r_title}]]")
                else:
                    exp_lines.append(f"- {exp_id[:8]}")

            # Build tag list
            tag_str = ", ".join(topic.tags[:5])

            content = f"""\
---
type: topic
name: {topic.name}
tags: [{tag_str}]
experience_count: {len(topic.experience_ids)}
---

# {topic.name}

> 自动发现的主题聚类，包含 {len(topic.experience_ids)} 条相关经验。

## 核心实体

{chr(10).join(entity_lines) if entity_lines else "（无）"}

## 关联经验

{chr(10).join(exp_lines) if exp_lines else "（无）"}
"""
            async with aiofiles.open(topic_path, "w", encoding="utf-8") as f:
                await f.write(content)

        # Update index to include topics section
        await self._update_index_with_topics(result.topics)

    async def _get_cache_by_experience_id(self, experience_id: str) -> dict | None:
        """Return cache record by experience_id."""
        db = self._require_db()
        cursor = await db.execute(
            "SELECT * FROM wiki_cache WHERE experience_id = ?", (experience_id,)
        )
        row = await cursor.fetchone()
        return dict(row) if row else None

    async def _update_index_with_topics(self, topics: list) -> None:
        """Append topics section to index.md."""
        index_path = os.path.join(self._wiki_root, "index.md")
        if not os.path.exists(index_path):
            return

        async with aiofiles.open(index_path, "r", encoding="utf-8") as f:
            content = await f.read()

        # Append topics section
        topic_lines = ["\n## 主题聚类\n"]
        for topic in topics:
            slug = self._slugify(topic.name)
            topic_lines.append(
                f"- [[topics/{slug}|{topic.name}]] ({len(topic.experience_ids)} 条经验)"
            )

        content += "\n".join(topic_lines) + "\n"
        async with aiofiles.open(index_path, "w", encoding="utf-8") as f:
            await f.write(content)

    # ------------------------------------------------------------------
    # Internal: contradictions report
    # ------------------------------------------------------------------

    async def _generate_contradictions(self) -> None:
        """Generate wiki/contradictions.md from contradiction detection."""
        if not self._pg_url:
            return

        try:
            from team_memory.services.contradiction_detector import (
                detect_contradictions,
            )
        except ImportError:
            return

        pairs = await detect_contradictions(self._pg_url, max_pairs=20)
        if not pairs:
            return

        # Build the content
        lines = [
            "# 矛盾经验检测报告\n",
            f"> 自动生成，共检测到 {len(pairs)} 对潜在矛盾经验。\n",
            "",
        ]

        for i, pair in enumerate(pairs, 1):
            lines.append(f"## {i}. {pair.exp_a_title[:60]}\n")
            lines.append(f"**经验 A:** {pair.exp_a_title}")
            lines.append(f"- ID: `{pair.exp_a_id}`\n")
            lines.append(f"**经验 B:** {pair.exp_b_title}")
            lines.append(f"- ID: `{pair.exp_b_id}`\n")
            lines.append(f"**共享实体:** {', '.join(pair.shared_entities)}")
            lines.append(f"**矛盾原因:** {pair.reason}\n")

        contradictions_path = os.path.join(
            self._wiki_root, "contradictions.md"
        )
        async with aiofiles.open(
            contradictions_path, "w", encoding="utf-8"
        ) as f:
            await f.write("\n".join(lines))


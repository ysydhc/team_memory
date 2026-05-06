"""SearchLog writer — writes search logs and usage tracking to TM database.

This module provides lightweight functions to:
1. Log each retrieval with query + result_ids
2. Mark entries as was_used when [mem:xxx] markers are found in agent responses
3. Fuzzy match: check if agent response overlaps with search result keywords
4. Query stats for evaluation dashboards

Uses raw asyncpg for performance (no ORM overhead in the daemon).
"""
from __future__ import annotations

import json
import logging
import re
import uuid
from datetime import datetime, timezone
from typing import Any

import asyncpg

_MEM_MARKER_RE = re.compile(r"\[mem:([0-9a-f]{8})\]", re.IGNORECASE)

logger = logging.getLogger("daemon.search_log_writer")


class SearchLogWriter:
    """Lightweight async writer for search_logs table."""

    def __init__(self, dsn: str = "postgresql://developer:***@localhost:5433/team_memory") -> None:
        self._dsn = dsn
        self._pool: asyncpg.Pool | None = None

    async def _ensure_pool(self) -> asyncpg.Pool:
        if self._pool is None or self._pool._closed:
            self._pool = await asyncpg.create_pool(self._dsn, min_size=1, max_size=3)
        return self._pool

    async def log_search(
        self,
        *,
        query: str,
        project: str = "default",
        source: str = "daemon",
        result_ids: list[dict[str, Any]] | None = None,
    ) -> str | None:
        """Write a search log entry. Returns the log ID or None on failure."""
        try:
            pool = await self._ensure_pool()
            log_id = uuid.uuid4()
            async with pool.acquire() as conn:
                await conn.execute(
                    """
                    INSERT INTO search_logs (id, query, project, source, result_ids, created_at, updated_at)
                    VALUES ($1, $2, $3, $4, $5, $6, $6)
                    """,
                    log_id,
                    query[:500],
                    project,
                    source,
                    json.dumps(result_ids) if result_ids else None,
                    datetime.now(timezone.utc),
                )
            return str(log_id)
        except Exception as e:
            logger.debug("log_search failed: %s", e)
            return None

    async def mark_used_from_response(
        self,
        agent_response: str,
        *,
        project: str | None = None,
        hours_window: int = 24,
    ) -> int:
        """Scan agent response for [mem:xxx] markers and mark matching search logs.

        Sets judgment_source = 'marker' for marker-based matches.
        Also increments used_count on the corresponding experience.

        Returns the number of logs marked as used.
        """
        markers = _MEM_MARKER_RE.findall(agent_response)
        if not markers:
            return 0

        try:
            pool = await self._ensure_pool()
            marked = 0
            async with pool.acquire() as conn:
                for marker_prefix in markers:
                    # Find search logs whose result_ids contain an experience
                    # with an id starting with this marker prefix
                    rows = await conn.fetch(
                        """
                        SELECT id, result_ids FROM search_logs
                        WHERE was_used IS NULL
                          AND created_at > NOW() - make_interval(hours => $1)
                          AND ($2::text IS NULL OR project = $2)
                        """,
                        hours_window,
                        project,
                    )
                    for row in rows:
                        result_ids = row["result_ids"]
                        if result_ids is None:
                            continue
                        if isinstance(result_ids, str):
                            result_ids = json.loads(result_ids)
                        for r in result_ids:
                            rid = str(r.get("id", ""))
                            if rid.startswith(marker_prefix):
                                now = datetime.now(timezone.utc)
                                await conn.execute(
                                    """
                                    UPDATE search_logs
                                    SET was_used = TRUE, judgment_source = 'marker', updated_at = $2
                                    WHERE id = $1
                                    """,
                                    row["id"],
                                    now,
                                )
                                # Increment used_count on the experience
                                try:
                                    exp_uuid = uuid.UUID(rid)
                                    await conn.execute(
                                        """
                                        UPDATE experiences
                                        SET used_count = used_count + 1, updated_at = $2
                                        WHERE id = $1
                                        """,
                                        exp_uuid,
                                        now,
                                    )
                                except (ValueError, Exception):
                                    pass
                                marked += 1
                                break
            return marked
        except Exception as e:
            logger.debug("mark_used_from_response failed: %s", e)
            return 0

    async def get_recent_with_results(
        self,
        *,
        project: str | None = None,
        hours_window: int = 1,
    ) -> dict | None:
        """Find the most recent search_log with results.

        Returns dict with keys: id, query, result_ids, or None.
        Tries project-specific first, falls back to any project.
        """
        try:
            pool = await self._ensure_pool()
            async with pool.acquire() as conn:
                # Try project-specific first
                if project:
                    row = await conn.fetchrow(
                        """
                        SELECT id::text, query, result_ids
                        FROM search_logs
                        WHERE result_ids IS NOT NULL
                          AND result_ids::text NOT IN ('[]', 'null')
                          AND created_at > NOW() - make_interval(hours => $1)
                          AND project = $2
                        ORDER BY created_at DESC
                        LIMIT 1
                        """,
                        hours_window,
                        project,
                    )
                    if row:
                        return {
                            "id": row["id"],
                            "query": row["query"],
                            "result_ids": json.loads(row["result_ids"]) if isinstance(row["result_ids"], str) else row["result_ids"],
                        }

                # Fallback: any project
                row = await conn.fetchrow(
                    """
                    SELECT id::text, query, result_ids
                    FROM search_logs
                    WHERE result_ids IS NOT NULL
                      AND result_ids::text NOT IN ('[]', 'null')
                      AND created_at > NOW() - make_interval(hours => $1)
                    ORDER BY created_at DESC
                    LIMIT 1
                    """,
                    hours_window,
                )
            if row:
                return {
                    "id": row["id"],
                    "query": row["query"],
                    "result_ids": json.loads(row["result_ids"]) if isinstance(row["result_ids"], str) else row["result_ids"],
                }
            return None
        except Exception as e:
            logger.debug("get_recent_with_results failed: %s", e)
            return None

    async def mark_used_fuzzy(
        self,
        agent_response: str,
        *,
        project: str | None = None,
        hours_window: int = 24,
        threshold: float = 0.8,
    ) -> int:
        """Fuzzy match: check if agent response overlaps with search result keywords.

        For each unjudged search log, extracts keywords from the result experience's
        description/solution, then computes overlap ratio with agent_response.
        If overlap > threshold → was_used = True, judgment_source = 'fuzzy'.

        Also increments used_count on the corresponding experience.

        Returns the number of logs marked as used.
        """
        try:
            pool = await self._ensure_pool()
            response_lower = agent_response.lower()
            marked = 0

            async with pool.acquire() as conn:
                # Get unjudged search logs with results
                rows = await conn.fetch(
                    """
                    SELECT id, result_ids FROM search_logs
                    WHERE was_used IS NULL
                      AND result_ids IS NOT NULL
                      AND result_ids::text NOT IN ('[]', 'null')
                      AND created_at > NOW() - make_interval(hours => $1)
                      AND ($2::text IS NULL OR project = $2)
                    """,
                    hours_window,
                    project,
                )

                for row in rows:
                    result_ids = row["result_ids"]
                    if isinstance(result_ids, str):
                        result_ids = json.loads(result_ids)
                    if not result_ids:
                        continue

                    # For each result in this search log, fetch the experience text
                    # and check keyword overlap
                    fuzzy_hit = False
                    for r in result_ids:
                        rid = str(r.get("id", ""))
                        if not rid:
                            continue

                        try:
                            exp_uuid = uuid.UUID(rid)
                        except ValueError:
                            continue

                        exp_row = await conn.fetchrow(
                            """
                            SELECT description, solution FROM experiences WHERE id = $1
                            """,
                            exp_uuid,
                        )
                        if not exp_row:
                            continue

                        text = exp_row["solution"] or exp_row["description"] or ""
                        if not text:
                            continue

                        # Tokenize: split by spaces and punctuation, keep tokens with length > 2
                        keywords = [
                            tok.lower()
                            for tok in re.split(r"[^\w]+", text)
                            if len(tok) > 2
                        ]
                        if not keywords:
                            continue

                        matched = sum(1 for kw in keywords if kw in response_lower)
                        overlap_ratio = matched / len(keywords)

                        if overlap_ratio > threshold:
                            fuzzy_hit = True
                            # Mark search log as used with fuzzy source
                            now = datetime.now(timezone.utc)
                            await conn.execute(
                                """
                                UPDATE search_logs
                                SET was_used = TRUE, judgment_source = 'fuzzy', updated_at = $2
                                WHERE id = $1
                                """,
                                row["id"],
                                now,
                            )
                            # Increment used_count on the experience
                            try:
                                await conn.execute(
                                    """
                                    UPDATE experiences
                                    SET used_count = used_count + 1, updated_at = $2
                                    WHERE id = $1
                                    """,
                                    exp_uuid,
                                    now,
                                )
                            except Exception:
                                pass
                            marked += 1
                            break  # One fuzzy hit is enough for this search log

            return marked
        except Exception as e:
            logger.debug("mark_used_fuzzy failed: %s", e)
            return 0

    async def get_stats(self, days: int = 7) -> dict[str, Any]:
        """Get evaluation statistics for the last N days."""
        try:
            pool = await self._ensure_pool()
            async with pool.acquire() as conn:
                row = await conn.fetchrow(
                    """
                    SELECT
                        COUNT(*) AS total,
                        COUNT(*) FILTER (WHERE result_ids IS NOT NULL AND result_ids::text NOT IN ('[]', 'null')) AS hit,
                        COUNT(*) FILTER (WHERE was_used = TRUE) AS used,
                        COUNT(*) FILTER (WHERE was_used IS NULL AND result_ids IS NOT NULL AND result_ids::text NOT IN ('[]', 'null')) AS unjudged,
                        COUNT(*) FILTER (WHERE judgment_source = 'marker') AS used_marker,
                        COUNT(*) FILTER (WHERE judgment_source = 'fuzzy') AS used_fuzzy
                    FROM search_logs
                    WHERE created_at > NOW() - make_interval(days => $1)
                    """,
                    days,
                )
                total = row["total"] or 0
                hit = row["hit"] or 0
                used = row["used"] or 0
                unjudged = row["unjudged"] or 0
                used_marker = row["used_marker"] or 0
                used_fuzzy = row["used_fuzzy"] or 0
                use_rate = used / hit if hit > 0 else 0.0

                # Per-project breakdown
                project_rows = await conn.fetch(
                    """
                    SELECT
                        project,
                        COUNT(*) AS total,
                        COUNT(*) FILTER (WHERE was_used = TRUE) AS used
                    FROM search_logs
                    WHERE created_at > NOW() - make_interval(days => $1)
                      AND result_ids IS NOT NULL AND result_ids::text NOT IN ('[]', 'null')
                    GROUP BY project
                    ORDER BY total DESC
                    """,
                    days,
                )
                by_project = {r["project"]: {"total": r["total"], "used": r["used"]} for r in project_rows}

                # Top queries (most frequent)
                top_queries = await conn.fetch(
                    """
                    SELECT query, COUNT(*) as cnt
                    FROM search_logs
                    WHERE created_at > NOW() - make_interval(days => $1)
                    GROUP BY query
                    ORDER BY cnt DESC
                    LIMIT 10
                    """,
                    days,
                )

                return {
                    "days": days,
                    "total_searches": total,
                    "hit": hit,
                    "used": used,
                    "used_marker": used_marker,
                    "used_fuzzy": used_fuzzy,
                    "unjudged": unjudged,
                    "use_rate": round(use_rate, 3),
                    "by_project": by_project,
                    "top_queries": [{"query": r["query"][:60], "count": r["cnt"]} for r in top_queries],
                }
        except Exception as e:
            return {"error": str(e)}

    async def get_trends(
        self,
        days: int = 30,
        granularity: str = "day",
    ) -> list[dict[str, Any]]:
        """Get daily/hourly search trends for the last N days."""
        try:
            pool = await self._ensure_pool()
            async with pool.acquire() as conn:
                if granularity == "hour":
                    trunc = "hour"
                elif granularity == "week":
                    trunc = "week"
                else:
                    trunc = "day"

                rows = await conn.fetch(
                    f"""
                    SELECT
                        date_trunc($1, created_at) AS period,
                        COUNT(*) AS total,
                        COUNT(*) FILTER (WHERE result_ids IS NOT NULL AND result_ids::text NOT IN ('[]', 'null')) AS hit,
                        COUNT(*) FILTER (WHERE was_used = TRUE) AS used,
                        COUNT(*) FILTER (WHERE judgment_source = 'marker') AS used_marker,
                        COUNT(*) FILTER (WHERE judgment_source = 'fuzzy') AS used_fuzzy
                    FROM search_logs
                    WHERE created_at > NOW() - make_interval(days => $2)
                    GROUP BY period
                    ORDER BY period
                    """,
                    trunc,
                    days,
                )

                result = []
                for r in rows:
                    total = r["total"] or 0
                    hit = r["hit"] or 0
                    used = r["used"] or 0
                    use_rate = used / hit if hit > 0 else 0.0
                    result.append({
                        "period": r["period"].isoformat() if r["period"] else None,
                        "total": total,
                        "hit": hit,
                        "used": used,
                        "used_marker": r["used_marker"] or 0,
                        "used_fuzzy": r["used_fuzzy"] or 0,
                        "use_rate": round(use_rate, 3),
                    })
                return result
        except Exception as e:
            return [{"error": str(e)}]

    async def get_miss_queries(
        self,
        days: int = 30,
        limit: int = 10,
    ) -> list[dict[str, Any]]:
        """Get queries that returned no results (miss queries)."""
        try:
            pool = await self._ensure_pool()
            async with pool.acquire() as conn:
                rows = await conn.fetch(
                    """
                    SELECT query, COUNT(*) as cnt, MAX(created_at) as last_seen
                    FROM search_logs
                    WHERE created_at > NOW() - make_interval(days => $1)
                      AND (result_ids IS NULL OR result_ids::text IN ('[]', 'null'))
                    GROUP BY query
                    ORDER BY cnt DESC
                    LIMIT $2
                    """,
                    days,
                    limit,
                )
                return [
                    {
                        "query": r["query"][:80],
                        "count": r["cnt"],
                        "last_seen": r["last_seen"].isoformat() if r["last_seen"] else None,
                    }
                    for r in rows
                ]
        except Exception as e:
            return [{"error": str(e)}]

    async def get_low_use_queries(
        self,
        days: int = 30,
        limit: int = 10,
    ) -> list[dict[str, Any]]:
        """Get queries that returned results but were rarely used."""
        try:
            pool = await self._ensure_pool()
            async with pool.acquire() as conn:
                rows = await conn.fetch(
                    """
                    SELECT
                        query,
                        COUNT(*) AS total,
                        COUNT(*) FILTER (WHERE was_used = TRUE) AS used
                    FROM search_logs
                    WHERE created_at > NOW() - make_interval(days => $1)
                      AND result_ids IS NOT NULL AND result_ids::text NOT IN ('[]', 'null')
                    GROUP BY query
                    HAVING COUNT(*) FILTER (WHERE was_used = TRUE) = 0
                    ORDER BY total DESC
                    LIMIT $2
                    """,
                    days,
                    limit,
                )
                return [
                    {
                        "query": r["query"][:80],
                        "count": r["total"],
                        "used": r["used"],
                        "use_rate": 0.0,
                    }
                    for r in rows
                ]
        except Exception as e:
            return [{"error": str(e)}]

    async def close(self) -> None:
        if self._pool and not self._pool._closed:
            await self._pool.close()

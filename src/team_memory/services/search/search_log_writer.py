"""SearchLog writer — writes search logs and usage tracking to TM database.

This module provides lightweight functions to:
1. Log each retrieval with query + result_ids
2. Fuzzy match: check if agent response overlaps with search result keywords
3. Faithfulness: buffer responses for LLM-based evaluation
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

logger = logging.getLogger(__name__)

# Exclude noise from quality/usage stats. Two layers:
#   1. is_noise flag set by the entry gate (rule/llm) on new rows.
#   2. Legacy fallback for rows logged before the flag existed: short query
#      (<12 chars) with no 4+ letter identifier run.
# A row counts as a real search only when it passes BOTH.
_NOISE_EXCLUSION = "is_noise IS NOT TRUE AND NOT (char_length(query) < 12 AND query !~ '[A-Za-z]{4,}')"

_ADOPTION_STOPWORDS: frozenset[str] = frozenset({
    "方案", "优化", "问题", "修复", "实现", "分析", "总结", "设计", "功能", "支持",
    "经验", "记录", "指南", "规范", "开发", "测试", "配置", "使用", "模块", "系统",
    "the", "and", "for", "with", "from", "that", "this", "fix", "add", "update",
    "completed", "exec-plan", "exec", "plan", "team_memory", "team_doc", "general",
    "true", "false", "null", "none",
})

_CJK_PARTICLE_SPLIT_RE = re.compile(r"[的与及在中和或对为了于之跟被把从向以并且]+")
_TECH_FILE_RE = re.compile(
    r"(?<![a-zA-Z0-9_.-])[a-zA-Z0-9_.-]+\.(?:py|ts|js|md|yaml|toml|json|sh|sql)(?![a-zA-Z0-9_])",
    re.IGNORECASE,
)
_TECH_CLI_RE = re.compile(r"(?<![a-zA-Z0-9_-])tm-cli\s+[a-zA-Z0-9_-]+", re.IGNORECASE)
_TECH_FLAG_RE = re.compile(r"(?<![a-zA-Z0-9_])--[a-zA-Z0-9_-]{3,}(?![a-zA-Z0-9_-])", re.IGNORECASE)
_TECH_SNAKE_RE = re.compile(r"(?<![a-zA-Z0-9_])[a-zA-Z_][a-zA-Z0-9_]*_[a-zA-Z0-9_]+(?![a-zA-Z0-9_])")
_TECH_CAMEL_RE = re.compile(r"(?<![a-zA-Z0-9_])[a-zA-Z]+[a-z0-9]+[A-Z][a-zA-Z0-9]*(?![a-zA-Z0-9_])")
_ASCII_TOKEN_RE = re.compile(r"[a-zA-Z0-9_-]{3,}")
_CJK_RUN_RE = re.compile(r"[\u4e00-\u9fff]{2,}")
_CJK_STOPWORDS_SORTED: tuple[str, ...] = tuple(
    sorted((w for w in _ADOPTION_STOPWORDS if "\u4e00" <= w[0] <= "\u9fff" and w != "系统"), key=len, reverse=True)
)


def _extract_high_specificity_tokens(title: str) -> list[str]:
    """Extract high-specificity technical artifacts (filenames, CLI flags, code symbols) from title."""
    tokens: list[str] = []
    seen: set[str] = set()

    for regex in (_TECH_FILE_RE, _TECH_CLI_RE, _TECH_FLAG_RE):
        for m in regex.finditer(title):
            tok = m.group(0).lower().strip()
            if tok and tok not in _ADOPTION_STOPWORDS and tok not in seen:
                seen.add(tok)
                tokens.append(tok)

    for regex in (_TECH_SNAKE_RE, _TECH_CAMEL_RE):
        for m in regex.finditer(title):
            tok = m.group(0).lower().strip("_")
            if len(tok) >= 6 and tok not in _ADOPTION_STOPWORDS and tok not in seen:
                seen.add(tok)
                tokens.append(tok)

    return tokens


def _strip_cjk_stopwords(chunk: str) -> list[str]:
    """Strip or split common generic CJK stopwords while preserving domain terms like '鸿蒙系统'."""
    if not chunk or len(chunk) < 2 or chunk in _ADOPTION_STOPWORDS:
        return []

    # Strip leading/trailing generic stopwords iteratively
    changed = True
    cur = chunk
    while changed and len(cur) > 2:
        changed = False
        for sw in _CJK_STOPWORDS_SORTED:
            if len(cur) > len(sw) and cur.startswith(sw):
                cur = cur[len(sw):]
                changed = True
                break
            if len(cur) > len(sw) and cur.endswith(sw):
                cur = cur[:-len(sw)]
                changed = True
                break
        if len(cur) > 4 and cur.endswith("系统"):
            cur = cur[:-2]
            changed = True

    if len(cur) < 2 or cur in _ADOPTION_STOPWORDS:
        return []

    if len(cur) <= 4:
        return [cur]

    # For longer chunks (> 4 chars), try splitting on internal stopwords first
    for sw in _CJK_STOPWORDS_SORTED:
        if sw in cur:
            parts = [p for p in cur.split(sw) if len(p) >= 2 and p not in _ADOPTION_STOPWORDS]
            if parts:
                out: list[str] = []
                for p in parts:
                    out.extend(_strip_cjk_stopwords(p))
                return out

    # Otherwise split into non-overlapping 2-4 char sub-grams so compound phrases match cleanly
    if len(cur) == 5:
        sub_parts = [cur[:2], cur[-2:]]
    elif len(cur) == 6:
        sub_parts = [cur[:4], cur[4:]]
    else:
        sub_parts = [cur[i : i + 4] for i in range(0, len(cur), 4) if len(cur[i : i + 4]) >= 2]
    return [p for p in sub_parts if len(p) >= 2 and p not in _ADOPTION_STOPWORDS]


def _extract_title_tokens(title: str) -> list[str]:
    """Extract deduplicated meaningful ASCII and CJK tokens from an experience title."""
    tokens: list[str] = []
    seen: set[str] = set()

    for m in _ASCII_TOKEN_RE.finditer(title):
        tok = m.group(0).lower()
        if tok not in _ADOPTION_STOPWORDS and not tok.isdigit() and tok not in seen:
            seen.add(tok)
            tokens.append(tok)

    for m in _CJK_RUN_RE.finditer(title):
        run = m.group(0)
        for seg in _CJK_PARTICLE_SPLIT_RE.split(run):
            for cleaned in _strip_cjk_stopwords(seg):
                if 2 <= len(cleaned) <= 6 and cleaned not in _ADOPTION_STOPWORDS and cleaned not in seen:
                    seen.add(cleaned)
                    tokens.append(cleaned)

    return tokens


def evaluate_rule_based_adoption(
    results: list[dict[str, Any]],
    response_text: str,
    tool_observations: list[dict[str, Any]] | None = None,
) -> tuple[bool, list[str]]:
    """Evaluate whether recalled experiences were adopted in the response or tool trajectory.

    Returns ``(was_used, matched_exp_ids)``.
    """
    parts: list[str] = [response_text or ""]
    for obs in tool_observations or []:
        if not isinstance(obs, dict):
            continue
        for key in ("tool_name", "summary", "command_summary", "command", "commands", "target_files", "target_file"):
            val = obs.get(key)
            if isinstance(val, str) and val.strip():
                parts.append(val)
            elif isinstance(val, (list, tuple, set)):
                parts.extend(str(item) for item in val if item)

    haystack = " ".join(parts)
    haystack_lower = haystack.lower()
    if len(haystack_lower.strip()) < 10 or not results:
        return (False, [])

    matched_ids: list[str] = []
    seen_ids: set[str] = set()

    def _mark_matched(exp_id_val: str, title_val: str) -> None:
        ident = exp_id_val or title_val or "matched"
        if ident not in seen_ids:
            seen_ids.add(ident)
            matched_ids.append(ident)

    for r in results:
        if not isinstance(r, dict):
            continue
        exp_id = str(r.get("id", "") or "").strip()
        title = str(r.get("title", "") or "").strip()
        tags = [str(t).strip() for t in (r.get("tags") or []) if t and str(t).strip()]

        # Rule 1: Explicit ID / URI reference
        if exp_id:
            exp_id_lower = exp_id.lower()
            if exp_id_lower in haystack_lower or (len(exp_id_lower) >= 8 and exp_id_lower[:8] in haystack_lower):
                _mark_matched(exp_id, title)
                continue

        # Rule 2: High-specificity technical token match from title
        tech_tokens = _extract_high_specificity_tokens(title)
        if any(tok in haystack_lower for tok in tech_tokens):
            _mark_matched(exp_id, title)
            continue

        # Rule 3: Multi-keyword conceptual match from title & tags
        title_tokens = _extract_title_tokens(title)
        matched_title_tokens = sum(1 for tok in title_tokens if tok in haystack_lower)

        valid_tag_tokens: list[str] = []
        seen_tags: set[str] = set()
        for t in tags:
            t_low = t.lower()
            is_cjk_tag = any("\u4e00" <= ch <= "\u9fff" for ch in t_low)
            min_len = 2 if is_cjk_tag else 3
            if len(t_low) >= min_len and t_low not in _ADOPTION_STOPWORDS and t_low not in seen_tags:
                seen_tags.add(t_low)
                valid_tag_tokens.append(t_low)

        matched_tag_tokens = sum(1 for tok in valid_tag_tokens if tok in haystack_lower)

        if matched_title_tokens >= 2 or (
            matched_title_tokens >= 1 and matched_tag_tokens >= 1 and len(title_tokens) <= 2
        ):
            _mark_matched(exp_id, title)

    return (bool(matched_ids), matched_ids)


class SearchLogWriter:
    """Lightweight async writer for search_logs table."""

    def __init__(self, dsn: str | None = None) -> None:
        if dsn is None:
            import os

            dsn = os.environ.get("TEAM_MEMORY_DB_URL", "")
            if not dsn:
                raise RuntimeError("SearchLogWriter requires a DSN. Set TEAM_MEMORY_DB_URL or pass dsn= explicitly.")

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
        intent_type: str | None = None,
        result_ids: list[dict[str, Any]] | None = None,
        is_noise: bool | None = None,
        noise_reason: str | None = None,
        stage_metrics: dict[str, Any] | None = None,
        stage_counts: dict[str, Any] | None = None,
        returned_tokens: int | None = None,
    ) -> str | None:
        """Write a search log entry. Returns the log ID or None on failure.

        ``is_noise`` marks utterances with no real retrieval intent so that
        quality/usage stats can exclude them; ``noise_reason`` is 'rule' or 'llm'.
        """
        try:
            pool = await self._ensure_pool()
            log_id = uuid.uuid4()
            now = datetime.now(timezone.utc)
            eff_intent = (intent_type or "unknown")[:50]
            async with pool.acquire() as conn:
                if (
                    stage_metrics is not None
                    or stage_counts is not None
                    or returned_tokens is not None
                    or intent_type is not None
                ):
                    try:
                        await conn.execute(
                            """
                            INSERT INTO search_logs
                                (id, query, project, source, intent_type, result_ids, is_noise, noise_reason,
                                 stage_metrics, stage_counts, returned_tokens, created_at, updated_at)
                            VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9::jsonb, $10::jsonb, $11, $12, $12)
                            """,
                            log_id,
                            query[:500],
                            project,
                            source,
                            eff_intent,
                            json.dumps(result_ids) if result_ids else None,
                            is_noise,
                            noise_reason,
                            json.dumps(stage_metrics) if stage_metrics is not None else None,
                            json.dumps(stage_counts) if stage_counts is not None else None,
                            returned_tokens,
                            now,
                        )
                        return str(log_id)
                    except Exception:
                        pass
                await conn.execute(
                    """
                    INSERT INTO search_logs
                        (id, query, project, source, result_ids, is_noise, noise_reason, created_at, updated_at)
                    VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $8)
                    """,
                    log_id,
                    query[:500],
                    project,
                    source,
                    json.dumps(result_ids) if result_ids else None,
                    is_noise,
                    noise_reason,
                    now,
                )
            return str(log_id)
        except Exception as e:
            logger.debug("log_search failed: %s", e)
            return None

    async def record_rule_adoption(
        self,
        search_log_id: int | str | uuid.UUID,
        matched_exp_ids: list[str] | None = None,
    ) -> bool:
        """Record rule-based adoption on search_logs and increment experiences.used_count."""
        if not search_log_id:
            return False
        try:
            log_id_param: Any = search_log_id
            if isinstance(search_log_id, str):
                try:
                    log_id_param = uuid.UUID(search_log_id.strip())
                except ValueError:
                    log_id_param = search_log_id

            pool = await self._ensure_pool()
            async with pool.acquire() as conn:
                await conn.execute(
                    """
                    UPDATE search_logs
                    SET was_used = TRUE,
                        judgment_source = 'rule_match'
                    WHERE id = $1 AND (was_used IS NULL OR was_used = FALSE)
                    """,
                    log_id_param,
                )

                if matched_exp_ids:
                    seen_uuids: set[uuid.UUID] = set()
                    for eid in matched_exp_ids:
                        if not eid:
                            continue
                        try:
                            parsed_uid = eid if isinstance(eid, uuid.UUID) else uuid.UUID(str(eid).strip())
                        except (ValueError, TypeError, AttributeError):
                            continue
                        if parsed_uid in seen_uuids:
                            continue
                        seen_uuids.add(parsed_uid)
                        await conn.execute(
                            """
                            UPDATE experiences
                            SET used_count = COALESCE(used_count, 0) + 1
                            WHERE id = $1
                            """,
                            parsed_uid,
                        )
            return True
        except Exception as e:
            logger.debug("record_rule_adoption failed: %s", e)
            return False

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
                            "result_ids": (
                                json.loads(row["result_ids"])
                                if isinstance(row["result_ids"], str)
                                else row["result_ids"]
                            ),
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
                    "result_ids": (
                        json.loads(row["result_ids"]) if isinstance(row["result_ids"], str) else row["result_ids"]
                    ),
                }
            return None
        except Exception as e:
            logger.debug("get_recent_with_results failed: %s", e)
            return None

    async def find_unjudged_in_window(
        self,
        *,
        t_start: datetime,
        t_end: datetime,
        project: str | None = None,
        source: str = "mcp",
    ) -> list[dict]:
        """Find unjudged search_logs within a time window.

        Used as fallback when recent_searches is lost (daemon restart).
        Returns list of dicts with: id, query, result_ids.
        """
        try:
            pool = await self._ensure_pool()
            async with pool.acquire() as conn:
                if project:
                    rows = await conn.fetch(
                        """
                        SELECT id::text, query, result_ids
                        FROM search_logs
                        WHERE source = $1
                          AND was_used IS NULL
                          AND result_ids IS NOT NULL
                          AND result_ids::text NOT IN ('[]', 'null')
                          AND created_at BETWEEN $2 AND $3
                          AND project = $4
                        ORDER BY created_at DESC
                        LIMIT 10
                        """,
                        source,
                        t_start,
                        t_end,
                        project,
                    )
                else:
                    rows = await conn.fetch(
                        """
                        SELECT id::text, query, result_ids
                        FROM search_logs
                        WHERE source = $1
                          AND was_used IS NULL
                          AND result_ids IS NOT NULL
                          AND result_ids::text NOT IN ('[]', 'null')
                          AND created_at BETWEEN $2 AND $3
                        ORDER BY created_at DESC
                        LIMIT 10
                        """,
                        source,
                        t_start,
                        t_end,
                    )
            return [
                {
                    "id": r["id"],
                    "query": r["query"],
                    "result_ids": (
                        json.loads(r["result_ids"]) if isinstance(r["result_ids"], str) else r["result_ids"]
                    ),
                }
                for r in rows
            ]
        except Exception as e:
            logger.debug("find_unjudged_in_window failed: %s", e)
            return []

    async def get_stats(self, days: int = 7) -> dict[str, Any]:
        """Get evaluation statistics for the last N days."""
        try:
            pool = await self._ensure_pool()
            async with pool.acquire() as conn:
                row = await conn.fetchrow(
                    f"""
                    SELECT
                        COUNT(*) AS total,
                        COUNT(*) FILTER (
                            WHERE result_ids IS NOT NULL
                            AND result_ids::text NOT IN ('[]', 'null')
                        ) AS hit,
                        COUNT(*) FILTER (
                            WHERE was_used = TRUE
                            AND result_ids IS NOT NULL AND result_ids::text NOT IN ('[]', 'null')
                        ) AS used,
                        COUNT(*) FILTER (
                            WHERE was_used IS NOT NULL
                            AND result_ids IS NOT NULL
                            AND result_ids::text NOT IN ('[]', 'null')
                        ) AS judged,
                        COUNT(*) FILTER (
                            WHERE was_used IS NULL
                            AND result_ids IS NOT NULL
                            AND result_ids::text NOT IN ('[]', 'null')
                            AND (judgment_source IS NULL
                                 OR judgment_source != 'legacy_unjudged')
                        ) AS unjudged,
                        COUNT(*) FILTER (WHERE judgment_source = 'legacy_unjudged') AS legacy_unjudged,
                        COUNT(*) FILTER (WHERE judgment_source = 'faithfulness') AS used_faithfulness
                    FROM search_logs
                    WHERE created_at > NOW() - make_interval(days => $1)
                      AND {_NOISE_EXCLUSION}
                    """,
                    days,
                )
                total = row["total"] or 0
                hit = row["hit"] or 0
                used = row["used"] or 0
                judged = row["judged"] or 0
                unjudged = row["unjudged"] or 0
                legacy_unjudged = row["legacy_unjudged"] or 0
                used_faithfulness = row["used_faithfulness"] or 0
                # Use rate = adoption among JUDGED recalls (not all hits), so the
                # number reflects real quality and isn't dragged down by recalls
                # still awaiting evaluation.
                use_rate = used / judged if judged > 0 else 0.0

                # Per-project breakdown
                project_rows = await conn.fetch(
                    f"""
                    SELECT
                        project,
                        COUNT(*) AS total,
                        COUNT(*) FILTER (WHERE was_used IS NOT NULL) AS judged,
                        COUNT(*) FILTER (WHERE was_used = TRUE) AS used
                    FROM search_logs
                    WHERE created_at > NOW() - make_interval(days => $1)
                      AND result_ids IS NOT NULL AND result_ids::text NOT IN ('[]', 'null')
                      AND {_NOISE_EXCLUSION}
                    GROUP BY project
                    ORDER BY total DESC
                    """,
                    days,
                )
                by_project = {
                    r["project"]: {
                        "total": r["total"],
                        "judged": r["judged"] if "judged" in r and r["judged"] is not None else r["total"],
                        "used": r["used"],
                    }
                    for r in project_rows
                }

                # Top queries (most frequent)
                top_queries = await conn.fetch(
                    f"""
                    SELECT query, COUNT(*) as cnt
                    FROM search_logs
                    WHERE created_at > NOW() - make_interval(days => $1)
                      AND {_NOISE_EXCLUSION}
                    GROUP BY query
                    ORDER BY cnt DESC
                    LIMIT 10
                    """,
                    days,
                )

                # WoW comparison
                prev_row = await conn.fetchrow(
                    f"""
                    SELECT
                        COUNT(*) FILTER (
                            WHERE result_ids IS NOT NULL
                            AND result_ids::text NOT IN ('[]', 'null')
                        ) AS hit,
                        COUNT(*) FILTER (
                            WHERE was_used IS NOT NULL
                            AND result_ids IS NOT NULL
                            AND result_ids::text NOT IN ('[]', 'null')
                        ) AS judged,
                        COUNT(*) FILTER (WHERE was_used = TRUE) AS used
                    FROM search_logs
                    WHERE created_at > NOW() - make_interval(days => $1 * 2)
                      AND created_at <= NOW() - make_interval(days => $1)
                      AND {_NOISE_EXCLUSION}
                    """,
                    days,
                )
                prev_hit = prev_row["hit"] or 0 if prev_row else 0
                prev_judged = prev_row["judged"] or 0 if prev_row else 0
                prev_used = prev_row["used"] or 0 if prev_row else 0
                prev_use_rate = prev_used / prev_judged if prev_judged > 0 else 0.0

                return {
                    "days": days,
                    "total_searches": total,
                    "hit": hit,
                    "used": used,
                    "judged": judged,
                    "used_faithfulness": used_faithfulness,
                    "unjudged": unjudged,
                    "legacy_unjudged": legacy_unjudged,
                    "use_rate": round(use_rate, 3),
                    "prev_hit": prev_hit,
                    "prev_judged": prev_judged,
                    "prev_used": prev_used,
                    "prev_use_rate": round(prev_use_rate, 3),
                    "use_rate_delta": round(use_rate - prev_use_rate, 3),
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
                        COUNT(*) FILTER (
                            WHERE result_ids IS NOT NULL
                            AND result_ids::text NOT IN ('[]', 'null')
                        ) AS hit,
                        COUNT(*) FILTER (
                            WHERE was_used = TRUE
                            AND result_ids IS NOT NULL AND result_ids::text NOT IN ('[]', 'null')
                        ) AS used,
                        COUNT(*) FILTER (
                            WHERE was_used IS NOT NULL
                            AND result_ids IS NOT NULL
                            AND result_ids::text NOT IN ('[]', 'null')
                        ) AS judged,
                        COUNT(*) FILTER (WHERE judgment_source = 'faithfulness') AS used_faithfulness,
                        AVG(relevance_score) FILTER (WHERE relevance_score IS NOT NULL) AS avg_relevance,
                        AVG(helpfulness_score) FILTER (WHERE helpfulness_score IS NOT NULL) AS avg_helpfulness
                    FROM search_logs
                    WHERE created_at > NOW() - make_interval(days => $2)
                      AND {_NOISE_EXCLUSION}
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
                    judged = r["judged"] or 0
                    use_rate = used / judged if judged > 0 else 0.0
                    result.append(
                        {
                            "period": r["period"].isoformat() if r["period"] else None,
                            "total": total,
                            "hit": hit,
                            "used": used,
                            "judged": judged,
                            "used_faithfulness": r["used_faithfulness"] or 0,
                            "use_rate": round(use_rate, 3),
                            "avg_relevance": (
                                round(float(r["avg_relevance"]), 3) if r["avg_relevance"] is not None else None
                            ),
                            "avg_helpfulness": (
                                round(float(r["avg_helpfulness"]), 3) if r["avg_helpfulness"] is not None else None
                            ),
                        }
                    )
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
                    f"""
                    SELECT query, COUNT(*) as cnt, MAX(created_at) as last_seen
                    FROM search_logs
                    WHERE created_at > NOW() - make_interval(days => $1)
                      AND (result_ids IS NULL OR result_ids::text IN ('[]', 'null'))
                      AND {_NOISE_EXCLUSION}
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
                        "last_seen": (r["last_seen"].isoformat() if r["last_seen"] else None),
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
                    f"""
                    SELECT
                        query,
                        COUNT(*) AS total,
                        COUNT(*) FILTER (WHERE was_used = TRUE) AS used
                    FROM search_logs
                    WHERE created_at > NOW() - make_interval(days => $1)
                      AND result_ids IS NOT NULL AND result_ids::text NOT IN ('[]', 'null')
                      AND {_NOISE_EXCLUSION}
                    GROUP BY query
                    HAVING COUNT(*) FILTER (WHERE was_used = TRUE) = 0
                       AND COUNT(*) FILTER (WHERE was_used IS NOT NULL) > 0
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

    async def get_weekly_comparison(self, weeks: int = 4) -> list[dict[str, Any]]:
        """Get weekly aggregated stats for week-over-week comparison.

        Returns list of dicts with: week_start, total, hit, used, use_rate,
        hit_rate, avg_relevance, avg_helpfulness.
        """
        try:
            pool = await self._ensure_pool()
            async with pool.acquire() as conn:
                rows = await conn.fetch(
                    f"""
                    SELECT
                        date_trunc('week', created_at) AS week_start,
                        COUNT(*) AS total,
                        COUNT(*) FILTER (
                            WHERE result_ids IS NOT NULL AND result_ids::text NOT IN ('[]', 'null')
                        ) AS hit,
                        COUNT(*) FILTER (
                            WHERE was_used = TRUE
                            AND result_ids IS NOT NULL AND result_ids::text NOT IN ('[]', 'null')
                        ) AS used,
                        COUNT(*) FILTER (
                            WHERE was_used IS NOT NULL
                            AND result_ids IS NOT NULL AND result_ids::text NOT IN ('[]', 'null')
                        ) AS judged,
                        AVG(relevance_score) FILTER (WHERE relevance_score IS NOT NULL) AS avg_relevance,
                        AVG(helpfulness_score) FILTER (WHERE helpfulness_score IS NOT NULL) AS avg_helpfulness
                    FROM search_logs
                    WHERE created_at > NOW() - make_interval(weeks => $1)
                      AND {_NOISE_EXCLUSION}
                    GROUP BY week_start
                    ORDER BY week_start
                    """,
                    weeks,
                )
                result = []
                for r in rows:
                    total = r["total"] or 0
                    hit = r["hit"] or 0
                    used = r["used"] or 0
                    judged = r["judged"] or 0
                    result.append(
                        {
                            "week_start": r["week_start"].strftime("%m-%d"),
                            "total": total,
                            "hit": hit,
                            "used": used,
                            "judged": judged,
                            "hit_rate": round(hit / total, 3) if total > 0 else 0.0,
                            "use_rate": round(used / judged, 3) if judged > 0 else 0.0,
                            "avg_relevance": (
                                round(float(r["avg_relevance"]), 3) if r["avg_relevance"] is not None else None
                            ),
                            "avg_helpfulness": (
                                round(float(r["avg_helpfulness"]), 3) if r["avg_helpfulness"] is not None else None
                            ),
                        }
                    )
                return result
        except Exception as e:
            return [{"error": str(e)}]

    async def get_top_used_queries(self, days: int = 7, limit: int = 10) -> list[dict[str, Any]]:
        """Get queries with highest use rate (hit and used frequently)."""
        try:
            pool = await self._ensure_pool()
            async with pool.acquire() as conn:
                rows = await conn.fetch(
                    f"""
                    SELECT
                        query,
                        COUNT(*) AS total,
                        COUNT(*) FILTER (WHERE was_used = TRUE) AS used,
                        AVG(relevance_score) FILTER (WHERE relevance_score IS NOT NULL) AS avg_rel
                    FROM search_logs
                    WHERE created_at > NOW() - make_interval(days => $1)
                      AND result_ids IS NOT NULL AND result_ids::text NOT IN ('[]', 'null')
                      AND {_NOISE_EXCLUSION}
                    GROUP BY query
                    HAVING COUNT(*) >= 2 AND COUNT(*) FILTER (WHERE was_used = TRUE) > 0
                    ORDER BY COUNT(*) FILTER (WHERE was_used = TRUE) DESC, total DESC
                    LIMIT $2
                    """,
                    days,
                    limit,
                )
                return [
                    {
                        "query": r["query"][:80],
                        "total": r["total"],
                        "used": r["used"],
                        "use_rate": round(r["used"] / r["total"], 3) if r["total"] > 0 else 0.0,
                        "avg_relevance": round(float(r["avg_rel"]), 3) if r["avg_rel"] is not None else None,
                    }
                    for r in rows
                ]
        except Exception as e:
            return [{"error": str(e)}]

    async def get_arena_stats(self, days: int = 7) -> dict[str, Any]:
        """Get arena evaluation statistics for the last N days.

        Returns avg relevance/helpfulness scores and score distribution
        from search_logs entries judged by the arena system.
        """
        try:
            pool = await self._ensure_pool()
            async with pool.acquire() as conn:
                row = await conn.fetchrow(
                    f"""
                    SELECT
                        avg(relevance_score) AS avg_relevance,
                        avg(helpfulness_score) AS avg_helpfulness,
                        COUNT(*) FILTER (WHERE helpfulness_score >= 0.7) AS high,
                        COUNT(*) FILTER (WHERE helpfulness_score >= 0.3 AND helpfulness_score < 0.7) AS medium,
                        COUNT(*) FILTER (WHERE helpfulness_score < 0.3) AS low
                    FROM search_logs
                    WHERE created_at > NOW() - make_interval(days => $1)
                      AND (relevance_score IS NOT NULL OR helpfulness_score IS NOT NULL)
                      AND {_NOISE_EXCLUSION}
                    """,
                    days,
                )
                return {
                    "avg_relevance": (
                        round(float(row["avg_relevance"]), 3) if row["avg_relevance"] is not None else None
                    ),
                    "avg_helpfulness": (
                        round(float(row["avg_helpfulness"]), 3) if row["avg_helpfulness"] is not None else None
                    ),
                    "score_distribution": {
                        "high": row["high"] or 0,
                        "medium": row["medium"] or 0,
                        "low": row["low"] or 0,
                    },
                }
        except Exception as e:
            return {"error": str(e)}

    async def get_telemetry_summary(self, days: int = 7) -> dict[str, Any]:
        """Get stage telemetry summary for the last N days."""
        try:
            pool = await self._ensure_pool()
            async with pool.acquire() as conn:
                row = await conn.fetchrow(
                    f"""
                    SELECT
                        AVG(returned_tokens) FILTER (WHERE returned_tokens IS NOT NULL) AS avg_returned_tokens,
                        AVG(
                            COALESCE(
                                (stage_metrics->>'total_ms')::float,
                                (
                                    COALESCE((stage_metrics->>'embedding_ms')::float, 0)
                                    + COALESCE((stage_metrics->>'retrieve_fuse_ms')::float, 0)
                                )
                            )
                        ) FILTER (
                            WHERE stage_metrics IS NOT NULL
                        ) AS avg_total_ms,
                        COUNT(*) FILTER (
                            WHERE stage_counts IS NOT NULL
                            AND COALESCE((stage_counts->>'graph_unique_added')::int, 0) > 0
                        ) AS graph_assisted_searches
                    FROM search_logs
                    WHERE created_at > NOW() - make_interval(days => $1)
                      AND {_NOISE_EXCLUSION}
                    """,
                    days,
                )
                return {
                    "days": days,
                    "avg_returned_tokens": (
                        round(float(row["avg_returned_tokens"]), 1)
                        if row and row["avg_returned_tokens"] is not None
                        else None
                    ),
                    "avg_total_ms": (
                        round(float(row["avg_total_ms"]), 2)
                        if row and row["avg_total_ms"] is not None
                        else None
                    ),
                    "graph_assisted_searches": (row["graph_assisted_searches"] or 0) if row else 0,
                }
        except Exception as e:
            return {"error": str(e)}

    async def close(self) -> None:
        if self._pool and not self._pool._closed:
            await self._pool.close()

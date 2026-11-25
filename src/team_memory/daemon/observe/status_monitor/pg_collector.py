"""PostgreSQL knowledge base, graph scale, and recent recalls collector."""

from __future__ import annotations

import asyncio
from typing import Any

from team_memory.daemon.observe.status_monitor.collectors import (
    _clean_query_display,
    _lookup_local_wiki_entries,
    _parse_json_field,
    _parse_list_field,
)
from team_memory.daemon.observe.tm_stats import (
    _bitemporal_health,
    _embedding_coverage,
    _faithfulness_avg,
    _resolve_dsn,
    format_telemetry_panel_data,
)


async def _fetch_recent_recalls_from_conn(conn: Any, limit: int = 12) -> list[dict[str, Any]]:
    """Fetch recent recall rows from search_logs, merging twin mcp+daemon rows."""
    fetch_n = max(30, limit * 3)
    rows = await conn.fetch(
        """
        SELECT id::text AS id,
               query,
               COALESCE(intent_type, 'general') AS intent_type,
               COALESCE(project, 'default') AS project,
               COALESCE(source, 'mcp') AS source,
               result_ids,
               was_used,
               judgment_source,
               faithfulness_score,
               relevance_score,
               helpfulness_score,
               COALESCE(is_noise, FALSE) AS is_noise,
               noise_reason,
               stage_metrics,
               stage_counts,
               returned_tokens,
               created_at
        FROM search_logs
        ORDER BY created_at DESC
        LIMIT $1
        """,
        fetch_n,
    )

    merged: list[dict[str, Any]] = []
    for r in rows:
        q_raw = str(r["query"] or "")
        q_clean_full = _clean_query_display(q_raw, 240).removeprefix("[Ctx] ").strip()
        q_norm = " ".join(q_clean_full.split())
        created_dt = r["created_at"]
        created_ts = created_dt.timestamp() if hasattr(created_dt, "timestamp") else 0.0
        created_str = (
            created_dt.astimezone().strftime("%Y-%m-%d %H:%M:%S")
            if hasattr(created_dt, "astimezone")
            else str(created_dt or "")[:19]
        )
        r_ids = _parse_list_field(r["result_ids"])
        s_metrics = _parse_json_field(r["stage_metrics"])
        s_counts = _parse_json_field(r["stage_counts"])

        # Check if this is a twin log (mcp + daemon logged within 15s for identical query or results)
        matched_prev: dict[str, Any] | None = None
        for prev_candidate in merged:
            if abs(prev_candidate["_ts"] - created_ts) <= 15.0 and prev_candidate["project"] == str(
                r["project"] or "default"
            ):
                same_q = (
                    prev_candidate["_q_norm"] == q_norm
                    or (len(q_norm) >= 6 and q_norm in prev_candidate["_q_norm"])
                    or (len(prev_candidate["_q_norm"]) >= 6 and prev_candidate["_q_norm"] in q_norm)
                )
                same_res = bool(r_ids and prev_candidate["result_ids"] == r_ids)
                if same_q or same_res:
                    matched_prev = prev_candidate
                    break

        if matched_prev is not None:
            prev = matched_prev
            # Merge telemetry & adoption fields into the existing entry
            if not prev["stage_metrics"] and s_metrics:
                prev["stage_metrics"] = s_metrics
            if not prev["stage_counts"] and s_counts:
                prev["stage_counts"] = s_counts
            if prev["returned_tokens"] is None and r["returned_tokens"] is not None:
                prev["returned_tokens"] = int(r["returned_tokens"])
            if not prev["result_ids"] and r_ids:
                prev["result_ids"] = r_ids
            if r["was_used"]:
                prev["was_used"] = True
            if not prev["judgment_source"] and r["judgment_source"]:
                prev["judgment_source"] = str(r["judgment_source"])
            if prev["faithfulness_score"] is None and r["faithfulness_score"] is not None:
                prev["faithfulness_score"] = round(float(r["faithfulness_score"]), 2)
            if r["source"] not in prev["_sources"]:
                prev["_sources"].append(str(r["source"]))
                prev["source"] = "+".join(prev["_sources"])
            # Prefer the row ID that actually has stage_counts or result_ids
            if s_counts and not _parse_json_field(prev.get("_raw_has_counts")):
                prev["id"] = str(r["id"])
                prev["short_id"] = str(r["id"])[:8]
            continue

        merged.append(
            {
                "id": str(r["id"]),
                "short_id": str(r["id"])[:8],
                "query": q_raw,
                "query_display": _clean_query_display(q_raw, 48),
                "intent_type": str(r["intent_type"] or "general"),
                "project": str(r["project"] or "default"),
                "source": str(r["source"] or "mcp"),
                "_sources": [str(r["source"] or "mcp")],
                "result_ids": r_ids,
                "hit_count": len(r_ids),
                "was_used": bool(r["was_used"]),
                "judgment_source": str(r["judgment_source"] or ""),
                "faithfulness_score": (
                    round(float(r["faithfulness_score"]), 2)
                    if r["faithfulness_score"] is not None
                    else None
                ),
                "relevance_score": (
                    round(float(r["relevance_score"]), 2)
                    if r["relevance_score"] is not None
                    else None
                ),
                "helpfulness_score": (
                    round(float(r["helpfulness_score"]), 2)
                    if r["helpfulness_score"] is not None
                    else None
                ),
                "is_noise": bool(r["is_noise"]),
                "noise_reason": str(r["noise_reason"] or ""),
                "stage_metrics": s_metrics,
                "stage_counts": s_counts,
                "returned_tokens": (
                    int(r["returned_tokens"]) if r["returned_tokens"] is not None else None
                ),
                "created_at": created_str,
                "_ts": created_ts,
                "_q_norm": q_norm,
                "_raw_has_counts": s_counts,
            }
        )
        if len(merged) >= limit:
            break

    # Batch-resolve Top-1 experience titles for quick human recognition
    top_exp_ids: list[str] = []
    for item in merged:
        item.pop("_ts", None)
        item.pop("_q_norm", None)
        item.pop("_sources", None)
        item.pop("_raw_has_counts", None)
        if item["result_ids"]:
            top_exp_ids.append(item["result_ids"][0])

    title_map: dict[str, str] = {}
    if top_exp_ids:
        uniq_top_ids = list(set(top_exp_ids))
        try:
            exp_rows = await conn.fetch(
                "SELECT id::text AS id, title FROM experiences WHERE id::text = ANY($1::text[])",
                uniq_top_ids,
            )
            for er in exp_rows:
                title_map[str(er["id"])] = str(er["title"] or "")
            missing_ids = [eid for eid in uniq_top_ids if eid not in title_map]
            if missing_ids:
                arch_rows = await conn.fetch(
                    "SELECT id::text AS id, title FROM archives WHERE id::text = ANY($1::text[])",
                    missing_ids,
                )
                for ar in arch_rows:
                    title_map[str(ar["id"])] = f"[Archive] {ar['title'] or ''}"
            still_missing = [eid for eid in uniq_top_ids if eid not in title_map]
            if still_missing:
                wiki_hits = _lookup_local_wiki_entries(still_missing)
                for wid, winfo in wiki_hits.items():
                    title_map[wid] = f"[📘Wiki] {winfo['title']}"
        except Exception:
            pass

    for idx, item in enumerate(merged, start=1):
        item["index"] = idx
        top_id = item["result_ids"][0] if item["result_ids"] else ""
        item["top_hit_title"] = title_map.get(top_id, "")
        item["hit_count"] = len(item["result_ids"])

    return merged


async def _inspect_postgres_knowledge(
    days: int = 7,
    timeout: float = 3.0,
    limit: int = 12,
) -> dict[str, Any]:
    """Inspect PostgreSQL knowledge base, bi-temporal stats, graph scale, and search telemetry."""
    result: dict[str, Any] = {
        "reachable": False,
        "experiences": {
            "active": 0,
            "superseded": 0,
            "verified": 0,
            "draft_or_review": 0,
            "embedding_covered": 0,
            "embedding_total": 0,
            "embedding_ratio": 1.0,
        },
        "knowledge_graph": {
            "entity_nodes": 0,
            "entity_relations": 0,
            "projects": 0,
        },
        "archives_total": 0,
        "background_tasks_pending": 0,
        "search_telemetry": {
            "days": days,
            "total_searches": 0,
            "hits": 0,
            "hit_rate": 0.0,
            "used": 0,
            "use_rate": 0.0,
            "avg_total_ms": 0.0,
            "avg_returned_tokens": 0.0,
            "graph_assisted_searches": 0,
            "faithfulness_avg": 0.0,
        },
        "recent_experiences": [],
        "recent_recalls": [],
    }

    try:
        dsn = _resolve_dsn()
    except Exception as exc:
        result["error"] = str(exc)[:120]
        return result

    try:
        import asyncpg

        conn = await asyncio.wait_for(asyncpg.connect(dsn), timeout=timeout)
        try:
            result["reachable"] = True

            # 1. Draft/review count
            draft_cnt = await conn.fetchval(
                """
                SELECT count(*) FROM experiences
                WHERE exp_status IN ('draft', 'review') AND is_deleted = FALSE
                """
            )
            result["experiences"]["draft_or_review"] = int(draft_cnt or 0)

            # 2. Knowledge graph & archives counts
            tables = {
                r["tablename"]
                for r in await conn.fetch(
                    "SELECT tablename FROM pg_tables WHERE schemaname = 'public'"
                )
            }
            if "entities" in tables:
                row_en = await conn.fetchrow(
                    "SELECT count(*) AS cnt, count(DISTINCT project) AS projs FROM entities"
                )
                if row_en:
                    result["knowledge_graph"]["entity_nodes"] = int(row_en["cnt"] or 0)
                    result["knowledge_graph"]["projects"] = int(row_en["projs"] or 0)
            if "relationships" in tables:
                rel_cnt = await conn.fetchval(
                    "SELECT count(*) FROM relationships WHERE valid_until IS NULL"
                )
                result["knowledge_graph"]["entity_relations"] = int(rel_cnt or 0)
            if "archives" in tables:
                arch_cnt = await conn.fetchval("SELECT count(*) FROM archives")
                result["archives_total"] = int(arch_cnt or 0)
            if "background_tasks" in tables:
                bg_cnt = await conn.fetchval(
                    "SELECT count(*) FROM background_tasks WHERE status IN ('pending', 'running')"
                )
                result["background_tasks_pending"] = int(bg_cnt or 0)

            # 3. Recent published experiences
            rec_rows = await conn.fetch(
                """
                SELECT id::text, title, COALESCE(project, 'default') AS project,
                       exp_status, created_at::text
                FROM experiences
                WHERE exp_status IN ('published', 'promoted') AND is_deleted = FALSE
                ORDER BY created_at DESC
                LIMIT 3
                """
            )
            result["recent_experiences"] = [
                {
                    "id": str(r["id"])[:8],
                    "title": str(r["title"] or "")[:60],
                    "project": str(r["project"]),
                    "status": str(r["exp_status"]),
                    "created_at": str(r["created_at"] or "")[:19],
                }
                for r in rec_rows
            ]

            # 4. Recent recalls list (with twin mcp/daemon log deduplication)
            if "search_logs" in tables:
                result["recent_recalls"] = await _fetch_recent_recalls_from_conn(
                    conn, limit=max(1, int(limit))
                )
        finally:
            await conn.close()

        # Reuse battle-tested helpers from tm_stats
        cov, bitemp, faith_avg = await asyncio.gather(
            _embedding_coverage(dsn),
            _bitemporal_health(dsn),
            _faithfulness_avg(dsn, days),
            return_exceptions=True,
        )
        if isinstance(cov, dict):
            result["experiences"]["embedding_covered"] = int(cov.get("covered", 0))
            result["experiences"]["embedding_total"] = int(cov.get("total", 0))
            result["experiences"]["embedding_ratio"] = float(cov.get("ratio", 1.0))
        if isinstance(bitemp, dict):
            result["experiences"]["active"] = int(bitemp.get("active_count", 0))
            result["experiences"]["superseded"] = int(bitemp.get("superseded_count", 0))
            result["experiences"]["verified"] = int(bitemp.get("verified_count", 0))
        if isinstance(faith_avg, (int, float)):
            result["search_telemetry"]["faithfulness_avg"] = round(float(faith_avg), 2)

        from team_memory.services.search.search_log_writer import SearchLogWriter

        writer = SearchLogWriter(dsn=dsn)
        try:
            stats, telemetry = await asyncio.gather(
                writer.get_stats(days=days),
                writer.get_telemetry_summary(days=days),
                return_exceptions=True,
            )
            if isinstance(stats, dict) and "error" not in stats:
                total_s = int(stats.get("total_searches", 0))
                hits = int(stats.get("hit", 0))
                used = int(stats.get("used", 0))
                use_rate = float(stats.get("use_rate", 0.0))
                result["search_telemetry"].update(
                    {
                        "total_searches": total_s,
                        "hits": hits,
                        "hit_rate": round((hits / total_s) if total_s > 0 else 0.0, 3),
                        "used": used,
                        "use_rate": round(use_rate, 3),
                    }
                )
            if isinstance(telemetry, dict):
                panel_data = format_telemetry_panel_data(
                    telemetry,
                    bitemp if isinstance(bitemp, dict) else None,
                )
                result["search_telemetry"].update(
                    {
                        "avg_total_ms": panel_data["avg_total_ms"],
                        "avg_returned_tokens": panel_data["avg_returned_tokens"],
                        "graph_assisted_searches": panel_data["graph_assisted_searches"],
                    }
                )
        finally:
            await writer.close()

    except Exception as exc:
        result["error"] = str(exc)[:120]

    return result

"""Helper functions for recall trace target lookup, query analysis, and graph path discovery."""

from __future__ import annotations

import re as _re
from typing import Any

from team_memory.daemon.observe.status_monitor.collectors import (
    _parse_json_field,
    _parse_list_field,
)
from team_memory.daemon.observe.status_monitor.pg_collector import _fetch_recent_recalls_from_conn


async def _resolve_trace_target(
    conn: Any,
    selector: str | int,
    recent_recalls: list[dict[str, Any]] | None,
    limit: int,
) -> dict[str, Any] | None:
    """Resolve a recall log row by 1-based index, short ID, or UUID prefix."""
    sel_str = str(selector).strip()
    needed_limit = max(limit, int(sel_str) + 5) if sel_str.isdigit() else max(limit, 30)
    if not recent_recalls or (sel_str.isdigit() and len(recent_recalls) < int(sel_str)):
        recent_recalls = await _fetch_recent_recalls_from_conn(conn, limit=needed_limit)

    target: dict[str, Any] | None = None
    if sel_str.isdigit():
        idx_num = int(sel_str)
        for item in recent_recalls:
            if item.get("index") == idx_num:
                target = dict(item)
                break

    if target is None:
        for item in recent_recalls:
            if item["id"].startswith(sel_str) or item.get("short_id") == sel_str:
                target = dict(item)
                break

    if target is None:
        row = await conn.fetchrow(
            """
            SELECT id::text AS id, query, COALESCE(intent_type, 'general') AS intent_type,
                   COALESCE(project, 'default') AS project, COALESCE(source, 'mcp') AS source,
                   result_ids, was_used, agent_response_snippet, judgment_source,
                   faithfulness_score, relevance_score, helpfulness_score,
                   COALESCE(is_noise, FALSE) AS is_noise, noise_reason,
                   stage_metrics, stage_counts, returned_tokens, created_at
            FROM search_logs
            WHERE id::text LIKE $1
            ORDER BY created_at DESC
            LIMIT 1
            """,
            f"{sel_str}%",
        )
        if not row:
            return None
        created_dt = row["created_at"]
        created_str = (
            created_dt.astimezone().strftime("%Y-%m-%d %H:%M:%S")
            if hasattr(created_dt, "astimezone")
            else str(created_dt or "")[:19]
        )
        r_ids = _parse_list_field(row["result_ids"])
        return {
            "index": 0,
            "id": str(row["id"]),
            "short_id": str(row["id"])[:8],
            "query": str(row["query"] or ""),
            "intent_type": str(row["intent_type"] or "general"),
            "project": str(row["project"] or "default"),
            "source": str(row["source"] or "mcp"),
            "result_ids": r_ids,
            "hit_count": len(r_ids),
            "was_used": bool(row["was_used"]),
            "agent_response_snippet": str(row["agent_response_snippet"] or ""),
            "judgment_source": str(row["judgment_source"] or ""),
            "faithfulness_score": (
                round(float(row["faithfulness_score"]), 2)
                if row["faithfulness_score"] is not None
                else None
            ),
            "relevance_score": (
                round(float(row["relevance_score"]), 2)
                if row["relevance_score"] is not None
                else None
            ),
            "helpfulness_score": (
                round(float(row["helpfulness_score"]), 2)
                if row["helpfulness_score"] is not None
                else None
            ),
            "is_noise": bool(row["is_noise"]),
            "noise_reason": str(row["noise_reason"] or ""),
            "stage_metrics": _parse_json_field(row["stage_metrics"]),
            "stage_counts": _parse_json_field(row["stage_counts"]),
            "returned_tokens": (
                int(row["returned_tokens"]) if row["returned_tokens"] is not None else None
            ),
            "created_at": created_str,
        }

    # Fetch agent_response_snippet if not loaded in list view
    snip = await conn.fetchval(
        "SELECT agent_response_snippet FROM search_logs WHERE id::text = $1",
        target["id"],
    )
    target["agent_response_snippet"] = str(snip or "")
    return target


async def _analyze_trace_query_and_graph(
    conn: Any,
    query_raw: str,
) -> tuple[
    str,
    list[str],
    list[str],
    list[dict[str, Any]],
    list[dict[str, Any]],
    set[str],
    set[str],
    int,
]:
    """Run query synonym expansion, entity extraction, graph path lookup, and FTS channel estimation."""
    from team_memory.services.entity.entity_search import extract_entities_from_query
    from team_memory.services.search.query_expansion import expand_query_synonyms

    default_synonyms = {
        "数据库": "database",
        "连接池": "connection pool",
        "缓存": "cache",
        "向量": "vector embedding",
        "图谱": "entity graph",
        "重排": "reranker",
        "钩子": "hook",
        "守护进程": "daemon",
        "精炼": "refinement",
        "死锁": "deadlock",
        "超时": "timeout",
        "迁移": "migration alembic",
        "召回": "recall",
        "状态": "status",
        "命令行": "cli",
        "知识库": "wiki",
        "tm": "team_memory",
    }
    try:
        from team_memory.config import get_settings

        cfg_syns = getattr(get_settings(), "tag_synonyms", None) or {}
        if isinstance(cfg_syns, dict):
            default_synonyms.update(cfg_syns)
    except Exception:
        pass
    expanded_query = expand_query_synonyms(query_raw, default_synonyms)
    added_synonyms = (
        expanded_query[len(query_raw.strip()) :].strip().split()
        if len(expanded_query) > len(query_raw.strip())
        else []
    )
    db_lexicon: set[str] = set()
    try:
        lex_rows = await conn.fetch(
            "SELECT DISTINCT lower(name) AS name FROM entities WHERE length(name) >= 2"
        )
        db_lexicon = {str(lr["name"]).strip().lower() for lr in lex_rows if lr["name"]}
    except Exception:
        pass
    candidate_entities = extract_entities_from_query(query_raw, lexicon=db_lexicon or None)

    for short_tok in _re.findall(r"[A-Za-z]{2,}", query_raw):
        if short_tok.lower() in {"js", "ui", "db", "ip", "ad", "ai", "ci", "cd", "os", "gc"}:
            if short_tok.lower() not in {c.lower() for c in candidate_entities}:
                candidate_entities.append(short_tok)

    matched_entities: list[dict[str, Any]] = []
    graph_paths: list[dict[str, Any]] = []
    graph_exp_ids: set[str] = set()

    if candidate_entities:
        cands_lower = [c.lower() for c in candidate_entities]
        ent_rows = await conn.fetch(
            """
            SELECT id::text AS id, name, entity_type, COALESCE(project, 'default') AS project,
                   COALESCE(source_count, 1) AS source_count
            FROM entities
            WHERE lower(name) = ANY($1::text[])
            ORDER BY source_count DESC
            LIMIT 12
            """,
            cands_lower,
        )
        matched_entities = [
            {
                "id": str(er["id"]),
                "name": str(er["name"]),
                "entity_type": str(er["entity_type"] or "concept"),
                "project": str(er["project"]),
                "source_count": int(er["source_count"] or 1),
            }
            for er in ent_rows
        ]

        seed_ids = [m["id"] for m in matched_entities]
        if seed_ids:
            rel_rows = await conn.fetch(
                """
                SELECT e1.name AS src_name, r.relation_type, e2.name AS tgt_name,
                       COALESCE(r.weight, 1.0) AS weight,
                       e2.id::text AS tgt_entity_id
                FROM relationships r
                JOIN entities e1 ON e1.id = r.source_entity_id
                JOIN entities e2 ON e2.id = r.target_entity_id
                WHERE r.valid_until IS NULL
                  AND (
                      r.source_entity_id::text = ANY($1::text[])
                      OR r.target_entity_id::text = ANY($1::text[])
                  )
                ORDER BY r.weight DESC
                LIMIT 8
                """,
                seed_ids,
            )
            for rr in rel_rows:
                graph_paths.append(
                    {
                        "src": str(rr["src_name"]),
                        "rel": str(rr["relation_type"]),
                        "tgt": str(rr["tgt_name"]),
                        "weight": round(float(rr["weight"] or 1.0), 2),
                    }
                )

            neighbor_ids = list({str(rr["tgt_entity_id"]) for rr in rel_rows} | set(seed_ids))
            ge_rows = await conn.fetch(
                """
                SELECT DISTINCT experience_id::text AS exp_id
                FROM experience_entities
                WHERE entity_id::text = ANY($1::text[])
                LIMIT 50
                """,
                neighbor_ids,
            )
            graph_exp_ids = {str(gr["exp_id"]) for gr in ge_rows}

    fts_exp_ids: set[str] = set()
    fts_total_estimate = 0
    q_words = [
        w
        for w in candidate_entities + query_raw.replace("?", " ").replace("，", " ").split()
        if len(w) >= 2
    ][:6]
    if q_words:
        like_patterns = [f"%{w}%" for w in q_words[:4]]
        try:
            fts_rows = await conn.fetch(
                """
                SELECT id::text AS id
                FROM experiences
                WHERE is_deleted = FALSE
                  AND exp_status IN ('published', 'promoted')
                  AND (
                      title ILIKE ANY($1::text[])
                      OR description ILIKE ANY($1::text[])
                      OR solution ILIKE ANY($1::text[])
                  )
                LIMIT 30
                """,
                like_patterns,
            )
            fts_exp_ids = {str(fr["id"]) for fr in fts_rows}
            fts_total_estimate = len(fts_exp_ids)
        except Exception:
            pass

    return (
        expanded_query,
        added_synonyms,
        candidate_entities,
        matched_entities,
        graph_paths,
        graph_exp_ids,
        fts_exp_ids,
        fts_total_estimate,
    )

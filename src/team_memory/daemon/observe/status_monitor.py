"""Panoramic status monitor and live TUI dashboard for TeamMemory.

Collects real-time health and telemetry across all four architectural planes:
1. Core Server & Infrastructure (:9111/health, PostgreSQL, Embedding, LLM, Cache)
2. TM Daemon & Client Hooks (:3901/status, RefinementWorker, Heartbeat, Hook Payloads)
3. Local Refinement Pipeline (SQLite ConversationBuffer funnel, PendingQueue, Tool Observations)
4. Knowledge Graph, Bi-Temporal Memory & Search Telemetry (PostgreSQL experiences, entities, search_logs)

Usage via CLI:
    tm-cli status                  # One-shot panoramic status report
    tm-cli status --watch          # Continuous live-updating terminal dashboard (Ctrl+C to exit)
    tm-cli status -w --interval 3  # Refresh every 3 seconds
    tm-cli status --json           # Machine-readable JSON snapshot
"""

from __future__ import annotations

import asyncio
import json
import os
import sqlite3
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import httpx

from team_memory.daemon.observe.tm_stats import (
    _bitemporal_health,
    _embedding_coverage,
    _faithfulness_avg,
    _resolve_dsn,
    format_telemetry_panel_data,
)


def _resolve_endpoints_and_paths(
    server_url: str | None = None,
    daemon_url: str | None = None,
    db_path: str | None = None,
) -> tuple[str, str, str, str]:
    """Resolve (server_url, daemon_url, drafts_db_path, capture_db_path)."""
    resolved_server = server_url or os.environ.get("TM_BASE_URL", "")
    resolved_daemon = daemon_url or os.environ.get("TM_DAEMON_URL", "")
    resolved_db = db_path or os.environ.get("TM_PIPELINE_DB_PATH", "")

    try:
        from team_memory.client_config import ClientConfig

        cfg = ClientConfig.load()
        if not resolved_server:
            resolved_server = cfg.server.url
        if not resolved_daemon:
            resolved_daemon = f"http://{cfg.daemon.host}:{cfg.daemon.port}"
        if not resolved_db:
            resolved_db = cfg.conversation.db_path
    except Exception:
        pass

    if not resolved_server:
        resolved_server = "http://localhost:9111"
    if not resolved_daemon:
        daemon_port = os.environ.get("DAEMON_PORT", "3901")
        resolved_daemon = f"http://127.0.0.1:{daemon_port}"
    if not resolved_db:
        resolved_db = str(Path.home() / ".cache" / "tm-pipeline" / "drafts.db")

    p_db = Path(resolved_db)
    capture_db = str(p_db.with_name(p_db.stem + "_capture.db"))
    return (
        resolved_server.rstrip("/"),
        resolved_daemon.rstrip("/"),
        resolved_db,
        capture_db,
    )


async def _probe_web_health(server_url: str, timeout: float = 2.5) -> dict[str, Any]:
    """Probe Core Web/MCP Server /health endpoint."""
    url = f"{server_url}/health"
    start = time.monotonic()
    try:
        async with httpx.AsyncClient(timeout=timeout) as client:
            resp = await client.get(url)
            latency_ms = round((time.monotonic() - start) * 1000, 1)
            data = resp.json() if resp.content else {}
            return {
                "reachable": True,
                "url": server_url,
                "http_status": resp.status_code,
                "latency_ms": latency_ms,
                "status": data.get("status", "unknown"),
                "version": data.get("version", "?"),
                "checks": data.get("checks", {}),
                "heartbeat": data.get("heartbeat", {}),
            }
    except Exception as exc:
        return {
            "reachable": False,
            "url": server_url,
            "status": "offline",
            "error": str(exc)[:120],
            "checks": {},
            "heartbeat": {},
        }


async def _probe_daemon_status(daemon_url: str, timeout: float = 2.5) -> dict[str, Any]:
    """Probe TM Daemon /status endpoint."""
    url = f"{daemon_url}/status"
    start = time.monotonic()
    try:
        async with httpx.AsyncClient(timeout=timeout) as client:
            resp = await client.get(url)
            latency_ms = round((time.monotonic() - start) * 1000, 1)
            data = resp.json() if resp.content else {}
            return {
                "reachable": True,
                "url": daemon_url,
                "http_status": resp.status_code,
                "latency_ms": latency_ms,
                "status": data.get("status", "running"),
                "tm_mode": data.get("tm_mode", "unknown"),
                "refinement_worker": data.get("refinement_worker", {}),
                "heartbeat": data.get("heartbeat", {}),
            }
    except Exception as exc:
        return {
            "reachable": False,
            "url": daemon_url,
            "status": "offline",
            "error": str(exc)[:120],
            "refinement_worker": {},
            "heartbeat": {},
        }


def _inspect_local_sqlite(drafts_db_path: str, capture_db_path: str) -> dict[str, Any]:
    """Inspect local SQLite ConversationBuffer, PendingQueue, and PayloadCapture."""
    result: dict[str, Any] = {
        "db_path": drafts_db_path,
        "db_exists": False,
        "db_size_mb": 0.0,
        "sessions_by_status": {},
        "total_sessions": 0,
        "ready_to_converge": 0,
        "stuck_no_assistant": 0,
        "total_messages": 0,
        "total_tool_observations": 0,
        "pending_queue": {"pending": 0, "retrying": 0, "failed": 0},
        "recent_sessions": [],
        "hooks_capture": {
            "db_exists": False,
            "total_payloads": 0,
            "by_endpoint": {},
            "latest_event": None,
        },
    }

    p_drafts = Path(drafts_db_path)
    if p_drafts.exists():
        result["db_exists"] = True
        try:
            result["db_size_mb"] = round(p_drafts.stat().st_size / (1024 * 1024), 2)
        except OSError:
            pass

        try:
            conn = sqlite3.connect(f"file:{drafts_db_path}?mode=ro", uri=True, timeout=2.0)
            try:
                tables = {
                    r[0]
                    for r in conn.execute(
                        "SELECT name FROM sqlite_master WHERE type='table'"
                    ).fetchall()
                }
                if "sessions" in tables:
                    status_rows = conn.execute(
                        "SELECT COALESCE(status, 'unknown'), COUNT(*) FROM sessions GROUP BY status"
                    ).fetchall()
                    by_status = {str(st): int(cnt) for st, cnt in status_rows}
                    result["sessions_by_status"] = by_status
                    result["total_sessions"] = sum(by_status.values())

                    if "messages" in tables:
                        result["total_messages"] = int(
                            conn.execute("SELECT COUNT(*) FROM messages").fetchone()[0] or 0
                        )
                        result["ready_to_converge"] = int(
                            conn.execute(
                                """
                                SELECT COUNT(*) FROM sessions s
                                WHERE s.status IN ('active', 'collecting')
                                AND (
                                    SELECT COUNT(*) FROM messages m
                                    WHERE m.session_id = s.id AND m.role = 'assistant'
                                ) >= 3
                                """
                            ).fetchone()[0]
                            or 0
                        )
                        result["stuck_no_assistant"] = int(
                            conn.execute(
                                """
                                SELECT COUNT(*) FROM sessions s
                                WHERE s.status IN ('active', 'collecting')
                                AND (
                                    SELECT COUNT(*) FROM messages m
                                    WHERE m.session_id = s.id AND m.role = 'assistant'
                                ) = 0
                                """
                            ).fetchone()[0]
                            or 0
                        )

                    has_tool_obs = "tool_observations" in tables
                    if has_tool_obs:
                        result["total_tool_observations"] = int(
                            conn.execute("SELECT COUNT(*) FROM tool_observations").fetchone()[0] or 0
                        )

                    tool_subquery = (
                        "(SELECT COUNT(*) FROM tool_observations t WHERE t.session_id = s.id)"
                        if has_tool_obs
                        else "0"
                    )
                    msg_user_subquery = (
                        "(SELECT COUNT(*) FROM messages m WHERE m.session_id = s.id AND m.role = 'user')"
                        if "messages" in tables
                        else "0"
                    )
                    msg_asst_subquery = (
                        "(SELECT COUNT(*) FROM messages m WHERE m.session_id = s.id AND m.role = 'assistant')"
                        if "messages" in tables
                        else "0"
                    )
                    recent_rows = conn.execute(
                        f"""
                        SELECT s.id, COALESCE(s.project, 'default'), COALESCE(s.status, 'active'),
                               s.updated_at,
                               {msg_user_subquery} AS user_n,
                               {msg_asst_subquery} AS asst_n,
                               {tool_subquery} AS tool_n
                        FROM sessions s
                        ORDER BY s.updated_at DESC
                        LIMIT 5
                        """
                    ).fetchall()
                    result["recent_sessions"] = [
                        {
                            "id": str(r[0]),
                            "project": str(r[1]),
                            "status": str(r[2]),
                            "updated_at": str(r[3] or ""),
                            "user_msgs": int(r[4] or 0),
                            "asst_msgs": int(r[5] or 0),
                            "tool_obs": int(r[6] or 0),
                        }
                        for r in recent_rows
                    ]

                if "pending_ops" in tables:
                    pq_rows = conn.execute(
                        "SELECT COALESCE(status, 'pending'), COUNT(*) FROM pending_ops GROUP BY status"
                    ).fetchall()
                    pq_map = {"pending": 0, "retrying": 0, "failed": 0}
                    for st, cnt in pq_rows:
                        pq_map[str(st)] = int(cnt)
                    result["pending_queue"] = pq_map
            finally:
                conn.close()
        except Exception as exc:
            result["error"] = str(exc)[:120]

    p_cap = Path(capture_db_path)
    if p_cap.exists():
        result["hooks_capture"]["db_exists"] = True
        try:
            conn = sqlite3.connect(f"file:{capture_db_path}?mode=ro", uri=True, timeout=2.0)
            try:
                tables = {
                    r[0]
                    for r in conn.execute(
                        "SELECT name FROM sqlite_master WHERE type='table'"
                    ).fetchall()
                }
                if "raw_payloads" in tables:
                    ep_rows = conn.execute(
                        "SELECT endpoint, COUNT(*) FROM raw_payloads GROUP BY endpoint ORDER BY COUNT(*) DESC"
                    ).fetchall()
                    by_ep = {str(ep): int(cnt) for ep, cnt in ep_rows}
                    result["hooks_capture"]["by_endpoint"] = by_ep
                    result["hooks_capture"]["total_payloads"] = sum(by_ep.values())

                    latest = conn.execute(
                        "SELECT endpoint, source, created_at FROM raw_payloads ORDER BY id DESC LIMIT 1"
                    ).fetchone()
                    if latest:
                        result["hooks_capture"]["latest_event"] = {
                            "endpoint": str(latest[0]),
                            "source": str(latest[1]),
                            "created_at": str(latest[2]),
                        }
            finally:
                conn.close()
        except Exception as exc:
            result["hooks_capture"]["error"] = str(exc)[:120]

    return result


def _parse_json_field(val: Any) -> dict[str, Any]:
    """Parse JSONB or JSON string field into a dictionary."""
    if isinstance(val, dict):
        return dict(val)
    if isinstance(val, str) and val.strip():
        try:
            parsed = json.loads(val)
            return dict(parsed) if isinstance(parsed, dict) else {}
        except Exception:
            return {}
    return {}


def _parse_list_field(val: Any) -> list[str]:
    """Safely parse a JSONB / array / text field from asyncpg into a list of experience ID strings."""
    raw_items: list[Any] = []
    if isinstance(val, (list, tuple)):
        raw_items = list(val)
    elif isinstance(val, str) and val.strip():
        s = val.strip()
        if s.startswith("["):
            try:
                parsed = json.loads(s)
                if isinstance(parsed, list):
                    raw_items = parsed
            except Exception:
                raw_items = [s]
        else:
            raw_items = [s]

    out: list[str] = []
    for item in raw_items:
        if isinstance(item, dict):
            eid = str(item.get("id") or item.get("experience_id") or "").strip()
            if eid:
                out.append(eid)
        elif item:
            s_item = str(item).strip()
            if s_item:
                out.append(s_item)
    return out


_WIKI_DB_CANDIDATES: tuple[str, ...] = (
    os.path.join(os.path.expanduser("~"), ".cache", "tm-pipeline", "wiki.db"),
    os.path.join(os.path.expanduser("~"), ".team_memory", "wiki.db"),
)


def _lookup_local_wiki_entries(
    ids: list[str],
    candidate_dbs: tuple[str, ...] | list[str] | None = None,
) -> dict[str, dict[str, Any]]:
    """Look up Wiki entry metadata from local SQLite wiki.db files by ID."""
    if not ids:
        return {}
    dbs = candidate_dbs if candidate_dbs is not None else _WIKI_DB_CANDIDATES
    found: dict[str, dict[str, Any]] = {}
    placeholders = ",".join("?" for _ in ids)
    for db_p in dbs:
        if not os.path.exists(db_p):
            continue
        try:
            conn = sqlite3.connect(f"file:{db_p}?mode=ro", uri=True, timeout=1.0)
            conn.row_factory = sqlite3.Row
            try:
                cur = conn.execute(
                    f"""
                    SELECT *
                    FROM wiki_entries
                    WHERE id IN ({placeholders})
                    """,
                    tuple(ids),
                )
                for row in cur.fetchall():
                    keys = set(row.keys())
                    wid = str(row["id"])
                    if wid in found:
                        continue
                    raw_tags = row["tags"] if "tags" in keys else ""
                    tags_list: list[str] = []
                    if raw_tags:
                        try:
                            tags_list = [str(t) for t in json.loads(raw_tags)]
                        except Exception:
                            tags_list = [str(raw_tags)]
                    found[wid] = {
                        "id": wid,
                        "title": str(row["title"] or "") if "title" in keys else "",
                        "content": str(row["content"] or "") if "content" in keys else "",
                        "project": str(row["project"] or "default") if "project" in keys else "default",
                        "tags": tags_list[:6],
                        "source_url": str(row["source_url"] or "") if "source_url" in keys else "",
                    }
            finally:
                conn.close()
        except Exception:
            continue
    return found


def _clean_query_display(query: str, max_len: int = 58) -> str:
    """Clean up single-line display of a recall query."""
    raw = query or ""
    if "<CONTEXT_SUMMARY>" in raw or "# User Requests" in raw:
        # Extract the last numbered user request before # Previous Session Summary
        sec = raw.split("# User Requests")[-1].split("# Previous Session Summary")[0]
        req_lines = [
            ln.strip()
            for ln in sec.splitlines()
            if ln.strip() and ln.strip()[0].isdigit() and ". " in ln
        ]
        if req_lines:
            last_req = req_lines[-1].split(". ", 1)[-1].strip()
            raw = f"[Ctx] {last_req}"
    q = " ".join(raw.replace("\n", " ").split())
    if len(q) > max_len:
        return q[: max_len - 1] + "…"
    return q


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


async def collect_status_snapshot(
    *,
    server_url: str | None = None,
    daemon_url: str | None = None,
    db_path: str | None = None,
    days: int = 7,
    limit: int = 12,
) -> dict[str, Any]:
    """Collect a complete status snapshot across all 4 TeamMemory runtime planes."""
    s_url, d_url, drafts_db, capture_db = _resolve_endpoints_and_paths(
        server_url=server_url,
        daemon_url=daemon_url,
        db_path=db_path,
    )

    web_task = _probe_web_health(s_url)
    daemon_task = _probe_daemon_status(d_url)
    pg_task = _inspect_postgres_knowledge(days=days, limit=limit)
    sqlite_task = asyncio.to_thread(_inspect_local_sqlite, drafts_db, capture_db)

    web_info, daemon_info, pg_info, sqlite_info = await asyncio.gather(
        web_task,
        daemon_task,
        pg_task,
        sqlite_task,
    )

    # Derive overall health summary
    anomalies: list[str] = []
    if not web_info.get("reachable") and not pg_info.get("reachable"):
        anomalies.append("Core Server & PostgreSQL unreachable")
    elif not pg_info.get("reachable"):
        anomalies.append("PostgreSQL unreachable")

    if not daemon_info.get("reachable"):
        anomalies.append(f"TM Daemon offline ({d_url})")
    else:
        rw = daemon_info.get("refinement_worker") or {}
        if rw and not rw.get("healthy", True):
            anomalies.append(f"RefinementWorker unhealthy (fails={rw.get('fail_count', 0)})")

    pq = sqlite_info.get("pending_queue") or {}
    if pq.get("failed", 0) > 0:
        anomalies.append(f"PendingQueue has {pq['failed']} failed ops")

    emb_ratio = pg_info.get("experiences", {}).get("embedding_ratio", 1.0)
    if pg_info.get("experiences", {}).get("embedding_total", 0) > 0 and emb_ratio < 0.95:
        anomalies.append(f"Embedding coverage low ({emb_ratio * 100:.0f}%)")

    overall = "healthy" if not anomalies else ("degraded" if pg_info.get("reachable") else "unhealthy")

    return {
        "timestamp": datetime.now(timezone.utc).astimezone().strftime("%Y-%m-%d %H:%M:%S %Z"),
        "overall_status": overall,
        "anomalies": anomalies,
        "core_server": web_info,
        "daemon": daemon_info,
        "pipeline_local": sqlite_info,
        "knowledge_base": pg_info,
    }


def _badge_plain(ok: bool, warn: bool = False, label_ok: str = "UP", label_down: str = "DOWN") -> str:
    if ok and not warn:
        return f"🟢 {label_ok}"
    if warn:
        return f"🟡 {label_ok}"
    return f"🔴 {label_down}"


def render_status_plain(snapshot: dict[str, Any], watch_meta: dict[str, Any] | None = None) -> str:
    """Format status snapshot as clean plain/ANSI text (works without Rich)."""
    lines: list[str] = []
    ts = snapshot.get("timestamp", "")
    overall = snapshot.get("overall_status", "unknown").upper()
    overall_icon = {"HEALTHY": "🟢", "DEGRADED": "🟡", "UNHEALTHY": "🔴"}.get(overall, "⚪")

    header = f"=== TeamMemory Panoramic Status  [{overall_icon} {overall}]  ({ts}) ==="
    if watch_meta:
        tick = watch_meta.get("tick", 1)
        interval = watch_meta.get("interval", 2.0)
        header += f"  [Watch #{tick} | every {interval:g}s | Ctrl+C to exit]"
    lines.append(header)

    anomalies = snapshot.get("anomalies") or []
    if anomalies:
        lines.append("  ⚠ Anomalies: " + " ; ".join(anomalies))
    lines.append("")

    # 1. Core Server & Infrastructure
    web = snapshot.get("core_server", {})
    pg = snapshot.get("knowledge_base", {})
    checks = web.get("checks") or {}
    db_chk = checks.get("database") or {}
    emb_chk = checks.get("embedding_health") or checks.get("embedding_provider") or {}
    llm_chk = checks.get("llm_provider") or {}
    cache_chk = checks.get("cache") or {}

    web_ok = bool(web.get("reachable")) and web.get("status") in ("healthy", "degraded", "up")
    db_ok = (db_chk.get("status") == "up") or bool(pg.get("reachable"))
    emb_ok = emb_chk.get("status") == "up"
    llm_ok = llm_chk.get("status") == "up"

    lines.append("┌─ 1. 核心服务与基础设施 (Core Server & Infra) ─────────────────────────────────")
    lines.append(
        f"│ Web / MCP Server ({web.get('url', ':9111')}): "
        f"{_badge_plain(web_ok, warn=(web.get('status') == 'degraded'))} "
        f"(v{web.get('version', '?')}, latency={web.get('latency_ms', '-')}ms)"
    )
    lines.append(
        f"│ PostgreSQL + pgvector: {_badge_plain(db_ok)} "
        f"(latency={db_chk.get('latency_ms', '-')}ms)"
    )
    if web_ok:
        lines.append(
            f"│ Embedding Provider:    {_badge_plain(emb_ok, label_down='IDLE/DOWN')} "
            f"(provider={emb_chk.get('provider', '?')}, latency={emb_chk.get('latency_ms', '-')}ms)"
        )
        lines.append(
            f"│ LLM Provider:          {_badge_plain(llm_ok)} "
            f"(provider={llm_chk.get('provider', '?')}, model={llm_chk.get('model', '-')})"
        )
        lines.append(
            f"│ Search Cache:          {_badge_plain(cache_chk.get('status') == 'up')} "
            f"(backend={cache_chk.get('backend', 'none')})"
        )
    lines.append("└───────────────────────────────────────────────────────────────────────────────")

    # 2. TM Daemon & Hooks
    dm = snapshot.get("daemon", {})
    dm_ok = bool(dm.get("reachable"))
    rw = dm.get("refinement_worker") or {}
    rw_ok = bool(rw.get("healthy", False)) if dm_ok else False
    hb = dm.get("heartbeat") or {}
    loc = snapshot.get("pipeline_local", {})
    cap = loc.get("hooks_capture") or {}
    latest_ev = cap.get("latest_event")

    lines.append("┌─ 2. 守护进程与客户端感知 (TM Daemon & IDE Hooks) ─────────────────────────────")
    lines.append(
        f"│ TM Daemon ({dm.get('url', ':3901')}):      "
        f"{_badge_plain(dm_ok)} (mode={dm.get('tm_mode', '-')}, latency={dm.get('latency_ms', '-')}ms)"
    )
    last_ok_str = str(rw.get("last_tick_ok") or "-")[:19]
    lines.append(
        f"│ RefinementWorker:      {_badge_plain(rw_ok, label_ok='HEALTHY', label_down='DOWN/UNHEALTHY')} "
        f"(ticks={rw.get('tick_count', 0)}, fails={rw.get('fail_count', 0)}, last_ok={last_ok_str})"
    )
    lines.append(
        f"│ Sleep/Wake Heartbeat:  checks={hb.get('check_count', 0)}, "
        f"wake_recoveries={hb.get('sleep_wake_count', 0)}"
    )
    if latest_ev:
        lines.append(
            f"│ IDE Hook Capture:      total={cap.get('total_payloads', 0)} events | "
            f"latest={latest_ev.get('endpoint')} ({latest_ev.get('source')}) @ {latest_ev.get('created_at', '')[:19]}"
        )
    else:
        lines.append(f"│ IDE Hook Capture:      total={cap.get('total_payloads', 0)} events")
    lines.append("└───────────────────────────────────────────────────────────────────────────────")

    # 3. Local Refinement Funnel
    by_st = loc.get("sessions_by_status") or {}
    active_n = by_st.get("active", 0) + by_st.get("collecting", 0)
    needs_n = by_st.get("needs_refinement", 0)
    refining_n = by_st.get("refining", 0) + by_st.get("publishing", 0)
    done_n = by_st.get("published", 0) + by_st.get("refined", 0)
    skip_n = by_st.get("skipped", 0) + by_st.get("empty", 0)
    pq = loc.get("pending_queue") or {}

    lines.append("┌─ 3. 本地精炼管线漏斗 (Refinement Pipeline Funnel — SQLite) ───────────────────")
    lines.append(
        f"│ Session Funnel:  🟢 Collecting={active_n} (Ready≥3 turns: {loc.get('ready_to_converge', 0)}) "
        f"→ 🟡 Queued={needs_n} → 🔄 Refining={refining_n} → ✅ Refined={done_n} (Skipped={skip_n})"
    )
    lines.append(
        f"│ Buffer Stats:    Messages={loc.get('total_messages', 0)} | "
        f"Tool Observations={loc.get('total_tool_observations', 0)} | "
        f"SQLite Size={loc.get('db_size_mb', 0.0):.2f} MB"
    )
    pq_warn = pq.get("failed", 0) > 0 or pq.get("retrying", 0) > 0
    pq_retry_only = pq.get("retrying", 0) > 0 and pq.get("failed", 0) == 0
    pq_badge = _badge_plain(not pq_warn, warn=pq_retry_only, label_ok="CLEAR", label_down="BACKLOG")
    lines.append(
        f"│ PendingQueue:    {pq_badge} "
        f"(pending={pq.get('pending', 0)}, retrying={pq.get('retrying', 0)}, failed={pq.get('failed', 0)})"
    )
    rec_sess = loc.get("recent_sessions") or []
    if rec_sess:
        lines.append("│ Recent Sessions:")
        for s in rec_sess[:3]:
            lines.append(
                f"│   • [{s['project']:<14}] {s['id'][:16]}  status={s['status']:<16} "
                f"msgs={s['user_msgs']}u/{s['asst_msgs']}a  tools={s['tool_obs']}  ({s['updated_at'][:19]})"
            )
    lines.append("└───────────────────────────────────────────────────────────────────────────────")

    # 4. Knowledge Base, Bi-Temporal Graph & Search Telemetry
    exp = pg.get("experiences") or {}
    kg = pg.get("knowledge_graph") or {}
    st = pg.get("search_telemetry") or {}
    emb_pct = exp.get("embedding_ratio", 1.0) * 100

    lines.append("┌─ 4. 知识库、双时态图谱与检索指标 (Knowledge Graph & Search Telemetry) ────────")
    lines.append(
        f"│ Experiences:     Active={exp.get('active', 0)} | "
        f"Superseded (History)={exp.get('superseded', 0)} | "
        f"Verified={exp.get('verified', 0)} | Draft/Review={exp.get('draft_or_review', 0)}"
    )
    lines.append(
        f"│ Vector Coverage: {exp.get('embedding_covered', 0)}/{exp.get('embedding_total', 0)} ({emb_pct:.0f}%) | "
        f"Archives={pg.get('archives_total', 0)} | BG Tasks={pg.get('background_tasks_pending', 0)}"
    )
    lines.append(
        f"│ Knowledge Graph: Entity Nodes={kg.get('entity_nodes', 0)} | "
        f"Relations (Edges)={kg.get('entity_relations', 0)} | Projects={kg.get('projects', 0)}"
    )
    lines.append(
        f"│ Search ({st.get('days', 7)}d):     Searches={st.get('total_searches', 0)} | "
        f"Hit={st.get('hit_rate', 0.0) * 100:.0f}% | Use={st.get('use_rate', 0.0) * 100:.1f}% | "
        f"Faithfulness={st.get('faithfulness_avg', 0.0):.2f}"
    )
    lines.append(
        f"│ Pipeline Perf:   Avg Latency={st.get('avg_total_ms', 0.0):.1f}ms | "
        f"Avg Tokens={st.get('avg_returned_tokens', 0.0):.0f} | "
        f"2-Hop Graph Assisted={st.get('graph_assisted_searches', 0)}"
    )
    rec_exps = pg.get("recent_experiences") or []
    if rec_exps:
        lines.append("│ Latest Published Experiences:")
        for e in rec_exps[:3]:
            lines.append(f"│   • [{e['project']:<14}] {e['id']}  {e['title']} ({e['created_at']})")
    lines.append("└───────────────────────────────────────────────────────────────────────────────")

    # 5. Recent Recalls List (numbered for interactive / CLI trace selection)
    rec_recalls = pg.get("recent_recalls") or []
    lines.append("┌─ 5. 最近召回记录列表 (Recent Recalls — 选择序号查看完整召回链路图) ──────────")
    if not rec_recalls:
        lines.append("│ (暂无召回记录 / No search logs found)")
    else:
        lines.append(
            "│ [#]   Time                Source      Project          "
            "4-Way(V/F/T/G) → Hit  Latency   Adopted  Query / Top Hit"
        )
        lines.append(
            "│ ────  ──────────────────  ──────────  ───────────────  "
            "────────────────────  ────────  ───────  ────────────────────────────"
        )
        for r in rec_recalls:
            idx = r.get("index", 0)
            sc = r.get("stage_counts") or {}
            sm = r.get("stage_metrics") or {}
            hit_n = r.get("hit_count", 0)
            if sc:
                v_n = sc.get("vec_count", 0)
                f_n = sc.get("fts_count", 0)
                t_n = sc.get("topic_count", 0)
                g_n = sc.get("graph_count", 0)
                way_str = f"{v_n}/{f_n}/{t_n}/{g_n} → {hit_n}"
            else:
                way_str = f"-/-/-/- → {hit_n}"

            lat_val = sm.get("total_ms")
            lat_str = f"{float(lat_val):.0f}ms" if lat_val is not None else "-"
            used_str = "✅Yes" if r.get("was_used") else ("🚫Noise" if r.get("is_noise") else "⏳-")
            q_disp = r.get("query_display") or _clean_query_display(r.get("query", ""), 42)
            top_title = r.get("top_hit_title") or ""
            top_suffix = f"  🎯 {top_title[:28]}" if top_title else ""
            lines.append(
                f"│ [{idx:<2}]  {r.get('created_at', '')[:19]:<18}  "
                f"{r.get('source', 'mcp')[:10]:<10}  "
                f"{r.get('project', 'default')[:15]:<15}  "
                f"{way_str:<20}  {lat_str:<8}  {used_str:<7}  {q_disp}{top_suffix}"
            )
        lines.append(
            "│ 💡 查看全链路节点图: `tm-cli status <序号>` | 查看更多条数: `tm-cli status -n 30` 或交互输入 `m`"
        )
    lines.append("└───────────────────────────────────────────────────────────────────────────────")

    return "\n".join(lines)


def build_rich_renderable(snapshot: dict[str, Any], watch_meta: dict[str, Any] | None = None) -> Any:
    """Build a Rich Group renderable for one-shot or Live TUI display."""
    from rich.columns import Columns
    from rich.console import Group
    from rich.panel import Panel
    from rich.table import Table
    from rich.text import Text

    ts = snapshot.get("timestamp", "")
    overall = snapshot.get("overall_status", "unknown").upper()
    overall_style = {"HEALTHY": "bold green", "DEGRADED": "bold yellow", "UNHEALTHY": "bold red"}.get(
        overall, "bold white"
    )
    overall_icon = {"HEALTHY": "🟢", "DEGRADED": "🟡", "UNHEALTHY": "🔴"}.get(overall, "⚪")

    header_text = Text()
    header_text.append(f" {overall_icon} TeamMemory Status: {overall} ", style=overall_style)
    header_text.append(f"  |  Updated: {ts}", style="cyan")
    if watch_meta:
        tick = watch_meta.get("tick", 1)
        interval = watch_meta.get("interval", 2.0)
        header_text.append(f"  |  🔄 Live #{tick} ({interval:g}s) — Press Ctrl+C to exit", style="bold magenta")

    anomalies = snapshot.get("anomalies") or []
    if anomalies:
        header_text.append("\n ⚠ Anomalies: " + "  •  ".join(anomalies), style="bold red")

    # Panel 1: Core Server & Infra
    web = snapshot.get("core_server", {})
    pg = snapshot.get("knowledge_base", {})
    checks = web.get("checks") or {}
    db_chk = checks.get("database") or {}
    emb_chk = checks.get("embedding_health") or checks.get("embedding_provider") or {}
    llm_chk = checks.get("llm_provider") or {}
    cache_chk = checks.get("cache") or {}

    web_ok = bool(web.get("reachable")) and web.get("status") in ("healthy", "degraded", "up")
    db_ok = (db_chk.get("status") == "up") or bool(pg.get("reachable"))

    t_infra = Table.grid(padding=(0, 2))
    t_infra.add_column(style="bold")
    t_infra.add_column()
    t_infra.add_column(style="dim")

    t_infra.add_row(
        "Web / MCP (:9111)",
        "[green]● UP[/green]" if web_ok else "[red]● OFFLINE[/red]",
        f"v{web.get('version', '?')} | {web.get('latency_ms', '-')} ms",
    )
    t_infra.add_row(
        "PostgreSQL + pgvector",
        "[green]● UP[/green]" if db_ok else "[red]● DOWN[/red]",
        f"latency: {db_chk.get('latency_ms', 'direct OK' if db_ok else '-')} ms",
    )
    if web_ok:
        emb_up = emb_chk.get("status") == "up"
        llm_up = llm_chk.get("status") == "up"
        t_infra.add_row(
            "Embedding Service",
            "[green]● UP[/green]" if emb_up else "[yellow]● IDLE/WARN[/yellow]",
            f"{emb_chk.get('provider', '?')} ({emb_chk.get('latency_ms', '-')} ms)",
        )
        t_infra.add_row(
            "LLM Provider",
            "[green]● UP[/green]" if llm_up else "[red]● DOWN[/red]",
            f"{llm_chk.get('provider', '?')} {llm_chk.get('model', '')}".strip(),
        )
        t_infra.add_row(
            "Search Cache",
            "[green]● UP[/green]" if cache_chk.get("status") == "up" else "[yellow]● NONE[/yellow]",
            f"backend: {cache_chk.get('backend', 'none')}",
        )
    else:
        t_infra.add_row("Sub-checks", "[dim]Skipped (Web offline)[/dim]", "Run: make web")

    p_infra = Panel(t_infra, title="1. 核心服务与基础设施 (Core & Infra)", border_style="blue")

    # Panel 2: TM Daemon & Hooks
    dm = snapshot.get("daemon", {})
    dm_ok = bool(dm.get("reachable"))
    rw = dm.get("refinement_worker") or {}
    rw_ok = bool(rw.get("healthy", False)) if dm_ok else False
    hb = dm.get("heartbeat") or {}
    loc = snapshot.get("pipeline_local", {})
    cap = loc.get("hooks_capture") or {}
    latest_ev = cap.get("latest_event")

    t_daemon = Table.grid(padding=(0, 2))
    t_daemon.add_column(style="bold")
    t_daemon.add_column()
    t_daemon.add_column(style="dim")

    t_daemon.add_row(
        "TM Daemon (:3901)",
        "[green]● RUNNING[/green]" if dm_ok else "[red]● OFFLINE[/red]",
        f"mode: {dm.get('tm_mode', '-')} | {dm.get('latency_ms', '-')} ms",
    )
    t_daemon.add_row(
        "RefinementWorker",
        "[green]● HEALTHY[/green]" if rw_ok else "[red]● DOWN/STALLED[/red]",
        f"ticks: {rw.get('tick_count', 0)} | fails: {rw.get('fail_count', 0)}",
    )
    t_daemon.add_row(
        "Last Worker Tick",
        str(rw.get("last_tick_ok") or "never")[:19],
        f"wake recoveries: {hb.get('sleep_wake_count', 0)}",
    )
    t_daemon.add_row(
        "IDE Hook Payloads",
        f"[cyan]{cap.get('total_payloads', 0)} captured[/cyan]",
        (
            f"latest: {latest_ev.get('endpoint')} ({latest_ev.get('created_at', '')[:19]})"
            if latest_ev
            else "no hook events yet"
        ),
    )

    p_daemon = Panel(t_daemon, title="2. 守护进程与 Hook 感知 (Daemon & Hooks)", border_style="magenta")

    # Panel 3: Local Refinement Pipeline Funnel
    by_st = loc.get("sessions_by_status") or {}
    active_n = by_st.get("active", 0) + by_st.get("collecting", 0)
    needs_n = by_st.get("needs_refinement", 0)
    refining_n = by_st.get("refining", 0) + by_st.get("publishing", 0)
    done_n = by_st.get("published", 0) + by_st.get("refined", 0)
    skip_n = by_st.get("skipped", 0) + by_st.get("empty", 0)
    pq = loc.get("pending_queue") or {}

    funnel_summary = Text()
    funnel_summary.append(f"  🟢 Collecting: {active_n} ", style="green")
    funnel_summary.append(f"(Ready≥3 turns: {loc.get('ready_to_converge', 0)})  →  ", style="dim")
    funnel_summary.append(f"🟡 Queued: {needs_n}  →  ", style="yellow")
    funnel_summary.append(f"🔄 Refining: {refining_n}  →  ", style="cyan")
    funnel_summary.append(f"✅ Refined: {done_n} ", style="bold green")
    funnel_summary.append(f"(Skipped: {skip_n})\n", style="dim")
    funnel_summary.append(
        f"  💬 Messages: {loc.get('total_messages', 0)}   "
        f"🔧 Tool Observations: {loc.get('total_tool_observations', 0)}   "
        f"💾 SQLite: {loc.get('db_size_mb', 0.0):.2f} MB   "
        f"⏳ PendingQueue: pending={pq.get('pending', 0)} retrying={pq.get('retrying', 0)} ",
    )
    failed_n = pq.get("failed", 0)
    funnel_summary.append(
        f"failed={failed_n}",
        style="bold red" if failed_n > 0 else "dim",
    )

    rec_sess = loc.get("recent_sessions") or []
    t_sess = Table(show_header=True, header_style="bold cyan", box=None, padding=(0, 1))
    t_sess.add_column("Session ID")
    t_sess.add_column("Project")
    t_sess.add_column("Status")
    t_sess.add_column("Turns (U/A)")
    t_sess.add_column("Tools")
    t_sess.add_column("Updated At")
    for s in rec_sess[:4]:
        st_val = s["status"]
        st_color = {
            "active": "green",
            "collecting": "green",
            "needs_refinement": "yellow",
            "refining": "cyan",
            "publishing": "cyan",
            "published": "bold green",
            "refined": "bold green",
        }.get(st_val, "dim")
        t_sess.add_row(
            s["id"][:16],
            s["project"][:18],
            f"[{st_color}]{st_val}[/{st_color}]",
            f"{s['user_msgs']}/{s['asst_msgs']}",
            str(s["tool_obs"]),
            s["updated_at"][:19],
        )

    p_funnel = Panel(
        Group(funnel_summary, t_sess) if rec_sess else funnel_summary,
        title="3. 本地精炼管线漏斗 (Refinement Pipeline Funnel — SQLite)",
        border_style="yellow",
    )

    # Panel 4: Knowledge Base, Bi-Temporal Graph & Search Telemetry
    exp = pg.get("experiences") or {}
    kg = pg.get("knowledge_graph") or {}
    st = pg.get("search_telemetry") or {}
    emb_pct = exp.get("embedding_ratio", 1.0) * 100

    kb_text = Text()
    kb_text.append(
        f"  📚 Experiences:  Active={exp.get('active', 0)}  |  "
        f"Superseded (Bi-Temporal)={exp.get('superseded', 0)}  |  "
        f"Verified={exp.get('verified', 0)}  |  Draft/Review={exp.get('draft_or_review', 0)}\n",
        style="bold",
    )
    kb_text.append(
        f"  🕸  Knowledge Graph:  Entity Nodes={kg.get('entity_nodes', 0)}  |  "
        f"Relations (Edges)={kg.get('entity_relations', 0)}  |  "
        f"Projects={kg.get('projects', 0)}  |  Archives={pg.get('archives_total', 0)}\n",
        style="cyan",
    )
    kb_text.append(
        f"  🔍 Search ({st.get('days', 7)}d):  Total={st.get('total_searches', 0)}  |  "
        f"Hit={st.get('hit_rate', 0.0) * 100:.0f}%  |  "
        f"Adopted={st.get('use_rate', 0.0) * 100:.1f}%  |  "
        f"Faithfulness={st.get('faithfulness_avg', 0.0):.2f}  |  "
        f"Vector Cov={emb_pct:.0f}%\n",
        style="green",
    )
    kb_text.append(
        f"  ⚡ Pipeline Perf:  Avg Latency={st.get('avg_total_ms', 0.0):.1f} ms  |  "
        f"Avg Returned Tokens={st.get('avg_returned_tokens', 0.0):.0f}  |  "
        f"2-Hop Graph Assisted={st.get('graph_assisted_searches', 0)}",
        style="dim",
    )

    rec_exps = pg.get("recent_experiences") or []
    if rec_exps:
        kb_text.append("\n  🆕 Latest Published: ", style="bold")
        kb_text.append(
            "  •  ".join(f"[{e['project']}] {e['title'][:36]}" for e in rec_exps[:3]),
            style="dim",
        )

    p_kb = Panel(
        kb_text,
        title="4. 知识库、双时态图谱与检索遥测 (Knowledge Graph & Search Telemetry)",
        border_style="green",
    )

    # Panel 5: Recent Recalls List (numbered for interactive / CLI trace selection)
    rec_recalls = pg.get("recent_recalls") or []
    t_recalls = Table(show_header=True, header_style="bold magenta", box=None, padding=(0, 1))
    t_recalls.add_column("#", style="bold yellow", justify="right", no_wrap=True, min_width=4)
    t_recalls.add_column("Time", style="dim", no_wrap=True)
    t_recalls.add_column("Source", no_wrap=True)
    t_recalls.add_column("Project", style="cyan", no_wrap=True)
    t_recalls.add_column("4-Way(V/F/T/G)→Hit", justify="center", no_wrap=True)
    t_recalls.add_column("Perf / State", justify="right", no_wrap=True)
    t_recalls.add_column("Query & Top Hit Experience", ratio=1, min_width=26)

    for r in rec_recalls:
        idx = r.get("index", 0)
        sc = r.get("stage_counts") or {}
        sm = r.get("stage_metrics") or {}
        hit_n = r.get("hit_count", 0)
        if sc:
            v_n = sc.get("vec_count", 0)
            f_n = sc.get("fts_count", 0)
            t_n = sc.get("topic_count", 0)
            g_n = sc.get("graph_count", 0)
            way_str = (
                f"[blue]{v_n}[/blue]/[magenta]{f_n}[/magenta]/"
                f"[yellow]{t_n}[/yellow]/[green]{g_n}[/green]→[bold]{hit_n}[/bold]"
            )
        else:
            way_str = f"[dim]-/-/-/-[/dim]→[bold]{hit_n}[/bold]"

        lat_val = sm.get("total_ms")
        lat_str = f"{float(lat_val):.0f}ms" if lat_val is not None else "-"

        if r.get("is_noise"):
            used_str = "[red]🚫Noise[/red]"
        elif r.get("was_used"):
            used_str = "[bold green]✅Yes[/bold green]"
        else:
            used_str = "[dim]⏳-[/dim]"

        perf_state = f"{lat_str} {used_str}"
        q_disp = r.get("query_display") or _clean_query_display(r.get("query", ""), 42)
        top_title = r.get("top_hit_title") or ""
        q_cell = q_disp
        if top_title:
            q_cell += f"\n[dim]🎯 {top_title[:42]}[/dim]"

        t_recalls.add_row(
            f"[{idx}]",
            r.get("created_at", "")[11:19] or r.get("created_at", "")[:10],
            r.get("source", "mcp")[:10],
            r.get("project", "default")[:14],
            way_str,
            perf_state,
            q_cell,
        )

    hint_text = Text(
        "\n  💡 提示: 运行 `tm-cli status <序号>` (如 `tm-cli status 1`) 查看链路图；"
        "查看展示更多条数可用 `tm-cli status -n 30` 或在下方输入 `m`",
        style="bold yellow",
    )
    p_recalls = Panel(
        Group(t_recalls, hint_text) if rec_recalls else Text("  暂无召回记录 (No search logs found)", style="dim"),
        title=f"5. 最近召回记录列表 (Recent Recalls Top {len(rec_recalls)} — 选中序号查看完整召回链路)",
        border_style="bright_magenta",
    )

    return Group(
        Panel(header_text, border_style="cyan"),
        Columns([p_infra, p_daemon], expand=True),
        p_funnel,
        p_kb,
        p_recalls,
    )


def _draw_bar(count: int, max_val: int, width: int = 16, fill: str = "█", empty: str = "░") -> str:
    """Render an intuitive Unicode horizontal bar for stage funnel comparison."""
    if max_val <= 0 or count <= 0:
        return empty * width
    filled = max(1, min(width, int(round((count / max_val) * width))))
    return (fill * filled) + (empty * (width - filled))


async def inspect_recall_trace(
    selector: str | int,
    recent_recalls: list[dict[str, Any]] | None = None,
    timeout: float = 4.0,
    limit: int = 20,
) -> dict[str, Any]:
    """Inspect a selected recall log and reconstruct its complete 6-node pipeline trace."""
    dsn = _resolve_dsn()
    import asyncpg

    conn = await asyncio.wait_for(asyncpg.connect(dsn), timeout=timeout)
    try:
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
                return {"error": f"未找到对应的召回记录: {selector}"}
            created_dt = row["created_at"]
            created_str = (
                created_dt.astimezone().strftime("%Y-%m-%d %H:%M:%S")
                if hasattr(created_dt, "astimezone")
                else str(created_dt or "")[:19]
            )
            r_ids = _parse_list_field(row["result_ids"])
            target = {
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
        else:
            # Fetch agent_response_snippet if not loaded in list view
            snip = await conn.fetchval(
                "SELECT agent_response_snippet FROM search_logs WHERE id::text = $1",
                target["id"],
            )
            target["agent_response_snippet"] = str(snip or "")

        raw_log_query = str(target.get("query") or "")
        query_raw = _clean_query_display(raw_log_query, 240).removeprefix("[Ctx] ").strip()
        project = str(target.get("project") or "default")
        result_ids = list(target.get("result_ids") or [])

        # Node ②: Query Expansion & Rule-based Entity Extraction
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
        import re as _re

        for short_tok in _re.findall(r"[A-Za-z]{2,}", query_raw):
            if short_tok.lower() in {"js", "ui", "db", "ip", "ad", "ai", "ci", "cd", "os", "gc"}:
                if short_tok.lower() not in {c.lower() for c in candidate_entities}:
                    candidate_entities.append(short_tok)

        # Match extracted candidate entities against PostgreSQL `entities` table
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
                # 1-hop & 2-hop active relationships + linked experiences
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

                # Experiences linked to seed or 1-hop neighbor entities
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

        # Check FTS matches among result_ids / database for channel attribution & fallback counts
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

        # Node ③ & ④: Stage counts and metrics (with intelligent reconstruction if historical None)
        raw_counts = dict(target.get("stage_counts") or {})
        raw_metrics = dict(target.get("stage_metrics") or {})
        counts_reconstructed = False

        if not raw_counts:
            counts_reconstructed = True
            hit_n = len(result_ids)
            vec_est = max(hit_n * 2, 10) if hit_n > 0 else 0
            fts_est = fts_total_estimate
            graph_est = len(graph_exp_ids)
            graph_uniq = len(graph_exp_ids - fts_exp_ids)
            fused_est = max(hit_n, len(set(result_ids) | fts_exp_ids | graph_exp_ids))
            raw_counts = {
                "vec_count": vec_est,
                "fts_count": fts_est,
                "topic_count": 0,
                "graph_count": graph_est,
                "graph_unique_added": graph_uniq,
                "fused_count": fused_est,
                "final_count": hit_n,
            }

        # Node ⑤: Detailed experience cards for result_ids
        recalled_experiences: list[dict[str, Any]] = []
        superseded_hits = 0
        if result_ids:
            exp_rows = await conn.fetch(
                """
                SELECT id::text AS id,
                       title,
                       COALESCE(project, 'default') AS project,
                       COALESCE(exp_status, 'published') AS exp_status,
                       COALESCE(experience_type, 'general') AS experience_type,
                       tags,
                       description,
                       solution,
                       valid_from,
                       valid_until,
                       superseded_by_id::text AS superseded_by_id,
                       COALESCE(used_count, 0) AS use_count,
                       COALESCE(recall_count, 0) AS recall_count,
                       (last_verified_at IS NOT NULL) AS verified
                FROM experiences
                WHERE id::text = ANY($1::text[])
                """,
                result_ids,
            )
            exp_map = {str(er["id"]): er for er in exp_rows}

            missing_arch_ids = [eid for eid in result_ids if eid not in exp_map]
            arch_map: dict[str, Any] = {}
            pm_map: dict[str, Any] = {}
            wiki_map: dict[str, dict[str, Any]] = {}
            if missing_arch_ids:
                try:
                    arch_rows = await conn.fetch(
                        """
                        SELECT id::text AS id,
                               title,
                               COALESCE(project, 'default') AS project,
                               COALESCE(status, 'published') AS exp_status,
                               COALESCE(content_type, 'archive') AS experience_type,
                               tags,
                               COALESCE(overview, value_summary, '') AS description,
                               COALESCE(solution_doc, '') AS solution
                        FROM archives
                        WHERE id::text = ANY($1::text[])
                        """,
                        missing_arch_ids,
                    )
                    arch_map = {str(ar["id"]): ar for ar in arch_rows}
                except Exception:
                    pass

                still_missing = [eid for eid in missing_arch_ids if eid not in arch_map]
                if still_missing:
                    try:
                        pm_rows = await conn.fetch(
                            """
                            SELECT id::text AS id,
                                   user_id,
                                   content,
                                   COALESCE(scope, 'generic') AS scope,
                                   COALESCE(persona, 'general') AS persona,
                                   COALESCE(context_hint, '') AS context_hint
                            FROM personal_memories
                            WHERE id::text = ANY($1::text[])
                            """,
                            still_missing,
                        )
                        pm_map = {str(pr["id"]): pr for pr in pm_rows}
                    except Exception:
                        pass
                    wiki_missing = [eid for eid in still_missing if eid not in pm_map]
                    if wiki_missing:
                        wiki_map = _lookup_local_wiki_entries(wiki_missing)

            # Fetch linked entities per experience
            ee_rows = await conn.fetch(
                """
                SELECT ee.experience_id::text AS exp_id,
                       e.id::text AS entity_id,
                       e.name,
                       e.entity_type,
                       COALESCE(e.source_count, 1) AS source_count,
                       ee.role
                FROM experience_entities ee
                JOIN entities e ON e.id = ee.entity_id
                WHERE ee.experience_id::text = ANY($1::text[])
                ORDER BY e.source_count DESC
                """,
                result_ids,
            )
            exp_entities_map: dict[str, list[str]] = {}
            for eer in ee_rows:
                eid_str = str(eer["exp_id"])
                exp_entities_map.setdefault(eid_str, [])
                ename = str(eer["name"])
                if ename not in exp_entities_map[eid_str] and len(exp_entities_map[eid_str]) < 6:
                    exp_entities_map[eid_str].append(ename)

            # If direct query entity match was empty but graph_count > 0 or experiences have entities,
            # populate active graph paths from the top entities bridged by these results.
            if not matched_entities and ee_rows and int(raw_counts.get("graph_count", 0)) > 0:
                seen_eids: set[str] = set()
                for eer in ee_rows:
                    en_id = str(eer["entity_id"])
                    if en_id not in seen_eids and len(matched_entities) < 4:
                        seen_eids.add(en_id)
                        matched_entities.append(
                            {
                                "id": en_id,
                                "name": str(eer["name"]),
                                "entity_type": str(eer["entity_type"] or "concept"),
                                "project": project,
                                "source_count": int(eer["source_count"] or 1),
                            }
                        )
                if seen_eids and not graph_paths:
                    rel_rows2 = await conn.fetch(
                        """
                        SELECT e1.name AS src_name, r.relation_type, e2.name AS tgt_name,
                               COALESCE(r.weight, 1.0) AS weight
                        FROM relationships r
                        JOIN entities e1 ON e1.id = r.source_entity_id
                        JOIN entities e2 ON e2.id = r.target_entity_id
                        WHERE r.valid_until IS NULL
                          AND (
                              r.source_entity_id::text = ANY($1::text[])
                              OR r.target_entity_id::text = ANY($1::text[])
                          )
                        ORDER BY r.weight DESC
                        LIMIT 6
                        """,
                        list(seen_eids),
                    )
                    for rr in rel_rows2:
                        graph_paths.append(
                            {
                                "src": str(rr["src_name"]),
                                "rel": str(rr["relation_type"]),
                                "tgt": str(rr["tgt_name"]),
                                "weight": round(float(rr["weight"] or 1.0), 2),
                            }
                        )

            for rank, exp_id in enumerate(result_ids, start=1):
                er = exp_map.get(exp_id)
                if not er and exp_id in arch_map:
                    ar = arch_map[exp_id]
                    prob_a = " ".join(str(ar["description"] or "").split())
                    sol_a = " ".join(str(ar["solution"] or "").split())
                    recalled_experiences.append(
                        {
                            "rank": rank,
                            "id": exp_id,
                            "short_id": exp_id[:8],
                            "title": f"[📦L3归档] {ar['title'] or ''}",
                            "project": str(ar["project"] or project),
                            "exp_status": str(ar["exp_status"] or "published"),
                            "experience_type": str(ar["experience_type"] or "archive"),
                            "channels": ["🔵Vector(Archive)"],
                            "is_superseded": False,
                            "verified": False,
                            "tags": [str(t) for t in (ar["tags"] or [])][:6],
                            "entities": [],
                            "problem_snippet": (
                                (prob_a[:120] + "…") if len(prob_a) > 120 else prob_a
                            ),
                            "solution_snippet": (
                                (sol_a[:140] + "…") if len(sol_a) > 140 else sol_a
                            ),
                            "use_count": 0,
                            "recall_count": 1,
                        }
                    )
                    continue
                if not er and exp_id in pm_map:
                    pr = pm_map[exp_id]
                    pm_txt = " ".join(str(pr["content"] or "").split())
                    recalled_experiences.append(
                        {
                            "rank": rank,
                            "id": exp_id,
                            "short_id": exp_id[:8],
                            "title": f"[👤L1个人记忆] {pm_txt[:56]}",
                            "project": f"user:{pr['user_id']}",
                            "exp_status": "active",
                            "experience_type": str(pr["persona"] or "personal"),
                            "channels": ["🟣L1Profile"],
                            "is_superseded": False,
                            "verified": True,
                            "tags": [str(pr["scope"]), str(pr["persona"])],
                            "entities": [],
                            "problem_snippet": str(pr["context_hint"] or "用户个人偏好/全局记忆注入"),
                            "solution_snippet": (
                                (pm_txt[:140] + "…") if len(pm_txt) > 140 else pm_txt
                            ),
                            "use_count": 1,
                            "recall_count": 1,
                        }
                    )
                    continue
                if not er and exp_id in wiki_map:
                    wr = wiki_map[exp_id]
                    w_txt = " ".join(str(wr.get("content") or "").split())
                    recalled_experiences.append(
                        {
                            "rank": rank,
                            "id": exp_id,
                            "short_id": exp_id[:8],
                            "title": f"[📘Wiki卡片] {wr.get('title') or ''}",
                            "project": str(wr.get("project") or project),
                            "exp_status": "published",
                            "experience_type": "wiki_card",
                            "channels": ["📘Wiki(FTS5)"],
                            "is_superseded": False,
                            "verified": True,
                            "tags": list(wr.get("tags") or []),
                            "entities": [],
                            "problem_snippet": str(wr.get("source_url") or "本地 SQLite LLMWiki 知识卡片检索"),
                            "solution_snippet": (
                                (w_txt[:140] + "…") if len(w_txt) > 140 else w_txt
                            ),
                            "use_count": 0,
                            "recall_count": 1,
                        }
                    )
                    continue
                if not er:
                    recalled_experiences.append(
                        {
                            "rank": rank,
                            "id": exp_id,
                            "short_id": exp_id[:8],
                            "title": "(已删除或未找到的经验条目)",
                            "project": project,
                            "channels": ["🔵Vector"],
                            "is_superseded": False,
                            "verified": False,
                            "tags": [],
                            "entities": [],
                            "problem_snippet": "",
                            "solution_snippet": "",
                            "use_count": 0,
                            "recall_count": 0,
                        }
                    )
                    continue

                is_sup = er["valid_until"] is not None or er["superseded_by_id"] is not None
                if is_sup:
                    superseded_hits += 1

                channels = ["🔵Vector"]
                if exp_id in fts_exp_ids:
                    channels.append("🟣FTS")
                if exp_id in graph_exp_ids:
                    channels.append("🟢Graph(2-Hop)")

                tags_val = er["tags"]
                if isinstance(tags_val, str):
                    try:
                        tags_val = json.loads(tags_val)
                    except Exception:
                        tags_val = [tags_val]
                tags_list = [str(t) for t in (tags_val or [])][:6]

                prob_clean = " ".join(str(er["description"] or "").split())
                sol_clean = " ".join(str(er["solution"] or "").split())

                recalled_experiences.append(
                    {
                        "rank": rank,
                        "id": exp_id,
                        "short_id": exp_id[:8],
                        "title": str(er["title"] or ""),
                        "project": str(er["project"] or "default"),
                        "exp_status": str(er["exp_status"] or "published"),
                        "experience_type": str(er["experience_type"] or "general"),
                        "channels": channels,
                        "is_superseded": is_sup,
                        "superseded_by": str(er["superseded_by_id"] or "")[:8],
                        "verified": bool(er["verified"]),
                        "tags": tags_list,
                        "entities": exp_entities_map.get(exp_id, []),
                        "problem_snippet": (
                            (prob_clean[:120] + "…") if len(prob_clean) > 120 else prob_clean
                        ),
                        "solution_snippet": (
                            (sol_clean[:140] + "…") if len(sol_clean) > 140 else sol_clean
                        ),
                        "use_count": int(er["use_count"] or 0),
                        "recall_count": int(er["recall_count"] or 0),
                    }
                )

        # Estimate token count if historical row had returned_tokens=None
        ret_tokens = target.get("returned_tokens")
        tokens_estimated = False
        if ret_tokens is None:
            tokens_estimated = True
            total_chars = sum(
                len(x.get("title", ""))
                + len(x.get("problem_snippet", ""))
                + len(x.get("solution_snippet", ""))
                for x in recalled_experiences
            )
            ret_tokens = (
                max(60 * len(recalled_experiences), int(total_chars * 0.6))
                if recalled_experiences
                else 0
            )

        return {
            "log": target,
            "query_analysis": {
                "raw_query": query_raw,
                "expanded_query": expanded_query,
                "added_synonyms": added_synonyms,
                "candidate_entities": candidate_entities,
                "matched_entities": matched_entities,
                "graph_paths": graph_paths,
            },
            "stage_counts": raw_counts,
            "counts_reconstructed": counts_reconstructed,
            "stage_metrics": raw_metrics,
            "superseded_hits": superseded_hits,
            "returned_tokens": ret_tokens,
            "tokens_estimated": tokens_estimated,
            "recalled_experiences": recalled_experiences,
        }
    finally:
        await conn.close()


def render_recall_trace_plain(trace: dict[str, Any]) -> str:
    """Render the complete recall pipeline trace as human-intuitive ASCII flowchart + 6-node narrative."""
    if "error" in trace:
        return f"❌ {trace['error']}"

    log = trace["log"]
    qa = trace["query_analysis"]
    sc = trace["stage_counts"]
    sm = trace["stage_metrics"]
    exps = trace["recalled_experiences"]

    vec_n = int(sc.get("vec_count", 0))
    fts_n = int(sc.get("fts_count", 0))
    top_n = int(sc.get("topic_count", 0))
    grp_n = int(sc.get("graph_count", 0))
    grp_u = int(sc.get("graph_unique_added", 0))
    fused_n = int(sc.get("fused_count", 0))
    final_n = int(sc.get("final_count", len(exps)))

    emb_ms = sm.get("embedding_ms", "-")
    fuse_ms = sm.get("retrieve_fuse_ms", "-")
    boost_ms = sm.get("popularity_boost_ms", "-")
    rerank_ms = sm.get("rerank_ms", "-")
    filt_ms = sm.get("adaptive_filter_ms", "-")
    total_ms = sm.get("total_ms", "-")

    ret_tok = trace.get("returned_tokens", 0)
    cand_ents = qa.get("candidate_entities") or []
    match_ents = qa.get("matched_entities") or []
    g_paths = qa.get("graph_paths") or []
    added_syns = qa.get("added_synonyms") or []

    noise_label = "🚫 拦截(Noise)" if log.get("is_noise") else "🟢 放行(Pass)"
    used_label = "✅ 已采纳(Adopted)" if log.get("was_used") else "⏳ 未标记采纳"
    faith_val = log.get("faithfulness_score")
    faith_str = f"{faith_val:.2f}" if faith_val is not None else "-"

    lines: list[str] = []
    lines.append("═" * 92)
    lines.append(
        f" 🔎 TeamMemory 召回全链路深度追踪 (Recall Pipeline Trace)  "
        f"[#{log.get('index', '?')} | Log ID: {log.get('short_id')}]"
    )
    lines.append(
        f" 🕒 时间: {log.get('created_at')}  |  📍 入口: {log.get('source')}  |  "
        f"📁 项目: {log.get('project')}  |  ⏱ 总耗时: {total_ms} ms"
    )
    lines.append("═" * 92)
    lines.append("")

    # ── Part A: Visual End-to-End ASCII Pipeline Flowchart ──
    lines.append("【一、全链路可视化拓扑图 (End-to-End Recall Pipeline Flowchart)】")
    lines.append("")
    q_short = _clean_query_display(qa.get("raw_query", ""), 46)
    lines.append("  ┌─────────────────────────────────────────────────────────────────────────┐")
    lines.append("  │ ① 入口与意图路由 (Entry & Intent Gate)                                  │")
    lines.append(f'  │   Query: "{q_short}"')
    lines.append(
        f"  │   Source: {log.get('source'):<10} | Project: {log.get('project'):<12} | "
        f"Intent: {log.get('intent_type'):<8} | {noise_label}"
    )
    lines.append("  └────────────────────────────────────┬────────────────────────────────────┘")
    lines.append("                                       ▼")
    lines.append("  ┌─────────────────────────────────────────────────────────────────────────┐")
    lines.append(
        f"  │ ② 查询扩展、实体识别与向量化 (Query Expansion & Embedding) [{str(emb_ms):>4}ms]   │"
    )
    lines.append(
        f"  │   同义词扩展: +{len(added_syns)} 词  |  规则提取实体: {len(cand_ents)} 个  |  "
        f"命中图谱锚点: {len(match_ents)} 个        │"
    )
    lines.append("  └──────┬──────────────────┬───────────────────┬───────────────────┬───────┘")
    lines.append("         │                  │                   │                   │")
    lines.append("         ▼                  ▼                   ▼                   ▼")
    lines.append("  ┌──────────────┐   ┌──────────────┐   ┌──────────────┐   ┌────────────────┐")
    lines.append("  │ ③-A 向量检索 │   │ ③-B 全文检索 │   │ ③-C 主题聚类 │   │ ③-D 2-Hop 图谱 │")
    lines.append("  │ pgvector HNSW│   │  tsvector/词 │   │ Topic Cluster│   │ Degree-PPR扩散 │")
    lines.append(
        f"  │  命中: {vec_n:>3} 条 │   │  命中: {fts_n:>3} 条 │   "
        f"│  命中: {top_n:>3} 条 │   │ 命中:{grp_n:>2}(独+{grp_u:<2}) │"
    )
    lines.append("  └──────┬───────┘   └──────┬───────┘   └──────┬───────┘   └────────┬───────┘")
    lines.append("         └──────────────────┴─────────┬────────┴────────────────────┘")
    lines.append("                                      ▼")
    lines.append("  ┌─────────────────────────────────────────────────────────────────────────┐")
    lines.append("  │ ④ 多路 RRF 融合、双时态降权与断崖过滤 (Fusion, Bi-Temporal & Elbow)     │")
    lines.append(
        f"  │   RRF合并池: {fused_n:>2}条 ──► 热度加权({boost_ms}ms) ──► "
        f"Rerank({rerank_ms}ms) ──► 断崖截断: {final_n:>2}条 │"
    )
    lines.append("  └────────────────────────────────────┬────────────────────────────────────┘")
    lines.append("                                       ▼")
    lines.append("  ┌─────────────────────────────────────────────────────────────────────────┐")
    lines.append("  │ ⑤ 渐进式披露组装与 Prompt 注入 (Progressive Disclosure)                 │")
    lines.append(
        f"  │   最终注入经验: {final_n:>2} 条  |  消耗 Token: ~{ret_tok}  |  "
        f"双时态降权(0.3x): {trace.get('superseded_hits', 0)} 条      │"
    )
    lines.append("  └────────────────────────────────────┬────────────────────────────────────┘")
    lines.append("                                       ▼")
    lines.append("  ┌─────────────────────────────────────────────────────────────────────────┐")
    lines.append("  │ ⑥ 事后采纳与忠实度评估闭环 (Post-Recall Adoption & Faithfulness)        │")
    lines.append(
        f"  │   状态: {used_label:<16} | 判定源: {log.get('judgment_source') or '-':<8} | "
        f"忠实度(Faithfulness): {faith_str:<5} │"
    )
    lines.append("  └─────────────────────────────────────────────────────────────────────────┘")
    lines.append("")

    # ── Part B: Node-by-Node Detailed Breakdown & Human-Intuitive Explanation ──
    lines.append("【二、逐节点详情拆解与诊断说明 (Node-by-Node Detailed Breakdown)】")
    lines.append("")

    # Node 1
    lines.append("─── 节点 ①：入口与意图路由 (Entry & Intent Gate) ───────────────────────────────────")
    lines.append(f"  • 原始查询 (Query):   {qa.get('raw_query', '').strip()}")
    lines.append(
        f"  • 调用来源 (Source):  {log.get('source')}   |   项目作用域 (Project): {log.get('project')}   |   "
        f"意图分类 (Intent): {log.get('intent_type')}"
    )
    if log.get("is_noise"):
        noise_rsn = log.get("noise_reason") or "chit-chat/noise"
        lines.append(f"  • 噪音门禁 (Gate):    🚫 已拦截 (原因: {noise_rsn})")
    else:
        lines.append(
            "  • 噪音门禁 (Gate):    🟢 正常放行 — 非闲聊/确认类噪音，允许进入语义与图谱检索管线"
        )
    lines.append("")

    # Node 2
    lines.append("─── 节点 ②：查询扩展、实体提取与向量化 (Query Expansion & Entity Extraction) ────────")
    syn_desc = ("+" + ", ".join(added_syns)) if added_syns else "无额外同义词触发 (保持原查询语义)"
    lines.append(f"  • 同义词扩展:         {syn_desc}")
    cand_desc = ", ".join(cand_ents) if cand_ents else "(未提取到驼峰/缩写/技术词典实体)"
    lines.append(f"  • 规则提取候选实体:   {cand_desc}")
    if match_ents:
        ent_badges = [
            f"{m['name']} ({m['entity_type']}, 关联度={m['source_count']})" for m in match_ents
        ]
        lines.append(f"  • 知识图谱命中锚点:   🎯 {', '.join(ent_badges)}")
    else:
        lines.append("  • 知识图谱命中锚点:   ⚪ 未命中图谱已有实体节点 (本次依赖向量+全文双路检索)")
    lines.append(f"  • 向量编码耗时:       {emb_ms} ms")
    lines.append("")

    # Node 3
    max_c = max(vec_n, fts_n, top_n, grp_n, fused_n, 1)
    recon_note = (
        " (注: 历史日志未落盘分路计数，已根据当前库实时还原)"
        if trace.get("counts_reconstructed")
        else ""
    )
    lines.append(f"─── 节点 ③：四路并行召回漏斗对比 (4-Way Parallel Retrieval){recon_note} ────────")
    lines.append(
        f"  • 🔵 向量语义路 (Vector HNSW):   {_draw_bar(vec_n, max_c)}  {vec_n:>3} 条  "
        "(捕捉语义相似与跨表述经验)"
    )
    lines.append(
        f"  • 🟣 全文关键词 (FTS tsvector):  {_draw_bar(fts_n, max_c)}  {fts_n:>3} 条  "
        "(精确命中报错名/函数名/配置项)"
    )
    lines.append(
        f"  • 🟡 主题聚类路 (Topic Cluster): {_draw_bar(top_n, max_c)}  {top_n:>3} 条  "
        "(按高频主题簇补充上下文)"
    )
    lines.append(
        f"  • 🟢 2-Hop 图谱 (Entity PPR):    {_draw_bar(grp_n, max_c)}  {grp_n:>3} 条  "
        f"(图谱独有增补: +{grp_u} 条跨跳关联经验)"
    )
    if g_paths:
        lines.append("  • 🕸  激活的 1~2 跳实体关系路径 (Active Graph Paths):")
        for gp in g_paths[:5]:
            lines.append(f"       [{gp['src']}] ──({gp['rel']}, w={gp['weight']})──► [{gp['tgt']}]")
    lines.append("")

    # Node 4
    lines.append("─── 节点 ④：多路 RRF 融合、双时态降权与断崖截断 (RRF Fusion & Adaptive Filter) ────")
    lines.append(
        f"  • 漏斗收敛轨迹:       四路去重合并池 {fused_n} 条  ──►  "
        f"自适应断崖过滤 (Elbow Detection)  ──►  最终保留 {final_n} 条"
    )
    lines.append(
        f"  • 分阶段耗时拆解:     检索+RRF融合={fuse_ms}ms  |  热度/标题加权={boost_ms}ms  |  "
        f"重排(Rerank)={rerank_ms}ms  |  断崖截断={filt_ms}ms"
    )
    sup_n = trace.get("superseded_hits", 0)
    if sup_n > 0:
        lines.append(
            f"  • 双时态演进保护:     ⚠ 检测到 {sup_n} 条已演进旧经验 (valid_until IS NOT NULL)，"
            "已自动施加 0.3x 降权并打标"
        )
    else:
        lines.append(
            "  • 双时态演进保护:     ✅ 命中的经验均为当前最新有效版本 (Active Bi-Temporal State)"
        )
    lines.append("")

    # Node 5
    tok_note = " (估算)" if trace.get("tokens_estimated") else ""
    lines.append(
        f"─── 节点 ⑤：最终命中经验逐条拆解 "
        f"(Recalled Experiences — 共 {len(exps)} 条, ~{ret_tok}{tok_note} tokens) ───"
    )
    if not exps:
        lines.append("  ⚪ 本次召回未返回任何经验条目 (0 hits)。")
        lines.append(
            "     诊断建议: 检查查询词是否过于宽泛、项目名隔离是否正确，或知识库中尚无相关领域沉淀。"
        )
    else:
        for exp in exps:
            ch_str = " + ".join(exp["channels"])
            ver_badge = " [✓已验证]" if exp.get("verified") else ""
            sup_badge = (
                f" [⚠已演进→{exp.get('superseded_by')}]"
                if exp.get("is_superseded")
                else " [🟢现行有效]"
            )
            lines.append(f"  ┌─ Top #{exp['rank']}  {exp['title']}{ver_badge}{sup_badge}")
            lines.append(
                f"  │  ID: {exp['id']}  |  Project: {exp['project']}  |  "
                f"命中通道: {ch_str}  |  历史复用: {exp['use_count']}/{exp['recall_count']} 次"
            )
            if exp.get("tags") or exp.get("entities"):
                tags_s = ", ".join(exp.get("tags") or []) or "-"
                ents_s = ", ".join(exp.get("entities") or []) or "-"
                lines.append(f"  │  标签(Tags): {tags_s}   |   图谱实体(Entities): {ents_s}")
            if exp.get("problem_snippet"):
                lines.append(f"  │  问题摘要: {exp['problem_snippet']}")
            if exp.get("solution_snippet"):
                lines.append(f"  └─ 方案摘要: {exp['solution_snippet']}")
            else:
                lines.append("  └─ (无方案摘要)")
    lines.append("")

    # Node 6
    lines.append("─── 节点 ⑥：事后采纳与忠实度评估闭环 (Post-Recall Adoption & Faithfulness) ──────────")
    judg_src = log.get("judgment_source") or "尚无事后回复触发评估"
    lines.append(f"  • 采纳状态 (was_used): {used_label}   |   判定来源: {judg_src}")
    rel_val = log.get("relevance_score")
    help_val = log.get("helpfulness_score")
    lines.append(
        f"  • 三维质量评分:        忠实度(Faithfulness)={faith_str}  |  "
        f"相关性(Relevance)={rel_val if rel_val is not None else '-'}  |  "
        f"有用性(Helpfulness)={help_val if help_val is not None else '-'}"
    )
    snip = log.get("agent_response_snippet") or ""
    if snip:
        snip_clean = " ".join(snip.split())[:140]
        lines.append(f'  • Agent 回复抽样:      "{snip_clean}…"')
    else:
        lines.append(
            "  • 诊断说明:            当会话停止 (Stop Hook) 或调用 memory_submit_response 时，"
            "系统会自动比对 Agent 最终回复与召回经验，回填采纳率与忠实度。"
        )
    lines.append("═" * 92)

    return "\n".join(lines)


def build_recall_trace_rich(trace: dict[str, Any]) -> Any:
    """Build a Rich renderable for the selected recall's complete pipeline trace."""
    from rich.panel import Panel
    from rich.text import Text

    plain_text = render_recall_trace_plain(trace)
    if "error" in trace:
        return Panel(Text(plain_text, style="bold red"), title="Recall Trace Error", border_style="red")

    log = trace["log"]
    title = (
        f"🔎 召回全链路深度追踪 (Recall Pipeline Trace) — "
        f"[#{log.get('index', '?')} | Log ID: {log.get('short_id')} | Project: {log.get('project')}]"
    )
    return Panel(
        Text(plain_text),
        title=title,
        border_style="bright_cyan",
    )


def _make_console() -> Any:
    """Create a Rich Console with comfortable minimum width even in 80-col PTYs/pipes."""
    import shutil

    from rich.console import Console

    term_w = shutil.get_terminal_size((120, 40)).columns
    width = max(term_w, 118)
    return Console(width=width, _environ={})


def _print_trace_output(trace: dict[str, Any], as_json: bool = False) -> None:
    """Print recall trace as JSON or full-width ASCII flowchart + node breakdown."""
    if as_json:
        print(json.dumps(trace, indent=2, ensure_ascii=False))
        return
    print(render_recall_trace_plain(trace))


async def run_status_monitor(
    *,
    watch: bool = False,
    interval: float = 2.0,
    as_json: bool = False,
    days: int = 7,
    limit: int = 12,
    server_url: str | None = None,
    daemon_url: str | None = None,
    db_path: str | None = None,
    max_ticks: int | None = None,
    trace_target: str | None = None,
    interactive: bool | None = None,
) -> int:
    """Execute one-shot status, recall pipeline trace, or continuous live status monitor."""
    interval = max(0.5, float(interval))
    current_limit = max(1, int(limit))

    if trace_target:
        trace = await inspect_recall_trace(trace_target, limit=max(20, current_limit))
        _print_trace_output(trace, as_json=as_json)
        return 0 if "error" not in trace else 1

    if not watch:
        snapshot = await collect_status_snapshot(
            server_url=server_url,
            daemon_url=daemon_url,
            db_path=db_path,
            days=days,
            limit=current_limit,
        )
        if as_json:
            print(json.dumps(snapshot, indent=2, ensure_ascii=False))
            return 0 if snapshot.get("overall_status") != "unhealthy" else 1

        try:
            _make_console().print(build_rich_renderable(snapshot))
        except ImportError:
            print(render_status_plain(snapshot))

        rec_recalls = (snapshot.get("knowledge_base") or {}).get("recent_recalls") or []
        should_prompt = (
            interactive
            if interactive is not None
            else (sys.stdin.isatty() and sys.stdout.isatty() and bool(rec_recalls))
        )
        if should_prompt and rec_recalls:
            while True:
                max_idx = len(rec_recalls)
                try:
                    prompt_msg = (
                        f"\n👉 输入召回序号 [1-{max_idx}]、Log ID，或输入 m 加载更多"
                        " (直接回车 / q 退出): "
                    )
                    choice = input(prompt_msg).strip()
                except (EOFError, KeyboardInterrupt):
                    print()
                    break
                if not choice or choice.lower() in ("q", "quit", "exit"):
                    break
                low_c = choice.lower()
                if low_c in ("m", "more") or low_c.startswith(("m ", "more ", "+")):
                    parts = low_c.replace("+", " ").split()
                    if len(parts) >= 2 and parts[-1].isdigit():
                        added_or_target = int(parts[-1])
                        current_limit = (
                            current_limit + added_or_target
                            if choice.startswith("+")
                            else max(current_limit + 10, added_or_target)
                        )
                    else:
                        current_limit += 20
                    snapshot = await collect_status_snapshot(
                        server_url=server_url,
                        daemon_url=daemon_url,
                        db_path=db_path,
                        days=days,
                        limit=current_limit,
                    )
                    rec_recalls = (snapshot.get("knowledge_base") or {}).get("recent_recalls") or []
                    try:
                        _make_console().print(build_rich_renderable(snapshot))
                    except ImportError:
                        print(render_status_plain(snapshot))
                    continue

                trace = await inspect_recall_trace(choice, recent_recalls=rec_recalls, limit=current_limit)
                print()
                _print_trace_output(trace, as_json=False)

        return 0 if snapshot.get("overall_status") != "unhealthy" else 1

    # Watch mode (continuous live dashboard)
    tick = 0
    if as_json:
        try:
            while max_ticks is None or tick < max_ticks:
                tick += 1
                snapshot = await collect_status_snapshot(
                    server_url=server_url,
                    daemon_url=daemon_url,
                    db_path=db_path,
                    days=days,
                    limit=current_limit,
                )
                print(json.dumps(snapshot, ensure_ascii=False), flush=True)
                if max_ticks is not None and tick >= max_ticks:
                    break
                await asyncio.sleep(interval)
        except KeyboardInterrupt:
            pass
        return 0

    try:
        from rich.console import Console
        from rich.live import Live

        console = Console()
        first_snapshot = await collect_status_snapshot(
            server_url=server_url,
            daemon_url=daemon_url,
            db_path=db_path,
            days=days,
            limit=current_limit,
        )
        tick = 1
        with Live(
            build_rich_renderable(first_snapshot, {"tick": tick, "interval": interval}),
            console=console,
            refresh_per_second=2,
            screen=False,
        ) as live:
            while max_ticks is None or tick < max_ticks:
                await asyncio.sleep(interval)
                tick += 1
                snapshot = await collect_status_snapshot(
                    server_url=server_url,
                    daemon_url=daemon_url,
                    db_path=db_path,
                    days=days,
                    limit=current_limit,
                )
                live.update(build_rich_renderable(snapshot, {"tick": tick, "interval": interval}))
        return 0
    except ImportError:
        try:
            while max_ticks is None or tick < max_ticks:
                tick += 1
                snapshot = await collect_status_snapshot(
                    server_url=server_url,
                    daemon_url=daemon_url,
                    db_path=db_path,
                    days=days,
                    limit=current_limit,
                )
                sys.stdout.write("\033[2J\033[H")
                sys.stdout.write(render_status_plain(snapshot, {"tick": tick, "interval": interval}) + "\n")
                sys.stdout.flush()
                if max_ticks is not None and tick >= max_ticks:
                    break
                await asyncio.sleep(interval)
        except KeyboardInterrupt:
            pass
        return 0
    except KeyboardInterrupt:
        return 0

"""Runtime probes, SQLite inspection, Wiki lookup, and snapshot collection for status_monitor."""

from __future__ import annotations

import asyncio
import os
import sqlite3
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import httpx

from team_memory.daemon.observe.status_monitor.wiki_utils import (  # noqa: F401
    _WIKI_DB_CANDIDATES,
    _clean_query_display,
    _lookup_local_wiki_entries,
    _parse_json_field,
    _parse_list_field,
    _pkg_attr,
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

    resolved_db = os.path.expanduser(resolved_db)
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
                "maintenance": data.get("maintenance", {}),
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


def _sqlite_total_size_bytes(db_path: Path) -> tuple[int, int]:
    """Return (main_bytes, wal_plus_shm_bytes) for a SQLite database."""
    main_b = db_path.stat().st_size if db_path.exists() else 0
    wal_p = Path(f"{db_path}-wal")
    shm_p = Path(f"{db_path}-shm")
    wal_b = (wal_p.stat().st_size if wal_p.exists() else 0) + (
        shm_p.stat().st_size if shm_p.exists() else 0
    )
    return main_b, wal_b


def _inspect_local_sqlite(drafts_db_path: str, capture_db_path: str) -> dict[str, Any]:
    """Inspect local SQLite ConversationBuffer, PendingQueue, and PayloadCapture."""
    resolved_drafts = os.path.expanduser(drafts_db_path)
    resolved_capture = os.path.expanduser(capture_db_path)
    result: dict[str, Any] = {
        "db_path": resolved_drafts,
        "db_exists": False,
        "db_size_mb": 0.0,
        "db_main_size_mb": 0.0,
        "wal_size_mb": 0.0,
        "capture_db_size_mb": 0.0,
        "total_sqlite_size_mb": 0.0,
        "sessions_by_status": {},
        "total_sessions": 0,
        "ready_to_converge": 0,
        "stuck_no_assistant": 0,
        "total_messages": 0,
        "total_tool_observations": 0,
        "total_metrics": 0,
        "total_drafts": 0,
        "pending_queue": {"pending": 0, "retrying": 0, "failed": 0},
        "recent_sessions": [],
        "hooks_capture": {
            "db_exists": False,
            "db_size_mb": 0.0,
            "total_payloads": 0,
            "by_endpoint": {},
            "latest_event": None,
        },
    }

    p_drafts = Path(resolved_drafts)
    if p_drafts.exists():
        result["db_exists"] = True
        try:
            main_b, wal_b = _sqlite_total_size_bytes(p_drafts)
            result["db_main_size_mb"] = round(main_b / (1024 * 1024), 2)
            result["wal_size_mb"] = round(wal_b / (1024 * 1024), 2)
            result["db_size_mb"] = round((main_b + wal_b) / (1024 * 1024), 2)
        except OSError:
            pass

        try:
            conn = sqlite3.connect(f"file:{resolved_drafts}?mode=ro", uri=True, timeout=2.0)
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

                if "metrics" in tables:
                    result["total_metrics"] = int(
                        conn.execute("SELECT COUNT(*) FROM metrics").fetchone()[0] or 0
                    )
                if "drafts" in tables:
                    result["total_drafts"] = int(
                        conn.execute("SELECT COUNT(*) FROM drafts").fetchone()[0] or 0
                    )
            finally:
                conn.close()
        except Exception as exc:
            result["error"] = str(exc)[:120]
    else:
        result["error"] = f"SQLite DB not found: {resolved_drafts}"

    p_cap = Path(resolved_capture)
    if p_cap.exists():
        result["hooks_capture"]["db_exists"] = True
        try:
            cap_main_b, cap_wal_b = _sqlite_total_size_bytes(p_cap)
            cap_mb = round((cap_main_b + cap_wal_b) / (1024 * 1024), 2)
            result["capture_db_size_mb"] = cap_mb
            result["hooks_capture"]["db_size_mb"] = cap_mb
        except OSError:
            pass
        try:
            conn = sqlite3.connect(f"file:{resolved_capture}?mode=ro", uri=True, timeout=2.0)
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

    result["total_sqlite_size_mb"] = round(
        float(result["db_size_mb"]) + float(result["capture_db_size_mb"]),
        2,
    )
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
    from team_memory.daemon.observe.status_monitor.pg_collector import (
        _inspect_postgres_knowledge as _default_pg_inspect,
    )

    s_url, d_url, drafts_db, capture_db = _resolve_endpoints_and_paths(
        server_url=server_url,
        daemon_url=daemon_url,
        db_path=db_path,
    )

    probe_web = _pkg_attr("_probe_web_health", _probe_web_health)
    probe_daemon = _pkg_attr("_probe_daemon_status", _probe_daemon_status)
    inspect_pg = _pkg_attr("_inspect_postgres_knowledge", _default_pg_inspect)

    web_task = probe_web(s_url)
    daemon_task = probe_daemon(d_url)
    pg_task = inspect_pg(days=days, limit=limit)
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

    if not sqlite_info.get("db_exists") or sqlite_info.get("error"):
        anomalies.append(
            f"Local SQLite unavailable: {sqlite_info.get('error') or sqlite_info.get('db_path')}"
        )

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

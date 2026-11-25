"""Full-pipeline observability aggregation & exception-driven ops digest.

Aggregates 5 subsystems into a single lightweight payload (zero Prometheus/Grafana):
  1. Ingestion (daily writes, 7d trend, consecutive zero-write workdays, daemon queue)
  2. Maintenance (AutoDream & Janitor last run age + real business outputs)
  3. Knowledge Graph (entities, relationships, isolated node rate, 7d graph assist, Graph Lift)
  4. Retrieval (daily searches, hit rate, P50/P95 latency, avg tokens, Golden Eval ablation & trend)
  5. Infra (embedding coverage, PG latency, embedding/daemon probe status)
"""

from __future__ import annotations

import json
import logging
import os
import time
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import httpx
from sqlalchemy import text as sa_text

from team_memory.services.analysis.golden_eval import (
    DEFAULT_HISTORY_PATH,
    compute_percentile,
    load_eval_history,
)
from team_memory.storage.database import get_session

logger = logging.getLogger("team_memory.observability")

DEFAULT_CACHE_DIR = Path.home() / ".cache" / "tm-pipeline"
DEFAULT_DAEMON_STATUS_URL = "http://127.0.0.1:3901/status"

# Redline thresholds (shared by Web Dashboard and `tm-cli ops-digest`)
ZERO_WRITE_WORKDAYS_WARN = 2
MAINTENANCE_STALE_HOURS = 48.0
P95_LATENCY_RED_MS = 8000.0
EVAL_RECALL_DROP_WARN = 0.05


def _parse_iso_datetime(raw: str | None) -> datetime | None:
    if not raw or not isinstance(raw, str):
        return None
    s = raw.strip()
    if not s:
        return None
    try:
        if s.endswith("Z"):
            s = s[:-1] + "+00:00"
        dt = datetime.fromisoformat(s)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt.astimezone(timezone.utc)
    except ValueError:
        return None


def _read_text_timestamp(path: Path) -> datetime | None:
    if not path.exists():
        return None
    try:
        return _parse_iso_datetime(path.read_text(encoding="utf-8").strip())
    except OSError:
        return None


def _read_json_file(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else {}
    except (OSError, json.JSONDecodeError):
        return {}


def count_consecutive_zero_write_workdays(
    daily_series: list[dict[str, Any]],
    today: date | None = None,
) -> int:
    """Count consecutive trailing Mon-Fri workdays (excluding today) with 0 total writes."""
    if not daily_series:
        return 0
    ref_today = today or datetime.now(timezone.utc).date()
    by_date = {str(d["date"]): int(d.get("total", 0)) for d in daily_series}
    streak = 0
    cursor = ref_today - timedelta(days=1)
    for _ in range(14):
        if cursor.weekday() < 5:  # Mon=0 .. Fri=4
            if by_date.get(cursor.isoformat(), 0) == 0:
                streak += 1
            else:
                break
        cursor -= timedelta(days=1)
    return streak


def _extract_total_latency_ms(stage_metrics: Any) -> float | None:
    if not isinstance(stage_metrics, dict) or not stage_metrics:
        return None
    for key in ("total_ms", "total", "duration_ms"):
        val = stage_metrics.get(key)
        if isinstance(val, (int, float)) and val >= 0:
            return float(val)
    total = 0.0
    found = False
    for k, v in stage_metrics.items():
        is_timing_key = k.endswith("_ms") or k in ("embedding", "retrieve_fuse", "rerank")
        if isinstance(v, (int, float)) and v >= 0 and is_timing_key:
            total += float(v)
            found = True
    return total if found else None


def _is_graph_assisted(stage_counts: Any) -> bool:
    if not isinstance(stage_counts, dict):
        return False
    return int(stage_counts.get("graph_count") or 0) > 0 or int(stage_counts.get("graph_unique_added") or 0) > 0


def _is_search_hit(result_ids: Any) -> bool:
    if isinstance(result_ids, list):
        return len(result_ids) > 0
    return False


def _status_rank(status: str) -> int:
    return {"green": 0, "yellow": 1, "red": 2}.get(status, 0)


def _worst_status(*statuses: str) -> str:
    return max(statuses, key=_status_rank) if statuses else "green"


def evaluate_subsystem_statuses(
    *,
    ingestion: dict[str, Any],
    maintenance: dict[str, Any],
    knowledge_graph: dict[str, Any],
    retrieval: dict[str, Any],
    infra: dict[str, Any],
) -> tuple[str, list[dict[str, str]]]:
    """Apply the 5 health redlines to compute subsystem status lamps & alerts."""
    alerts: list[dict[str, str]] = []

    # 1. Ingestion
    zero_wd = int(ingestion.get("zero_write_workdays", 0))
    queue_failed = int(ingestion.get("queue_failed", 0))
    queue_pending = int(ingestion.get("queue_pending", 0))
    if zero_wd >= ZERO_WRITE_WORKDAYS_WARN:
        ing_status = "red" if zero_wd >= 4 else "yellow"
        msg = f"连续 {zero_wd} 个工作日零写入"
        ingestion["status"] = ing_status
        ingestion["reason"] = msg
        alerts.append({"subsystem": "ingestion", "level": ing_status, "message": msg})
    elif queue_failed >= 10:
        ingestion["status"] = "yellow"
        msg = f"精炼队列存在 {queue_failed} 条失败任务"
        ingestion["reason"] = msg
        alerts.append({"subsystem": "ingestion", "level": "yellow", "message": msg})
    else:
        ingestion["status"] = "green"
        ingestion["reason"] = f"近7日均 {ingestion.get('avg_7d', 0)}/天 · 队列 {queue_pending}"

    # 2. Maintenance (AutoDream & Janitor)
    dream = maintenance.get("autodream") or {}
    janitor = maintenance.get("janitor") or {}
    dream_age = dream.get("age_hours")
    janitor_age = janitor.get("age_hours")
    dream_output = int(dream.get("total_output", 0))
    dream_zero_streak = int(dream.get("zero_output_streak", 0))

    maint_status = "green"
    maint_reasons: list[str] = []
    if dream_age is None or dream_age > MAINTENANCE_STALE_HOURS:
        maint_status = _worst_status(maint_status, "red")
        msg = (
            "AutoDream 从未运行"
            if dream_age is None
            else f"AutoDream 已超 {dream_age:.1f}h 未运行 (>48h)"
        )
        maint_reasons.append(msg)
        alerts.append({"subsystem": "maintenance", "level": "red", "message": msg})
    elif dream_zero_streak >= 2:
        maint_status = _worst_status(maint_status, "yellow")
        msg = f"AutoDream 连续 {dream_zero_streak} 周期零业务产出"
        maint_reasons.append(msg)
        alerts.append({"subsystem": "maintenance", "level": "yellow", "message": msg})

    if janitor_age is None or janitor_age > MAINTENANCE_STALE_HOURS:
        maint_status = _worst_status(maint_status, "yellow")
        msg = (
            "Janitor 从未运行"
            if janitor_age is None
            else f"Janitor 已超 {janitor_age:.1f}h 未运行 (>48h)"
        )
        maint_reasons.append(msg)
        alerts.append({"subsystem": "maintenance", "level": "yellow", "message": msg})

    maintenance["status"] = maint_status
    if maint_reasons:
        maintenance["reason"] = "；".join(maint_reasons)
    else:
        maintenance["reason"] = (
            f"Dream {dream_age:.1f}h 前产出 {dream_output} 条 · Janitor {janitor_age:.1f}h 前"
        )

    # 3. Knowledge Graph
    graph_lift = knowledge_graph.get("latest_graph_lift")
    isolated_rate = float(knowledge_graph.get("isolated_rate", 0.0))
    kg_status = "green"
    kg_reasons: list[str] = []
    if graph_lift is not None and float(graph_lift) < 0.0:
        kg_status = "red"
        msg = f"Graph Lift 为负 ({float(graph_lift):+.4f})，图谱当前拖累召回"
        kg_reasons.append(msg)
        alerts.append({"subsystem": "knowledge_graph", "level": "red", "message": msg})
    elif isolated_rate > 0.5:
        kg_status = "yellow"
        msg = f"孤立节点率偏高 ({isolated_rate * 100:.1f}%)"
        kg_reasons.append(msg)
        alerts.append({"subsystem": "knowledge_graph", "level": "yellow", "message": msg})

    knowledge_graph["status"] = kg_status
    if kg_reasons:
        knowledge_graph["reason"] = "；".join(kg_reasons)
    else:
        lift_str = f"Lift {float(graph_lift):+.4f}" if graph_lift is not None else "待首测"
        knowledge_graph["reason"] = (
            f"{knowledge_graph.get('entities', 0)}点/{knowledge_graph.get('relationships', 0)}边 · {lift_str}"
        )

    # 4. Retrieval
    p95_ms = float(retrieval.get("p95_ms_7d", 0.0))
    recall_delta = retrieval.get("eval_recall_delta")
    ret_status = "green"
    ret_reasons: list[str] = []
    if p95_ms > P95_LATENCY_RED_MS:
        ret_status = "red"
        msg = f"近7日 P95 延迟 {p95_ms:.0f}ms 超过 8000ms 红线"
        ret_reasons.append(msg)
        alerts.append({"subsystem": "retrieval", "level": "red", "message": msg})
    if recall_delta is not None and float(recall_delta) < -EVAL_RECALL_DROP_WARN:
        ret_status = _worst_status(ret_status, "yellow")
        msg = f"Golden Eval Recall@5 较上次下降 {abs(float(recall_delta)) * 100:.1f}%"
        ret_reasons.append(msg)
        alerts.append({"subsystem": "retrieval", "level": "yellow", "message": msg})

    retrieval["status"] = ret_status
    if ret_reasons:
        retrieval["reason"] = "；".join(ret_reasons)
    else:
        latest_eval = retrieval.get("latest_eval") or {}
        full_m = (latest_eval.get("modes") or {}).get("full") or {}
        rec_str = f"Recall@5 {full_m['recall_at_5']:.3f}" if "recall_at_5" in full_m else "未评测"
        retrieval["reason"] = f"P95 {p95_ms:.0f}ms · {rec_str}"

    # 5. Infra
    coverage = float(infra.get("embedding_coverage", 1.0))
    total_exps = int(infra.get("total_experiences", 0))
    pg_ok = infra.get("pg_status") == "up"
    infra_status = "green"
    infra_reasons: list[str] = []
    if not pg_ok:
        infra_status = "red"
        msg = "PostgreSQL 探针异常"
        infra_reasons.append(msg)
        alerts.append({"subsystem": "infra", "level": "red", "message": msg})
    if total_exps > 0 and coverage < 1.0:
        infra_status = "red"
        missing = total_exps - int(infra.get("embedded_experiences", 0))
        msg = f"向量覆盖率 {coverage * 100:.2f}% < 100%（缺失 {missing} 条）"
        infra_reasons.append(msg)
        alerts.append({"subsystem": "infra", "level": "red", "message": msg})
    if infra.get("daemon_status") == "down":
        infra_status = _worst_status(infra_status, "yellow")
        msg = "tm-daemon (:3901) 未响应"
        infra_reasons.append(msg)
        alerts.append({"subsystem": "infra", "level": "yellow", "message": msg})

    infra["status"] = infra_status
    if infra_reasons:
        infra["reason"] = "；".join(infra_reasons)
    else:
        infra["reason"] = f"向量 {coverage * 100:.1f}% · PG {infra.get('pg_latency_ms', 0)}ms"

    overall = _worst_status(
        ingestion["status"],
        maintenance["status"],
        knowledge_graph["status"],
        retrieval["status"],
        infra["status"],
    )
    return overall, alerts


async def _fetch_daemon_status(url: str = DEFAULT_DAEMON_STATUS_URL) -> dict[str, Any]:
    try:
        async with httpx.AsyncClient(timeout=1.5) as client:
            resp = await client.get(url)
            if resp.status_code == 200:
                data = resp.json()
                return data if isinstance(data, dict) else {}
    except Exception:
        pass
    return {}


async def get_observability_overview(
    db_url: str,
    *,
    days: int = 30,
    cache_dir: Path = DEFAULT_CACHE_DIR,
    history_path: Path = DEFAULT_HISTORY_PATH,
    daemon_status_url: str = DEFAULT_DAEMON_STATUS_URL,
) -> dict[str, Any]:
    """Build the unified 5-subsystem observability overview payload."""
    days = 180 if int(days) >= 90 else 30
    now_utc = datetime.now(timezone.utc)
    today = now_utc.date()
    start_date = today - timedelta(days=days - 1)
    start_ts = datetime(start_date.year, start_date.month, start_date.day, tzinfo=timezone.utc)
    seven_days_ago = now_utc - timedelta(days=7)

    # Probe daemon concurrently with DB queries
    daemon_data = await _fetch_daemon_status(daemon_status_url)
    daemon_up = bool(daemon_data)

    pg_start = time.monotonic()
    pg_status = "up"
    pg_latency_ms = 0.0

    # Pre-initialize date buckets
    date_keys = [(start_date + timedelta(days=i)).isoformat() for i in range(days)]
    write_buckets: dict[str, dict[str, int]] = {
        d: {"date": d, "experiences": 0, "personal_memories": 0, "archives": 0, "total": 0}
        for d in date_keys
    }
    search_buckets: dict[str, dict[str, Any]] = {
        d: {"date": d, "searches": 0, "hits": 0, "graph_hits": 0, "latencies": [], "tokens": []}
        for d in date_keys
    }

    total_exps = 0
    embedded_exps = 0
    entity_count = 0
    rel_count = 0
    isolated_entities = 0
    searches_7d = 0
    hits_7d = 0
    graph_hits_7d = 0
    latencies_7d: list[float] = []
    tokens_7d: list[int] = []
    feedback_stats: dict[str, int] = {
        "explicit_total": 0,
        "implicit_total": 0,
        "explicit_7d": 0,
        "implicit_7d": 0,
        "implicit_positive_7d": 0,
        "implicit_negative_7d": 0,
    }

    try:
        async with get_session(db_url) as session:
            # PG probe
            await session.execute(sa_text("SELECT 1"))
            pg_latency_ms = round((time.monotonic() - pg_start) * 1000.0, 1)

            # 1. Daily writes: experiences, personal_memories, archives
            for table, field_name, extra_where in (
                ("experiences", "experiences", "AND is_deleted = false"),
                ("personal_memories", "personal_memories", ""),
                ("archives", "archives", ""),
            ):
                rows = (
                    await session.execute(
                        sa_text(
                            f"SELECT DATE(created_at AT TIME ZONE 'UTC') AS day, COUNT(*) "
                            f"FROM {table} "
                            f"WHERE created_at >= :start_ts {extra_where} "
                            f"GROUP BY day"
                        ),
                        {"start_ts": start_ts},
                    )
                ).fetchall()
                for day_val, cnt in rows:
                    dk = str(day_val)
                    if dk in write_buckets:
                        write_buckets[dk][field_name] = int(cnt)
                        write_buckets[dk]["total"] += int(cnt)

            # 2. Vector coverage
            cov_row = (
                await session.execute(
                    sa_text(
                        "SELECT COUNT(*) AS total, COUNT(embedding) AS embedded "
                        "FROM experiences WHERE is_deleted = false"
                    )
                )
            ).fetchone()
            if cov_row:
                total_exps = int(cov_row[0] or 0)
                embedded_exps = int(cov_row[1] or 0)

            # 3. Knowledge Graph stats (entities, relationships, isolated nodes)
            kg_row = (
                await session.execute(
                    sa_text(
                        "SELECT "
                        "(SELECT COUNT(*) FROM entities) AS ent_cnt, "
                        "(SELECT COUNT(*) FROM relationships) AS rel_cnt, "
                        "(SELECT COUNT(*) FROM entities e "
                        " WHERE NOT EXISTS (SELECT 1 FROM relationships r "
                        " WHERE r.source_entity_id = e.id OR r.target_entity_id = e.id)) AS iso_cnt"
                    )
                )
            ).fetchone()
            if kg_row:
                entity_count = int(kg_row[0] or 0)
                rel_count = int(kg_row[1] or 0)
                isolated_entities = int(kg_row[2] or 0)

            # 4. Search logs within window
            sl_rows = (
                await session.execute(
                    sa_text(
                        "SELECT created_at, result_ids, stage_metrics, stage_counts, returned_tokens "
                        "FROM search_logs "
                        "WHERE created_at >= :start_ts "
                        "ORDER BY created_at ASC"
                    ),
                    {"start_ts": start_ts},
                )
            ).fetchall()

            for created_at, result_ids, stage_metrics, stage_counts, ret_tokens in sl_rows:
                if created_at is None:
                    continue
                dt = created_at if created_at.tzinfo else created_at.replace(tzinfo=timezone.utc)
                dt_utc = dt.astimezone(timezone.utc)
                dk = dt_utc.date().isoformat()
                hit = _is_search_hit(result_ids)
                g_hit = _is_graph_assisted(stage_counts)
                lat = _extract_total_latency_ms(stage_metrics)
                tok = int(ret_tokens) if isinstance(ret_tokens, (int, float)) and ret_tokens > 0 else None

                if dk in search_buckets:
                    b = search_buckets[dk]
                    b["searches"] += 1
                    if hit:
                        b["hits"] += 1
                    if g_hit:
                        b["graph_hits"] += 1
                    if lat is not None:
                        b["latencies"].append(lat)
                    if tok is not None:
                        b["tokens"].append(tok)

                if dt_utc >= seven_days_ago:
                    searches_7d += 1
                    if hit:
                        hits_7d += 1
                    if g_hit:
                        graph_hits_7d += 1
                    if lat is not None:
                        latencies_7d.append(lat)
                    if tok is not None:
                        tokens_7d.append(tok)

            # 5. Feedback loop stats (explicit vs implicit)
            try:
                fb_row = (
                    await session.execute(
                        sa_text(
                            """
                            SELECT
                                COUNT(*) FILTER (
                                    WHERE COALESCE(source, 'explicit_user') != 'implicit_judge'
                                ) AS exp_tot,
                                COUNT(*) FILTER (WHERE source = 'implicit_judge') AS imp_tot,
                                COUNT(*) FILTER (
                                    WHERE COALESCE(source, 'explicit_user') != 'implicit_judge'
                                      AND created_at >= :seven_days_ago
                                ) AS exp_7d,
                                COUNT(*) FILTER (
                                    WHERE source = 'implicit_judge'
                                      AND created_at >= :seven_days_ago
                                ) AS imp_7d,
                                COUNT(*) FILTER (
                                    WHERE source = 'implicit_judge' AND rating >= 4
                                      AND created_at >= :seven_days_ago
                                ) AS imp_pos_7d,
                                COUNT(*) FILTER (
                                    WHERE source = 'implicit_judge' AND rating <= 2
                                      AND created_at >= :seven_days_ago
                                ) AS imp_neg_7d
                            FROM experience_feedbacks
                            """
                        ),
                        {"seven_days_ago": seven_days_ago},
                    )
                ).fetchone()
                if fb_row:
                    feedback_stats = {
                        "explicit_total": int(fb_row[0] or 0),
                        "implicit_total": int(fb_row[1] or 0),
                        "explicit_7d": int(fb_row[2] or 0),
                        "implicit_7d": int(fb_row[3] or 0),
                        "implicit_positive_7d": int(fb_row[4] or 0),
                        "implicit_negative_7d": int(fb_row[5] or 0),
                    }
            except Exception as fb_exc:
                logger.debug("Feedback stats query skipped: %s", fb_exc)

    except Exception as exc:
        logger.warning("Observability DB query failed: %s", exc)
        pg_status = "down"

    # Assemble Ingestion
    write_series = [write_buckets[d] for d in date_keys]
    last_7_totals = [b["total"] for b in write_series[-7:]]
    prev_7_totals = [b["total"] for b in write_series[-14:-7]]
    avg_7d = round(sum(last_7_totals) / max(1, len(last_7_totals)), 1)
    prev_7d_avg = round(sum(prev_7_totals) / max(1, len(prev_7_totals)), 1) if prev_7_totals else 0.0
    zero_write_workdays = count_consecutive_zero_write_workdays(write_series, today=today)

    # Extract daemon queue stats if present
    pending_queue = daemon_data.get("pending_queue") or daemon_data.get("queue") or {}
    queue_pending = int(pending_queue.get("pending", 0) or 0)
    queue_failed = int(pending_queue.get("failed", 0) or 0)

    ingestion = {
        "today_total": write_series[-1]["total"] if write_series else 0,
        "avg_7d": avg_7d,
        "prev_7d_avg": prev_7d_avg,
        "zero_write_workdays": zero_write_workdays,
        "queue_pending": queue_pending,
        "queue_failed": queue_failed,
        "daily_series": write_series,
    }

    # Assemble Maintenance (AutoDream & Janitor)
    dream_stats = _read_json_file(cache_dir / "last_dream_stats.json")
    dream_ts = _parse_iso_datetime(dream_stats.get("ran_at")) or _read_text_timestamp(cache_dir / "last_dream.txt")
    janitor_ts = _read_text_timestamp(cache_dir / "last_janitor.txt")

    dream_age_h = round((now_utc - dream_ts).total_seconds() / 3600.0, 1) if dream_ts else None
    janitor_age_h = round((now_utc - janitor_ts).total_seconds() / 3600.0, 1) if janitor_ts else None
    merged = int(dream_stats.get("merged", 0) or 0)
    superseded = int(dream_stats.get("superseded", 0) or 0)
    pruned = int(dream_stats.get("pruned", 0) or 0)
    distilled = int(dream_stats.get("distilled", 0) or 0)
    total_dream_output = merged + superseded + pruned + distilled
    zero_output_streak = int(dream_stats.get("zero_output_streak", 0) or 0)
    if total_dream_output == 0 and dream_stats and zero_output_streak == 0:
        zero_output_streak = 1

    maintenance = {
        "autodream": {
            "last_run_at": dream_ts.isoformat() if dream_ts else None,
            "age_hours": dream_age_h,
            "merged": merged,
            "superseded": superseded,
            "pruned": pruned,
            "distilled": distilled,
            "total_output": total_dream_output,
            "zero_output_streak": zero_output_streak,
            "success": dream_stats.get("success", True if dream_ts else None),
        },
        "janitor": {
            "last_run_at": janitor_ts.isoformat() if janitor_ts else None,
            "age_hours": janitor_age_h,
            "decayed": int(dream_stats.get("decayed", 0) or 0),
            "purged": pruned,
        },
    }

    # Assemble Golden Eval history
    eval_history = load_eval_history(history_path, limit=20)
    latest_eval = eval_history[-1] if eval_history else None
    prev_eval = eval_history[-2] if len(eval_history) >= 2 else None
    latest_graph_lift = latest_eval.get("graph_lift") if latest_eval else None

    eval_recall_delta: float | None = None
    if latest_eval and prev_eval:
        cur_r = ((latest_eval.get("modes") or {}).get("full") or {}).get("recall_at_5")
        prev_r = ((prev_eval.get("modes") or {}).get("full") or {}).get("recall_at_5")
        if cur_r is not None and prev_r is not None:
            eval_recall_delta = round(float(cur_r) - float(prev_r), 4)

    # Assemble Knowledge Graph
    isolated_rate = round(isolated_entities / entity_count, 4) if entity_count > 0 else 0.0
    graph_assist_rate_7d = round(graph_hits_7d / searches_7d, 4) if searches_7d > 0 else 0.0
    knowledge_graph = {
        "entities": entity_count,
        "relationships": rel_count,
        "isolated_entities": isolated_entities,
        "isolated_rate": isolated_rate,
        "graph_assist_rate_7d": graph_assist_rate_7d,
        "latest_graph_lift": latest_graph_lift,
    }

    # Assemble Retrieval
    retrieval_series: list[dict[str, Any]] = []
    for d in date_keys:
        b = search_buckets[d]
        cnt = b["searches"]
        retrieval_series.append(
            {
                "date": d,
                "searches": cnt,
                "hit_rate": round(b["hits"] / cnt, 4) if cnt > 0 else 0.0,
                "p50_ms": round(compute_percentile(b["latencies"], 50), 1) if b["latencies"] else 0.0,
                "p95_ms": round(compute_percentile(b["latencies"], 95), 1) if b["latencies"] else 0.0,
                "graph_assist_rate": round(b["graph_hits"] / cnt, 4) if cnt > 0 else 0.0,
                "avg_tokens": round(sum(b["tokens"]) / len(b["tokens"]), 1) if b["tokens"] else 0.0,
            }
        )

    retrieval = {
        "searches_7d": searches_7d,
        "hit_rate_7d": round(hits_7d / searches_7d, 4) if searches_7d > 0 else 0.0,
        "p50_ms_7d": round(compute_percentile(latencies_7d, 50), 1) if latencies_7d else 0.0,
        "p95_ms_7d": round(compute_percentile(latencies_7d, 95), 1) if latencies_7d else 0.0,
        "avg_tokens_7d": round(sum(tokens_7d) / len(tokens_7d), 1) if tokens_7d else 0.0,
        "eval_recall_delta": eval_recall_delta,
        "feedback": feedback_stats,
        "daily_series": retrieval_series,
        "latest_eval": latest_eval,
        "eval_history": eval_history,
    }

    # Assemble Infra
    embedding_coverage = round(embedded_exps / total_exps, 4) if total_exps > 0 else 1.0
    infra = {
        "total_experiences": total_exps,
        "embedded_experiences": embedded_exps,
        "embedding_coverage": embedding_coverage,
        "pg_status": pg_status,
        "pg_latency_ms": pg_latency_ms,
        "daemon_status": "up" if daemon_up else "down",
    }

    overall_status, alerts = evaluate_subsystem_statuses(
        ingestion=ingestion,
        maintenance=maintenance,
        knowledge_graph=knowledge_graph,
        retrieval=retrieval,
        infra=infra,
    )

    return {
        "generated_at": now_utc.isoformat(),
        "days": days,
        "overall_status": overall_status,
        "alerts": alerts,
        "subsystems": {
            "ingestion": ingestion,
            "maintenance": maintenance,
            "knowledge_graph": knowledge_graph,
            "retrieval": retrieval,
            "infra": infra,
        },
    }


# ============================================================
# Exception-Driven Ops Digest Formatting & Webhook Push
# ============================================================

_LAMP_ICON = {"green": "🟢", "yellow": "🟡", "red": "🔴"}


def format_ops_digest(
    overview: dict[str, Any],
    *,
    alert_only: bool = False,
    weekly: bool = False,
) -> str:
    """Format observability overview for CLI / Feishu ops digest.

    - `alert_only=True`: returns empty string when overall_status == "green".
    - `weekly=True`: compact <= 10-line health digest with WoW deltas.
    """
    overall = overview.get("overall_status", "green")
    alerts = overview.get("alerts") or []
    subs = overview.get("subsystems") or {}

    if alert_only and overall == "green":
        return ""

    if alert_only:
        lines = [f"{_LAMP_ICON.get(overall, '⚪')} TeamMemory 异常告警 ({len(alerts)} 项)"]
        for a in alerts:
            icon = _LAMP_ICON.get(a.get("level", "yellow"), "🟡")
            lines.append(f"- {icon} [{a.get('subsystem')}] {a.get('message')}")
        return "\n".join(lines)

    ing = subs.get("ingestion") or {}
    maint = subs.get("maintenance") or {}
    kg = subs.get("knowledge_graph") or {}
    ret = subs.get("retrieval") or {}
    inf = subs.get("infra") or {}

    lamp_bar = " ".join(
        f"{_LAMP_ICON.get((subs.get(k) or {}).get('status', 'green'), '⚪')}{label}"
        for k, label in (
            ("ingestion", "写入"),
            ("maintenance", "维护"),
            ("knowledge_graph", "图谱"),
            ("retrieval", "召回"),
            ("infra", "底座"),
        )
    )

    avg_7d = float(ing.get("avg_7d", 0.0))
    prev_7d = float(ing.get("prev_7d_avg", 0.0))
    wow_str = f"{((avg_7d - prev_7d) / prev_7d * 100):+.0f}%" if prev_7d > 0 else "N/A"

    latest_eval = ret.get("latest_eval") or {}
    full_m = (latest_eval.get("modes") or {}).get("full") or {}
    eval_line = (
        f"Recall@5={full_m.get('recall_at_5', 0):.3f} MRR@5={full_m.get('mrr_at_5', 0):.3f} "
        f"GraphLift={float(latest_eval.get('graph_lift', 0)):+.4f}"
        if full_m
        else "尚未运行 Golden Eval"
    )

    iso_pct = float(kg.get("isolated_rate", 0)) * 100
    assist_pct = float(kg.get("graph_assist_rate_7d", 0)) * 100
    hit_pct = float(ret.get("hit_rate_7d", 0)) * 100
    cov_pct = float(inf.get("embedding_coverage", 1)) * 100
    p50_ms = float(ret.get("p50_ms_7d", 0))
    p95_ms = float(ret.get("p95_ms_7d", 0))

    lines = [
        f"TeamMemory 健康周报 | 总状态 {_LAMP_ICON.get(overall, '⚪')} {overall.upper()}",
        f"状态灯: {lamp_bar}",
        (
            f"写入: 近7日均 {avg_7d}/天 (环比 {wow_str}) · 今日 {ing.get('today_total', 0)} · "
            f"队列 {ing.get('queue_pending', 0)}/{ing.get('queue_failed', 0)}失败"
        ),
        f"维护: {maint.get('reason', '-')}",
        (
            f"图谱: {kg.get('entities', 0)}点/{kg.get('relationships', 0)}边 · "
            f"孤立率 {iso_pct:.1f}% · 图谱辅助 {assist_pct:.1f}%"
        ),
        (
            f"召回: 7d搜索 {ret.get('searches_7d', 0)}次 · 命中率 {hit_pct:.1f}% · "
            f"P50/P95 {p50_ms:.0f}/{p95_ms:.0f}ms"
        ),
        f"评测: {eval_line}",
        (
            f"底座: 向量覆盖 {cov_pct:.2f}% ({inf.get('embedded_experiences', 0)}/"
            f"{inf.get('total_experiences', 0)}) · PG {inf.get('pg_latency_ms', 0)}ms · "
            f"Daemon {inf.get('daemon_status', 'down')}"
        ),
    ]
    if alerts:
        top_alerts = "；".join(str(a.get("message", "")) for a in alerts[:2])
        lines.append(f"异常: {top_alerts}")
    return "\n".join(lines[:10] if weekly else lines)


async def send_feishu_webhook(webhook_url: str, text: str) -> bool:
    """Send plain text digest to a Feishu bot webhook."""
    url = (webhook_url or os.environ.get("TEAM_MEMORY_FEISHU_WEBHOOK") or "").strip()
    if not url or not text:
        return False
    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            resp = await client.post(url, json={"msg_type": "text", "content": {"text": text}})
            return resp.status_code < 300
    except Exception as exc:
        logger.warning("Failed to send Feishu webhook: %s", exc)
        return False

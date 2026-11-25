"""Automatic implicit feedback generation from faithfulness/adoption signals.

Generates weighted implicit feedback (weight=0.5, source='implicit_judge') so
retrieval ranking improves continuously without relying on manual user/agent
ratings, while preserving strict multi-tenant isolation and anti-spam guards:

1. Implicit positive feedback:
   - Triggered when was_used=True and faithfulness_score >= 0.7 (and
     relevance_score >= 0.5 when available).
   - Attributed strictly to `used_indices` returned by FaithfulnessJudge
     (1-based indices into contexts); if empty, conservatively attributed
     only to Top-1 (never blind-broadcast to Top-3).
   - Idempotent per (tenant_id, experience_id, search_log_id, source).

2. Implicit negative feedback:
   - Triggered when an experience appears in Top-3 >= 5 times within 7 days
     (non-noise, judged) with ALL `was_used == False` AND high relevance
     (avg relevance_score >= 0.5, or avg retrieval score >= 0.5 when
     relevance_score is unavailable) — distinguishing genuinely stale/unhelpful
     entries from retriever false-positives.
   - Anti-false-kill guards:
     * Skips pinned or deleted experiences.
     * Skips experiences with explicit positive feedback (rating >= 4) in the
       lookback window.
     * Rate-limited to <= 1 per 24h and <= 2 per 7d per experience per tenant.
"""

from __future__ import annotations

import json
import logging
import uuid as uuid_mod
from collections import defaultdict
from datetime import datetime, timedelta, timezone
from typing import Any

from sqlalchemy import text as sa_text

from team_memory.storage.database import get_session
from team_memory.storage.tenant_context import get_current_tenant_id

logger = logging.getLogger("team_memory.implicit_feedback")

IMPLICIT_FEEDBACK_SOURCE = "implicit_judge"
IMPLICIT_FEEDBACK_WEIGHT = 0.5
POSITIVE_FAITHFULNESS_THRESHOLD = 0.7
HIGH_RELEVANCE_THRESHOLD = 0.5
NEGATIVE_MIN_RECALLS = 5
NEGATIVE_LOOKBACK_DAYS = 7
NEGATIVE_DAILY_CAP = 1
NEGATIVE_WEEKLY_CAP = 2


def _extract_result_id(entry: Any) -> str | None:
    """Extract a valid UUID string from a result_ids list item."""
    raw = entry.get("id") if isinstance(entry, dict) else entry
    if not raw:
        return None
    try:
        return str(uuid_mod.UUID(str(raw).strip()))
    except (ValueError, TypeError, AttributeError):
        return None


def _extract_result_score(entry: Any) -> float | None:
    if isinstance(entry, dict):
        val = entry.get("score")
        if isinstance(val, (int, float)):
            return float(val)
    return None


def resolve_positive_target_ids(
    result_ids: list[Any] | None,
    used_indices: list[int] | None = None,
) -> list[str]:
    """Resolve which experience IDs should receive implicit positive feedback.

    Uses 1-based `used_indices` from FaithfulnessJudge when provided; otherwise
    conservatively attributes only to Top-1 to prevent free-riding on rank 2/3.
    """
    if not result_ids:
        return []

    ordered_ids: list[str] = []
    for item in result_ids:
        eid = _extract_result_id(item)
        if eid:
            ordered_ids.append(eid)

    if not ordered_ids:
        return []

    if used_indices:
        targets: list[str] = []
        for idx in used_indices:
            if isinstance(idx, int) and 1 <= idx <= len(ordered_ids):
                eid = ordered_ids[idx - 1]
                if eid not in targets:
                    targets.append(eid)
        if targets:
            return targets

    return [ordered_ids[0]]


async def record_implicit_positive_feedback(
    *,
    db_url: str,
    result_ids: list[Any] | None,
    faithfulness_score: float,
    was_used: bool = True,
    used_indices: list[int] | None = None,
    relevance_score: float | None = None,
    search_log_id: str | None = None,
    tenant_id: str | None = None,
) -> dict[str, Any]:
    """Record idempotent implicit positive feedback (weight=0.5) for adopted experiences."""
    if not was_used or faithfulness_score < POSITIVE_FAITHFULNESS_THRESHOLD:
        return {"recorded": 0, "skipped_reason": "below_threshold"}
    if relevance_score is not None and relevance_score < HIGH_RELEVANCE_THRESHOLD:
        return {"recorded": 0, "skipped_reason": "low_relevance"}

    target_ids = resolve_positive_target_ids(result_ids, used_indices)
    if not target_ids:
        return {"recorded": 0, "skipped_reason": "no_valid_targets"}

    sl_uuid: uuid_mod.UUID | None = None
    if search_log_id:
        try:
            sl_uuid = uuid_mod.UUID(str(search_log_id))
        except (ValueError, TypeError):
            sl_uuid = None

    recorded = 0
    try:
        async with get_session(db_url) as session:
            eff_tenant_id: uuid_mod.UUID | None = None
            if tenant_id:
                eff_tenant_id = uuid_mod.UUID(str(tenant_id))
            elif sl_uuid is not None:
                row = (
                    await session.execute(
                        sa_text("SELECT tenant_id FROM search_logs WHERE id = :sl_id"),
                        {"sl_id": sl_uuid},
                    )
                ).fetchone()
                if row and row[0]:
                    eff_tenant_id = row[0] if isinstance(row[0], uuid_mod.UUID) else uuid_mod.UUID(str(row[0]))

            if eff_tenant_id is None:
                eff_tenant_id = get_current_tenant_id()

            comment = f"implicit_positive: faithfulness={faithfulness_score:.2f}"
            for eid_str in target_ids:
                exp_uuid = uuid_mod.UUID(eid_str)
                # Ensure experience belongs to the same tenant and is active
                exp_row = (
                    await session.execute(
                        sa_text(
                            "SELECT id FROM experiences "
                            "WHERE id = :eid AND tenant_id = :tid AND is_deleted = false"
                        ),
                        {"eid": exp_uuid, "tid": eff_tenant_id},
                    )
                ).fetchone()
                if not exp_row:
                    continue

                if sl_uuid is not None:
                    ins = await session.execute(
                        sa_text(
                            """
                            INSERT INTO experience_feedbacks
                                (tenant_id, experience_id, rating, comment, feedback_by,
                                 source, weight, search_log_id, created_at)
                            VALUES
                                (:tid, :eid, 4, :comment, 'implicit_judge',
                                 :source, :weight, :sl_id, NOW())
                            ON CONFLICT (tenant_id, experience_id, search_log_id, source)
                            WHERE search_log_id IS NOT NULL
                            DO NOTHING
                            RETURNING id
                            """
                        ),
                        {
                            "tid": eff_tenant_id,
                            "eid": exp_uuid,
                            "comment": comment,
                            "source": IMPLICIT_FEEDBACK_SOURCE,
                            "weight": IMPLICIT_FEEDBACK_WEIGHT,
                            "sl_id": sl_uuid,
                        },
                    )
                else:
                    ins = await session.execute(
                        sa_text(
                            """
                            INSERT INTO experience_feedbacks
                                (tenant_id, experience_id, rating, comment, feedback_by,
                                 source, weight, search_log_id, created_at)
                            VALUES
                                (:tid, :eid, 4, :comment, 'implicit_judge',
                                 :source, :weight, NULL, NOW())
                            RETURNING id
                            """
                        ),
                        {
                            "tid": eff_tenant_id,
                            "eid": exp_uuid,
                            "comment": comment,
                            "source": IMPLICIT_FEEDBACK_SOURCE,
                            "weight": IMPLICIT_FEEDBACK_WEIGHT,
                        },
                    )

                inserted = ins.fetchone()
                if inserted is not None:
                    await session.execute(
                        sa_text(
                            """
                            UPDATE experiences
                            SET implicit_positive_count = COALESCE(implicit_positive_count, 0) + 1,
                                quality_score = LEAST(100.0, COALESCE(quality_score, 100.0) + :delta)
                            WHERE id = :eid AND tenant_id = :tid
                            """
                        ),
                        {"eid": exp_uuid, "tid": eff_tenant_id, "delta": 1.0 * IMPLICIT_FEEDBACK_WEIGHT},
                    )
                    recorded += 1

            await session.commit()
    except Exception as exc:
        logger.warning("record_implicit_positive_feedback failed: %s", exc)
        return {"recorded": recorded, "error": str(exc)}

    return {"recorded": recorded, "target_ids": target_ids}


def evaluate_negative_candidates(
    search_log_rows: list[dict[str, Any]],
    *,
    min_recalls: int = NEGATIVE_MIN_RECALLS,
    min_high_relevance: float = HIGH_RELEVANCE_THRESHOLD,
) -> list[tuple[str, str, int, float]]:
    """Pure aggregation over judged search_logs to identify repeated high-relevance non-adoptions.

    Returns list of `(tenant_id_str, experience_id_str, recall_count, avg_relevance)`.
    """
    # (tenant_id, exp_id) -> {"total": int, "any_used": bool, "relevances": list[float], "scores": list[float]}
    stats: dict[tuple[str, str], dict[str, Any]] = defaultdict(
        lambda: {"total": 0, "any_used": False, "relevances": [], "scores": []}
    )

    for row in search_log_rows:
        if row.get("is_noise") is True:
            continue
        was_used = row.get("was_used")
        if was_used is None:
            continue

        raw_rids = row.get("result_ids")
        if isinstance(raw_rids, str):
            try:
                raw_rids = json.loads(raw_rids)
            except json.JSONDecodeError:
                continue
        if not isinstance(raw_rids, list) or not raw_rids:
            continue

        tid = str(row.get("tenant_id") or "")
        if not tid:
            continue

        rel = row.get("relevance_score")
        rel_val = float(rel) if isinstance(rel, (int, float)) else None

        seen_in_log: set[str] = set()
        for entry in raw_rids[:3]:  # Top-3 only
            eid = _extract_result_id(entry)
            if not eid or eid in seen_in_log:
                continue
            seen_in_log.add(eid)
            bucket = stats[(tid, eid)]
            bucket["total"] += 1
            if bool(was_used):
                bucket["any_used"] = True
            if rel_val is not None:
                bucket["relevances"].append(rel_val)
            score_val = _extract_result_score(entry)
            if score_val is not None:
                bucket["scores"].append(score_val)

    candidates: list[tuple[str, str, int, float]] = []
    for (tid, eid), b in stats.items():
        if b["total"] < min_recalls or b["any_used"]:
            continue
        # High-relevance guard: distinguish genuinely unhelpful/stale experiences
        # from unrelated retriever false positives.
        if b["relevances"]:
            avg_rel = sum(b["relevances"]) / len(b["relevances"])
            if avg_rel < min_high_relevance:
                continue
        elif b["scores"]:
            avg_rel = sum(b["scores"]) / len(b["scores"])
            if avg_rel < min_high_relevance:
                continue
        else:
            continue
        candidates.append((tid, eid, b["total"], round(avg_rel, 3)))

    return candidates


async def scan_and_record_implicit_negative_feedback(
    *,
    db_url: str,
    lookback_days: int = NEGATIVE_LOOKBACK_DAYS,
    min_recalls: int = NEGATIVE_MIN_RECALLS,
    daily_cap: int = NEGATIVE_DAILY_CAP,
    weekly_cap: int = NEGATIVE_WEEKLY_CAP,
    tenant_id: str | None = None,
) -> dict[str, Any]:
    """Scan recent judged search_logs and record rate-limited implicit negative feedback."""
    now = datetime.now(timezone.utc)
    since_ts = now - timedelta(days=lookback_days)
    day_ago = now - timedelta(days=1)

    recorded = 0
    skipped_capped = 0
    skipped_protected = 0

    try:
        async with get_session(db_url) as session:
            tenant_filter = "AND tenant_id = :tid" if tenant_id else ""
            params: dict[str, Any] = {"since_ts": since_ts}
            if tenant_id:
                params["tid"] = uuid_mod.UUID(str(tenant_id))

            rows = (
                await session.execute(
                    sa_text(
                        f"""
                        SELECT tenant_id, result_ids, was_used, relevance_score, is_noise
                        FROM search_logs
                        WHERE created_at >= :since_ts
                          AND was_used IS NOT NULL
                          AND is_noise IS NOT TRUE
                          AND result_ids IS NOT NULL
                          {tenant_filter}
                        """
                    ),
                    params,
                )
            ).fetchall()

            log_dicts = [
                {
                    "tenant_id": str(r[0]),
                    "result_ids": r[1],
                    "was_used": r[2],
                    "relevance_score": r[3],
                    "is_noise": r[4],
                }
                for r in rows
            ]
            candidates = evaluate_negative_candidates(log_dicts, min_recalls=min_recalls)

            for tid_str, eid_str, recall_cnt, avg_rel in candidates:
                tid_uuid = uuid_mod.UUID(tid_str)
                exp_uuid = uuid_mod.UUID(eid_str)

                # Guard 1: experience exists in same tenant, active, not pinned
                exp_row = (
                    await session.execute(
                        sa_text(
                            "SELECT is_pinned FROM experiences "
                            "WHERE id = :eid AND tenant_id = :tid AND is_deleted = false"
                        ),
                        {"eid": exp_uuid, "tid": tid_uuid},
                    )
                ).fetchone()
                if not exp_row or bool(exp_row[0]):
                    skipped_protected += 1
                    continue

                # Guard 2: skip if explicit positive feedback received in lookback window
                explicit_pos = (
                    await session.execute(
                        sa_text(
                            """
                            SELECT 1 FROM experience_feedbacks
                            WHERE tenant_id = :tid
                              AND experience_id = :eid
                              AND source LIKE 'explicit_%'
                              AND rating >= 4
                              AND created_at >= :since_ts
                            LIMIT 1
                            """
                        ),
                        {"tid": tid_uuid, "eid": exp_uuid, "since_ts": since_ts},
                    )
                ).fetchone()
                if explicit_pos is not None:
                    skipped_protected += 1
                    continue

                # Guard 3: daily & weekly anti-spam caps
                cap_row = (
                    await session.execute(
                        sa_text(
                            """
                            SELECT
                                COUNT(*) FILTER (WHERE created_at >= :day_ago) AS cnt_24h,
                                COUNT(*) AS cnt_7d
                            FROM experience_feedbacks
                            WHERE tenant_id = :tid
                              AND experience_id = :eid
                              AND source = :source
                              AND rating <= 2
                              AND created_at >= :since_ts
                            """
                        ),
                        {
                            "tid": tid_uuid,
                            "eid": exp_uuid,
                            "source": IMPLICIT_FEEDBACK_SOURCE,
                            "day_ago": day_ago,
                            "since_ts": since_ts,
                        },
                    )
                ).fetchone()
                cnt_24h = int(cap_row[0] or 0) if cap_row else 0
                cnt_7d = int(cap_row[1] or 0) if cap_row else 0
                if cnt_24h >= daily_cap or cnt_7d >= weekly_cap:
                    skipped_capped += 1
                    continue

                comment = (
                    f"implicit_negative: top3_recalls={recall_cnt} ({lookback_days}d), "
                    f"was_used=0, avg_relevance={avg_rel:.2f}"
                )
                await session.execute(
                    sa_text(
                        """
                        INSERT INTO experience_feedbacks
                            (tenant_id, experience_id, rating, comment, feedback_by,
                             source, weight, search_log_id, created_at)
                        VALUES
                            (:tid, :eid, 2, :comment, 'implicit_judge',
                             :source, :weight, NULL, NOW())
                        """
                    ),
                    {
                        "tid": tid_uuid,
                        "eid": exp_uuid,
                        "comment": comment,
                        "source": IMPLICIT_FEEDBACK_SOURCE,
                        "weight": IMPLICIT_FEEDBACK_WEIGHT,
                    },
                )
                await session.execute(
                    sa_text(
                        """
                        UPDATE experiences
                        SET implicit_negative_count = COALESCE(implicit_negative_count, 0) + 1,
                            quality_score = GREATEST(0.0, COALESCE(quality_score, 100.0) - :delta)
                        WHERE id = :eid AND tenant_id = :tid
                        """
                    ),
                    {"eid": exp_uuid, "tid": tid_uuid, "delta": 3.0 * IMPLICIT_FEEDBACK_WEIGHT},
                )
                recorded += 1

            await session.commit()
    except Exception as exc:
        logger.warning("scan_and_record_implicit_negative_feedback failed: %s", exc)
        return {"recorded": recorded, "error": str(exc)}

    return {
        "recorded": recorded,
        "skipped_capped": skipped_capped,
        "skipped_protected": skipped_protected,
    }

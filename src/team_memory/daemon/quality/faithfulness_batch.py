#!/usr/bin/env python3
"""Attribution & Faithfulness Batch Evaluator + Persistent Daemon Worker.

Processes pending entries in ``response_buffer`` using ``AttributionJudge``
(per-experience relevance, adoption with Evidence-Required gate, and
contradiction detection), writes back ``relevance_score``, ``was_used``,
``faithfulness_score``, and ``judgment_source='attribution'`` to
``search_logs``, and triggers RLS-safe implicit positive/negative feedback
and contradiction staleness updates.

Usage:
    python -m team_memory.daemon.quality.faithfulness_batch [OPTIONS]
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import sys
import uuid as uuid_mod
from typing import Any

from sqlalchemy import select as sa_select
from sqlalchemy import text as sa_text

from team_memory.project_resolver import find_project_root, load_dotenv_if_exists
from team_memory.storage.database import get_session
from team_memory.storage.tenant_context import get_current_tenant_id, parse_tenant_uuid

load_dotenv_if_exists(find_project_root() or os.path.dirname(__file__))

logger = logging.getLogger("faithfulness_batch")
worker_logger = logging.getLogger("daemon.eval_worker")

DEFAULT_DAILY_EVAL_LIMIT = int(os.environ.get("ATTRIBUTION_DAILY_LIMIT", "200"))


async def get_daily_eval_count(
    db_url: str,
    *,
    tenant_id: str | uuid_mod.UUID | None = None,
) -> int:
    """Count attribution_judge LLM calls recorded today (UTC) in llm_usage_logs."""
    eff_tid = parse_tenant_uuid(tenant_id) or get_current_tenant_id()
    try:
        async with get_session(db_url, tenant_id=eff_tid) as session:
            row = (
                await session.execute(
                    sa_text(
                        """
                        SELECT COUNT(*)
                        FROM llm_usage_logs
                        WHERE caller = 'attribution_judge'
                          AND created_at >= date_trunc('day', NOW() AT TIME ZONE 'UTC') AT TIME ZONE 'UTC'
                        """
                    )
                )
            ).fetchone()
            return int(row[0] or 0) if row else 0
    except Exception as exc:
        logger.debug("get_daily_eval_count degraded to 0: %s", exc)
        return 0


async def load_structured_contexts(
    db_url: str,
    result_ids: list[Any] | None,
    *,
    tenant_id: str | uuid_mod.UUID | None = None,
) -> list[dict[str, Any]]:
    """Load structured ``{id, title, problem, solution}`` contexts under RLS.

    Preserves the order of ``result_ids`` and supports both ``experiences``
    and ``archives`` tables.
    """
    if not result_ids:
        return []

    valid_ids: list[str] = []
    seen: set[str] = set()
    for rid in result_ids:
        raw = rid.get("id", "") if isinstance(rid, dict) else str(rid)
        if not raw:
            continue
        try:
            norm = str(uuid_mod.UUID(str(raw).strip()))
            if norm not in seen:
                seen.add(norm)
                valid_ids.append(norm)
        except (ValueError, TypeError, AttributeError):
            continue

    if not valid_ids:
        return []

    eff_tid = parse_tenant_uuid(tenant_id) or get_current_tenant_id()
    uuids = [uuid_mod.UUID(v) for v in valid_ids]
    by_id: dict[str, dict[str, Any]] = {}

    async with get_session(db_url, tenant_id=eff_tid) as session:
        from team_memory.storage.models import Archive, Experience

        exp_result = await session.execute(
            sa_select(
                Experience.id,
                Experience.title,
                Experience.description,
                Experience.solution,
            ).where(Experience.id.in_(uuids))
        )
        for r in exp_result.mappings().all():
            eid = str(r["id"])
            by_id[eid] = {
                "id": eid,
                "title": r["title"] or "",
                "problem": (r["description"] or "")[:400],
                "solution": (r["solution"] or "")[:600],
            }

        missing_uuids = [u for u in uuids if str(u) not in by_id]
        if missing_uuids:
            arc_result = await session.execute(
                sa_select(
                    Archive.id,
                    Archive.title,
                    Archive.overview,
                    Archive.solution_doc,
                ).where(Archive.id.in_(missing_uuids))
            )
            for r in arc_result.mappings().all():
                aid = str(r["id"])
                by_id[aid] = {
                    "id": aid,
                    "title": r.get("title") or "",
                    "problem": (r.get("overview") or "")[:400],
                    "solution": (r.get("solution_doc") or "")[:600],
                }

    return [by_id[vid] for vid in valid_ids if vid in by_id]


async def run_batch(
    *,
    dsn: str,
    model: str | None = None,
    base_url: str | None = None,
    api_key: str | None = None,
    batch_size: int = 20,
    dry_run: bool = False,
    force: bool = False,
    checkpoint_path: str = "/tmp/tm-faithfulness-checkpoint.txt",
    daily_limit: int | None = None,
    tenant_id: str | None = None,
) -> dict[str, Any]:
    """Run a batch per-experience attribution evaluation."""
    from team_memory.daemon.buffers.response_buffer import ResponseBuffer
    from team_memory.daemon.quality.faithfulness_judge import AttributionJudge
    from team_memory.services.analysis.implicit_feedback import (
        handle_contradicted_experiences,
        record_implicit_positive_feedback,
        scan_and_record_implicit_negative_feedback,
    )

    raw_dsn = dsn.replace("postgresql+asyncpg://", "postgresql://", 1)
    async_dsn = dsn.replace("postgresql://", "postgresql+asyncpg://", 1) if "+asyncpg" not in dsn else dsn
    eff_tid = parse_tenant_uuid(tenant_id) or get_current_tenant_id()
    eff_tid_str = str(eff_tid)
    eff_daily_limit = daily_limit if daily_limit is not None else DEFAULT_DAILY_EVAL_LIMIT

    buffer = ResponseBuffer(dsn=raw_dsn, tenant_id=eff_tid_str)

    try:
        # Check if we should run
        pending = await buffer.pending_count()
        logger.info("Pending entries: %d", pending)

        if not force and pending < buffer._batch_threshold:
            logger.info(
                "Below threshold (%d < %d), skipping. Use --force to override.",
                pending,
                buffer._batch_threshold,
            )
            return {"skipped": True, "pending": pending}

        # Fetch unevaluated entries
        entries = await buffer.fetch_pending(limit=batch_size)
        if not entries:
            logger.info("No entries to evaluate.")
            return {"skipped": False, "evaluated": 0}

        # Check daily evaluation budget
        daily_used = 0
        if eff_daily_limit > 0:
            daily_used = await get_daily_eval_count(async_dsn, tenant_id=eff_tid_str)
            if daily_used >= eff_daily_limit:
                logger.warning(
                    "Daily attribution eval limit reached (%d/%d); skipping batch.",
                    daily_used,
                    eff_daily_limit,
                )
                return {
                    "skipped": True,
                    "budget_exhausted": True,
                    "daily_used": daily_used,
                    "daily_limit": eff_daily_limit,
                    "evaluated": 0,
                }

        logger.info("Evaluating %d entries (daily usage %d/%d)...", len(entries), daily_used, eff_daily_limit)

        judge = AttributionJudge(
            base_url=base_url,
            model=model,
            api_key=api_key,
            db_url=async_dsn,
            tenant_id=eff_tid_str,
        )

        results: list[dict[str, Any]] = []
        budget_exhausted = False
        contradicted_marked_total = 0
        implicit_pos_total = 0

        for entry in entries:
            entry_id = entry["id"]
            query = entry["query"]
            response = entry["agent_response"]
            result_ids = entry.get("result_ids")

            # Skip explicitly flagged noise entries
            if entry.get("is_noise") is True:
                if not dry_run:
                    await buffer.mark_evaluated(
                        entry_id,
                        faithfulness_score=0.0,
                        judge_reasoning="skipped: noise",
                    )
                logger.info("  [%s] skipped (is_noise=true) query=%s", entry_id[:8], query[:50])
                continue

            # Skip entries with no retrieval results — nothing to evaluate usage against
            if not result_ids:
                if not dry_run:
                    await buffer.mark_evaluated(
                        entry_id,
                        faithfulness_score=0.0,
                        judge_reasoning="skipped: no results",
                    )
                    await buffer.sync_to_search_log(
                        entry_id,
                        faithfulness_score=0.0,
                        relevance_score=0.0,
                        was_used=False,
                        judgment_source="attribution",
                    )
                logger.info("  [%s] skipped (no result_ids) query=%s", entry_id[:8], query[:50])
                results.append(
                    {
                        "id": entry_id,
                        "query": query[:100],
                        "score": 0.0,
                        "relevance": 0.0,
                        "was_used": False,
                        "adopted_ids": [],
                        "contradicted_ids": [],
                        "supported": 0,
                        "total": 0,
                    }
                )
                continue

            # Load structured experience/archive contexts under RLS
            contexts = await load_structured_contexts(async_dsn, result_ids, tenant_id=eff_tid_str)

            # Skip if contexts resolved to empty (all result_ids invalid/deleted)
            if not contexts:
                if not dry_run:
                    await buffer.mark_evaluated(
                        entry_id,
                        faithfulness_score=0.0,
                        judge_reasoning="skipped: no valid contexts",
                    )
                    await buffer.sync_to_search_log(
                        entry_id,
                        faithfulness_score=0.0,
                        relevance_score=0.0,
                        was_used=False,
                        judgment_source="attribution",
                    )
                logger.info("  [%s] skipped (empty contexts) query=%s", entry_id[:8], query[:50])
                results.append(
                    {
                        "id": entry_id,
                        "query": query[:100],
                        "score": 0.0,
                        "relevance": 0.0,
                        "was_used": False,
                        "adopted_ids": [],
                        "contradicted_ids": [],
                        "supported": 0,
                        "total": 0,
                    }
                )
                continue

            # Enforce daily LLM call cap mid-batch
            if eff_daily_limit > 0 and daily_used >= eff_daily_limit:
                logger.warning(
                    "Daily attribution eval limit reached mid-batch (%d/%d); stopping.",
                    daily_used,
                    eff_daily_limit,
                )
                budget_exhausted = True
                break

            daily_used += 1
            eval_result = await judge.evaluate(
                query=query,
                contexts=contexts,
                response=response,
            )

            score = eval_result.score
            max_rel = getattr(eval_result, "max_relevance", 0.0)
            adopted_ids = list(getattr(eval_result, "adopted_ids", []) or [])
            contradicted_ids = list(getattr(eval_result, "contradicted_ids", []) or [])
            was_used = len(adopted_ids) > 0
            reasoning = json.dumps(eval_result.claims_detail, ensure_ascii=False)[:4000]

            logger.info(
                "  [%s] score=%.2f rel=%.2f adopted=%d contradicted=%d query=%s",
                entry_id[:8],
                score,
                max_rel,
                len(adopted_ids),
                len(contradicted_ids),
                query[:50],
            )

            if not dry_run:
                await buffer.mark_evaluated(
                    entry_id,
                    faithfulness_score=score,
                    judge_reasoning=reasoning,
                )
                if score >= 0:
                    await buffer.sync_to_search_log(
                        entry_id,
                        faithfulness_score=score,
                        relevance_score=max_rel,
                        was_used=was_used,
                        judgment_source="attribution",
                    )
                    if was_used and score >= 0.7:
                        try:
                            pos_res = await record_implicit_positive_feedback(
                                db_url=async_dsn,
                                result_ids=result_ids,
                                faithfulness_score=score,
                                was_used=True,
                                used_indices=eval_result.used_indices,
                                adopted_ids=adopted_ids,
                                relevance_score=max_rel,
                                search_log_id=entry.get("search_log_id"),
                                tenant_id=eff_tid_str,
                            )
                            implicit_pos_total += int(pos_res.get("recorded", 0))
                        except Exception as fb_err:
                            logger.warning("Implicit positive feedback failed: %s", fb_err)

                    if contradicted_ids:
                        try:
                            contra_res = await handle_contradicted_experiences(
                                db_url=async_dsn,
                                contradicted_ids=contradicted_ids,
                                search_log_id=entry.get("search_log_id"),
                                tenant_id=eff_tid_str,
                            )
                            contradicted_marked_total += int(contra_res.get("marked_stale", 0))
                        except Exception as c_err:
                            logger.warning("Contradiction staleness handling failed: %s", c_err)

            results.append(
                {
                    "id": entry_id,
                    "query": query[:100],
                    "score": score,
                    "relevance": max_rel,
                    "was_used": was_used,
                    "adopted_ids": adopted_ids,
                    "contradicted_ids": contradicted_ids,
                    "supported": eval_result.supported_claims,
                    "total": eval_result.total_claims,
                }
            )

        # Run periodic implicit negative feedback scan after batch evaluation
        implicit_neg_summary: dict[str, Any] = {}
        if not dry_run and results:
            try:
                implicit_neg_summary = await scan_and_record_implicit_negative_feedback(
                    db_url=async_dsn,
                    tenant_id=eff_tid_str,
                )
            except Exception as fb_err:
                logger.warning("Implicit negative feedback scan failed: %s", fb_err)

        # Summary
        scores = [r["score"] for r in results if r["score"] >= 0]
        avg_score = sum(scores) / len(scores) if scores else 0.0
        high = sum(1 for s in scores if s >= 0.7)
        low = sum(1 for s in scores if 0 <= s < 0.7)
        errors = sum(1 for r in results if r["score"] < 0)

        summary: dict[str, Any] = {
            "evaluated": len(results),
            "avg_faithfulness": round(avg_score, 3),
            "high_faithfulness": high,
            "low_faithfulness": low,
            "errors": errors,
            "dry_run": dry_run,
            "budget_exhausted": budget_exhausted,
            "daily_used": daily_used,
            "daily_limit": eff_daily_limit,
            "implicit_positive_recorded": implicit_pos_total,
            "implicit_negative_recorded": implicit_neg_summary.get("recorded", 0),
            "contradicted_marked_stale": contradicted_marked_total,
        }

        logger.info(
            "[EVAL] batch complete: evaluated=%d avg=%.3f high=%d low=%d errors=%d "
            "pos_fb=%d neg_fb=%d stale=%d daily=%d/%d",
            summary["evaluated"],
            summary["avg_faithfulness"],
            high,
            low,
            errors,
            implicit_pos_total,
            summary["implicit_negative_recorded"],
            contradicted_marked_total,
            daily_used,
            eff_daily_limit,
        )

        return summary

    finally:
        await buffer.close()


class AttributionEvalWorker:
    """Persistent async background worker consuming ``response_buffer`` inside TM Daemon.

    Replaces the legacy ``subprocess.Popen(..., stdout=DEVNULL, stderr=DEVNULL)``
    pattern with an observable, in-process asyncio worker featuring:
    - Event-driven instant wake-up via ``trigger()`` + periodic fallback polling.
    - Concurrency guard via ``asyncio.Lock`` (never overlaps two batches).
    - Automatic retry with exponential backoff (up to ``max_retries=3``).
    - Pause/resume hooks for laptop sleep/wake recovery.
    - Structured logging to ``daemon.eval_worker``.
    """

    def __init__(
        self,
        *,
        dsn: str,
        batch_size: int = 20,
        poll_interval_seconds: float = 60.0,
        max_retries: int = 3,
        retry_base_delay: float = 2.0,
        daily_limit: int | None = None,
        model: str | None = None,
        base_url: str | None = None,
        api_key: str | None = None,
        tenant_id: str | None = None,
    ) -> None:
        self._dsn = dsn
        self._batch_size = batch_size
        self._poll_interval = poll_interval_seconds
        self._max_retries = max_retries
        self._retry_base_delay = retry_base_delay
        self._daily_limit = daily_limit
        self._model = model
        self._base_url = base_url
        self._api_key = api_key
        self._tenant_id = tenant_id

        self._trigger_event = asyncio.Event()
        self._run_lock = asyncio.Lock()
        self._stop_event = asyncio.Event()
        self._paused = False
        self._task: asyncio.Task[None] | None = None

    def start(self) -> asyncio.Task[None]:
        """Start the persistent worker loop as an asyncio Task."""
        if self._task is None or self._task.done():
            self._stop_event.clear()
            self._task = asyncio.create_task(self._loop(), name="attribution-eval-worker")
            worker_logger.info(
                "[EVAL-WORKER] started (poll=%.0fs, batch_size=%d, max_retries=%d, daily_limit=%s)",
                self._poll_interval,
                self._batch_size,
                self._max_retries,
                self._daily_limit if self._daily_limit is not None else DEFAULT_DAILY_EVAL_LIMIT,
            )
        return self._task

    def trigger(self) -> None:
        """Non-blocking signal to wake the worker immediately."""
        self._trigger_event.set()
        worker_logger.debug("[EVAL-WORKER] triggered by hook")

    def pause(self) -> None:
        self._paused = True

    def resume(self) -> None:
        self._paused = False
        self._trigger_event.set()

    def stop(self) -> None:
        self._stop_event.set()
        self._trigger_event.set()

    async def run_once_with_retry(self, *, force: bool = False) -> dict[str, Any] | None:
        """Execute one batch pass under lock, retrying up to ``max_retries`` times."""
        if self._paused:
            worker_logger.debug("[EVAL-WORKER] paused; skipping run")
            return None

        async with self._run_lock:
            last_exc: Exception | None = None
            for attempt in range(1, self._max_retries + 1):
                try:
                    result = await run_batch(
                        dsn=self._dsn,
                        model=self._model,
                        base_url=self._base_url,
                        api_key=self._api_key,
                        batch_size=self._batch_size,
                        force=force,
                        daily_limit=self._daily_limit,
                        tenant_id=self._tenant_id,
                    )
                    if not result.get("skipped"):
                        worker_logger.info(
                            "[EVAL-WORKER] completed attempt=%d evaluated=%d avg=%.3f pos_fb=%d neg_fb=%d stale=%d",
                            attempt,
                            result.get("evaluated", 0),
                            result.get("avg_faithfulness", 0.0),
                            result.get("implicit_positive_recorded", 0),
                            result.get("implicit_negative_recorded", 0),
                            result.get("contradicted_marked_stale", 0),
                        )
                    return result
                except asyncio.CancelledError:
                    raise
                except Exception as exc:
                    last_exc = exc
                    if attempt < self._max_retries:
                        delay = self._retry_base_delay * (2 ** (attempt - 1))
                        worker_logger.warning(
                            "[EVAL-WORKER] batch failed (attempt %d/%d), retrying in %.1fs: %s",
                            attempt,
                            self._max_retries,
                            delay,
                            exc,
                            exc_info=True,
                        )
                        await asyncio.sleep(delay)
                    else:
                        worker_logger.error(
                            "[EVAL-WORKER] batch failed after %d attempts: %s",
                            self._max_retries,
                            exc,
                            exc_info=True,
                        )
            return {"error": str(last_exc), "evaluated": 0}

    async def _loop(self) -> None:
        """Main background loop waiting for trigger events or periodic timeout."""
        try:
            while not self._stop_event.is_set():
                triggered = False
                try:
                    await asyncio.wait_for(self._trigger_event.wait(), timeout=self._poll_interval)
                    triggered = True
                except asyncio.TimeoutError:
                    triggered = False

                self._trigger_event.clear()
                if self._stop_event.is_set():
                    break

                await self.run_once_with_retry(force=triggered)
        except asyncio.CancelledError:
            worker_logger.info("[EVAL-WORKER] cancelled, shutting down cleanly")
            raise


async def show_recent(dsn: str, limit: int = 10) -> None:
    """Show recent buffer entries with their faithfulness scores."""
    import asyncpg

    conn = await asyncpg.connect(dsn)
    try:
        rows = await conn.fetch(
            """
            SELECT id::text, query, agent_response, result_ids,
                   evaluated, faithfulness_score, judge_reasoning, created_at::text
            FROM response_buffer
            ORDER BY created_at DESC
            LIMIT $1
            """,
            limit,
        )

        if not rows:
            print("No entries in response_buffer.")
            return

        print(f"\n{'=' * 60}")
        print(f"Recent {len(rows)} Response Buffer Entries")
        print(f"{'=' * 60}\n")

        for i, r in enumerate(rows, 1):
            score = r["faithfulness_score"]
            evaluated = r["evaluated"]
            query = r["query"][:80]
            response = (r["agent_response"] or "")[:150]
            created = (r["created_at"] or "")[:19]

            result_ids = r["result_ids"]
            if isinstance(result_ids, str):
                result_ids = json.loads(result_ids)
            n_results = len(result_ids) if result_ids else 0

            if evaluated and score is not None:
                if score >= 0.7:
                    score_str = f"\033[32m{score:.2f} ✓\033[0m"
                elif score >= 0:
                    score_str = f"\033[31m{score:.2f} ✗\033[0m"
                else:
                    score_str = "\033[33merror\033[0m"
            elif evaluated:
                score_str = "evaluated (no score)"
            else:
                score_str = "\033[33mpending\033[0m"

            print(f"  [{i}] {created}  score={score_str}  results={n_results}")
            print(f"      Query:    {query}")
            print(f"      Response: {response}...")
            print()

    finally:
        await conn.close()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    parser = argparse.ArgumentParser(description="Attribution & faithfulness batch evaluator")
    parser.add_argument("--model", default=None, help="LLM model name")
    parser.add_argument("--base-url", default=None, help="LLM base URL")
    parser.add_argument("--api-key", default=None, help="LLM API key")
    parser.add_argument("--batch-size", type=int, default=20, help="Max entries per run")
    parser.add_argument("--daily-limit", type=int, default=None, help="Daily LLM evaluation cap")
    parser.add_argument("--dry-run", action="store_true", help="Don't write results")
    parser.add_argument("--force", action="store_true", help="Ignore threshold")
    parser.add_argument(
        "--recent",
        type=int,
        default=0,
        help="Evaluate the N most recent buffer entries immediately",
    )
    parser.add_argument(
        "--show",
        type=int,
        default=0,
        help="Show the N most recent buffer entries with scores",
    )
    parser.add_argument("--db-url", default=None, help="Database URL")
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="/tmp/tm-faithfulness-checkpoint.txt",
        help="Checkpoint file (stores last evaluated entry ID)",
    )
    args = parser.parse_args()

    dsn = (args.db_url or os.environ.get("TEAM_MEMORY_DB_URL") or "").strip()
    if not dsn:
        parser.error("Database URL required: pass --db-url or set TEAM_MEMORY_DB_URL.")

    if args.show > 0:
        asyncio.run(show_recent(dsn=dsn, limit=args.show))
        return

    if args.recent > 0:
        result = asyncio.run(
            run_batch(
                dsn=dsn,
                model=args.model,
                base_url=args.base_url,
                api_key=args.api_key,
                batch_size=args.recent,
                dry_run=args.dry_run,
                force=True,
                checkpoint_path=args.checkpoint,
                daily_limit=args.daily_limit,
            )
        )
    else:
        result = asyncio.run(
            run_batch(
                dsn=dsn,
                model=args.model,
                base_url=args.base_url,
                api_key=args.api_key,
                batch_size=args.batch_size,
                dry_run=args.dry_run,
                checkpoint_path=args.checkpoint,
                force=args.force,
                daily_limit=args.daily_limit,
            )
        )

    if result.get("skipped"):
        sys.exit(0)


if __name__ == "__main__":
    main()

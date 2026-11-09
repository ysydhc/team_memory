#!/usr/bin/env python3
"""Faithfulness Batch Evaluator — processes pending entries in response_buffer.

Usage:
    python -m team_memory.daemon.quality.faithfulness_batch [OPTIONS]

Options:
    --model MODEL       LLM model for judge (default: from env or gbu-gemini-3.8-flash)
    --base-url URL      LLM base URL (default: from env or https://gw.gburouter.com/v1)
    --api-key KEY       LLM API key (default: from env GATEWAY_API_KEY)
    --batch-size N      Max entries per run (default: 20)
    --dry-run           Evaluate but don't write results to DB
    --force             Evaluate all pending, ignore threshold
    --db-url URL        Database URL
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import sys

from team_memory.project_resolver import find_project_root, load_dotenv_if_exists

load_dotenv_if_exists(find_project_root() or os.path.dirname(__file__))

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("faithfulness_batch")


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
) -> dict:
    """Run a batch faithfulness evaluation."""

    from team_memory.daemon.buffers.response_buffer import ResponseBuffer
    from team_memory.daemon.quality.faithfulness_judge import FaithfulnessJudge
    from team_memory.services.maintenance.batch_ops import BatchOpsService

    raw_dsn = dsn.replace("postgresql+asyncpg://", "postgresql://", 1)
    async_dsn = dsn.replace("postgresql://", "postgresql+asyncpg://", 1) if "+asyncpg" not in dsn else dsn
    buffer = ResponseBuffer(dsn=raw_dsn)
    batch_svc = BatchOpsService(db_url=async_dsn)

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

        # Fetch unevaluated entries (ordered by created_at ASC)
        entries = await buffer.fetch_pending(limit=batch_size)
        if not entries:
            logger.info("No entries to evaluate.")
            return {"skipped": False, "evaluated": 0}

        logger.info("Evaluating %d entries...", len(entries))

        # Init judge
        judge = FaithfulnessJudge(
            base_url=base_url,
            model=model,
            api_key=api_key,
        )

        results = []
        for entry in entries:
            entry_id = entry["id"]
            query = entry["query"]
            response = entry["agent_response"]
            result_ids = entry["result_ids"]

            # Skip entries with no retrieval results — nothing to evaluate usage against
            if not result_ids:
                if not dry_run:
                    await buffer.mark_evaluated(
                        entry_id,
                        faithfulness_score=0.0,
                        judge_reasoning="skipped: no results",
                    )
                    # Record the verdict on the search_log too, else it stays
                    # was_used=NULL forever and pollutes the unjudged count.
                    await buffer.sync_to_search_log(entry_id, faithfulness_score=0.0)
                logger.info("  [%s] skipped (no result_ids) query=%s", entry_id[:8], query[:50])
                results.append(
                    {
                        "id": entry_id,
                        "query": query[:100],
                        "score": 0.0,
                        "supported": 0,
                        "total": 0,
                    }
                )
                continue

            # Load experience context
            contexts = await batch_svc.load_experience_contexts(result_ids)

            # Skip if contexts resolved to empty (all result_ids invalid/deleted)
            if not contexts:
                if not dry_run:
                    await buffer.mark_evaluated(
                        entry_id,
                        faithfulness_score=0.0,
                        judge_reasoning="skipped: no valid contexts",
                    )
                    await buffer.sync_to_search_log(entry_id, faithfulness_score=0.0)
                logger.info("  [%s] skipped (empty contexts) query=%s", entry_id[:8], query[:50])
                results.append(
                    {
                        "id": entry_id,
                        "query": query[:100],
                        "score": 0.0,
                        "supported": 0,
                        "total": 0,
                    }
                )
                continue

            # Evaluate
            eval_result = await judge.evaluate(
                query=query,
                contexts=contexts,
                response=response,
            )

            score = eval_result.score
            reasoning = json.dumps(eval_result.claims_detail, ensure_ascii=False)[:2000]

            logger.info(
                "  [%s] score=%.2f claims=%d/%d query=%s",
                entry_id[:8],
                score,
                eval_result.supported_claims,
                eval_result.total_claims,
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
                    )

            results.append(
                {
                    "id": entry_id,
                    "query": query[:100],
                    "score": score,
                    "supported": eval_result.supported_claims,
                    "total": eval_result.total_claims,
                }
            )

        # Summary
        scores = [r["score"] for r in results if r["score"] >= 0]
        avg_score = sum(scores) / len(scores) if scores else 0
        high = sum(1 for s in scores if s >= 0.7)
        low = sum(1 for s in scores if s < 0.7 and s >= 0)
        errors = sum(1 for r in results if r["score"] < 0)

        summary = {
            "evaluated": len(results),
            "avg_faithfulness": round(avg_score, 3),
            "high_faithfulness": high,
            "low_faithfulness": low,
            "errors": errors,
            "dry_run": dry_run,
        }

        # Print report
        print("\n" + "=" * 50)
        print("Faithfulness Batch Report")
        print("=" * 50)
        print(f"Evaluated:    {summary['evaluated']}")
        print(f"Avg score:    {summary['avg_faithfulness']}")
        print(f"High (>=0.7): {summary['high_faithfulness']}")
        print(f"Low  (<0.7):  {summary['low_faithfulness']}")
        if errors:
            print(f"Errors:       {errors}")
        if dry_run:
            print("(DRY RUN — no results written to DB)")

        # Show low faithfulness entries
        low_entries = [r for r in results if 0 <= r["score"] < 0.7]
        if low_entries:
            print("\nLow faithfulness entries:")
            for r in low_entries:
                print(f"  score={r['score']:.2f} claims={r['supported']}/{r['total']} query={r['query']}")

        return summary

    finally:
        await buffer.close()


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

            # Parse result_ids
            result_ids = r["result_ids"]
            if isinstance(result_ids, str):
                import json

                result_ids = json.loads(result_ids)
            n_results = len(result_ids) if result_ids else 0

            # Score display
            if evaluated and score is not None:
                if score >= 0.7:
                    score_str = f"\033[32m{score:.2f} ✓\033[0m"  # green
                elif score >= 0:
                    score_str = f"\033[31m{score:.2f} ✗\033[0m"  # red
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


def main():
    parser = argparse.ArgumentParser(description="Faithfulness batch evaluator")
    parser.add_argument("--model", default=None, help="LLM model name")
    parser.add_argument("--base-url", default=None, help="LLM base URL")
    parser.add_argument("--api-key", default=None, help="LLM API key")
    parser.add_argument("--batch-size", type=int, default=20, help="Max entries per run")
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

    # Show mode: display recent entries
    if args.show > 0:
        asyncio.run(show_recent(dsn=dsn, limit=args.show))
        return

    # Recent mode: evaluate N most recent entries immediately
    if args.recent > 0:
        result = asyncio.run(
            run_batch(
                dsn=dsn,
                model=args.model,
                base_url=args.base_url,
                api_key=args.api_key,
                batch_size=args.recent,
                dry_run=args.dry_run,
                force=True,  # Always force when --recent is specified
                checkpoint_path=args.checkpoint,
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
            )
        )

    if result.get("skipped"):
        sys.exit(0)


if __name__ == "__main__":
    main()

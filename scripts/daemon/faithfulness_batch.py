#!/usr/bin/env python3
"""Faithfulness Batch Evaluator — processes pending entries in response_buffer.

Usage:
    python scripts/daemon/faithfulness_batch.py [OPTIONS]

Options:
    --model MODEL       LLM model for judge (default: from env or glm-4-flash)
    --base-url URL      LLM base URL (default: from env or http://localhost:4000/v1)
    --api-key KEY       LLM API key (default: from env LITELLM_MASTER_KEY)
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

# Add project paths
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../src"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("faithfulness_batch")


async def load_experience_contexts(
    dsn: str,
    result_ids: list[dict] | None,
) -> list[str]:
    """Load experience titles/solutions for the given result IDs."""
    if not result_ids:
        return []

    import asyncpg
    ids = []
    for r in result_ids:
        rid = r.get("id", "")
        if rid:
            try:
                import uuid as _uuid
                ids.append(_uuid.UUID(rid))
            except (ValueError, AttributeError):
                continue

    if not ids:
        return []

    conn = await asyncpg.connect(dsn)
    try:
        rows = await conn.fetch(
            """
            SELECT id, title, problem, solution
            FROM experiences
            WHERE id = ANY($1)
            """,
            ids,
        )
        contexts = []
        for r in rows:
            parts = []
            if r["title"]:
                parts.append(f"标题: {r['title']}")
            if r["problem"]:
                parts.append(f"问题: {r['problem'][:200]}")
            if r["solution"]:
                parts.append(f"方案: {r['solution'][:300]}")
            contexts.append(" | ".join(parts))
        return contexts
    finally:
        await conn.close()


async def run_batch(
    *,
    dsn: str,
    model: str | None = None,
    base_url: str | None = None,
    api_key: str | None = None,
    batch_size: int = 20,
    dry_run: bool = False,
    force: bool = False,
) -> dict:
    """Run a batch faithfulness evaluation."""
    from daemon.response_buffer import ResponseBuffer
    from daemon.faithfulness_judge import FaithfulnessJudge

    buffer = ResponseBuffer(dsn=dsn)

    try:
        # Check if we should run
        pending = await buffer.pending_count()
        logger.info("Pending entries: %d", pending)

        if not force and pending < buffer._batch_threshold:
            logger.info(
                "Below threshold (%d < %d), skipping. Use --force to override.",
                pending, buffer._batch_threshold,
            )
            return {"skipped": True, "pending": pending}

        # Fetch entries to evaluate
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

            # Load experience context
            contexts = await load_experience_contexts(dsn, result_ids)

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
                entry_id[:8], score,
                eval_result.supported_claims, eval_result.total_claims,
                query[:50],
            )

            if not dry_run:
                await buffer.mark_evaluated(
                    entry_id,
                    faithfulness_score=score,
                    judge_reasoning=reasoning,
                )
                await buffer.sync_to_search_log(
                    entry_id,
                    faithfulness_score=score,
                )

            results.append({
                "id": entry_id,
                "query": query[:100],
                "score": score,
                "supported": eval_result.supported_claims,
                "total": eval_result.total_claims,
            })

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
            print(f"\nLow faithfulness entries:")
            for r in low_entries:
                print(f"  score={r['score']:.2f} claims={r['supported']}/{r['total']} query={r['query']}")

        return summary

    finally:
        await buffer.close()


def main():
    parser = argparse.ArgumentParser(description="Faithfulness batch evaluator")
    parser.add_argument("--model", default=None, help="LLM model name")
    parser.add_argument("--base-url", default=None, help="LLM base URL")
    parser.add_argument("--api-key", default=None, help="LLM API key")
    parser.add_argument("--batch-size", type=int, default=20, help="Max entries per run")
    parser.add_argument("--dry-run", action="store_true", help="Don't write results")
    parser.add_argument("--force", action="store_true", help="Ignore threshold")
    parser.add_argument("--db-url", default=None, help="Database URL")
    args = parser.parse_args()

    dsn = args.db_url or "postgresql://developer:devpass@localhost:5433/team_memory"

    result = asyncio.run(run_batch(
        dsn=dsn,
        model=args.model,
        base_url=args.base_url,
        api_key=args.api_key,
        batch_size=args.batch_size,
        dry_run=args.dry_run,
        force=args.force,
    ))

    if result.get("skipped"):
        sys.exit(0)


if __name__ == "__main__":
    main()

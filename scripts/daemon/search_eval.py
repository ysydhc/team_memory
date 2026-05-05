#!/usr/bin/env python3
"""Search Quality Evaluator — tests recall precision against known queries.

Usage:
    python scripts/daemon/search_eval.py [--db-url URL] [--output FILE]

Reads test queries from config or stdin, runs them through the search pipeline,
and reports precision/recall metrics.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
from datetime import datetime, timezone

# Add project paths
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../src"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

# Default test queries — each has a query and expected experience ids (or titles)
DEFAULT_TEST_QUERIES = [
    {
        "query": "Docker PostgreSQL connection",
        "expected_keywords": ["docker", "postgresql", "postgres", "connection"],
        "min_results": 1,
    },
    {
        "query": "WKWebView cookie sync",
        "expected_keywords": ["wkwebview", "cookie", "webview"],
        "min_results": 1,
    },
    {
        "query": "embedding model configuration",
        "expected_keywords": ["embedding", "model", "config"],
        "min_results": 1,
    },
    {
        "query": "entity extraction",
        "expected_keywords": ["entity", "extraction"],
        "min_results": 1,
    },
    {
        "query": "task group completion",
        "expected_keywords": ["task", "group", "completion"],
        "min_results": 1,
    },
    {
        "query": "asyncpg pgvector",
        "expected_keywords": ["asyncpg", "pgvector", "postgres"],
        "min_results": 1,
    },
    {
        "query": "Makefile target",
        "expected_keywords": ["makefile", "target", "make"],
        "min_results": 1,
    },
    {
        "query": "refinement worker",
        "expected_keywords": ["refinement", "worker", "refine"],
        "min_results": 1,
    },
    {
        "query": "wiki compilation",
        "expected_keywords": ["wiki", "compil"],
        "min_results": 1,
    },
    {
        "query": "search pipeline RRF",
        "expected_keywords": ["search", "pipeline", "rrf", "fusion"],
        "min_results": 1,
    },
]


async def run_evaluation(db_url: str) -> dict:
    """Run evaluation and return metrics."""
    from team_memory.bootstrap import bootstrap
    from team_memory.services.memory_operations import op_recall

    # Bootstrap the AppContext
    bootstrap(enable_background=False)

    results = []
    for test in DEFAULT_TEST_QUERIES:
        query = test["query"]
        expected_keywords = [kw.lower() for kw in test["expected_keywords"]]
        min_results = test["min_results"]

        try:
            recall_result = await op_recall(
                "eval",
                query=query,
                max_results=10,
            )
            recall_results = recall_result.get("results", [])
        except Exception as e:
            results.append({
                "query": query,
                "status": "error",
                "error": str(e),
                "hits": 0,
                "precision": 0.0,
            })
            continue

        # Check if results contain expected keywords
        hits = 0
        for r in recall_results:
            title = (r.get("title") or "").lower()
            desc = (r.get("description") or "").lower()
            solution = (r.get("solution") or "").lower()
            tags = [t.lower() for t in (r.get("tags") or [])]
            text = f"{title} {desc} {solution} {' '.join(tags)}"

            # Check if any expected keyword appears in the result
            if any(kw in text for kw in expected_keywords):
                hits += 1

        precision = hits / len(recall_results) if recall_results else 0.0
        recall = 1.0 if hits >= min_results else 0.0

        results.append({
            "query": query,
            "status": "ok",
            "total_results": len(recall_results),
            "hits": hits,
            "precision": precision,
            "recall": recall,
            "top_result": recall_results[0].get("title", "") if recall_results else "",
        })

    # Compute aggregate metrics
    ok_results = [r for r in results if r["status"] == "ok"]
    avg_precision = sum(r["precision"] for r in ok_results) / len(ok_results) if ok_results else 0.0
    avg_recall = sum(r["recall"] for r in ok_results) / len(ok_results) if ok_results else 0.0
    hit_rate = sum(1 for r in ok_results if r["total_results"] > 0) / len(ok_results) if ok_results else 0.0

    return {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "total_queries": len(results),
        "successful_queries": len(ok_results),
        "avg_precision": round(avg_precision, 3),
        "avg_recall": round(avg_recall, 3),
        "hit_rate": round(hit_rate, 3),
        "details": results,
    }


def print_report(metrics: dict) -> None:
    """Print evaluation report in a readable format."""
    print(f"\n{'='*60}")
    print(f"  Search Quality Evaluation Report")
    print(f"  {metrics['timestamp']}")
    print(f"{'='*60}\n")

    print(f"  Queries tested:     {metrics['total_queries']}")
    print(f"  Successful:         {metrics['successful_queries']}")
    print(f"  Hit rate:           {metrics['hit_rate']:.1%}")
    print(f"  Avg precision:      {metrics['avg_precision']:.1%}")
    print(f"  Avg recall:         {metrics['avg_recall']:.1%}")

    print(f"\n{'─'*60}")
    print(f"  {'Query':<35} {'Hits':>5} {'Prec':>6} {'Recall':>7}")
    print(f"{'─'*60}")

    for r in metrics["details"]:
        if r["status"] == "error":
            print(f"  {r['query']:<35} {'ERR':>5}")
        else:
            print(f"  {r['query']:<35} {r['hits']:>5} {r['precision']:>6.1%} {r['recall']:>7.1%}")

    print(f"{'─'*60}\n")


async def main():
    parser = argparse.ArgumentParser(description="Evaluate search quality")
    parser.add_argument("--db-url", help="PostgreSQL connection URL")
    parser.add_argument("--output", help="Output JSON file path")
    parser.add_argument("--json", action="store_true", help="Output JSON to stdout")
    args = parser.parse_args()

    # Get db_url from args, env, or settings
    db_url = args.db_url
    if not db_url:
        db_url = os.environ.get("TEAM_MEMORY_DATABASE__URL")
    if not db_url:
        from team_memory.config import load_settings
        settings = load_settings()
        db_url = str(settings.database.url)

    metrics = await run_evaluation(db_url)

    if args.json:
        print(json.dumps(metrics, ensure_ascii=False, indent=2))
    else:
        print_report(metrics)

    if args.output:
        with open(args.output, "w") as f:
            json.dump(metrics, f, ensure_ascii=False, indent=2)
        print(f"Report saved to: {args.output}")


if __name__ == "__main__":
    asyncio.run(main())

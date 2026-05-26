#!/usr/bin/env python3
"""A/B Evaluation: Single-pass vs Adaptive Multi-hop Retrieval.

Runs complex queries through both retrieval modes and compares
FaithfulnessJudge scores to quantify the improvement.

Usage:
    PYTHONPATH=src python -m tests.eval_adaptive_ab
"""
from __future__ import annotations

import asyncio
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from team_memory.project_resolver import load_dotenv_if_exists

load_dotenv_if_exists(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

EVAL_QUERIES = [
    {
        "query": "为什么之前的搜索方案被废弃，现在的 RRF 融合方案是如何解决旧方案缺陷的？",
        "description": "Multi-hop: requires finding old approach + new approach + comparison",
    },
    {
        "query": "embedding 模型从旧版本迁移到新版本时遇到了什么问题，最终是怎么解决的？",
        "description": "Multi-hop: requires migration history + problem + solution chain",
    },
    {
        "query": "entity extraction 和 search pipeline 是如何协作的？二者的数据流是什么？",
        "description": "Multi-hop: requires understanding two subsystems and their interaction",
    },
]


async def run_single_pass(ctx, query: str, max_results: int = 10) -> dict:
    """Run standard single-pass retrieval."""
    from team_memory.services.memory.memory_operations import op_recall

    return await op_recall(
        "eval",
        search_orchestrator=ctx.search_orchestrator,
        db_url=ctx.db_url,
        settings=ctx.settings,
        embedding_provider=ctx.embedding,
        query=query,
        max_results=max_results,
        adaptive=False,
    )


async def run_adaptive(ctx, query: str, max_results: int = 10) -> dict:
    """Run adaptive multi-hop retrieval."""
    from team_memory.services.memory.memory_operations import op_recall

    return await op_recall(
        "eval",
        search_orchestrator=ctx.search_orchestrator,
        db_url=ctx.db_url,
        settings=ctx.settings,
        embedding_provider=ctx.embedding,
        query=query,
        max_results=max_results,
        adaptive=True,
        adaptive_retriever=ctx.adaptive_retriever,
    )


def format_context(results: list[dict]) -> list[str]:
    """Extract context strings from search results for FaithfulnessJudge."""
    contexts = []
    for r in results:
        title = r.get("title", "")
        solution = r.get("solution", r.get("description", ""))
        contexts.append(f"{title}: {solution}")
    return contexts


async def main():
    from team_memory.bootstrap import bootstrap, get_context
    from team_memory.daemon.quality.faithfulness_judge import FaithfulnessJudge

    bootstrap(enable_background=False)
    ctx = get_context()
    judge = FaithfulnessJudge()

    print(f"\n{'=' * 70}")
    print("  A/B Evaluation: Single-pass vs Adaptive Multi-hop Retrieval")
    print(f"{'=' * 70}\n")

    results_table = []

    for eq in EVAL_QUERIES:
        query = eq["query"]
        print(f"  Query: {query[:60]}...")

        try:
            single_result = await run_single_pass(ctx, query)
        except Exception as e:
            print(f"    [Single-pass ERROR] {e}")
            single_result = {"results": []}

        try:
            adaptive_result = await run_adaptive(ctx, query)
        except Exception as e:
            print(f"    [Adaptive ERROR] {e}")
            adaptive_result = {"results": [], "adaptive": {"hops": 0, "queries_used": []}}

        single_contexts = format_context(single_result.get("results", []))
        adaptive_contexts = format_context(adaptive_result.get("results", []))

        single_response = "\n".join(single_contexts) if single_contexts else "(no results)"
        adaptive_response = "\n".join(adaptive_contexts) if adaptive_contexts else "(no results)"

        try:
            single_score = await judge.evaluate(
                query=query, contexts=single_contexts, response=single_response
            )
            s_score = single_score.score
        except Exception as e:
            print(f"    [Judge ERROR single] {e}")
            s_score = -1.0

        try:
            adaptive_score = await judge.evaluate(
                query=query, contexts=adaptive_contexts, response=adaptive_response
            )
            a_score = adaptive_score.score
        except Exception as e:
            print(f"    [Judge ERROR adaptive] {e}")
            a_score = -1.0

        hops = adaptive_result.get("adaptive", {}).get("hops", 1)
        row = {
            "query": query[:50],
            "single_results": len(single_result.get("results", [])),
            "single_score": s_score,
            "adaptive_results": len(adaptive_result.get("results", [])),
            "adaptive_score": a_score,
            "adaptive_hops": hops,
            "delta": round(a_score - s_score, 3) if (a_score >= 0 and s_score >= 0) else None,
        }
        results_table.append(row)
        print(f"    Single: {row['single_results']} results, score={row['single_score']:.3f}")
        print(f"    Adaptive: {row['adaptive_results']} results ({hops} hops), score={row['adaptive_score']:.3f}")
        if row["delta"] is not None:
            print(f"    Delta: {row['delta']:+.3f}")
        print()

    # Summary
    valid_rows = [r for r in results_table if r["delta"] is not None]
    if valid_rows:
        avg_single = sum(r["single_score"] for r in valid_rows) / len(valid_rows)
        avg_adaptive = sum(r["adaptive_score"] for r in valid_rows) / len(valid_rows)
        avg_delta = avg_adaptive - avg_single
    else:
        avg_single = avg_adaptive = avg_delta = 0.0

    print(f"{'─' * 70}")
    print(f"  Average Single-pass Score:  {avg_single:.3f}")
    print(f"  Average Adaptive Score:     {avg_adaptive:.3f}")
    print(f"  Average Delta:              {avg_delta:+.3f}")
    print(f"{'─' * 70}\n")

    report = {
        "queries": results_table,
        "avg_single_score": round(avg_single, 3),
        "avg_adaptive_score": round(avg_adaptive, 3),
        "avg_delta": round(avg_delta, 3),
    }
    report_path = os.path.join(os.path.dirname(__file__), "..", "eval_adaptive_report.json")
    with open(report_path, "w") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(f"  Report saved to: {report_path}")


if __name__ == "__main__":
    asyncio.run(main())

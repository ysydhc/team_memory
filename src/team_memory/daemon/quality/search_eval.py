#!/usr/bin/env python3
"""Search Quality Evaluator — tests recall precision against known queries.

Usage:
    python -m team_memory.daemon.quality.search_eval [--db-url URL] [--output FILE]

Reads test queries from config or stdin, runs them through the search pipeline,
and reports precision/recall metrics.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import math
import os
from datetime import datetime, timezone
from typing import Any

from team_memory.project_resolver import load_dotenv_if_exists

load_dotenv_if_exists(os.path.dirname(__file__))


def compute_recall_at_k(retrieved_ids: list[str], relevant_ids: set[str], k: int = 5) -> float:
    """Compute Recall@K for a single query."""
    if not relevant_ids or k <= 0:
        return 0.0
    top_k = set(retrieved_ids[:k])
    hits = len(top_k & relevant_ids)
    return hits / len(relevant_ids)


def compute_mrr(retrieved_ids: list[str], relevant_ids: set[str]) -> float:
    """Compute Reciprocal Rank (RR) for a single query."""
    if not relevant_ids:
        return 0.0
    for rank, doc_id in enumerate(retrieved_ids, start=1):
        if doc_id in relevant_ids:
            return 1.0 / rank
    return 0.0


def compute_ndcg_at_k(
    retrieved_ids: list[str],
    relevant_ids: set[str] | dict[str, float],
    k: int = 5,
) -> float:
    """Compute Normalized Discounted Cumulative Gain (NDCG@K) for a single query."""
    if not relevant_ids or k <= 0:
        return 0.0

    if isinstance(relevant_ids, dict):
        gains = {doc_id: float(rel) for doc_id, rel in relevant_ids.items() if float(rel) > 0}
    else:
        gains = {doc_id: 1.0 for doc_id in relevant_ids}

    if not gains:
        return 0.0

    dcg = 0.0
    seen: set[str] = set()
    for idx, doc_id in enumerate(retrieved_ids[:k]):
        if doc_id in seen:
            continue
        seen.add(doc_id)
        rel = gains.get(doc_id, 0.0)
        if rel > 0.0:
            dcg += rel / math.log2(idx + 2)

    ideal_rels = sorted(gains.values(), reverse=True)[:k]
    idcg = sum(rel / math.log2(idx + 2) for idx, rel in enumerate(ideal_rels))
    return dcg / idcg if idcg > 0.0 else 0.0


def compute_fama_at_k(
    retrieved_ids: list[str],
    active_relevant_ids: set[str],
    superseded_ids: set[str],
    k: int = 5,
) -> float:
    """Compute Forgetting-Aware Memory Accuracy (FAMA@K).

    Returns 1.0 if at least one active relevant experience appears in top-K
    AND zero superseded/expired experiences appear in top-K; otherwise 0.0.
    """
    if not active_relevant_ids or k <= 0:
        return 0.0
    top_k = set(retrieved_ids[:k])
    has_active = bool(top_k & active_relevant_ids)
    has_superseded = bool(top_k & superseded_ids)
    return 1.0 if (has_active and not has_superseded) else 0.0


def evaluate_benchmark_suite(
    cases: list[dict[str, Any]],
    k: int = 5,
    *,
    include_recall_at_1: bool = False,
) -> dict[str, float]:
    """Aggregate IR and FAMA metrics across an offline benchmark suite."""
    if not cases:
        base_empty = {
            "avg_recall_at_5": 0.0,
            "mrr": 0.0,
            "avg_ndcg_at_5": 0.0,
            "fama_at_5": 0.0,
            "multihop_recall_at_5": 0.0,
        }
        if include_recall_at_1:
            base_empty["recall_at_1"] = 0.0
        return base_empty

    recalls: list[float] = []
    mrrs: list[float] = []
    ndcgs: list[float] = []
    famas: list[float] = []
    multihop_recalls: list[float] = []

    for case in cases:
        retrieved = [str(x) for x in (case.get("retrieved_ids") or [])]
        raw_rel = case.get("relevant_ids")
        if raw_rel is None:
            raw_rel = case.get("active_relevant_ids") or set()
        if isinstance(raw_rel, dict):
            rel_set = {str(doc_id) for doc_id, score in raw_rel.items() if float(score) > 0}
            rel_for_ndcg: set[str] | dict[str, float] = {str(doc_id): float(score) for doc_id, score in raw_rel.items()}
        else:
            rel_set = {str(x) for x in raw_rel}
            rel_for_ndcg = rel_set

        active_raw = case.get("active_relevant_ids")
        active_set = {str(x) for x in active_raw} if active_raw is not None else rel_set
        superseded_set = {str(x) for x in (case.get("superseded_ids") or [])}

        r_k = compute_recall_at_k(retrieved, rel_set, k=k)
        recalls.append(r_k)
        mrrs.append(compute_mrr(retrieved, rel_set))
        ndcgs.append(compute_ndcg_at_k(retrieved, rel_for_ndcg, k=k))
        famas.append(compute_fama_at_k(retrieved, active_set, superseded_set, k=k))

        if case.get("is_multihop") or case.get("category") == "multihop" or int(case.get("hops", 1) or 1) > 1:
            multihop_recalls.append(r_k)

    avg_recall = sum(recalls) / len(recalls)
    avg_multihop = sum(multihop_recalls) / len(multihop_recalls) if multihop_recalls else avg_recall
    recalls_at_1 = [
        compute_recall_at_k(
            [str(x) for x in (c.get("retrieved_ids") or [])],
            (
                {str(doc_id) for doc_id, s in c["relevant_ids"].items() if float(s) > 0}
                if isinstance(c.get("relevant_ids"), dict)
                else {str(x) for x in (c.get("relevant_ids") or c.get("active_relevant_ids") or [])}
            ),
            k=1,
        )
        for c in cases
    ]

    out = {
        "avg_recall_at_5": round(avg_recall, 4),
        "mrr": round(sum(mrrs) / len(mrrs), 4),
        "avg_ndcg_at_5": round(sum(ndcgs) / len(ndcgs), 4),
        "fama_at_5": round(sum(famas) / len(famas), 4),
        "multihop_recall_at_5": round(avg_multihop, 4),
    }
    if include_recall_at_1:
        out["recall_at_1"] = round(sum(recalls_at_1) / len(recalls_at_1), 4)
    return out


# Standard 24-query Ground-Truth IR Benchmark Suite across 4 retrieval categories:
# 1. semantic_paraphrase: natural-language symptom descriptions with low title lexical overlap
# 2. exact_identifier: exact code symbols, config keys, method names, and ticket IDs
# 3. multihop_graph: 2-hop entity-graph queries where upstream entity is not in target title
# 4. cross_project_rerank: dense cross-project queries testing reranking & score calibration
IR_BENCHMARK_SUITE: list[dict[str, Any]] = [
    # --- Category 1: semantic_paraphrase (6 queries) ---
    {
        "id": "q_sem_01",
        "category": "semantic_paraphrase",
        "project": "default",
        "query": "苹果客户端在后台全局队列刷新界面控件崩溃，或者主队列同步派发卡死怎么处理",
        "relevant_ids": ["057287ca-546e-467c-94a2-071e3cea5a7e"],
        "expected_keywords": ["uikit", "线程安全", "死锁"],
        "is_multihop": False,
        "hops": 1,
    },
    {
        "id": "q_sem_02",
        "category": "semantic_paraphrase",
        "project": "default",
        "query": "用户在广告展示中途直接划掉进程强退应用，数仓里怎么查这种生命周期中断事件",
        "relevant_ids": ["2cb62f69-bbda-43a7-ad6c-156b96d00626"],
        "expected_keywords": ["lifecycle_interrupt", "user_kill", "强杀"],
        "is_multihop": False,
        "hops": 1,
    },
    {
        "id": "q_sem_03",
        "category": "semantic_paraphrase",
        "project": "default",
        "query": "Mac 电脑上怎么把团结引擎或者 Unity 彻底删干净，不留历史缓存和配置文件",
        "relevant_ids": ["79798f04-3614-4187-a706-1c33a57e52a0"],
        "expected_keywords": ["卸载 unity", "残留配置", "tuanjie"],
        "is_multihop": False,
        "hops": 1,
    },
    {
        "id": "q_sem_04",
        "category": "semantic_paraphrase",
        "project": "default",
        "query": "安卓无障碍自动化抓取页面控件树时部分节点莫名消失不见了怎么解决",
        "relevant_ids": ["3a696797-6dee-4e54-bf92-b334b90f90b0"],
        "expected_keywords": ["uiautomator", "压缩模式", "节点静默丢失"],
        "is_multihop": False,
        "hops": 1,
    },
    {
        "id": "q_sem_05",
        "category": "semantic_paraphrase",
        "project": "team_memory",
        "query": "大模型网关高并发报 429 频率超限，后台异步提炼经验的队列卡住不动了",
        "relevant_ids": ["4134ed3a-3628-46bc-916c-3c049ede2476"],
        "expected_keywords": ["429", "glm-5-turbo", "精炼任务"],
        "is_multihop": False,
        "hops": 1,
    },
    {
        "id": "q_sem_06",
        "category": "semantic_paraphrase",
        "project": "default",
        "query": "苹果端点击页面按钮没反应或者手势事件直接透传到下层视图了怎么排查",
        "relevant_ids": ["3cbcd343-334a-4c40-ac07-ea98d287c7bb"],
        "expected_keywords": ["触摸事件穿透", "命中失效", "hittest"],
        "is_multihop": False,
        "hops": 1,
    },
    # --- Category 2: exact_identifier (6 queries) ---
    {
        "id": "q_lex_01",
        "category": "exact_identifier",
        "project": "team_memory",
        "query": "cold-start",
        "relevant_ids": ["e5227822-8326-4868-b1b9-3d809f5bcb5a"],
        "expected_keywords": ["冷启动", "step-0", "入口"],
        "is_multihop": False,
        "hops": 1,
    },
    {
        "id": "q_lex_02",
        "category": "exact_identifier",
        "project": "default",
        "query": "call_id Responses API 64",
        "relevant_ids": ["ef48294c-da71-497a-a7d9-013b3b0e68e3"],
        "expected_keywords": ["call_id", "responses api", "64"],
        "is_multihop": False,
        "hops": 1,
    },
    {
        "id": "q_lex_03",
        "category": "exact_identifier",
        "project": "default",
        "query": "InApp AdX auto_play",
        "relevant_ids": ["7b1b3a6c-22de-4144-b92f-d1e138e9e7ad"],
        "expected_keywords": ["auto_play", "autoplay", "inapp adx"],
        "is_multihop": False,
        "hops": 1,
    },
    {
        "id": "q_lex_04",
        "category": "exact_identifier",
        "project": "default",
        "query": "SelfHostedCrashHandler LRU 双闸门",
        "relevant_ids": ["bf2eb0e9-7ec9-45c3-be8c-73c4ca1275c6"],
        "expected_keywords": ["selfhostedcrashhandler", "lru", "双闸门"],
        "is_multihop": False,
        "hops": 1,
    },
    {
        "id": "q_lex_05",
        "category": "exact_identifier",
        "project": "team_memory",
        "query": "memory_save embedding_failed Ollama",
        "relevant_ids": ["cac8f55f-a65b-4c58-ad3d-37fb5863352a"],
        "expected_keywords": ["embedding_failed", "ollama", "memory_save"],
        "is_multihop": False,
        "hops": 1,
    },
    {
        "id": "q_lex_06",
        "category": "exact_identifier",
        "project": "default",
        "query": "EDM4U Resolver not enabled",
        "relevant_ids": ["d2dec0ac-9b2d-429a-a4ed-0bb03f8aff35"],
        "expected_keywords": ["edm4u", "resolver not enabled", "android resolver"],
        "is_multihop": False,
        "hops": 1,
    },
    # --- Category 3: multihop_graph (6 queries, 2-hop entity traversal) ---
    {
        "id": "q_hop_01",
        "category": "multihop_graph",
        "project": "adx_sdk_android",
        "query": "PublicConfigCache",
        "relevant_ids": ["cb0a82a8-df78-4973-9075-c8fab95a6723"],
        "expected_keywords": ["对外暴露接口", "npe", "url"],
        "is_multihop": True,
        "hops": 2,
    },
    {
        "id": "q_hop_02",
        "category": "multihop_graph",
        "project": "adx_sdk_android",
        "query": "ku-doc-manage",
        "relevant_ids": ["aadb3934-4659-4cd7-871d-62e739c3f069"],
        "expected_keywords": ["knowbase", "ugate", "token"],
        "is_multihop": True,
        "hops": 2,
    },
    {
        "id": "q_hop_03",
        "category": "multihop_graph",
        "project": "adx_sdk_android",
        "query": "EndCardGuideClickHandler",
        "relevant_ids": ["f6c15e9b-afb6-4ff2-bb0a-b1dca8840fd7"],
        "expected_keywords": ["漏斗打点", "硬编码实验上报", "单一职责"],
        "is_multihop": True,
        "hops": 2,
    },
    {
        "id": "q_hop_04",
        "category": "multihop_graph",
        "project": "adx_sdk_android",
        "query": "com.hkjtl.game.tpmzxly.mihw",
        "relevant_ids": ["7db17cdc-7c2c-492b-9642-9b31901159ae"],
        "expected_keywords": ["apk", "dex", "版本"],
        "is_multihop": True,
        "hops": 2,
    },
    {
        "id": "q_hop_05",
        "category": "multihop_graph",
        "project": "adx_sdk_android",
        "query": "design_guess_you_like_recall_component.md",
        "relevant_ids": ["87f3f431-e405-4da7-931f-81d67d52f9e0"],
        "expected_keywords": ["猜你喜欢", "渲染性能", "api"],
        "is_multihop": True,
        "hops": 2,
    },
    {
        "id": "q_hop_06",
        "category": "multihop_graph",
        "project": "inapp_game_crystal_mist",
        "query": "MobRewardVideoActivity",
        "relevant_ids": ["1c67e7f1-1952-471f-bdf9-ed443ed529c6"],
        "expected_keywords": ["预测性返回", "广告页", "android 13"],
        "is_multihop": True,
        "hops": 2,
    },
    # --- Category 4: cross_project_rerank (6 queries) ---
    {
        "id": "q_rnk_01",
        "category": "cross_project_rerank",
        "project": "default",
        "query": "通过 AppOps 与 Frida 探测 Android 第三方 SDK 读取全量已安装应用列表",
        "relevant_ids": ["0ad7b323-6550-4629-a4cd-156c91f76d1a"],
        "expected_keywords": ["appops", "frida", "全量应用列表"],
        "is_multihop": False,
        "hops": 1,
    },
    {
        "id": "q_rnk_02",
        "category": "cross_project_rerank",
        "project": "default",
        "query": "vanniktech maven publish 插件升级后 publish 任务不联网静默失败",
        "relevant_ids": ["86037bd1-1fd6-4560-b541-296330acfdb2"],
        "expected_keywords": ["vanniktech", "maven central", "静默失败"],
        "is_multihop": False,
        "hops": 1,
    },
    {
        "id": "q_rnk_03",
        "category": "cross_project_rerank",
        "project": "default",
        "query": "小米商店审核拒审：SDK 启动触发 Google Play Store 弹窗的 GMS 可用性检查修复",
        "relevant_ids": ["6f17e689-e9a9-4ba7-8ac4-3bd1fa973709"],
        "expected_keywords": ["小米商店", "gms", "google play store"],
        "is_multihop": False,
        "hops": 1,
    },
    {
        "id": "q_rnk_04",
        "category": "cross_project_rerank",
        "project": "default",
        "query": "Alembic 升级 010 迁移失败时易被掩盖的真实原因",
        "relevant_ids": ["21f2e050-1c16-4a3c-8a9f-7c01779011f4"],
        "expected_keywords": ["alembic", "010", "迁移失败"],
        "is_multihop": False,
        "hops": 1,
    },
    {
        "id": "q_rnk_05",
        "category": "cross_project_rerank",
        "project": "team_memory",
        "query": "Hermes v13 shell hook 4层修复：事件名与 stdin 字段迁移",
        "relevant_ids": ["c25461a7-7a4b-4a63-a8f8-cc3ef1ba8c30"],
        "expected_keywords": ["hermes", "v13", "stdin"],
        "is_multihop": False,
        "hops": 1,
    },
    {
        "id": "q_rnk_06",
        "category": "cross_project_rerank",
        "project": "default",
        "query": "Android Media3 SimpleCache 视频下载中断导致脏缓存残留及误判修复",
        "relevant_ids": ["1b0ab50e-455f-43f6-b921-cbbd2dd29387"],
        "expected_keywords": ["simplecache", "media3", "脏缓存"],
        "is_multihop": False,
        "hops": 1,
    },
]

ABLATION_CONFIGS: list[dict[str, Any]] = [
    {
        "key": "L1_vector_only",
        "label": "L1: Vector Only (Dense Cosine)",
        "mode": "vector",
        "enable_graph": False,
        "enable_topic": False,
        "enable_exact_boost": False,
        "popularity_enabled": False,
        "rerank_enabled": False,
    },
    {
        "key": "L2_hybrid_rrf2",
        "label": "L2: Hybrid RRF2 (Vector + Jieba FTS)",
        "mode": "hybrid",
        "enable_graph": False,
        "enable_topic": False,
        "enable_exact_boost": False,
        "popularity_enabled": False,
        "rerank_enabled": False,
    },
    {
        "key": "L3_hybrid_rrf4_graph",
        "label": "L3: Hybrid RRF4 + 2-Hop Graph PPR",
        "mode": "hybrid",
        "enable_graph": True,
        "enable_topic": True,
        "enable_exact_boost": False,
        "popularity_enabled": False,
        "rerank_enabled": False,
    },
    {
        "key": "L4_full_pipeline",
        "label": "L4: Full Pipeline (RRF4 + Graph + Rerank + Boost)",
        "mode": "hybrid",
        "enable_graph": True,
        "enable_topic": True,
        "enable_exact_boost": True,
        "popularity_enabled": True,
        "rerank_enabled": True,
    },
]

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
    from team_memory.bootstrap import bootstrap, get_context
    from team_memory.services.memory.memory_operations import op_recall

    # Bootstrap the AppContext
    bootstrap(enable_background=False)

    ctx = get_context()

    results = []
    for test in DEFAULT_TEST_QUERIES:
        query = test["query"]
        expected_keywords = [kw.lower() for kw in test["expected_keywords"]]
        min_results = test["min_results"]

        try:
            recall_result = await op_recall(
                "eval",
                search_orchestrator=ctx.search_orchestrator,
                db_url=ctx.db_url,
                settings=ctx.settings,
                embedding_provider=ctx.embedding,
                query=query,
                max_results=10,
            )
            recall_results = recall_result.get("results", [])
        except Exception as e:
            results.append(
                {
                    "query": query,
                    "status": "error",
                    "error": str(e),
                    "hits": 0,
                    "precision": 0.0,
                }
            )
            continue

        # Check if results contain expected keywords
        hits = 0
        retrieved_ids: list[str] = []
        matched_ids: set[str] = set()
        for idx, r in enumerate(recall_results):
            doc_id = str(r.get("id") or f"idx-{idx}")
            retrieved_ids.append(doc_id)
            title = (r.get("title") or "").lower()
            desc = (r.get("description") or "").lower()
            solution = (r.get("solution") or "").lower()
            tags = [t.lower() for t in (r.get("tags") or [])]
            text = f"{title} {desc} {solution} {' '.join(tags)}"

            # Check if any expected keyword appears in the result
            if any(kw in text for kw in expected_keywords):
                hits += 1
                matched_ids.add(doc_id)

        relevant_ids = matched_ids if matched_ids else {"__unmatched__"}
        precision = hits / len(recall_results) if recall_results else 0.0
        recall = 1.0 if hits >= min_results else 0.0
        mrr = compute_mrr(retrieved_ids, relevant_ids)
        ndcg_at_5 = compute_ndcg_at_k(retrieved_ids, relevant_ids, k=5)

        results.append(
            {
                "query": query,
                "status": "ok",
                "total_results": len(recall_results),
                "hits": hits,
                "precision": precision,
                "recall": recall,
                "mrr": mrr,
                "ndcg_at_5": ndcg_at_5,
                "top_result": (recall_results[0].get("title", "") if recall_results else ""),
            }
        )

    # Compute aggregate metrics
    ok_results = [r for r in results if r["status"] == "ok"]
    avg_precision = sum(r["precision"] for r in ok_results) / len(ok_results) if ok_results else 0.0
    avg_recall = sum(r["recall"] for r in ok_results) / len(ok_results) if ok_results else 0.0
    avg_mrr = sum(r["mrr"] for r in ok_results) / len(ok_results) if ok_results else 0.0
    avg_ndcg_at_5 = sum(r["ndcg_at_5"] for r in ok_results) / len(ok_results) if ok_results else 0.0
    hit_rate = sum(1 for r in ok_results if r["total_results"] > 0) / len(ok_results) if ok_results else 0.0

    return {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "total_queries": len(results),
        "successful_queries": len(ok_results),
        "avg_precision": round(avg_precision, 3),
        "avg_recall": round(avg_recall, 3),
        "mrr": round(avg_mrr, 3),
        "ndcg_at_5": round(avg_ndcg_at_5, 3),
        "hit_rate": round(hit_rate, 3),
        "details": results,
    }


def print_report(metrics: dict) -> None:
    """Print evaluation report in a readable format."""
    print(f"\n{'=' * 60}")
    print("  Search Quality Evaluation Report")
    print(f"  {metrics['timestamp']}")
    print(f"{'=' * 60}\n")

    print(f"  Queries tested:     {metrics['total_queries']}")
    print(f"  Successful:         {metrics['successful_queries']}")
    print(f"  Hit rate:           {metrics['hit_rate']:.1%}")
    print(f"  Avg precision:      {metrics['avg_precision']:.1%}")
    print(f"  Avg recall:         {metrics['avg_recall']:.1%}")
    if "mrr" in metrics:
        print(f"  MRR:                {metrics['mrr']:.3f}")
    if "ndcg_at_5" in metrics:
        print(f"  NDCG@5:             {metrics['ndcg_at_5']:.3f}")

    print(f"\n{'─' * 74}")
    print(f"  {'Query':<33} {'Hits':>5} {'Prec':>6} {'Recall':>7} {'MRR':>7} {'NDCG@5':>8}")
    print(f"{'─' * 74}")

    for r in metrics["details"]:
        if r["status"] == "error":
            print(f"  {r['query']:<33} {'ERR':>5}")
        else:
            mrr_val = r.get("mrr", 0.0)
            ndcg_val = r.get("ndcg_at_5", 0.0)
            print(
                f"  {r['query']:<33} {r['hits']:>5} {r['precision']:>6.1%} "
                f"{r['recall']:>7.1%} {mrr_val:>7.3f} {ndcg_val:>8.3f}"
            )

    print(f"{'─' * 74}\n")


async def run_ablation_benchmark(
    db_url: str,
    suite: list[dict[str, Any]] | None = None,
    k: int = 5,
) -> dict[str, Any]:
    """Run 4-stage IR ablation study across IR_BENCHMARK_SUITE.

    Compares:
      - L1_vector_only: Dense cosine similarity only
      - L2_hybrid_rrf2: Vector + Jieba FTS (2-way RRF)
      - L3_hybrid_rrf4_graph: Vector + FTS + Topic + 2-Hop PPR Entity Graph (4-way RRF)
      - L4_full_pipeline: Full TeamMemory Pipeline (+ Exact Match / Bayesian Boost / Reranker)
    """
    import time

    from team_memory.bootstrap import bootstrap, get_context
    from team_memory.services.search.search_pipeline import SearchRequest, get_session

    bootstrap(enable_background=False)
    ctx = get_context()
    pipeline = ctx.search_pipeline
    scfg = pipeline._search_config

    benchmark_cases = suite if suite is not None else IR_BENCHMARK_SUITE

    orig_mode = getattr(scfg, "mode", "hybrid")
    orig_pop = getattr(scfg, "popularity_enabled", True)
    orig_graph = getattr(scfg, "enable_graph", True)
    orig_topic = getattr(scfg, "enable_topic", True)
    orig_exact = getattr(scfg, "enable_exact_boost", True)
    orig_rerank = pipeline._rerank_enabled

    layers_out: dict[str, dict[str, Any]] = {}
    try:
        for cfg in ABLATION_CONFIGS:
            layer_key = cfg["key"]
            if hasattr(pipeline, "_cache") and hasattr(pipeline._cache, "_result_cache"):
                await pipeline._cache._result_cache.clear()

            scfg.mode = cfg["mode"]
            scfg.enable_graph = cfg["enable_graph"]
            scfg.enable_topic = cfg["enable_topic"]
            scfg.enable_exact_boost = cfg["enable_exact_boost"]
            scfg.popularity_enabled = cfg["popularity_enabled"]
            pipeline._rerank_enabled = bool(cfg["rerank_enabled"] and pipeline._reranker is not None)

            evaluated_cases: list[dict[str, Any]] = []
            latencies_ms: list[float] = []

            async with get_session(db_url) as session:
                for item in benchmark_cases:
                    proj = item.get("project") or "default"
                    fallback = ["*"] if proj in ("default", "team_memory") else []
                    req = SearchRequest(
                        query=item["query"],
                        project=proj,
                        fallback_projects=fallback,
                        max_results=10,
                        min_similarity=0.45,
                        include_promoted=True,
                    )
                    t0 = time.monotonic()
                    res = await pipeline.search(session, req)
                    elapsed_ms = (time.monotonic() - t0) * 1000.0
                    latencies_ms.append(elapsed_ms)

                    retrieved_ids: list[str] = []
                    superseded_leaked: set[str] = set()
                    kw_matched_ids: set[str] = set()
                    expected_kws = [kw.lower() for kw in (item.get("expected_keywords") or [])]

                    for idx, r in enumerate(res.results):
                        doc_id = str(r.get("id") or f"idx-{idx}")
                        retrieved_ids.append(doc_id)
                        title = str(r.get("title") or "")
                        if r.get("valid_until") is not None or not title.strip() or title.startswith("Draft "):
                            superseded_leaked.add(doc_id)
                        blob = (
                            f"{title} {r.get('description') or ''} "
                            f"{r.get('solution') or ''} {' '.join(r.get('tags') or [])}"
                        ).lower()
                        if expected_kws and any(kw in blob for kw in expected_kws):
                            kw_matched_ids.add(doc_id)

                    gt_ids = {str(x) for x in (item.get("relevant_ids") or [])}
                    # Strict Ground-Truth UUID matching when relevant_ids are specified;
                    # fallback to expected_keywords only for custom suites without relevant_ids.
                    effective_rel_ids = gt_ids if gt_ids else (kw_matched_ids or {"__unmatched__"})

                    evaluated_cases.append(
                        {
                            "id": item.get("id", ""),
                            "category": item.get("category", "general"),
                            "query": item["query"],
                            "retrieved_ids": retrieved_ids,
                            "relevant_ids": sorted(effective_rel_ids),
                            "active_relevant_ids": sorted(effective_rel_ids),
                            "superseded_ids": sorted(superseded_leaked),
                            "is_multihop": bool(item.get("is_multihop", False)),
                            "hops": int(item.get("hops", 1) or 1),
                            "top_title": res.results[0].get("title", "") if res.results else "",
                            "latency_ms": round(elapsed_ms, 1),
                        }
                    )

            agg = evaluate_benchmark_suite(evaluated_cases, k=k, include_recall_at_1=True)
            by_cat: dict[str, dict[str, float]] = {}
            categories = sorted({c["category"] for c in evaluated_cases})
            for cat in categories:
                cat_cases = [c for c in evaluated_cases if c["category"] == cat]
                by_cat[cat] = evaluate_benchmark_suite(cat_cases, k=k, include_recall_at_1=True)

            layers_out[layer_key] = {
                "label": cfg["label"],
                **agg,
                "avg_latency_ms": round(sum(latencies_ms) / len(latencies_ms), 1) if latencies_ms else 0.0,
                "by_category": by_cat,
                "cases": evaluated_cases,
            }
    finally:
        scfg.mode = orig_mode
        scfg.popularity_enabled = orig_pop
        scfg.enable_graph = orig_graph
        scfg.enable_topic = orig_topic
        scfg.enable_exact_boost = orig_exact
        pipeline._rerank_enabled = orig_rerank

    return {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "suite_size": len(benchmark_cases),
        "k": k,
        "layers": layers_out,
    }


def print_ablation_report(report: dict[str, Any]) -> None:
    """Print side-by-side 4-stage IR ablation benchmark report."""
    print(f"\n{'=' * 92}")
    print(f"  TeamMemory Standard IR Benchmark & 4-Stage Ablation Report (N={report['suite_size']} queries)")
    print(f"  {report['timestamp']}")
    print(f"{'=' * 92}\n")

    header = (
        f"  {'Pipeline Configuration':<44} "
        f"{'R@1':>6} {'R@5':>6} {'MRR':>7} {'NDCG@5':>7} {'FAMA@5':>7} {'HopR@5':>7} {'Lat(ms)':>8}"
    )
    print(header)
    print(f"  {'─' * 88}")
    for _key, layer in report["layers"].items():
        print(
            f"  {layer['label']:<44} "
            f"{layer['recall_at_1']:>6.1%} "
            f"{layer['avg_recall_at_5']:>6.1%} "
            f"{layer['mrr']:>7.3f} "
            f"{layer['avg_ndcg_at_5']:>7.3f} "
            f"{layer['fama_at_5']:>6.1%} "
            f"{layer['multihop_recall_at_5']:>6.1%} "
            f"{layer['avg_latency_ms']:>8.1f}"
        )
    print(f"  {'─' * 88}\n")

    full_layer = report["layers"].get("L4_full_pipeline") or next(iter(report["layers"].values()), None)
    if full_layer and full_layer.get("by_category"):
        print("  Category Breakdown (L4 Full Pipeline):")
        print(f"  {'Category':<26} {'R@1':>8} {'R@5':>8} {'MRR':>8} {'NDCG@5':>8} {'FAMA@5':>8}")
        print(f"  {'─' * 70}")
        for cat, cmetrics in full_layer["by_category"].items():
            print(
                f"  {cat:<26} "
                f"{cmetrics['recall_at_1']:>8.1%} "
                f"{cmetrics['avg_recall_at_5']:>8.1%} "
                f"{cmetrics['mrr']:>8.3f} "
                f"{cmetrics['avg_ndcg_at_5']:>8.3f} "
                f"{cmetrics['fama_at_5']:>8.1%}"
            )
        print(f"  {'─' * 70}\n")


async def main():
    parser = argparse.ArgumentParser(description="Evaluate search quality")
    parser.add_argument("--db-url", help="PostgreSQL connection URL")
    parser.add_argument("--output", help="Output JSON file path")
    parser.add_argument("--json", action="store_true", help="Output JSON to stdout")
    parser.add_argument(
        "--ablation",
        "--benchmark",
        dest="ablation",
        action="store_true",
        help="Run the 24-query Standard IR Benchmark & 4-Stage Ablation Study",
    )
    args = parser.parse_args()

    # Get db_url from args, env, or settings
    db_url = args.db_url
    if not db_url:
        db_url = os.environ.get("TEAM_MEMORY_DATABASE__URL")
    if not db_url:
        from team_memory.config import load_settings

        settings = load_settings()
        db_url = str(settings.database.url)

    if args.ablation:
        metrics = await run_ablation_benchmark(db_url)
        if args.json:
            print(json.dumps(metrics, ensure_ascii=False, indent=2))
        else:
            print_ablation_report(metrics)
    else:
        metrics = await run_evaluation(db_url)
        if args.json:
            print(json.dumps(metrics, ensure_ascii=False, indent=2))
        else:
            print_report(metrics)

    if args.output:
        from pathlib import Path

        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w") as f:
            json.dump(metrics, f, ensure_ascii=False, indent=2)
        print(f"Report saved to: {args.output}")


if __name__ == "__main__":
    asyncio.run(main())

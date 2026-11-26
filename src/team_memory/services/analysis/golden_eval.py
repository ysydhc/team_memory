"""Golden Eval — automated ablation benchmark (Vector vs Hybrid vs Full vs Full+Reranker).

Measures retrieval quality on a curated golden set (`evals/golden_set.json`)
using controlled SearchPipeline configurations:
  - A (vector_only): mode="vector", enable_graph=False, enable_topic=False, rerank_enabled=False
  - B (hybrid):      mode="hybrid", enable_graph=False, enable_topic=True,  rerank_enabled=<off or prod>
  - C (full):        mode="hybrid", enable_graph=True,  enable_topic=True,  rerank_enabled=<off or prod>
  - D (full_rerank): mode="hybrid", enable_graph=True,  enable_topic=True,  rerank_enabled=True (optional 4th arm)

Keeping `rerank_enabled` aligned between B and C isolates the pure marginal
contribution of the Knowledge Graph: `Graph Lift = MRR@5(C) - MRR@5(B)`.
When the 4th arm D is enabled, B and C run with `rerank_enabled=False` so that
`Rerank Lift = MRR@5(D) - MRR@5(C)` isolates the pure marginal contribution
of server-side reranking on top of Full (Hybrid+Graph).
"""

from __future__ import annotations

import json
import logging
import math
import re
import subprocess
import time
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from sqlalchemy import text as sa_text

from team_memory.config import CacheConfig, SearchConfig
from team_memory.storage.database import get_session

logger = logging.getLogger("team_memory.golden_eval")

REPO_ROOT = Path(__file__).resolve().parents[4]
DEFAULT_GOLDEN_SET_PATH = REPO_ROOT / "evals" / "golden_set.json"
DEFAULT_HISTORY_PATH = REPO_ROOT / "evals" / "history.jsonl"

_URL_ONLY_RE = re.compile(r"^\s*(https?://\S+|/\S+|\./\S+|~/\S+)\s*$", re.IGNORECASE)
_PURE_PUNCT_OR_NUM_RE = re.compile(r"^[\W\d_]+$")
_NOISE_PREFIX_RE = re.compile(
    r"^\s*(<[a-zA-Z0-9_-]+>|#{1,6}\s|\d{2}:\d{2}:\d{2}\.\d+|ERROR:|Traceback|Provider returned error:|\*\*)",
    re.IGNORECASE,
)
_PROBE_QUERIES = {
    "ping",
    "test",
    "hello",
    "hi",
    "health",
    "status",
    "smoke",
    "smoke test",
    "测试",
    "你好",
}


# ============================================================
# IR Metric Functions
# ============================================================


def compute_recall_at_k(retrieved_ids: list[str], expected_ids: set[str], k: int = 5) -> float:
    """Compute Recall@K for a single query."""
    if not expected_ids or k <= 0:
        return 0.0
    top_k = set(retrieved_ids[:k])
    return len(top_k & expected_ids) / len(expected_ids)


def compute_mrr_at_k(retrieved_ids: list[str], expected_ids: set[str], k: int = 5) -> float:
    """Compute Reciprocal Rank@K (RR@K) for a single query."""
    if not expected_ids or k <= 0:
        return 0.0
    for rank, doc_id in enumerate(retrieved_ids[:k], start=1):
        if doc_id in expected_ids:
            return 1.0 / rank
    return 0.0


def compute_ndcg_at_k(
    retrieved_ids: list[str],
    expected_ids: set[str] | dict[str, float],
    k: int = 5,
) -> float:
    """Compute Normalized Discounted Cumulative Gain (NDCG@K) for a single query."""
    if not expected_ids or k <= 0:
        return 0.0

    if isinstance(expected_ids, dict):
        gains = {str(doc_id): float(rel) for doc_id, rel in expected_ids.items() if float(rel) > 0}
    else:
        gains = {str(doc_id): 1.0 for doc_id in expected_ids}

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


def compute_percentile(values: list[float], pct: float) -> float:
    """Compute linear-interpolated percentile (0-100) for a list of numeric values."""
    if not values:
        return 0.0
    ordered = sorted(float(v) for v in values)
    if len(ordered) == 1:
        return ordered[0]
    rank = (max(0.0, min(100.0, pct)) / 100.0) * (len(ordered) - 1)
    lo = int(math.floor(rank))
    hi = int(math.ceil(rank))
    if lo == hi:
        return ordered[lo]
    weight = rank - lo
    return ordered[lo] * (1.0 - weight) + ordered[hi] * weight


def estimate_result_tokens(results: list[dict[str, Any]]) -> int:
    """Estimate returned token count for a list of experience result dicts."""
    total_chars = 0
    for item in results:
        if not isinstance(item, dict):
            continue
        for key in ("title", "problem", "description", "solution"):
            val = item.get(key)
            if isinstance(val, str):
                total_chars += len(val)
    # Mixed Chinese/English heuristic: ~2.5 chars per token
    return int(round(total_chars / 2.5)) if total_chars > 0 else 0


def summarize_mode_metrics(
    per_query_records: list[dict[str, Any]],
    k: int = 5,
) -> dict[str, Any]:
    """Aggregate Recall@K, MRR@K, NDCG@K, P50/P95 latency, and Avg Tokens."""
    if not per_query_records:
        return {
            "query_count": 0,
            "recall_at_5": 0.0,
            "mrr_at_5": 0.0,
            "ndcg_at_5": 0.0,
            "p50_latency_ms": 0.0,
            "p95_latency_ms": 0.0,
            "avg_tokens": 0.0,
            "hit_rate_at_5": 0.0,
        }

    recalls: list[float] = []
    mrrs: list[float] = []
    ndcgs: list[float] = []
    latencies: list[float] = []
    tokens: list[int] = []
    hits = 0

    for rec in per_query_records:
        retrieved = [str(x) for x in (rec.get("retrieved_ids") or [])]
        expected = {str(x) for x in (rec.get("expected_ids") or [])}
        r = compute_recall_at_k(retrieved, expected, k=k)
        m = compute_mrr_at_k(retrieved, expected, k=k)
        n = compute_ndcg_at_k(retrieved, expected, k=k)
        recalls.append(r)
        mrrs.append(m)
        ndcgs.append(n)
        if r > 0:
            hits += 1
        latencies.append(float(rec.get("latency_ms", 0.0)))
        tokens.append(int(rec.get("tokens", 0)))

    count = len(per_query_records)
    return {
        "query_count": count,
        "recall_at_5": round(sum(recalls) / count, 4),
        "mrr_at_5": round(sum(mrrs) / count, 4),
        "ndcg_at_5": round(sum(ndcgs) / count, 4),
        "p50_latency_ms": round(compute_percentile(latencies, 50), 1),
        "p95_latency_ms": round(compute_percentile(latencies, 95), 1),
        "avg_tokens": round(sum(tokens) / count, 1),
        "hit_rate_at_5": round(hits / count, 4),
    }


# ============================================================
# Noise Filtering & Golden Set Construction
# ============================================================


def is_noise_query(query: str, known_projects: set[str] | None = None) -> bool:
    """Return True if a search_logs query is low-signal noise."""
    if not query or not isinstance(query, str):
        return True
    if "\n" in query.strip():
        return True
    if _NOISE_PREFIX_RE.match(query):
        return True
    q = " ".join(query.strip().split())
    if len(q) < 6 or len(q) > 160:
        return True
    lower = q.lower()
    if lower in _PROBE_QUERIES:
        return True
    if _URL_ONLY_RE.match(q):
        return True
    if _PURE_PUNCT_OR_NUM_RE.match(q):
        return True
    if known_projects and lower in {p.lower() for p in known_projects if p}:
        return True
    # Pure single-word identifier with no spaces and no CJK (usually project/tag probe)
    has_cjk = any("\u4e00" <= ch <= "\u9fff" for ch in q)
    if not has_cjk and len(q.split()) == 1 and len(q) < 12:
        return True
    return False


_TOKEN_SPLIT_RE = re.compile(r"[a-zA-Z0-9_.-]{2,}|[\u4e00-\u9fff]{2,}")


def _has_query_title_overlap(query: str, title: str, tags: list[str] | None = None) -> bool:
    """Return True if query and candidate title/tags share at least one meaningful token."""
    q_low = (query or "").lower()
    t_low = (title or "").lower()
    if tags:
        t_low = t_low + " " + " ".join(str(tag).lower() for tag in tags if tag)
    tokens = _TOKEN_SPLIT_RE.findall(q_low)
    for tok in tokens:
        if len(tok) >= 3 and tok in t_low:
            return True
        # For 2-char CJK bi-grams inside longer CJK runs, check sliding 2-char windows
        if "\u4e00" <= tok[0] <= "\u9fff":
            for idx in range(len(tok) - 1):
                if tok[idx : idx + 2] in t_low:
                    return True
    return False
def _extract_candidate_ids(result_ids: Any, min_score: float = 0.0) -> list[tuple[str, float]]:
    """Extract (experience_id, score) pairs from search_logs.result_ids JSON."""
    if not result_ids:
        return []
    out: list[tuple[str, float]] = []
    if isinstance(result_ids, list):
        for item in result_ids:
            if isinstance(item, dict):
                eid = item.get("id") or item.get("group_id")
                score = float(item.get("score") or item.get("similarity") or 0.0)
                if eid and score >= min_score:
                    out.append((str(eid), score))
            elif isinstance(item, str) and item:
                out.append((item, 1.0))
    return out


async def build_golden_set_from_db(
    db_url: str,
    *,
    target_size: int = 40,
    min_score: float = 0.01,
    max_per_project: int = 10,
    output_path: Path | None = DEFAULT_GOLDEN_SET_PATH,
) -> dict[str, Any]:
    """Sample 30-50 high-value real queries from search_logs stratified by project."""
    async with get_session(db_url) as session:
        # 1. Active searchable experience metadata lookup (exclude deleted, expired, promoted)
        exp_rows = (
            await session.execute(
                sa_text(
                    "SELECT id::text, title, project, exp_status, tags "
                    "FROM experiences "
                    "WHERE is_deleted = false AND valid_until IS NULL AND exp_status != 'promoted'"
                )
            )
        ).fetchall()
        valid_exps: dict[str, dict[str, Any]] = {
            str(r[0]): {
                "title": str(r[1] or ""),
                "project": str(r[2] or "default"),
                "status": str(r[3] or ""),
                "tags": list(r[4] or []),
            }
            for r in exp_rows
        }
        known_projects = {info["project"] for info in valid_exps.values()}

        # 2. Fetch recent MCP search_logs with non-empty result_ids
        log_rows = (
            await session.execute(
                sa_text(
                    "SELECT query, project, result_ids, was_used, created_at "
                    "FROM search_logs "
                    "WHERE result_ids IS NOT NULL AND source = 'mcp' "
                    "ORDER BY created_at DESC "
                    "LIMIT 4000"
                )
            )
        ).fetchall()

    # 3. Aggregate & score unique normalized queries (preserving rank order via RRF-style votes)
    agg: dict[str, dict[str, Any]] = {}
    for row in log_rows:
        raw_query = str(row[0] or "").strip()
        project = str(row[1] or "default").strip() or "default"
        result_ids = row[2]
        was_used = row[3]

        if is_noise_query(raw_query, known_projects):
            continue

        candidates = [
            (eid, sc)
            for eid, sc in _extract_candidate_ids(result_ids, min_score=min_score)
            if eid in valid_exps
            and (
                was_used is True
                or _has_query_title_overlap(raw_query, valid_exps[eid]["title"], valid_exps[eid].get("tags"))
            )
        ]
        if not candidates:
            continue

        norm_key = " ".join(raw_query.lower().split())
        entry = agg.get(norm_key)
        if entry is None:
            entry = {
                "query": raw_query,
                "project": project,
                "freq": 0,
                "used_hits": 0,
                "candidate_votes": {},
                "candidate_scores": {},
            }
            agg[norm_key] = entry

        entry["freq"] += 1
        if was_used is True:
            entry["used_hits"] += 1
        for rank, (eid, sc) in enumerate(candidates[:5], start=1):
            entry["candidate_votes"][eid] = entry["candidate_votes"].get(eid, 0.0) + (1.0 / rank)
            prev = entry["candidate_scores"].get(eid, 0.0)
            if sc > prev:
                entry["candidate_scores"][eid] = sc

    # 4. Rank within each project by value_score = used_hits*3 + log1p(freq)*2 + min(top_score, 1.0)
    by_project: dict[str, list[dict[str, Any]]] = {}
    for entry in agg.values():
        sorted_cands = sorted(
            entry["candidate_votes"].items(),
            key=lambda kv: (kv[1], min(1.0, entry["candidate_scores"].get(kv[0], 0.0))),
            reverse=True,
        )
        if not sorted_cands:
            continue
        top_eid = sorted_cands[0][0]
        top_score = min(1.0, entry["candidate_scores"].get(top_eid, 0.0))
        value_score = entry["used_hits"] * 3.0 + math.log1p(entry["freq"]) * 2.0 + top_score
        entry["value_score"] = round(value_score, 4)
        entry["sorted_candidates"] = [
            (eid, entry["candidate_scores"].get(eid, 0.0)) for eid, _ in sorted_cands[:3]
        ]
        by_project.setdefault(entry["project"], []).append(entry)

    for items in by_project.values():
        items.sort(key=lambda x: x["value_score"], reverse=True)

    # 5. Stratified round-robin sampling across projects (largest projects first)
    selected: list[dict[str, Any]] = []
    project_order = sorted(by_project.keys(), key=lambda p: len(by_project[p]), reverse=True)
    per_proj_count: dict[str, int] = {p: 0 for p in project_order}

    made_progress = True
    while len(selected) < target_size and made_progress:
        made_progress = False
        for proj in project_order:
            if len(selected) >= target_size:
                break
            idx = per_proj_count[proj]
            if idx >= max_per_project or idx >= len(by_project[proj]):
                continue
            selected.append(by_project[proj][idx])
            per_proj_count[proj] = idx + 1
            made_progress = True

    # 6. Format golden set payload
    queries_out: list[dict[str, Any]] = []
    for i, item in enumerate(selected, start=1):
        cands = item["sorted_candidates"]
        expected = [eid for eid, _ in cands[:2]]

        previews = [
            {
                "id": eid,
                "score": round(sc, 4),
                "title": valid_exps[eid]["title"][:100],
            }
            for eid, sc in cands
        ]
        queries_out.append(
            {
                "id": f"gs_{i:03d}",
                "project": item["project"],
                "query": item["query"],
                "freq": item["freq"],
                "used_hits": item["used_hits"],
                "expected_ids": expected,
                "candidate_previews": previews,
            }
        )

    payload = {
        "version": 1,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "total_queries": len(queries_out),
        "project_distribution": {p: c for p, c in per_proj_count.items() if c > 0},
        "queries": queries_out,
    }

    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    return payload


# ============================================================
# Persistence Helpers (golden_set.json & history.jsonl)
# ============================================================


def load_golden_set(path: Path = DEFAULT_GOLDEN_SET_PATH) -> list[dict[str, Any]]:
    """Load golden queries from JSON file."""
    if not path.exists():
        return []
    raw = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(raw, dict):
        return list(raw.get("queries") or [])
    if isinstance(raw, list):
        return raw
    return []


def load_eval_history(path: Path = DEFAULT_HISTORY_PATH, limit: int = 30) -> list[dict[str, Any]]:
    """Load recent evaluation runs from JSONL history (oldest -> newest within limit)."""
    if not path.exists():
        return []
    records: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            records.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return records[-limit:] if limit > 0 else records


def append_eval_history(record: dict[str, Any], path: Path = DEFAULT_HISTORY_PATH) -> None:
    """Append a single evaluation run record to JSONL history."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(record, ensure_ascii=False) + "\n")


def _get_git_sha() -> str:
    try:
        res = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=str(REPO_ROOT),
            capture_output=True,
            text=True,
            timeout=3,
            check=False,
        )
        return res.stdout.strip() or "unknown"
    except Exception:
        return "unknown"


# ============================================================
# Ablation Runner (A: Vector-only, B: Hybrid, C: Full+Graph, D: Full+Reranker)
# ============================================================


@dataclass
class AblationModeSpec:
    name: str
    label: str
    mode: str
    enable_graph: bool
    enable_topic: bool
    rerank_enabled: bool


def build_ablation_specs(
    prod_rerank_enabled: bool,
    *,
    include_rerank_ablation: bool = False,
) -> list[AblationModeSpec]:
    """Return 3-way (A/B/C) or 4-way (A/B/C/D) controlled ablation specs.

    - 3-way (`include_rerank_ablation=False`):
      B (hybrid) and C (full) share `prod_rerank_enabled` so that
      `Graph Lift = MRR@5(C) - MRR@5(B)` isolates pure graph contribution.
    - 4-way (`include_rerank_ablation=True`):
      B and C disable reranking while D (`full_rerank`) enables it on top of C,
      isolating both `Graph Lift = MRR@5(C) - MRR@5(B)` and
      `Rerank Lift = MRR@5(D) - MRR@5(C)`.
    """
    bc_rerank = False if include_rerank_ablation else prod_rerank_enabled
    specs = [
        AblationModeSpec(
            name="vector_only",
            label="A: Vector-only",
            mode="vector",
            enable_graph=False,
            enable_topic=False,
            rerank_enabled=False,
        ),
        AblationModeSpec(
            name="hybrid",
            label="B: Hybrid (Vec+FTS)",
            mode="hybrid",
            enable_graph=False,
            enable_topic=True,
            rerank_enabled=bc_rerank,
        ),
        AblationModeSpec(
            name="full",
            label="C: Full (Hybrid+Graph)",
            mode="hybrid",
            enable_graph=True,
            enable_topic=True,
            rerank_enabled=bc_rerank,
        ),
    ]
    if include_rerank_ablation:
        specs.append(
            AblationModeSpec(
                name="full_rerank",
                label="D: Full + Reranker",
                mode="hybrid",
                enable_graph=True,
                enable_topic=True,
                rerank_enabled=True,
            )
        )
    return specs


def compute_ablation_lifts(modes_summary: dict[str, Any]) -> dict[str, Any]:
    """Compute Graph Lift (C - B) and optional Rerank Lift + latency delta (D - C)."""
    hybrid = modes_summary.get("hybrid") or {}
    full = modes_summary.get("full") or {}
    mrr_b = float(hybrid.get("mrr_at_5", 0.0))
    mrr_c = float(full.get("mrr_at_5", 0.0))
    recall_b = float(hybrid.get("recall_at_5", 0.0))
    recall_c = float(full.get("recall_at_5", 0.0))

    lifts: dict[str, Any] = {
        "graph_lift": round(mrr_c - mrr_b, 4),
        "graph_recall_lift": round(recall_c - recall_b, 4),
    }

    full_rerank = modes_summary.get("full_rerank")
    if full_rerank is not None:
        mrr_d = float(full_rerank.get("mrr_at_5", 0.0))
        recall_d = float(full_rerank.get("recall_at_5", 0.0))
        ndcg_c = float(full.get("ndcg_at_5", 0.0))
        ndcg_d = float(full_rerank.get("ndcg_at_5", 0.0))
        p50_c = float(full.get("p50_latency_ms", 0.0))
        p50_d = float(full_rerank.get("p50_latency_ms", 0.0))
        p95_c = float(full.get("p95_latency_ms", 0.0))
        p95_d = float(full_rerank.get("p95_latency_ms", 0.0))
        lifts.update(
            {
                "rerank_lift": round(mrr_d - mrr_c, 4),
                "rerank_recall_lift": round(recall_d - recall_c, 4),
                "rerank_ndcg_lift": round(ndcg_d - ndcg_c, 4),
                "rerank_p50_delta_ms": round(p50_d - p50_c, 1),
                "rerank_p95_delta_ms": round(p95_d - p95_c, 1),
            }
        )
    return lifts


async def run_ablation_eval(
    *,
    db_url: str,
    embedding_provider: Any,
    settings: Any,
    reranker_provider: Any = None,
    include_rerank: bool | None = None,
    golden_set_path: Path = DEFAULT_GOLDEN_SET_PATH,
    history_path: Path | None = DEFAULT_HISTORY_PATH,
    limit: int | None = None,
    k: int = 5,
) -> dict[str, Any]:
    """Execute 3-way or 4-way ablation evaluation on golden_set.json and persist results."""
    from team_memory.services.search.search_pipeline import SearchPipeline, SearchRequest

    queries = load_golden_set(golden_set_path)
    if not queries:
        raise FileNotFoundError(
            f"Golden set not found or empty: {golden_set_path}. "
            "Run `python scripts/build_golden_set.py` first."
        )
    if limit is not None and limit > 0:
        queries = queries[:limit]

    rerank_cfg = getattr(settings, "reranker", None)
    has_real_reranker = (
        reranker_provider is not None
        and getattr(reranker_provider, "provider_name", "noop") != "noop"
    )
    prod_rerank_enabled = bool(
        has_real_reranker and rerank_cfg is not None and getattr(rerank_cfg, "provider", "none") != "none"
    )

    eval_reranker = reranker_provider if has_real_reranker else None
    if include_rerank is True:
        use_4way = True
        if eval_reranker is None:
            from team_memory.reranker.cross_encoder_provider import CrossEncoderRerankerProvider

            ce_cfg = getattr(rerank_cfg, "cross_encoder", None)
            eval_reranker = CrossEncoderRerankerProvider(
                model_name=getattr(ce_cfg, "model_name", "cross-encoder/ms-marco-MiniLM-L6-v2"),
                device=getattr(ce_cfg, "device", "auto"),
                top_k=getattr(ce_cfg, "top_k", 10),
                timeout_seconds=getattr(ce_cfg, "timeout_seconds", 3.0),
            )
    elif include_rerank is False:
        use_4way = False
    else:
        use_4way = has_real_reranker

    specs = build_ablation_specs(prod_rerank_enabled, include_rerank_ablation=use_4way)

    # Pre-build pipelines with caching disabled to guarantee clean latency & recall
    disabled_cache = CacheConfig(enabled=False)
    base_search_cfg = settings.search
    pipelines: dict[str, SearchPipeline] = {}
    for spec in specs:
        cfg_data = base_search_cfg.model_dump() if hasattr(base_search_cfg, "model_dump") else dict(base_search_cfg)
        cfg_data["mode"] = spec.mode
        cfg_data["enable_graph"] = spec.enable_graph
        cfg_data["enable_topic"] = spec.enable_topic
        mode_search_cfg = SearchConfig(**cfg_data)
        active_reranker = eval_reranker if spec.rerank_enabled else None
        pipelines[spec.name] = SearchPipeline(
            embedding_provider=embedding_provider,
            search_config=mode_search_cfg,
            retrieval_config=settings.retrieval,
            cache_config=disabled_cache,
            llm_config=settings.llm,
            tag_synonyms=getattr(settings, "tag_synonyms", None) or {},
            db_url=db_url,
            reranker_provider=active_reranker,
            rerank_enabled=bool(spec.rerank_enabled and active_reranker is not None),
            reranker_signature=(
                rerank_cfg.cache_signature() if (spec.rerank_enabled and rerank_cfg is not None) else "none"
            ),
            rerank_max_document_chars=(getattr(rerank_cfg, "max_document_chars", 8000) if rerank_cfg else 8000),
        )

    mode_records: dict[str, list[dict[str, Any]]] = {spec.name: [] for spec in specs}
    eval_start = time.monotonic()

    async with get_session(db_url) as session:
        for item in queries:
            q_text = str(item.get("query") or "").strip()
            expected_ids = [str(x) for x in (item.get("expected_ids") or [])]
            project = item.get("project")
            if not q_text or not expected_ids:
                continue

            for spec in specs:
                req = SearchRequest(
                    query=q_text,
                    max_results=k,
                    min_similarity=0.35,
                    project=project,
                    source="eval",
                    fallback_projects=["*"] if project not in (None, "*", "all") else None,
                )
                t0 = time.monotonic()
                try:
                    res = await pipelines[spec.name].search(session, req)
                    latency_ms = round((time.monotonic() - t0) * 1000.0, 1)
                    retrieved_ids = [
                        str(r.get("group_id") or r.get("id") or "")
                        for r in res.results
                        if isinstance(r, dict) and (r.get("group_id") or r.get("id"))
                    ]
                    tokens = estimate_result_tokens(res.results)
                except Exception as exc:
                    logger.warning("Eval query failed mode=%s id=%s: %s", spec.name, item.get("id"), exc)
                    latency_ms = round((time.monotonic() - t0) * 1000.0, 1)
                    retrieved_ids = []
                    tokens = 0

                mode_records[spec.name].append(
                    {
                        "id": item.get("id"),
                        "project": project,
                        "bucket": item.get("bucket") or "general",
                        "retrieved_ids": retrieved_ids,
                        "expected_ids": expected_ids,
                        "latency_ms": latency_ms,
                        "tokens": tokens,
                    }
                )

    modes_summary: dict[str, Any] = {}
    for spec in specs:
        recs = mode_records[spec.name]
        summary = summarize_mode_metrics(recs, k=k)
        summary["label"] = spec.label
        summary["rerank_enabled"] = spec.rerank_enabled
        # Per-bucket breakdown for intent-adaptive ranking diagnostics
        buckets: dict[str, list[dict[str, Any]]] = {}
        for r in recs:
            buckets.setdefault(str(r.get("bucket") or "general"), []).append(r)
        if len(buckets) > 1 or "general" not in buckets:
            summary["by_bucket"] = {
                b_name: summarize_mode_metrics(b_recs, k=k)
                for b_name, b_recs in sorted(buckets.items())
            }
        modes_summary[spec.name] = summary

    lifts = compute_ablation_lifts(modes_summary)

    record: dict[str, Any] = {
        "run_id": f"eval-{uuid.uuid4().hex[:8]}",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "git_sha": _get_git_sha(),
        "golden_set": str(golden_set_path.name),
        "golden_count": len(mode_records["full"]),
        "k": k,
        "prod_rerank_enabled": prod_rerank_enabled,
        "rerank_ablation_enabled": use_4way,
        "reranker_provider": getattr(eval_reranker, "provider_name", None) if use_4way else None,
        "duration_s": round(time.monotonic() - eval_start, 2),
        "modes": modes_summary,
        **lifts,
    }

    if history_path is not None:
        append_eval_history(record, history_path)

    return record


def format_eval_report(record: dict[str, Any]) -> str:
    """Format an ablation evaluation record as a concise Markdown/CLI table."""
    modes = record.get("modes") or {}
    lines = [
        f"Golden Eval Run `{record.get('run_id')}` | queries={record.get('golden_count')} | "
        f"rerank={'ON' if record.get('prod_rerank_enabled') else 'OFF'} | "
        f"duration={record.get('duration_s')}s | sha={record.get('git_sha')}",
        "",
        f"{'Mode':<24} {'Recall@5':>9} {'MRR@5':>8} {'NDCG@5':>8} {'P50(ms)':>9} {'P95(ms)':>9} {'Tokens':>8}",
        "-" * 79,
    ]
    for key in ("vector_only", "hybrid", "full", "full_rerank"):
        m = modes.get(key)
        if not m:
            continue
        lines.append(
            f"{m.get('label', key):<24} "
            f"{m.get('recall_at_5', 0.0):>9.4f} "
            f"{m.get('mrr_at_5', 0.0):>8.4f} "
            f"{m.get('ndcg_at_5', 0.0):>8.4f} "
            f"{m.get('p50_latency_ms', 0.0):>9.1f} "
            f"{m.get('p95_latency_ms', 0.0):>9.1f} "
            f"{m.get('avg_tokens', 0.0):>8.1f}"
        )
    lift = float(record.get("graph_lift", 0.0))
    r_lift = float(record.get("graph_recall_lift", 0.0))
    verdict = "正增益 (+)" if lift > 0 else ("中性 (0)" if lift == 0 else "负拖累 (-)")
    lines.append("-" * 79)
    lines.append(f"Graph Lift (MRR@5 C-B): {lift:+.4f} [{verdict}] | Recall@5 Lift: {r_lift:+.4f}")

    if "rerank_lift" in record and record["rerank_lift"] is not None:
        rr_lift = float(record["rerank_lift"])
        rr_r_lift = float(record.get("rerank_recall_lift", 0.0))
        p50_delta = float(record.get("rerank_p50_delta_ms", 0.0))
        p95_delta = float(record.get("rerank_p95_delta_ms", 0.0))
        rr_verdict = "正增益 (+)" if rr_lift > 0 else ("中性 (0)" if rr_lift == 0 else "负拖累 (-)")
        lines.append(
            f"Rerank Lift (MRR@5 D-C): {rr_lift:+.4f} [{rr_verdict}] | "
            f"Recall@5 Lift: {rr_r_lift:+.4f} | "
            f"Latency ΔP50: {p50_delta:+.1f}ms, ΔP95: {p95_delta:+.1f}ms"
        )

    full_buckets = (modes.get("full") or {}).get("by_bucket") or {}
    rr_buckets = (modes.get("full_rerank") or {}).get("by_bucket") or {}
    if full_buckets:
        lines.append("")
        header = (
            "Per-Bucket Breakdown (MRR@5: Vector -> Hybrid -> Full -> Full+RR | Full Recall@5):"
            if rr_buckets
            else "Per-Bucket Breakdown (MRR@5: Vector -> Hybrid -> Full | Full Recall@5):"
        )
        lines.append(header)
        for b_name in sorted(full_buckets.keys()):
            vm = ((modes.get("vector_only") or {}).get("by_bucket") or {}).get(b_name, {})
            hm = ((modes.get("hybrid") or {}).get("by_bucket") or {}).get(b_name, {})
            fm = full_buckets[b_name]
            b_lift = float(fm.get("mrr_at_5", 0.0)) - float(hm.get("mrr_at_5", 0.0))
            rr_part = ""
            if b_name in rr_buckets:
                dm = rr_buckets[b_name]
                rr_b_lift = float(dm.get("mrr_at_5", 0.0)) - float(fm.get("mrr_at_5", 0.0))
                rr_part = f" -> Full+RR={dm.get('mrr_at_5', 0):.3f} (RR {rr_b_lift:+.3f})"
            lines.append(
                f"  {b_name:<18} (n={fm.get('query_count', 0):>2}) | "
                f"Vec={vm.get('mrr_at_5', 0):.3f} -> Hyb={hm.get('mrr_at_5', 0):.3f} -> "
                f"Full={fm.get('mrr_at_5', 0):.3f} (Lift {b_lift:+.3f}){rr_part} | R@5={fm.get('recall_at_5', 0):.3f}"
            )
    return "\n".join(lines)

"""Pure scoring and fusion algorithms for the search pipeline.

These are stateless functions that take search results + configuration
parameters and return modified results. No DB access or async needed.
"""

from __future__ import annotations

import math
import re
from datetime import datetime, timezone
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from team_memory.services.search.search_pipeline import SearchResultItem

_TECH_ID_RE = re.compile(
    r"(?:`([^`]{2,40})`"
    r"|\b[a-zA-Z][a-zA-Z0-9_]*(?:\.[a-zA-Z][a-zA-Z0-9_]*)+\b"
    r"|\b[a-zA-Z][a-zA-Z0-9]*(?:[-_][a-zA-Z0-9]+)+\b"
    r"|\b[A-Z][a-z]+[A-Z][A-Za-z0-9]*\b"
    r"|\b[a-z]+[A-Z][A-Za-z0-9]*\b"
    r"|\b[A-Z]{2,}[0-9]+[A-Za-z0-9]*\b"
    r"|\b[A-Z]{3,}\b)"
)


def rrf_fuse(
    vector_results: list[SearchResultItem],
    fts_results: list[SearchResultItem],
    limit: int,
    *,
    rrf_k: int = 60,
    vector_weight: float = 0.7,
    fts_weight: float = 0.3,
) -> list[SearchResultItem]:
    """Merge vector and FTS results using Reciprocal Rank Fusion.

    RRF formula: score = sum(weight / (rank + k)) for each source.
    The constant k=60 is empirically optimal (Cormack et al., 2009).
    """
    score_map: dict[str, dict] = {}

    for rank, item in enumerate(vector_results):
        exp_id = item.data.get("id") or item.data.get("group_id", "")
        rrf_contribution = vector_weight / (rank + rrf_k)
        if exp_id not in score_map:
            score_map[exp_id] = {
                "rrf_score": 0.0,
                "item": item,
                "similarity": item.similarity,
            }
        score_map[exp_id]["rrf_score"] += rrf_contribution
        if item.similarity > score_map[exp_id]["similarity"]:
            score_map[exp_id]["item"] = item
            score_map[exp_id]["similarity"] = item.similarity

    for rank, item in enumerate(fts_results):
        exp_id = item.data.get("id") or item.data.get("group_id", "")
        rrf_contribution = fts_weight / (rank + rrf_k)
        if exp_id not in score_map:
            score_map[exp_id] = {
                "rrf_score": 0.0,
                "item": item,
                "similarity": item.similarity,
            }
        score_map[exp_id]["rrf_score"] += rrf_contribution

    fused = []
    for _exp_id, data in score_map.items():
        item = data["item"]
        item.similarity = data["similarity"]
        item.rrf_score = data["rrf_score"]
        item.score = data["rrf_score"]
        item.source_type = "hybrid"
        fused.append(item)

    fused.sort(key=lambda x: x.score, reverse=True)
    return fused[:limit]


def rrf_fuse3(
    vector_results: list[SearchResultItem],
    fts_results: list[SearchResultItem],
    topic_results: list[SearchResultItem],
    limit: int,
    *,
    rrf_k: int = 60,
    vector_weight: float = 0.7,
    fts_weight: float = 0.3,
) -> list[SearchResultItem]:
    """Merge vector + FTS + topic results using 3-way RRF.

    Topic results get a lower weight since they are a supplementary signal.
    """
    t_weight = vector_weight * 0.15  # Topic is supplementary, low weight

    score_map: dict[str, dict] = {}

    for rank, item in enumerate(vector_results):
        exp_id = item.data.get("id") or item.data.get("group_id", "")
        sim_val = float(item.similarity or 0.0)
        sim_bonus = (2.20 * max(0.0, sim_val - 0.5)) / rrf_k
        rrf_contribution = vector_weight / (rank + rrf_k) + sim_bonus
        if exp_id not in score_map:
            score_map[exp_id] = {
                "rrf_score": 0.0,
                "item": item,
                "similarity": item.similarity,
            }
        score_map[exp_id]["rrf_score"] += rrf_contribution
        if item.similarity > score_map[exp_id]["similarity"]:
            score_map[exp_id]["item"] = item
            score_map[exp_id]["similarity"] = item.similarity

    for rank, item in enumerate(fts_results):
        exp_id = item.data.get("id") or item.data.get("group_id", "")
        fts_r = float(getattr(item, "fts_rank", 0.0) or item.data.get("fts_rank", 0.0) or 0.0)
        fts_scale = min(1.0, 2.5 * fts_r) if fts_r > 0 else 1.0
        rrf_contribution = (fts_weight * fts_scale) / (rank + rrf_k)
        if exp_id not in score_map:
            score_map[exp_id] = {
                "rrf_score": 0.0,
                "item": item,
                "similarity": item.similarity,
            }
        score_map[exp_id]["rrf_score"] += rrf_contribution

    for rank, item in enumerate(topic_results):
        exp_id = item.data.get("id") or item.data.get("group_id", "")
        if exp_id not in score_map:
            rrf_contribution = t_weight / (rank + rrf_k)
            score_map[exp_id] = {
                "rrf_score": 0.0,
                "item": item,
                "similarity": item.similarity,
            }
        else:
            rrf_contribution = (0.05 * t_weight) / (rank + rrf_k)
        score_map[exp_id]["rrf_score"] += rrf_contribution

    fused = []
    for _exp_id, data in score_map.items():
        item = data["item"]
        item.similarity = data["similarity"]
        item.rrf_score = data["rrf_score"]
        item.score = data["rrf_score"]
        item.source_type = "hybrid"
        fused.append(item)

    fused.sort(key=lambda x: x.score, reverse=True)
    return fused[:limit]


def rrf_fuse4(
    vector_results: list[SearchResultItem],
    fts_results: list[SearchResultItem],
    topic_results: list[SearchResultItem],
    graph_results: list[SearchResultItem],
    limit: int,
    *,
    rrf_k: int = 60,
    vector_weight: float = 0.7,
    fts_weight: float = 0.3,
    graph_weight: float | None = None,
) -> list[SearchResultItem]:
    """Merge vector + FTS + topic + entity-graph results using 4-way RRF.

    Graph results receive RRF rank contribution plus a direct specificity bonus
    proportional to their PPR graph_score, enabling independent multi-hop recall
    even when lexical/vector overlap is zero.
    """
    t_weight = vector_weight * 0.15
    g_weight = graph_weight if graph_weight is not None else (vector_weight * 0.6)

    score_map: dict[str, dict] = {}

    for rank, item in enumerate(vector_results):
        exp_id = item.data.get("id") or item.data.get("group_id", "")
        sim_val = float(item.similarity or 0.0)
        sim_bonus = (2.20 * max(0.0, sim_val - 0.5)) / rrf_k
        rrf_contribution = vector_weight / (rank + rrf_k) + sim_bonus
        if exp_id not in score_map:
            score_map[exp_id] = {
                "rrf_score": 0.0,
                "item": item,
                "similarity": item.similarity,
            }
        score_map[exp_id]["rrf_score"] += rrf_contribution
        if item.similarity > score_map[exp_id]["similarity"]:
            score_map[exp_id]["item"] = item
            score_map[exp_id]["similarity"] = item.similarity

    for rank, item in enumerate(fts_results):
        exp_id = item.data.get("id") or item.data.get("group_id", "")
        fts_r = float(getattr(item, "fts_rank", 0.0) or item.data.get("fts_rank", 0.0) or 0.0)
        fts_scale = min(1.0, 2.5 * fts_r) if fts_r > 0 else 1.0
        rrf_contribution = (fts_weight * fts_scale) / (rank + rrf_k)
        if exp_id not in score_map:
            score_map[exp_id] = {
                "rrf_score": 0.0,
                "item": item,
                "similarity": item.similarity,
            }
        score_map[exp_id]["rrf_score"] += rrf_contribution

    for rank, item in enumerate(topic_results):
        exp_id = item.data.get("id") or item.data.get("group_id", "")
        if exp_id not in score_map:
            rrf_contribution = t_weight / (rank + rrf_k)
            score_map[exp_id] = {
                "rrf_score": 0.0,
                "item": item,
                "similarity": item.similarity,
            }
        else:
            rrf_contribution = (0.05 * t_weight) / (rank + rrf_k)
        score_map[exp_id]["rrf_score"] += rrf_contribution

    for rank, item in enumerate(graph_results):
        exp_id = item.data.get("id") or item.data.get("group_id", "")
        g_score = float(item.data.get("graph_score", 0.0) or 0.0)
        g_factor = min(1.0, max(0.0, g_score))
        if exp_id not in score_map:
            rrf_contribution = (g_weight * g_factor) / (rank + rrf_k) + (0.85 / rrf_k) * g_factor
            score_map[exp_id] = {
                "rrf_score": 0.0,
                "item": item,
                "similarity": item.similarity,
            }
        else:
            score_map[exp_id]["item"].data["graph_score"] = round(g_score, 4)
            rrf_contribution = (0.05 * g_weight * g_factor) / (rank + rrf_k)
        score_map[exp_id]["rrf_score"] += rrf_contribution
        if item.similarity > score_map[exp_id]["similarity"]:
            score_map[exp_id]["similarity"] = item.similarity

    fused = []
    for _exp_id, data in score_map.items():
        item = data["item"]
        item.similarity = data["similarity"]
        item.rrf_score = data["rrf_score"]
        item.score = data["rrf_score"]
        item.source_type = "hybrid"
        fused.append(item)

    fused.sort(key=lambda x: x.score, reverse=True)
    return fused[:limit]


def apply_exact_match_boost(
    candidates: list[SearchResultItem],
    query: str,
) -> list[SearchResultItem]:
    """Boost score when query or extracted technical identifiers match title/content."""
    if not query or not query.strip():
        return candidates
    q = query.strip()
    q_lower = q.lower()
    exact_boost = 0.4
    contains_boost = 0.15

    tech_tokens: list[str] = []
    seen_tokens: set[str] = set()
    for m in _TECH_ID_RE.finditer(q):
        tok = (m.group(1) or m.group(0)).strip().lower()
        if len(tok) >= 3 and tok not in seen_tokens:
            seen_tokens.add(tok)
            tech_tokens.append(tok)

    for item in candidates:
        data = item.data
        titles_to_check: list[str] = []
        if data.get("title"):
            titles_to_check.append(data["title"])
        for child in data.get("children", []):
            if child.get("title"):
                titles_to_check.append(child["title"])

        match_type = ""
        for t in titles_to_check:
            t_stripped = (t or "").strip()
            if not t_stripped:
                continue
            if q == t_stripped:
                match_type = "exact"
                break
            if q_lower in t_stripped.lower():
                match_type = "contains"
                break

        old_score = float(item.score or 0.0)
        if match_type:
            data["exact_title_match"] = match_type
            if 0.0 < old_score < 0.1:
                mult = 1.55 if match_type == "exact" else 1.30
                item.score = old_score * mult
            else:
                boost = exact_boost if match_type == "exact" else contains_boost
                item.score = old_score + boost
        else:
            data["exact_title_match"] = ""

        if tech_tokens:
            tags_str = " ".join(str(t) for t in (data.get("tags") or []))
            title_blob = f"{' '.join(titles_to_check)} {tags_str}".lower()
            body_blob = f"{data.get('description') or ''} {data.get('solution') or ''}".lower()
            matched_weight = 0.0
            for tok in tech_tokens:
                if tok in title_blob:
                    matched_weight += 1.0
                elif tok in body_blob:
                    matched_weight += 0.65
            if matched_weight > 0:
                id_ratio = min(1.0, matched_weight / len(tech_tokens))
                data["identifier_match_score"] = round(id_ratio, 3)
                cur_score = float(item.score or 0.0)
                if 0.0 < cur_score < 0.1:
                    item.score = cur_score * (1.0 + 0.40 * id_ratio)
                else:
                    item.score = cur_score + 0.15 * id_ratio
        else:
            cjk_runs = re.findall(r"[\u4e00-\u9fff]{2,}", q)
            cjk_bigrams: set[str] = set()
            for run in cjk_runs:
                for i in range(len(run) - 1):
                    cjk_bigrams.add(run[i : i + 2])
            if len(cjk_bigrams) >= 3:
                tags_str = " ".join(str(t) for t in (data.get("tags") or []))
                title_blob = f"{' '.join(titles_to_check)} {tags_str}"
                hit_bg = sum(1 for bg in cjk_bigrams if bg in title_blob)
                if hit_bg >= 2:
                    bg_ratio = min(1.0, hit_bg / max(4, len(cjk_bigrams) * 0.5))
                    cur_score = float(item.score or 0.0)
                    if 0.0 < cur_score < 0.1:
                        item.score = cur_score * (1.0 + 0.25 * bg_ratio)
                    else:
                        item.score = cur_score + 0.08 * bg_ratio

    candidates.sort(key=lambda x: x.score, reverse=True)
    return candidates


def compute_bayesian_utility_rate(
    recall_count: int,
    used_count: int,
    negative_feedback_count: int = 0,
    alpha0: float = 1.0,
    beta0: float = 4.0,
) -> float:
    """Compute Bayesian-smoothed utility rate from recall, adoption, and negative feedback counts."""
    return (alpha0 + max(0, used_count)) / (
        alpha0 + beta0 + max(0, recall_count) + 3.0 * max(0, negative_feedback_count)
    )


def apply_popularity_boost(
    candidates: list[SearchResultItem],
    *,
    popularity_recall_weight: float = 0.02,
    popularity_used_weight: float = 0.1,
    popularity_decay_lambda: float = 0.05,
    popularity_age_floor_ratio: float = 0.3,
    popularity_scale: float = 0.3,
) -> list[SearchResultItem]:
    """Boost score based on recall_count and used_count with time decay.

    Formula (multiplicative boost + log compression):
        popularity = log(1 + alpha*recall_count + beta*used_count)
        effective_age = max(days(updated_at), days(created_at) * floor_ratio)
        decay = exp(-lambda * effective_age)
        boost = 1 + popularity * decay * gamma
        final_score = rrf_score * boost

    This ensures:
    - Experiences never recalled get no boost (boost=1.0).
    - Popular experiences are amplified proportionally, but log prevents
      recall_count=100 from dominating recall_count=10.
    - Time decay reduces boost for stale experiences.
    - Age floor prevents valid-but-unused experiences from decaying too fast.
    """
    alpha = popularity_recall_weight
    beta = popularity_used_weight
    lam = popularity_decay_lambda
    floor_ratio = popularity_age_floor_ratio
    gamma = popularity_scale

    now = datetime.now(timezone.utc)

    for item in candidates:
        data = item.data
        recall_count = int(data.get("recall_count", 0) or 0)
        used_count = int(data.get("used_count", 0) or 0)
        neg_count = int(data.get("negative_feedback_count", 0) or 0)

        if recall_count == 0 and used_count == 0 and neg_count == 0:
            # No popularity signal — skip computation
            data["popularity_boost"] = 1.0
            data["utility_rate"] = round(compute_bayesian_utility_rate(0, 0, 0), 4)
            continue

        utility_rate = compute_bayesian_utility_rate(recall_count, used_count, neg_count)
        recall_dampener = min(1.0, utility_rate / 0.2) if used_count == 0 else 1.0
        effective_alpha = alpha * recall_dampener
        effective_beta = beta * (1.0 + 3.0 * utility_rate)

        # Popularity with log compression and negative feedback penalty
        popularity = math.log(1.0 + effective_alpha * recall_count + effective_beta * used_count)
        neg_penalty = math.log(1.0 + 0.4 * neg_count)
        net_signal = popularity - neg_penalty

        # Time decay with age floor
        # raw_age is based on updated_at (most recent activity)
        age_days = 0.0
        updated_raw = data.get("updated_at")
        if updated_raw:
            try:
                if isinstance(updated_raw, str):
                    ts = datetime.fromisoformat(updated_raw)
                elif isinstance(updated_raw, datetime):
                    ts = updated_raw
                else:
                    ts = None
                if ts:
                    if ts.tzinfo is None:
                        ts = ts.replace(tzinfo=timezone.utc)
                    age_days = (now - ts).total_seconds() / 86400
            except (ValueError, TypeError):
                pass

        # Apply age floor: effective_age = max(raw_age, created_age * ratio)
        created_age = 0.0
        created_raw = data.get("created_at")
        if created_raw:
            try:
                if isinstance(created_raw, str):
                    cts = datetime.fromisoformat(created_raw)
                elif isinstance(created_raw, datetime):
                    cts = created_raw
                else:
                    cts = None
                if cts:
                    if cts.tzinfo is None:
                        cts = cts.replace(tzinfo=timezone.utc)
                    created_age = (now - cts).total_seconds() / 86400
            except (ValueError, TypeError):
                pass
        floor_age = created_age * floor_ratio
        effective_age = max(age_days, floor_age)

        decay = math.exp(-lam * effective_age)
        boost = max(0.5, 1.0 + net_signal * decay * gamma)
        if 0.0 < float(item.score or 0.0) < 0.1 and used_count == 0 and neg_count == 0:
            # Passive recall without verified adoption acts only as a gentle tie-breaker in RRF scale
            boost = min(1.005, boost)

        item.score = float(item.score) * boost
        data["popularity_boost"] = round(boost, 4)
        data["utility_rate"] = round(utility_rate, 4)

    candidates.sort(key=lambda x: x.score, reverse=True)
    return candidates


def apply_adaptive_filter(
    candidates: list[SearchResultItem],
    *,
    min_confidence_ratio: float = 0.6,
    score_gap_threshold: float = 0.15,
    adaptive_min_keep: int = 3,
) -> list[SearchResultItem]:
    """Apply adaptive score filtering using similarity scores.

    Two strategies:
    1. Dynamic threshold: filter results below min_confidence_ratio * top-1.
    2. Elbow detection: cut when score gap exceeds threshold.
    """
    if not candidates:
        return candidates

    has_similarity = any(c.similarity > 0 for c in candidates)
    is_hybrid = any(getattr(c, "rrf_score", 0.0) > 0 for c in candidates)
    top_raw_score = float(candidates[0].score or 1.0)
    top_sim = (
        max((c.similarity for c in candidates if c.similarity > 0), default=1.0)
        if has_similarity
        else top_raw_score
    )

    def _score(item: SearchResultItem) -> float:
        if not has_similarity:
            return float(item.score)
        if item.similarity > 0:
            return float(item.similarity)
        if top_raw_score > 0:
            return min(top_sim, (float(item.score) / top_raw_score) * top_sim)
        return 0.0

    threshold = top_sim * min_confidence_ratio
    min_keep = adaptive_min_keep

    filtered = [candidates[0]]

    for i in range(1, len(candidates)):
        curr = candidates[i]
        curr_s = _score(curr)

        if curr.data.get("exact_title_match") or curr.data.get("identifier_match_score"):
            filtered.append(curr)
            continue

        if is_hybrid:
            # In hybrid RRF mode, candidates are ordered by fused RRF score rather than
            # cosine similarity, so similarity is non-monotonic across ranks. Skip only
            # individual items below threshold instead of breaking early.
            raw_ratio = (float(curr.score) / top_raw_score) if top_raw_score > 0 else 0.0
            if curr_s < threshold and raw_ratio < min_confidence_ratio:
                continue
            filtered.append(curr)
            continue

        if curr_s < threshold:
            break

        prev_s = _score(candidates[i - 1])
        if (
            len(filtered) >= min_keep
            and prev_s > curr_s
            and (prev_s - curr_s) / prev_s > score_gap_threshold
        ):
            break

        filtered.append(curr)

    return filtered


def label_confidence(
    candidates: list[SearchResultItem],
) -> list[SearchResultItem]:
    """Label each result with a confidence level based on its score.

    - high: >= 80% of top score
    - medium: >= 50% of top score
    - low: < 50% of top score
    """
    if not candidates:
        return candidates

    top_score = candidates[0].score if candidates[0].score > 0 else 1.0

    for item in candidates:
        ratio = item.score / top_score if top_score > 0 else 0
        if ratio >= 0.8:
            item.confidence = "high"
        elif ratio >= 0.5:
            item.confidence = "medium"
        else:
            item.confidence = "low"

    return candidates

"""Pure scoring and fusion algorithms for the search pipeline.

These are stateless functions that take search results + configuration
parameters and return modified results. No DB access or async needed.
"""

from __future__ import annotations

import math
from datetime import datetime, timezone
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from team_memory.services.search.search_pipeline import SearchResultItem


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
    t_weight = vector_weight * 0.5  # Topic is supplementary, half of vector weight

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

    for rank, item in enumerate(topic_results):
        exp_id = item.data.get("id") or item.data.get("group_id", "")
        rrf_contribution = t_weight / (rank + rrf_k)
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
    """Boost score when query exactly matches or is contained in title."""
    if not query or not query.strip():
        return candidates
    q = query.strip()
    q_lower = q.lower()
    exact_boost = 0.4
    contains_boost = 0.15

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

        if match_type:
            data["exact_title_match"] = match_type
            boost = exact_boost if match_type == "exact" else contains_boost
            item.score = float(item.score) + boost
        else:
            data["exact_title_match"] = ""

    candidates.sort(key=lambda x: x.score, reverse=True)
    return candidates


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

        if recall_count == 0 and used_count == 0:
            # No popularity signal — skip computation
            data["popularity_boost"] = 1.0
            continue

        # Popularity with log compression
        popularity = math.log(1 + alpha * recall_count + beta * used_count)

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
        boost = 1.0 + popularity * decay * gamma

        item.score = float(item.score) * boost
        data["popularity_boost"] = round(boost, 4)

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

    def _score(item: SearchResultItem) -> float:
        return item.similarity if item.similarity > 0 else item.score

    top_sim = _score(candidates[0]) or 1.0
    threshold = top_sim * min_confidence_ratio
    min_keep = adaptive_min_keep

    filtered = [candidates[0]]

    for i in range(1, len(candidates)):
        curr = candidates[i]
        curr_s = _score(curr)

        if curr.data.get("exact_title_match"):
            filtered.append(curr)
            continue

        if curr_s < threshold:
            break

        prev_s = _score(candidates[i - 1])
        if len(filtered) >= min_keep and prev_s > 0 and (prev_s - curr_s) / prev_s > score_gap_threshold:
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

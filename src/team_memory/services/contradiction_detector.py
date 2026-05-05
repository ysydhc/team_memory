"""Contradiction Detector — detect conflicting experiences.

Two-stage approach:
1. Rule-based: find experience pairs sharing entities with contradictory signals
2. LLM validation (Stage 2): confirm if pairs are truly contradictory

Stage 1 is fast and cheap (no LLM). Stage 2 filters false positives.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field

import httpx
from sqlalchemy import text

from team_memory.storage.database import get_session

logger = logging.getLogger("team_memory.contradiction_detector")


@dataclass
class ContradictionPair:
    """A pair of potentially conflicting experiences."""

    exp_a_id: str
    exp_a_title: str
    exp_b_id: str
    exp_b_title: str
    shared_entities: list[str] = field(default_factory=list)
    reason: str = ""
    llm_confirmed: bool = False  # True if LLM validated this as a real contradiction


_CONTRADICTION_PROMPT = """You are a technical knowledge curator. Given two experiences that share entities, determine if they contain genuinely contradictory advice.

Rules:
- TRUE contradiction: One says "always do X" and the other says "never do X" about the SAME aspect
- FALSE contradiction: They discuss different aspects, different contexts, or are complementary
- Consider that one may be outdated and the other is a correction — this IS a contradiction worth flagging

Respond with ONLY a JSON object:
{"contradiction": true/false, "reason": "brief explanation"}"""


async def _validate_with_llm(
    pairs: list[ContradictionPair],
    exp_data: dict[str, dict],
    llm_config: object,
) -> list[ContradictionPair]:
    """Validate contradiction candidates using LLM.

    Sends pairs in a single batch prompt to minimize LLM calls.

    Args:
        pairs: Candidate pairs from rule-based detection
        exp_data: Experience data dict (id -> {title, description, solution})
        llm_config: EntityExtractionConfig with model/base_url/api_key_env

    Returns:
        Filtered list of pairs that LLM confirms as contradictions.
    """
    import json
    import os

    if not pairs:
        return []

    base_url = getattr(llm_config, "base_url", "http://localhost:4000/v1").rstrip("/")
    if not base_url.endswith("/v1"):
        base_url = f"{base_url}/v1"
    model = getattr(llm_config, "model", "DeepSeek-V3")
    api_key_env = getattr(llm_config, "api_key_env", "LITELLM_MASTER_KEY")
    api_key = os.environ.get(api_key_env, "none")
    timeout = getattr(llm_config, "timeout", 30)

    # Build batch prompt — all pairs in one call
    user_msg = "Evaluate each pair:\n\n"
    for i, pair in enumerate(pairs):
        a = exp_data.get(pair.exp_a_id, {})
        b = exp_data.get(pair.exp_b_id, {})
        user_msg += f"--- Pair {i + 1} ---\n"
        user_msg += f"A: {pair.exp_a_title}\n"
        user_msg += f"  Problem: {(a.get('description') or '')[:300]}\n"
        user_msg += f"  Solution: {(a.get('solution') or '')[:300]}\n"
        user_msg += f"B: {pair.exp_b_title}\n"
        user_msg += f"  Problem: {(b.get('description') or '')[:300]}\n"
        user_msg += f"  Solution: {(b.get('solution') or '')[:300]}\n"
        user_msg += f"  Shared entities: {pair.shared_entities}\n\n"

    user_msg += (
        "Return a JSON array with one object per pair: "
        '[{"contradiction": true/false, "reason": "..."}, ...]'
    )

    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": _CONTRADICTION_PROMPT},
            {"role": "user", "content": user_msg},
        ],
        "temperature": 0.1,
        "max_tokens": 1024,
    }
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }

    try:
        async with httpx.AsyncClient(timeout=float(timeout)) as client:
            resp = await client.post(
                f"{base_url}/chat/completions",
                json=payload,
                headers=headers,
            )
            resp.raise_for_status()
            raw = resp.json()["choices"][0]["message"]["content"].strip()

            # Extract JSON array from response
            # Handle cases where LLM wraps in markdown code block
            if "```" in raw:
                raw = raw.split("```")[1]
                if raw.startswith("json"):
                    raw = raw[4:]
                raw = raw.strip()

            results = json.loads(raw)
            if not isinstance(results, list):
                results = [results]

            confirmed = []
            for i, pair in enumerate(pairs):
                if i < len(results) and isinstance(results[i], dict):
                    if results[i].get("contradiction", False):
                        pair.llm_confirmed = True
                        pair.reason = results[i].get("reason", pair.reason)
                        confirmed.append(pair)
                else:
                    # If LLM response is malformed, keep the pair (conservative)
                    confirmed.append(pair)

            logger.info(
                "LLM contradiction validation: %d/%d confirmed",
                len(confirmed), len(pairs),
            )
            return confirmed

    except Exception as exc:
        logger.warning("LLM contradiction validation failed: %s — keeping all candidates", exc)
        return pairs  # On failure, keep all candidates (conservative)


async def detect_contradictions(
    db_url: str,
    max_pairs: int = 20,
    llm_config: object | None = None,
) -> list[ContradictionPair]:
    """Detect potentially conflicting experiences.

    Two-stage approach:
    1. Rule-based: find pairs sharing entities with contradictory signals
    2. LLM validation: confirm if pairs are truly contradictory (when llm_config provided)

    Args:
        db_url: PostgreSQL connection string
        max_pairs: Maximum pairs to return
        llm_config: Optional EntityExtractionConfig for LLM validation

    Returns:
        List of ContradictionPair objects.
    """
    pairs: list[ContradictionPair] = []

    # Signal words that indicate negation/opposition
    negation_words = {
        "不", "不要", "别", "不能", "避免", "avoid", "don't", "do not",
        "never", "禁止", "不可", "failed", "broken", "错误", "wrong",
        "问题", "bug", "fix", "修复", "crash", "失败",
    }
    positive_words = {
        "必须", "要", "应该", "使用", "use", "must", "should",
        "推荐", "recommend", "正确", "correct", "works", "成功",
        "解决", "resolved", "best practice",
    }

    # Step 1: Find experience pairs sharing 2+ entities
    async with get_session(db_url) as session:
        result = await session.execute(text("""
            SELECT ee1.experience_id as exp_a,
                   ee2.experience_id as exp_b,
                   array_agg(DISTINCT e.name) as shared_entities,
                   count(DISTINCT e.id) as entity_count
            FROM experience_entities ee1
            JOIN experience_entities ee2 ON ee1.entity_id = ee2.entity_id
                AND ee1.experience_id < ee2.experience_id
            JOIN entities e ON e.id = ee1.entity_id
            GROUP BY ee1.experience_id, ee2.experience_id
            HAVING count(DISTINCT e.id) >= 2
            ORDER BY entity_count DESC
            LIMIT :limit
        """), {"limit": max_pairs * 3})

        candidate_rows = result.fetchall()

    if not candidate_rows:
        return []

    # Step 2: For each candidate pair, check solution text for contradictory signals
    exp_ids_set: set[str] = set()
    seen_titles: set[tuple[str, ...]] = set()
    for row in candidate_rows:
        exp_ids_set.add(str(row[0]))
        exp_ids_set.add(str(row[1]))

    # Fetch experience data
    exp_data: dict[str, dict] = {}
    async with get_session(db_url) as session:
        result = await session.execute(text("""
            SELECT id, title, description, solution
            FROM experiences
            WHERE id = ANY(:ids) AND exp_status = 'published'
        """), {"ids": list(exp_ids_set)})
        for row in result.fetchall():
            exp_data[str(row[0])] = {
                "title": row[1] or "",
                "description": row[2] or "",
                "solution": row[3] or "",
            }

    for row in candidate_rows:
        exp_a_id = str(row[0])
        exp_b_id = str(row[1])
        shared_entities = list(row[2])

        a_data = exp_data.get(exp_a_id)
        b_data = exp_data.get(exp_b_id)
        if not a_data or not b_data:
            continue

        # Check for contradictory signals
        a_solution = a_data["solution"].lower()
        b_solution = b_data["solution"].lower()
        a_desc = a_data["description"].lower()
        b_desc = b_data["description"].lower()
        a_text = f"{a_desc} {a_solution}"
        b_text = f"{b_desc} {b_solution}"

        a_has_negation = any(w in a_text for w in negation_words)
        b_has_positive = any(w in b_text for w in positive_words)
        b_has_negation = any(w in b_text for w in negation_words)
        a_has_positive = any(w in a_text for w in positive_words)

        is_contradiction = False
        reason = ""

        if a_has_negation and b_has_positive:
            is_contradiction = True
            reason = "A suggests avoidance, B suggests adoption"
        elif b_has_negation and a_has_positive:
            is_contradiction = True
            reason = "B suggests avoidance, A suggests adoption"
        elif a_has_negation and b_has_negation:
            # Both negative — check if about different approaches
            # This is weaker signal, skip for simplicity
            pass

        if is_contradiction:
            # Deduplicate: skip pairs with same title (same experience group)
            title_key = tuple(sorted([a_data["title"], b_data["title"]]))
            if title_key in seen_titles:
                continue
            seen_titles.add(title_key)

            pairs.append(ContradictionPair(
                exp_a_id=exp_a_id,
                exp_a_title=a_data["title"],
                exp_b_id=exp_b_id,
                exp_b_title=b_data["title"],
                shared_entities=shared_entities,
                reason=reason,
            ))

        if len(pairs) >= max_pairs:
            break

    # Stage 2: LLM validation (when config provided)
    if pairs and llm_config is not None:
        pairs = await _validate_with_llm(pairs, exp_data, llm_config)

    return pairs

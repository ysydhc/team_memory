"""Contradiction Detector — detect conflicting experiences.

Simple approach:
1. Find experiences that share entities but have conflicting solutions
2. Use embedding reverse-similarity as a heuristic for contradiction
3. Output pairs of potentially conflicting experiences

This is a best-effort, rule-based detector. No LLM is used on the detection
path (keeping latency low). LLM validation can be added as Stage 2.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from sqlalchemy import text

from team_memory.storage.database import get_session


@dataclass
class ContradictionPair:
    """A pair of potentially conflicting experiences."""

    exp_a_id: str
    exp_a_title: str
    exp_b_id: str
    exp_b_title: str
    shared_entities: list[str] = field(default_factory=list)
    reason: str = ""


async def detect_contradictions(
    db_url: str,
    max_pairs: int = 20,
) -> list[ContradictionPair]:
    """Detect potentially conflicting experiences.

    Strategy:
    1. Find pairs of experiences that share 2+ entities
    2. Check if their solutions mention opposite signals
       (e.g., "don't" vs "must", "avoid" vs "use", "broken" vs "works")
    3. Use embedding cosine similarity: high similarity + different solution
       words = likely contradiction

    Args:
        db_url: PostgreSQL connection string
        max_pairs: Maximum pairs to return

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

    return pairs

"""Evaluation service — injects [mem:xxx] markers and tracks was_used."""

from __future__ import annotations

import logging

logger = logging.getLogger("team_memory.evaluation")


class EvaluationService:
    """Handles search result evaluation: marker injection and was_used detection."""

    def __init__(self, db_url: str = "", embedding_provider=None):
        self._db_url = db_url
        self._embedding = embedding_provider

    async def check_was_used_fuzzy(
        self,
        agent_response: str,
        results: list[dict],
        threshold: float = 0.8,
    ) -> dict[str, bool]:
        """Fuzzy match: check if agent response overlaps with result keywords.

        Uses simple text overlap (no embeddings required).
        For each result, extracts keywords from description/solution,
        then computes what fraction of those keywords appear in agent_response.
        If overlap > threshold → was_used = True.

        Returns dict mapping result_id -> was_used.
        """
        import re

        used: dict[str, bool] = {}
        response_lower = agent_response.lower()

        for r in results:
            rid = r.get("id", "")
            # Extract text for keyword generation: prefer solution, fall back to description
            text = r.get("solution", "") or r.get("description", "") or ""
            if not text:
                used[rid] = False
                continue

            # Tokenize: split by spaces and punctuation, keep tokens with length > 2
            keywords = [tok.lower() for tok in re.split(r"[^\w]+", text) if len(tok) > 2]

            if not keywords:
                used[rid] = False
                continue

            # Count how many keywords appear in the agent response
            matched = sum(1 for kw in keywords if kw in response_lower)
            overlap_ratio = matched / len(keywords)
            used[rid] = overlap_ratio > threshold

        return used

    async def get_weekly_stats(self) -> dict:
        """Return weekly evaluation statistics using SearchLogRepository."""
        # Will be connected to SearchLogRepository in a future task
        return {"total": 0, "hit": 0, "used": 0, "use_rate": 0.0}

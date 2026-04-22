"""Evaluation service — injects [mem:xxx] markers and tracks was_used."""
from __future__ import annotations

import logging

logger = logging.getLogger("team_memory.evaluation")


class EvaluationService:
    """Handles search result evaluation: marker injection and was_used detection."""

    def __init__(self, db_url: str = "", embedding_provider=None):
        self._db_url = db_url
        self._embedding = embedding_provider

    def inject_markers(self, results: list[dict]) -> list[dict]:
        """Inject [mem:xxx] markers into search results.

        Appends marker to 'solution' field if present and non-empty,
        otherwise to 'description'. Also stores marker in '_marker'
        field for internal tracking.
        """
        for r in results:
            exp_id = r.get("id", "")
            if not exp_id:
                continue
            marker = f"[mem:{exp_id}]"
            r["_marker"] = marker
            # Append to solution if it exists and is non-empty
            if "solution" in r and r["solution"] and marker not in r["solution"]:
                r["solution"] += f" {marker}"
            elif "description" in r and r["description"] and marker not in r["description"]:
                r["description"] += f" {marker}"
        return results

    def check_was_used(self, agent_response: str, result_ids: list[str]) -> dict[str, bool]:
        """Check if agent response contains [mem:xxx] markers.

        Returns dict mapping result_id -> was_used (True/False).
        """
        used: dict[str, bool] = {}
        for rid in result_ids:
            marker = f"[mem:{rid}]"
            used[rid] = marker in agent_response
        return used

    async def check_was_used_fuzzy(
        self,
        agent_response: str,
        results: list[dict],
        threshold: float = 0.8,
    ) -> dict[str, bool]:
        """Fuzzy match: check if agent response is semantically similar to any result.

        Only called when exact marker match fails.
        Returns dict mapping result_id -> was_used.
        """
        # If no embedding provider, can't do fuzzy matching
        if not self._embedding:
            return {r.get("id", ""): False for r in results}
        # Compute similarity between agent_response and each result.
        # If similarity > threshold, mark as used.
        # Placeholder — actual implementation needs embedding comparison.
        return {r.get("id", ""): False for r in results}

    async def get_weekly_stats(self) -> dict:
        """Return weekly evaluation statistics using SearchLogRepository."""
        # Will be connected to SearchLogRepository in a future task
        return {"total": 0, "hit": 0, "used": 0, "use_rate": 0.0}

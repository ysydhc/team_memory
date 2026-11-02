"""Bi-Temporal Memory State Machine Resolver (ADD / MERGE / SUPERSEDE / NOOP).

Resolves how a newly incoming experience interacts with existing active
experiences in the repository based on semantic similarity, contradiction/
deprecation signals, and complementary content analysis.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

_SUPERSEDE_MARKERS: tuple[str, ...] = (
    "deprecated",
    "obsolete",
    "replaced by",
    "supersede",
    "superseded",
    "breaking change",
    "no longer",
    "upgrade to",
    "migrated from",
    "废弃",
    "弃用",
    "不再使用",
    "已过时",
    "替代",
    "改为",
    "升级为",
    "旧版本",
)

_KV_ASSIGNMENT_RE = re.compile(r"\b([a-zA-Z_][a-zA-Z0-9_.-]*)\s*=\s*([^\s,;]+)")


@dataclass
class MutationDecision:
    """Decision output of the bi-temporal memory mutation state machine."""

    action: str  # "ADD" | "MERGE" | "SUPERSEDE" | "NOOP"
    target_id: str | None = None
    merged_solution: str | None = None
    merged_description: str | None = None
    reason: str = ""


def _extract_kv_pairs(text: str) -> dict[str, str]:
    """Extract key=value assignments from text for conflict detection."""
    pairs: dict[str, str] = {}
    for match in _KV_ASSIGNMENT_RE.finditer(text or ""):
        key = match.group(1).strip().lower()
        val = match.group(2).strip().strip("\"'")
        pairs[key] = val
    return pairs


def _has_supersede_signal(
    title: str,
    problem: str,
    solution: str | None,
    candidate: dict[str, Any],
) -> bool:
    """Detect whether the incoming experience supersedes or contradicts candidate."""
    combined_new = f"{title or ''}\n{problem or ''}\n{solution or ''}".lower()
    for marker in _SUPERSEDE_MARKERS:
        if marker in combined_new:
            return True

    old_text = f"{candidate.get('description') or ''}\n{candidate.get('solution') or ''}"
    new_text = f"{problem or ''}\n{solution or ''}"
    old_kv = _extract_kv_pairs(old_text)
    new_kv = _extract_kv_pairs(new_text)
    for k, old_val in old_kv.items():
        if k in new_kv and new_kv[k] != old_val:
            return True

    return False


def _normalize_text(text: str | None) -> str:
    return re.sub(r"\s+", " ", (text or "").strip().lower())


class MutationResolver:
    """Resolves memory write operations into ADD, MERGE, SUPERSEDE, or NOOP."""

    def __init__(
        self,
        llm_config: Any = None,
        noop_threshold: float = 0.96,
        mutation_threshold: float = 0.85,
    ) -> None:
        self._llm_config = llm_config
        self.noop_threshold = noop_threshold
        self.mutation_threshold = mutation_threshold

    def resolve_by_rules(
        self,
        *,
        title: str,
        problem: str,
        solution: str | None,
        candidates: list[dict[str, Any]],
    ) -> MutationDecision:
        """Deterministic rule-based state machine resolution."""
        if not candidates:
            return MutationDecision(action="ADD", reason="No similar active candidates found")

        sorted_candidates = sorted(
            candidates,
            key=lambda c: float(c.get("similarity", 0.0)),
            reverse=True,
        )
        top = sorted_candidates[0]
        sim = float(top.get("similarity", 0.0))
        target_id = str(top.get("id", ""))

        if sim < self.mutation_threshold or not target_id:
            return MutationDecision(
                action="ADD",
                reason=f"Top similarity {sim:.3f} below mutation threshold {self.mutation_threshold}",
            )

        if _has_supersede_signal(title, problem, solution, top):
            return MutationDecision(
                action="SUPERSEDE",
                target_id=target_id,
                reason="New experience supersedes or contradicts prior knowledge",
            )

        norm_new_sol = _normalize_text(solution)
        norm_top_sol = _normalize_text(top.get("solution"))
        if sim >= self.noop_threshold or not norm_new_sol or (norm_top_sol and norm_new_sol in norm_top_sol):
            return MutationDecision(
                action="NOOP",
                target_id=target_id,
                reason=f"Near-identical experience (similarity={sim:.3f}); refreshed verification timestamp",
            )

        top_sol = (top.get("solution") or "").strip()
        new_sol = (solution or "").strip()
        if top_sol and new_sol and new_sol not in top_sol:
            merged_solution = f"{top_sol}\n\n### 补充方案 (Merged Update)\n{new_sol}"
        else:
            merged_solution = new_sol or top_sol

        return MutationDecision(
            action="MERGE",
            target_id=target_id,
            merged_solution=merged_solution,
            reason=f"Merged complementary solution into existing experience (similarity={sim:.3f})",
        )

    async def resolve(
        self,
        *,
        title: str,
        problem: str,
        solution: str | None,
        candidates: list[dict[str, Any]],
    ) -> MutationDecision:
        """Resolve memory state transition (ADD / MERGE / SUPERSEDE / NOOP)."""
        return self.resolve_by_rules(
            title=title,
            problem=problem,
            solution=solution,
            candidates=candidates,
        )

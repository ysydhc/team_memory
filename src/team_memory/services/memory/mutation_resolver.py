"""Bi-Temporal Memory State Machine Resolver (ADD / UPDATE / MERGE / SUPERSEDE / NOOP).

Write-time gate inspired by Mem0:
  1. Fast 0ms deterministic rule path with strict anti-false-positive guards.
  2. Optional lightweight LLM adjudication for borderline candidates (0.82 <= sim < 0.95)
     with strict timeout (default 2.0s) and fail-safe degradation to ADD so write latency
     is never blocked and offline AutoDream can reconcile later.
  3. Protection guardrail: pinned or fresh verified experiences are never auto-superseded.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import re
from dataclasses import dataclass
from typing import Any

import httpx

logger = logging.getLogger("team_memory.mutation_resolver")

# Strong explicit replacement markers (narrowed to avoid false-positive supersedes).
_STRONG_SUPERSEDE_MARKERS: tuple[str, ...] = (
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
    "升级为",
)

# Broader markers that only suggest a potential conflict (trigger L2 LLM if configured).
_WEAK_CONFLICT_MARKERS: tuple[str, ...] = (
    "改为",
    "旧版本",
    "instead of",
    "changed from",
    "switch to",
    "migrate to",
    "修复了之前",
)

_KV_ASSIGNMENT_RE = re.compile(r"\b([a-zA-Z_][a-zA-Z0-9_.-]{2,40})\s*=\s*([^\s,;]+)")
_GENERIC_KV_KEYS: frozenset[str] = frozenset(
    {
        "id",
        "key",
        "val",
        "value",
        "name",
        "type",
        "status",
        "code",
        "msg",
        "message",
        "result",
        "count",
        "limit",
        "offset",
        "page",
        "size",
        "timeout",
    }
)

_LLM_ADJUDICATION_PROMPT = (
    "You are a write-time memory conflict resolver for a software engineering knowledge base.\n"
    "Given an INCOMING experience and up to 3 similar ACTIVE existing experiences, decide ONE action:\n"
    "- ADD: Distinct knowledge or different scope/component; keep both.\n"
    "- UPDATE: Incoming supplements/refines an existing item without invalidating it; "
    "provide target_id and concise updated_solution.\n"
    "- SUPERSEDE: Incoming corrects, replaces, or deprecates an existing item "
    "(e.g. upgraded API/config/workflow); provide target_id.\n"
    "  NEVER supersede an item with is_pinned=true or (epistemic_tier='verified' and staleness_status='fresh').\n"
    "- NOOP: Near-duplicate carrying no new actionable information; provide target_id.\n\n"
    "Respond with ONLY valid JSON:\n"
    '{"action": "ADD"|"UPDATE"|"SUPERSEDE"|"NOOP", "target_id": "<id or null>", '
    '"updated_solution": "<string or null>", "reason": "<brief reason>"}'
)


@dataclass
class MutationDecision:
    """Decision output of the write-time bi-temporal memory mutation state machine."""

    action: str  # "ADD" | "UPDATE" | "MERGE" | "SUPERSEDE" | "NOOP"
    target_id: str | None = None
    merged_solution: str | None = None
    merged_description: str | None = None
    reason: str = ""
    degraded: bool = False


def _extract_kv_pairs(text: str) -> dict[str, str]:
    """Extract non-generic key=value assignments from text for conflict detection."""
    pairs: dict[str, str] = {}
    for match in _KV_ASSIGNMENT_RE.finditer(text or ""):
        key = match.group(1).strip().lower()
        if key in _GENERIC_KV_KEYS:
            continue
        val = match.group(2).strip().strip("\"'")
        pairs[key] = val
    return pairs


def _is_protected_candidate(candidate: dict[str, Any]) -> bool:
    """Return True if candidate must not be auto-superseded at write time."""
    if candidate.get("is_pinned"):
        return True
    tier = str(candidate.get("epistemic_tier") or "auto_extract").lower()
    staleness = str(candidate.get("staleness_status") or "fresh").lower()
    return tier == "verified" and staleness == "fresh"


def _has_supersede_signal(
    title: str,
    problem: str,
    solution: str | None,
    candidate: dict[str, Any],
) -> bool:
    """Detect whether the incoming experience strongly supersedes or contradicts candidate."""
    combined_new = f"{title or ''}\n{problem or ''}\n{solution or ''}".lower()
    for marker in _STRONG_SUPERSEDE_MARKERS:
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


def _has_weak_conflict_signal(title: str, problem: str, solution: str | None) -> bool:
    combined_new = f"{title or ''}\n{problem or ''}\n{solution or ''}".lower()
    return any(marker in combined_new for marker in _WEAK_CONFLICT_MARKERS)


def _normalize_text(text: str | None) -> str:
    return re.sub(r"\s+", " ", (text or "").strip().lower())


class MutationResolver:
    """Resolves memory write operations into ADD, UPDATE/MERGE, SUPERSEDE, or NOOP."""

    def __init__(
        self,
        llm_config: Any = None,
        noop_threshold: float = 0.96,
        mutation_threshold: float = 0.85,
        *,
        llm_timeout_seconds: float = 2.0,
        llm_enabled: bool = False,
        llm_gray_zone_min: float = 0.86,
        update_action_name: str = "MERGE",
    ) -> None:
        self._llm_config = llm_config
        self.noop_threshold = noop_threshold
        self.mutation_threshold = mutation_threshold
        self.llm_timeout_seconds = max(0.1, float(llm_timeout_seconds))
        self.llm_enabled = bool(llm_enabled and llm_config is not None)
        self.llm_gray_zone_min = llm_gray_zone_min
        # Default "MERGE" preserves backward compatibility with existing rule-mode tests;
        # ExperienceService write-time gate passes "UPDATE" (handling both aliases identically).
        self.update_action_name = update_action_name

    def resolve_by_rules(
        self,
        *,
        title: str,
        problem: str,
        solution: str | None,
        candidates: list[dict[str, Any]],
    ) -> MutationDecision:
        """Deterministic 0ms rule-based state machine resolution."""
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
            if _is_protected_candidate(top):
                return MutationDecision(
                    action="ADD",
                    reason=(
                        "Guardrail: top candidate is pinned or fresh verified; "
                        "keeping both instead of auto-supersede"
                    ),
                )
            return MutationDecision(
                action="SUPERSEDE",
                target_id=target_id,
                reason="New experience supersedes or contradicts prior knowledge",
            )

        norm_new_sol = _normalize_text(solution)
        norm_top_sol = _normalize_text(top.get("solution"))
        if (
            sim >= self.noop_threshold
            or not norm_new_sol
            or (norm_top_sol and norm_new_sol in norm_top_sol)
        ):
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
            action=self.update_action_name,
            target_id=target_id,
            merged_solution=merged_solution,
            reason=f"Updated existing experience with complementary solution (similarity={sim:.3f})",
        )

    async def _call_llm_adjudicate(
        self,
        *,
        title: str,
        problem: str,
        solution: str | None,
        candidates: list[dict[str, Any]],
    ) -> dict[str, Any]:
        """Call lightweight LLM for write-time conflict adjudication."""
        base_url = getattr(self._llm_config, "base_url", "https://gw.gburouter.com/v1").rstrip("/")
        if not base_url.endswith("/v1"):
            base_url = f"{base_url}/v1"
        model = getattr(self._llm_config, "model", "gbu-gemini-3.8-flash")
        api_key_env = getattr(self._llm_config, "api_key_env", "GATEWAY_API_KEY")
        api_key = (
            os.environ.get(api_key_env, "")
            or getattr(self._llm_config, "api_key", "")
            or "none"
        )

        cand_payload = [
            {
                "id": str(c.get("id", "")),
                "title": (c.get("title") or "")[:200],
                "problem": (c.get("description") or "")[:400],
                "solution": (c.get("solution") or "")[:600],
                "similarity": round(float(c.get("similarity", 0.0)), 4),
                "is_pinned": bool(c.get("is_pinned", False)),
                "epistemic_tier": c.get("epistemic_tier", "auto_extract"),
                "staleness_status": c.get("staleness_status", "fresh"),
            }
            for c in candidates[:3]
        ]
        user_msg = json.dumps(
            {
                "incoming": {
                    "title": (title or "")[:200],
                    "problem": (problem or "")[:600],
                    "solution": (solution or "")[:1000],
                },
                "candidates": cand_payload,
            },
            ensure_ascii=False,
        )

        payload = {
            "model": model,
            "messages": [
                {"role": "system", "content": _LLM_ADJUDICATION_PROMPT},
                {"role": "user", "content": user_msg},
            ],
            "temperature": 0.0,
            "max_tokens": 512,
        }
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }

        async with httpx.AsyncClient(timeout=self.llm_timeout_seconds) as client:
            resp = await client.post(
                f"{base_url}/chat/completions",
                json=payload,
                headers=headers,
            )
            resp.raise_for_status()
            raw = resp.json()["choices"][0]["message"]["content"].strip()

        if "```" in raw:
            parts = raw.split("```")
            raw = parts[1] if len(parts) > 1 else raw
            if raw.startswith("json"):
                raw = raw[4:]
            raw = raw.strip()
        parsed = json.loads(raw)
        if not isinstance(parsed, dict):
            raise ValueError("LLM adjudication returned non-object JSON")
        return parsed

    def _validate_llm_decision(
        self,
        raw: dict[str, Any],
        candidates: list[dict[str, Any]],
    ) -> MutationDecision:
        by_id = {str(c.get("id", "")): c for c in candidates if c.get("id")}
        action = str(raw.get("action") or "ADD").upper().strip()
        reason = str(raw.get("reason") or "LLM write-time adjudication").strip()
        target_id = str(raw.get("target_id") or "").strip() or None

        if action == "MERGE":
            action = self.update_action_name

        if action not in ("ADD", "UPDATE", "MERGE", "SUPERSEDE", "NOOP"):
            return MutationDecision(
                action="ADD",
                reason=f"Invalid LLM action '{action}'; fell back to ADD",
                degraded=True,
            )

        if action == "ADD":
            return MutationDecision(action="ADD", reason=reason)

        if not target_id or target_id not in by_id:
            return MutationDecision(
                action="ADD",
                reason="LLM decision missing valid target_id; fell back to ADD",
                degraded=True,
            )

        target = by_id[target_id]
        if action == "SUPERSEDE" and _is_protected_candidate(target):
            return MutationDecision(
                action="ADD",
                reason="Guardrail: protected (pinned/fresh verified) target cannot be auto-superseded",
            )

        if action in ("UPDATE", "MERGE"):
            updated_sol = raw.get("updated_solution")
            return MutationDecision(
                action=self.update_action_name,
                target_id=target_id,
                merged_solution=str(updated_sol).strip() if updated_sol else None,
                reason=reason,
            )

        return MutationDecision(action=action, target_id=target_id, reason=reason)

    async def resolve(
        self,
        *,
        title: str,
        problem: str,
        solution: str | None,
        candidates: list[dict[str, Any]],
    ) -> MutationDecision:
        """Resolve memory state transition with fast rule path and bounded-timeout LLM adjudication."""
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

        # Fast 0ms path: near-identical NOOP
        norm_new_sol = _normalize_text(solution)
        norm_top_sol = _normalize_text(top.get("solution"))
        if sim >= self.noop_threshold or (
            sim >= 0.88 and (not norm_new_sol or (norm_top_sol and norm_new_sol in norm_top_sol))
        ):
            if not _has_supersede_signal(title, problem, solution, top):
                return MutationDecision(
                    action="NOOP",
                    target_id=target_id,
                    reason=f"Near-identical experience (similarity={sim:.3f}); refreshed verification timestamp",
                )

        # Fast 0ms path: high-confidence strong SUPERSEDE
        if sim >= 0.88 and _has_supersede_signal(title, problem, solution, top):
            if _is_protected_candidate(top):
                return MutationDecision(
                    action="ADD",
                    reason="Guardrail: top candidate is pinned or fresh verified; keeping both",
                )
            return MutationDecision(
                action="SUPERSEDE",
                target_id=target_id,
                reason="New experience strongly supersedes prior knowledge",
            )

        # L2: Borderline LLM adjudication with strict timeout and fail-safe ADD fallback
        should_use_llm = self.llm_enabled and (
            sim >= self.llm_gray_zone_min
            or _has_supersede_signal(title, problem, solution, top)
            or _has_weak_conflict_signal(title, problem, solution)
        )
        if should_use_llm:
            try:
                raw = await asyncio.wait_for(
                    self._call_llm_adjudicate(
                        title=title,
                        problem=problem,
                        solution=solution,
                        candidates=sorted_candidates[:3],
                    ),
                    timeout=self.llm_timeout_seconds,
                )
                return self._validate_llm_decision(raw, sorted_candidates[:3])
            except asyncio.TimeoutError:
                logger.warning(
                    "Write-time LLM mutation adjudication timed out after %.1fs; degrading safely to ADD",
                    self.llm_timeout_seconds,
                )
                return MutationDecision(
                    action="ADD",
                    reason=f"LLM adjudication timed out after {self.llm_timeout_seconds:.1f}s; degraded to ADD",
                    degraded=True,
                )
            except Exception as exc:
                logger.warning(
                    "Write-time LLM mutation adjudication failed (%s); degrading safely to ADD",
                    exc,
                )
                return MutationDecision(
                    action="ADD",
                    reason=f"LLM adjudication failed ({type(exc).__name__}); degraded to ADD",
                    degraded=True,
                )

        return self.resolve_by_rules(
            title=title,
            problem=problem,
            solution=solution,
            candidates=sorted_candidates,
        )

"""Shared helper functions and constants for MemoryJanitor."""

from __future__ import annotations

import math
from datetime import datetime, timezone

_CANONICAL_ARCHIVE_CONTENT_TYPES = {
    "session_archive",
    "tech_design",
    "incident_review",
    "decision_record",
    "bugfix",
    "engineering_experience",
}


def _cosine_similarity(vec_a: object, vec_b: object) -> float:
    """Compute cosine similarity between two vector-like sequences."""
    if vec_a is None or vec_b is None:
        return 0.0
    try:
        a = [float(x) for x in vec_a]  # type: ignore[attr-defined]
        b = [float(x) for x in vec_b]  # type: ignore[attr-defined]
    except (TypeError, ValueError):
        return 0.0
    if len(a) != len(b) or not a:
        return 0.0
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(x * x for x in b))
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return dot / (norm_a * norm_b)


def _normalize_archive_content_type(raw: str | None) -> str:
    """Map free-form archive content_type strings to the 6 canonical types."""
    if not raw or not isinstance(raw, str):
        return "session_archive"
    ct = raw.strip().lower()
    if ct in _CANONICAL_ARCHIVE_CONTENT_TYPES:
        return ct
    if any(
        k in ct
        for k in (
            "incident",
            "postmortem",
            "retrospective",
            "audit",
            "analysis",
            "test_report",
            "troubleshooting",
            "verification",
        )
    ):
        return "incident_review"
    if any(k in ct for k in ("decision", "adr")):
        return "decision_record"
    if any(k in ct for k in ("bug", "fix", "hotfix", "patch")):
        return "bugfix"
    if any(
        k in ct
        for k in ("design", "architecture", "refactor", "spec", "plan", "migration", "feature", "optimization")
    ):
        return "tech_design"
    if any(
        k in ct
        for k in ("guide", "sop", "experience", "best_practice", "onboarding", "deployment", "workflow", "handbook")
    ):
        return "engineering_experience"
    return "session_archive"


def _utcnow() -> datetime:
    """Get current UTC datetime."""
    return datetime.now(timezone.utc)


def _is_eligible_for_promotion(exp, threshold: int, require_used: bool) -> bool:
    """Check whether an experience is eligible for promotion."""
    if isinstance(getattr(exp, "valid_until", None), datetime):
        return False
    if isinstance(getattr(exp, "negative_feedback_count", None), int) and exp.negative_feedback_count > 0:
        return False
    if require_used and isinstance(getattr(exp, "used_count", None), int) and exp.used_count < threshold:
        return False
    return True

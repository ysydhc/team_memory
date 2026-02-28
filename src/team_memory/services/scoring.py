"""Experience quality scoring engine.

Implements time-based decay, reference/rating boosts, and tier calculation.
"""

from __future__ import annotations

import json
import logging
from datetime import date, timedelta
from pathlib import Path

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

logger = logging.getLogger(__name__)

DEFAULT_SCORING_CONFIG = {
    "initial_score": 100,
    "max_score": 300,
    "protection_days": 10,
    "decay_rate": 1.0,
    "slow_decay_threshold": 50,
    "slow_decay_rate": 0.5,
    "reference_boost": 2,
    "high_rating_boost": 1,
    "high_rating_threshold": 4,
    "outdated_threshold": 0,
    "tiers": {"gold": 120, "silver": 60, "bronze": 20},
}

_CFG_PATH = Path.home() / ".team_memory" / "scoring_config.json"


def get_scoring_config() -> dict:
    if _CFG_PATH.exists():
        try:
            with open(_CFG_PATH) as f:
                cfg = json.load(f)
            merged = {**DEFAULT_SCORING_CONFIG, **cfg}
            merged["tiers"] = {**DEFAULT_SCORING_CONFIG["tiers"], **cfg.get("tiers", {})}
            return merged
        except Exception:
            pass
    return dict(DEFAULT_SCORING_CONFIG)


def save_scoring_config(cfg: dict) -> None:
    _CFG_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(_CFG_PATH, "w") as f:
        json.dump(cfg, f, ensure_ascii=False, indent=2)


def calculate_decay(
    current_score: int,
    created_at_date: date,
    last_decay: date | None,
    pinned: bool,
    config: dict | None = None,
) -> tuple[int, int]:
    """Calculate decay and return (decay_amount, new_score).

    Uses stepped decay: protection period, then -1/day, then -0.5/day below threshold.
    """
    if pinned:
        return 0, current_score

    cfg = config or get_scoring_config()
    today = date.today()
    protection_days = cfg["protection_days"]
    decay_rate = cfg["decay_rate"]
    slow_threshold = cfg["slow_decay_threshold"]
    slow_rate = cfg["slow_decay_rate"]
    max_score = cfg["max_score"]

    start_date = last_decay or created_at_date
    days_to_process = (today - start_date).days
    if days_to_process <= 0:
        return 0, min(current_score, max_score)

    protection_end = created_at_date + timedelta(days=protection_days)

    total_decay = 0.0
    score = float(current_score)
    for i in range(days_to_process):
        check_date = start_date + timedelta(days=i + 1)
        if check_date <= protection_end:
            continue
        if score - total_decay <= slow_threshold:
            total_decay += slow_rate
        else:
            total_decay += decay_rate

    new_score = max(0, current_score - int(total_decay))
    return int(total_decay), min(new_score, max_score)


def apply_reference_boost(current_score: int, config: dict | None = None) -> int:
    cfg = config or get_scoring_config()
    return min(current_score + cfg["reference_boost"], cfg["max_score"])


def apply_rating_boost(
    current_score: int, rating: float, config: dict | None = None
) -> int:
    cfg = config or get_scoring_config()
    if rating >= cfg["high_rating_threshold"]:
        return min(current_score + cfg["high_rating_boost"], cfg["max_score"])
    return current_score


def get_tier(score: int, config: dict | None = None) -> str:
    cfg = config or get_scoring_config()
    tiers = cfg["tiers"]
    if score >= tiers.get("gold", 120):
        return "gold"
    if score >= tiers.get("silver", 60):
        return "silver"
    if score >= tiers.get("bronze", 20):
        return "bronze"
    return "outdated"


async def run_decay_batch(session: AsyncSession, config: dict | None = None) -> int:
    """Apply decay to all non-pinned experiences. Returns count of updated rows."""
    from team_memory.storage.models import Experience

    cfg = config or get_scoring_config()
    today = date.today()

    q = (
        select(Experience)
        .where(Experience.pinned == False)  # noqa: E712
        .where(Experience.is_deleted == False)  # noqa: E712
        .where(Experience.quality_score > 0)
    )
    result = await session.execute(q)
    experiences = list(result.scalars().all())

    updated = 0
    for exp in experiences:
        decay_amount, new_score = calculate_decay(
            exp.quality_score,
            exp.created_at.date() if exp.created_at else today,
            exp.last_decay_date,
            exp.pinned,
            cfg,
        )
        if decay_amount > 0:
            exp.quality_score = new_score
            exp.last_decay_date = today
            updated += 1

    if updated > 0:
        await session.flush()

    return updated

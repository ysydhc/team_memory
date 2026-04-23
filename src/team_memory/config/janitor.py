"""Janitor service configuration."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class JanitorConfig:
    """Configuration for the memory janitor service.

    Controls automated cleanup and quality management operations:
    - Quality score decay over time
    - Outdated experience cleanup
    - Soft-deleted record purging
    - Draft expiration
    - Personal memory pruning
    """

    enabled: bool = True
    """Whether janitor operations are enabled."""

    interval_hours: int = 24
    """Hours between scheduled janitor runs."""

    protection_period_days: int = 10
    """Days to protect new experiences from quality decay."""

    decay_per_day: float = 1.0
    """Quality score decay per day for scores >= 50."""

    decay_per_day_low: float = 0.5
    """Quality score decay per day for scores < 50."""

    decay_low_threshold: float = 50.0
    """Score threshold below which slow decay applies."""

    outdated_threshold: float = 0.0
    """Quality score threshold for marking experiences as outdated."""

    auto_soft_delete_outdated: bool = False
    """Whether to automatically soft-delete outdated experiences."""

    purge_soft_deleted_days: int = 30
    """Days after which soft-deleted experiences are permanently purged."""

    draft_expiry_days: int = 30
    """Days after which draft experiences expire and are soft-deleted."""

    personal_memory_retention_days: int = 90
    """Days to retain dynamic personal memory entries."""

    promotion_enabled: bool = True
    """Whether L2→L3 auto-promotion is enabled."""

    promotion_use_count_threshold: int = 3
    """use_count reaching this value triggers promotion."""

    promotion_group_key_threshold: int = 5
    """Same group_key reaching this count triggers promotion."""

    tier_gold: float = 120.0
    """Quality score threshold for Gold tier."""

    tier_silver: float = 60.0
    """Quality score threshold for Silver tier."""

    tier_bronze: float = 20.0
    """Quality score threshold for Bronze tier."""

    promotion_output_dirs: dict[str, str] = field(default_factory=dict)
    """project → Obsidian output directory mapping for promoted Markdown files."""

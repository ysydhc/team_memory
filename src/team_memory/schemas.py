"""Pydantic validation schemas for team_memory (MVP).

Minimal schema layer — type system and structured_data removed.
"""

from __future__ import annotations

# Legacy constants kept as simple labels (no schema binding)
EXPERIENCE_TYPES = (
    "general", "feature", "bugfix", "tech_design",
    "incident", "best_practice", "learning",
)

"""Memory janitor service for automated cleanup and quality management.

Provides automated maintenance operations for team memory:
- Quality score decay over time
- Outdated experience cleanup
- Soft-deleted record purging
- Draft expiration
- Personal memory pruning
"""

from __future__ import annotations

import subprocess

from team_memory.services.maintenance.janitor.helpers import (
    _CANONICAL_ARCHIVE_CONTENT_TYPES,
    _cosine_similarity,
    _is_eligible_for_promotion,
    _normalize_archive_content_type,
    _utcnow,
)
from team_memory.services.maintenance.janitor.orchestrator import MemoryJanitor
from team_memory.services.maintenance.promotion_compiler import PromotionCompiler

__all__ = [
    "MemoryJanitor",
    "PromotionCompiler",
    "subprocess",
    "_CANONICAL_ARCHIVE_CONTENT_TYPES",
    "_cosine_similarity",
    "_normalize_archive_content_type",
    "_utcnow",
    "_is_eligible_for_promotion",
]

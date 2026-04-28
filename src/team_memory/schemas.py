"""Pydantic validation schemas for team_memory (MVP).

Minimal schema layer — type system and structured_data removed.
"""

from __future__ import annotations

from pydantic import BaseModel, Field, field_validator

# Legacy constants kept as simple labels (no schema binding)
EXPERIENCE_TYPES = (
    "general",
    "feature",
    "bugfix",
    "tech_design",
    "incident",
    "best_practice",
    "learning",
)

SEVERITY_LEVELS = (
    "critical",
    "high",
    "medium",
    "low",
    "info",
)

CATEGORIES = (
    "backend",
    "frontend",
    "mobile",
    "infra",
    "data",
    "security",
    "process",
    "product",
    "other",
)


class ArchiveCreateRequest(BaseModel):
    """Request body for POST /api/v1/archives (create or upsert)."""

    title: str = Field(..., max_length=500)
    solution_doc: str = Field(..., max_length=64_000)
    content_type: str = "session_archive"
    value_summary: str | None = Field(None, max_length=500)
    tags: list[str] | None = None
    overview: str | None = Field(None, max_length=4_000)
    conversation_summary: str | None = Field(None, max_length=6_400)
    raw_conversation: str | None = Field(None, max_length=640_000)
    linked_experience_ids: list[str] | None = None
    project: str | None = None
    scope: str = "session"
    scope_ref: str | None = None

    @field_validator("tags")
    @classmethod
    def validate_tags(cls, v: list[str] | None) -> list[str] | None:
        if v is None:
            return v
        if len(v) > 20:
            raise ValueError("Maximum 20 tags allowed")
        for tag in v:
            if len(tag) > 50:
                raise ValueError(f"Tag too long (max 50 chars): {tag[:20]}...")
        return v

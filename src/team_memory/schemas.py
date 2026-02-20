"""Pydantic validation schemas for JSONB fields and API requests.

Provides type-safe validation for structured_data, git_refs, and related_links
before they are persisted to the database. Each experience_type has its own
structured data schema.

Also provides the **SchemaRegistry** — the single source of truth for which
experience_types, categories, severity_levels, progress_states and structured
fields are available at runtime.  All code should read from the registry (via
``get_schema_registry()``) instead of the legacy constants.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Literal

from pydantic import BaseModel

if TYPE_CHECKING:
    from team_memory.config import CustomSchemaConfig, ExperienceTypeDef

logger = logging.getLogger("team_memory.schemas")

# ---------------------------------------------------------------------------
# Legacy constants (kept for backward compatibility — prefer SchemaRegistry)
# ---------------------------------------------------------------------------

EXPERIENCE_TYPES = (
    "general", "feature", "bugfix", "tech_design",
    "incident", "best_practice", "learning",
)

SEVERITY_LEVELS = ("P0", "P1", "P2", "P3", "P4")

CATEGORY_OPTIONS = (
    "frontend", "backend", "database", "infra",
    "performance", "security", "mobile", "other",
)

PROGRESS_STATES: dict[str, list[str]] = {
    "bugfix": ["open", "investigating", "fixed", "verified"],
    "feature": ["planning", "developing", "testing", "released"],
    "tech_design": ["researching", "reviewing", "implementing", "completed"],
    "incident": ["detected", "mitigating", "resolved", "post_mortem"],
}

TYPE_DEFAULT_NO_SOLUTION: dict[str, str] = {
    "bugfix": "open",
    "feature": "planning",
    "tech_design": "researching",
    "incident": "detected",
}

TYPE_DEFAULT_WITH_SOLUTION: dict[str, str] = {
    "bugfix": "fixed",
    "feature": "developing",
    "tech_design": "implementing",
    "incident": "mitigating",
}

TEMPLATE_FIELD_COUNTS: dict[str, int] = {
    "bugfix": 5,
    "feature": 4,
    "tech_design": 6,
    "incident": 7,
}


# ---------------------------------------------------------------------------
# Git Refs & Related Links — cross-type, top-level JSONB fields
# ---------------------------------------------------------------------------

class GitRef(BaseModel):
    """A reference to a git commit, PR, or branch."""
    type: Literal["commit", "pr", "branch"]
    url: str | None = None
    hash: str | None = None
    description: str | None = None


class RelatedLink(BaseModel):
    """A link to an issue tracker, document, wiki, or other URL."""
    type: Literal["issue", "doc", "wiki", "other"] = "other"
    url: str
    title: str | None = None


# ---------------------------------------------------------------------------
# Structured data schemas — one per experience_type
# ---------------------------------------------------------------------------

class BugfixData(BaseModel):
    """Structured data for bugfix experiences."""
    reproduction_steps: str | None = None
    environment: str | None = None
    error_logs: str | None = None
    impact_scope: str | None = None
    verification_result: str | None = None


class FeatureData(BaseModel):
    """Structured data for feature/requirement experiences."""
    requirements: str | None = None
    acceptance_criteria: list[str] | None = None
    test_summary: str | None = None
    release_notes: str | None = None


class AlternativeOption(BaseModel):
    """A single alternative in a tech design comparison."""
    name: str = ""
    pros: str = ""
    cons: str = ""
    chosen: bool = False


class TechDesignData(BaseModel):
    """Structured data for tech design experiences."""
    alternatives: list[AlternativeOption] | None = None
    rollback_plan: str | None = None
    data_migration: str | None = None
    performance_data: str | None = None
    upstream_downstream: str | None = None
    monitoring: str | None = None


class IncidentData(BugfixData):
    """Structured data for incident experiences (extends bugfix)."""
    timeline: str | None = None
    prevention: str | None = None


# Map experience_type to its Pydantic schema
STRUCTURED_DATA_SCHEMAS: dict[str, type[BaseModel]] = {
    "bugfix": BugfixData,
    "feature": FeatureData,
    "tech_design": TechDesignData,
    "incident": IncidentData,
}


def validate_structured_data(experience_type: str, data: dict | None) -> dict | None:
    """Validate structured_data against the schema for the given experience_type.

    Returns the validated dict (with defaults applied) or None.
    For types without a schema (general, best_practice, learning),
    the data is passed through unchanged.
    """
    if data is None:
        return None
    schema = STRUCTURED_DATA_SCHEMAS.get(experience_type)
    if schema is None:
        # No schema for this type — pass through as-is
        return data
    validated = schema.model_validate(data)
    return validated.model_dump(exclude_none=False)


def validate_git_refs(refs: list | None) -> list[dict] | None:
    """Validate a list of git references."""
    if not refs:
        return refs
    validated = []
    for ref in refs:
        validated.append(GitRef.model_validate(ref).model_dump(exclude_none=False))
    return validated


def validate_related_links(links: list | None) -> list[dict] | None:
    """Validate a list of related links."""
    if not links:
        return links
    validated = []
    for link in links:
        validated.append(RelatedLink.model_validate(link).model_dump(exclude_none=False))
    return validated


def compute_completeness_score(
    title: str | None,
    description: str | None,
    solution: str | None,
    root_cause: str | None,
    code_snippets: str | None,
    tags: list | None,
    git_refs: list | None,
    related_links: list | None,
    structured_data: dict | None,
    experience_type: str = "general",
    avg_rating: float = 0.0,
) -> int:
    """Calculate a 0-100 completeness score for an experience.

    Encourages teams to fill in more fields over time.
    """
    score = 0
    if title:
        score += 10
    if description:
        score += 10
    if solution:
        score += 15
    if root_cause:
        score += 10
    if code_snippets:
        score += 10
    if tags:
        score += 5
    if git_refs:
        score += 10
    if related_links:
        score += 5

    # structured_data fill rate (up to 20 points)
    # Try SchemaRegistry first, fall back to legacy constant
    registry = _get_registry_safe()
    if registry:
        type_def = registry.get_type_def(experience_type)
        type_fields = len(type_def.structured_fields) if type_def else 0
    else:
        type_fields = TEMPLATE_FIELD_COUNTS.get(experience_type, 0)

    if type_fields > 0 and structured_data:
        filled = sum(1 for v in structured_data.values() if v)
        score += int(20 * min(filled / type_fields, 1.0))

    # Has feedback
    if avg_rating > 0:
        score += 5

    return min(score, 100)


# =========================================================================
# SchemaRegistry — runtime source of truth for types / categories / severity
# =========================================================================


class SchemaRegistry:
    """Runtime schema registry that merges preset defaults with YAML overrides.

    Usage:
        registry = SchemaRegistry(settings.custom_schema)
        registry.is_valid_type("bugfix")        # True
        registry.get_experience_types()          # list[ExperienceTypeDef]
        registry.to_dict()                       # full export for API / prompt
    """

    def __init__(self, config: "CustomSchemaConfig | None" = None) -> None:
        from team_memory.config import CategoryDef, CustomSchemaConfig
        from team_memory.schema_presets import get_preset

        if config is None:
            config = CustomSchemaConfig()

        preset = get_preset(config.preset)

        # Start with a copy of preset types (keyed by id)
        type_map: dict[str, ExperienceTypeDef] = {
            t.id: t.model_copy() for t in preset["experience_types"]
        }
        # Overlay user-defined types (same id = override, new id = append)
        for t in config.experience_types:
            type_map[t.id] = t

        self._types: list[ExperienceTypeDef] = list(type_map.values())

        # Categories
        cat_map: dict[str, CategoryDef] = {
            c.id: c for c in preset["categories"]
        }
        for c in config.categories:
            cat_map[c.id] = c
        self._categories: list[CategoryDef] = list(cat_map.values())

        # Severity levels
        if config.severity_levels:
            self._severity: list[str] = list(config.severity_levels)
        else:
            self._severity = list(preset["severity_levels"])

        # Build lookup caches
        self._type_map: dict[str, ExperienceTypeDef] = {t.id: t for t in self._types}
        self._cat_set: set[str] = {c.id for c in self._categories}
        self._sev_set: set[str] = set(self._severity)

        # Progress state helpers (derived from type definitions)
        self._progress_states: dict[str, list[str]] = {}
        self._default_no_solution: dict[str, str] = {}
        self._default_with_solution: dict[str, str] = {}
        for t in self._types:
            if t.progress_states:
                self._progress_states[t.id] = list(t.progress_states)
                self._default_no_solution[t.id] = t.progress_states[0]
                mid = len(t.progress_states) // 2
                self._default_with_solution[t.id] = t.progress_states[mid]

        self._preset_name = config.preset

    # ── Getters ──

    def get_experience_types(self) -> list["ExperienceTypeDef"]:
        return list(self._types)

    def get_categories(self) -> list:
        return list(self._categories)

    def get_severity_levels(self) -> list[str]:
        return list(self._severity)

    def get_type_def(self, type_id: str) -> "ExperienceTypeDef | None":
        return self._type_map.get(type_id)

    def get_progress_states(self, type_id: str) -> list[str]:
        return self._progress_states.get(type_id, [])

    def get_default_progress(self, type_id: str, has_solution: bool) -> str | None:
        if has_solution:
            return self._default_with_solution.get(type_id)
        return self._default_no_solution.get(type_id)

    # ── Validators ──

    def is_valid_type(self, type_id: str) -> bool:
        return type_id in self._type_map

    def is_valid_category(self, cat_id: str) -> bool:
        return cat_id in self._cat_set

    def is_valid_severity(self, sev: str) -> bool:
        return sev in self._sev_set

    def is_valid_progress(self, type_id: str, status: str) -> bool:
        states = self._progress_states.get(type_id)
        if not states:
            return True  # no constraints
        return status in states

    def validate_structured_data_for_type(
        self, type_id: str, data: dict | None
    ) -> dict | None:
        """Validate structured_data against Pydantic schema (if exists) or
        against the registry's structured_fields definitions."""
        if data is None:
            return None
        # Try Pydantic schema first (backward compat)
        schema = STRUCTURED_DATA_SCHEMAS.get(type_id)
        if schema is not None:
            validated = schema.model_validate(data)
            return validated.model_dump(exclude_none=False)
        # Fall through: pass unknown-type data as-is
        return data

    # ── Export ──

    def to_dict(self) -> dict:
        """Full schema export for API responses and prompt variable injection."""
        return {
            "preset": self._preset_name,
            "experience_types": [
                {
                    "id": t.id,
                    "label": t.label,
                    "severity": t.severity,
                    "progress_states": t.progress_states,
                    "structured_fields": [
                        {"name": f.name, "type": f.type, "label": f.label, "required": f.required}
                        for f in t.structured_fields
                    ],
                }
                for t in self._types
            ],
            "categories": [
                {"id": c.id, "label": c.label} for c in self._categories
            ],
            "severity_levels": list(self._severity),
        }

    def types_for_prompt(self) -> str:
        """Format experience types as a readable list for LLM prompts."""
        lines = []
        for t in self._types:
            label = f" ({t.label})" if t.label else ""
            lines.append(f'- "{t.id}": {t.label or t.id}{label}')
        return "\n".join(lines)

    def categories_for_prompt(self) -> str:
        """Format categories as a readable list for LLM prompts."""
        return "/".join(c.id for c in self._categories)

    def severity_for_prompt(self) -> str:
        """Format severity levels for LLM prompts."""
        if not self._severity:
            return "无"
        return "/".join(self._severity)


# ── Global singleton ──

_schema_registry: SchemaRegistry | None = None


def get_schema_registry() -> SchemaRegistry:
    """Return the global SchemaRegistry singleton.

    Lazily initialized from the global Settings the first time it is called.
    """
    global _schema_registry
    if _schema_registry is None:
        _schema_registry = _build_registry()
    return _schema_registry


def init_schema_registry(config: "CustomSchemaConfig | None" = None) -> SchemaRegistry:
    """Explicitly (re-)initialize the global SchemaRegistry.

    Call this on application startup after settings are loaded.
    """
    global _schema_registry
    _schema_registry = SchemaRegistry(config)
    return _schema_registry


def reset_schema_registry() -> None:
    """Reset the global SchemaRegistry (for testing)."""
    global _schema_registry
    _schema_registry = None


def _build_registry() -> SchemaRegistry:
    """Build a SchemaRegistry from the global Settings singleton."""
    try:
        from team_memory.config import get_settings
        settings = get_settings()
        return SchemaRegistry(settings.custom_schema)
    except Exception:
        logger.debug("Settings not available, using default SchemaRegistry")
        return SchemaRegistry()


def _get_registry_safe() -> SchemaRegistry | None:
    """Return the global registry if already initialized, else None."""
    return _schema_registry

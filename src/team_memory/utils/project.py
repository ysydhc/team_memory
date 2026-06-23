"""Project name resolution utilities.

Shared by both MCP server and Web layer to normalize and resolve
the active project from explicit param > env > config default.
"""

from __future__ import annotations

import os


def normalize_project_name(project: str | None) -> str:
    """Normalize a project name to its canonical slug, then resolve aliases.

    Canonicalization (lowercase, path-basename, separators → "_") is applied
    first so that every write path produces a single stable form regardless of
    whether the caller sent "adx-sdk-android", a full path, or "MyGuguGaga".
    Alias resolution runs on the canonical slug.
    """
    if not project:
        return ""
    try:
        from team_memory.project_resolver import canonicalize_project_name, resolve_alias

        canonical = canonicalize_project_name(project)
        return resolve_alias(canonical)
    except ImportError:
        # Fallback when project_resolver is unavailable: minimal canonicalization.
        value = project.strip().lower().replace("-", "_")
        alias_map = {"team_memory_project": "team_memory"}
        return alias_map.get(value, value)


def resolve_project(project: str | None = None) -> str:
    """Resolve project from explicit param > env > settings default.

    Uses normalize_project_name at each stage. Falls back to "default"
    if nothing is configured.
    """
    from team_memory.bootstrap import get_context

    normalized = normalize_project_name(project)
    if normalized:
        return normalized
    env_project = normalize_project_name(os.environ.get("TEAM_MEMORY_PROJECT", ""))
    if env_project:
        return env_project
    ctx = get_context()
    default_project = normalize_project_name(ctx.settings.default_project)
    return default_project or "default"

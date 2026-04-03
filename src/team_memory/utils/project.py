"""Project name resolution utilities.

Shared by both MCP server and Web layer to normalize and resolve
the active project from explicit param > env > config default.
"""

from __future__ import annotations

import os


def normalize_project_name(project: str | None) -> str:
    """Normalize legacy project aliases to a canonical project name."""
    if not project:
        return ""
    value = project.strip()
    alias_map = {
        "team-memory": "team_memory",
        "team_doc": "team_memory",
    }
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

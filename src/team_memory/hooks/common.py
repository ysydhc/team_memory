"""Shared utility functions for Agent memory pipeline hook scripts.

Hook scripts (invoked by Cursor / Claude Code) read JSON from stdin,
use these helpers to call TeamMemory's MCP-over-HTTP interface, and
load pipeline configuration.
"""

from __future__ import annotations

import io
import json
import sys
from pathlib import Path
from typing import Any

import httpx
import yaml

# Directory where this module lives — used to locate config.yaml
_HOOKS_DIR = Path(__file__).resolve().parent

# Cached config so we don't hit the filesystem on every call
_config_cache: dict | None = None


# ---------------------------------------------------------------------------
# parse_hook_input
# ---------------------------------------------------------------------------

def parse_hook_input() -> dict:
    """Read JSON payload from stdin (Cursor/Claude Code hooks send data via stdin).

    Returns the parsed dict, or an empty dict on parse failure.
    """
    try:
        raw = sys.stdin.read()
        return json.loads(raw) if raw.strip() else {}
    except (json.JSONDecodeError, ValueError):
        return {}


# ---------------------------------------------------------------------------
# load_config
# ---------------------------------------------------------------------------

def load_config() -> dict:
    """Load config.yaml from the same directory as this module.

    Returns the parsed config dict.  Results are cached for the process
    lifetime.
    """
    global _config_cache
    if _config_cache is not None:
        return _config_cache

    config_path = _HOOKS_DIR / "config.yaml"
    with open(config_path, "r", encoding="utf-8") as fh:
        _config_cache = yaml.safe_load(fh)
    return _config_cache  # type: ignore[return-value]


# ---------------------------------------------------------------------------
# get_project_from_path
# ---------------------------------------------------------------------------

def get_project_from_path(workspace_root: str, config: dict | None = None) -> str | None:
    """Map a workspace path to a TM project name using config patterns.

    Args:
        workspace_root: Absolute or relative workspace path string.
        config: Optional pre-loaded config dict.  If *None*, load_config() is called.

    Returns:
        The matching project name, or None if no pattern matches.
    """
    if config is None:
        config = load_config()

    projects = config.get("projects", {})
    # Support both dict format {"name": {"path_patterns": [...]}} and list format [{"name": ..., "path_patterns": [...]}]
    if isinstance(projects, list):
        for entry in projects:
            project_name = entry.get("name", "")
            patterns = entry.get("path_patterns", [])
            for pattern in patterns:
                if pattern in workspace_root:
                    return project_name
    else:
        for project_name, project_cfg in projects.items():
            patterns: list[str] = project_cfg.get("path_patterns", [])
            for pattern in patterns:
                if pattern in workspace_root:
                    return project_name
    return None


# ---------------------------------------------------------------------------
# call_mcp_tool
# ---------------------------------------------------------------------------

def call_mcp_tool(tool_name: str, arguments: dict) -> dict:
    """Call a TeamMemory MCP tool via HTTP POST.

    TM exposes an MCP-over-HTTP interface at the base_url configured in
    config.yaml (default: http://localhost:3900).

    Args:
        tool_name: Name of the MCP tool (e.g. "memory_search").
        arguments: Dict of arguments to pass to the tool.

    Returns:
        Parsed JSON response as a dict.
    """
    config = load_config()
    base_url: str = config["tm"]["base_url"].rstrip("/")
    url = f"{base_url}/{tool_name}"

    payload = {"arguments": arguments}
    response = httpx.post(url, json=payload, timeout=30.0)
    response.raise_for_status()
    return response.json()

"""Path normalization utilities for node_key alignment."""

from __future__ import annotations


def normalize_node_key(path: str) -> str:
    """Normalize a file path to a consistent relative path format.

    - Output unified as relative path (e.g. src/team_memory/server.py)
    - Remove ./ and leading / prefix
    - Convert backslash to forward slash
    - Empty string or whitespace returns ""

    Args:
        path: Raw path string (may have ./ prefix, leading /, or backslashes).

    Returns:
        Normalized relative path string.
    """
    if not path or not path.strip():
        return ""
    s = path.strip()
    s = s.replace("\\", "/")
    if s.startswith("./"):
        s = s[2:]
    elif s.startswith("/"):
        s = s[1:]
    return s

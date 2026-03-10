"""Tests for path normalization utilities."""

from __future__ import annotations

import pytest

from team_memory.utils.path_utils import normalize_node_key


class TestNormalizeNodeKey:
    """Tests for normalize_node_key function."""

    def test_normalize_strips_leading_dot_slash(self) -> None:
        """Remove ./ prefix."""
        assert normalize_node_key("./src/team_memory/server.py") == "src/team_memory/server.py"

    def test_normalize_strips_leading_slash(self) -> None:
        """Remove leading / prefix."""
        assert normalize_node_key("/src/team_memory/server.py") == "src/team_memory/server.py"

    def test_normalize_backslash_to_forward(self) -> None:
        """Convert backslash to forward slash."""
        assert normalize_node_key("src\\team_memory\\server.py") == "src/team_memory/server.py"

    def test_normalize_empty_returns_empty(self) -> None:
        """Empty string or whitespace returns empty string."""
        assert normalize_node_key("") == ""
        assert normalize_node_key("   ") == ""

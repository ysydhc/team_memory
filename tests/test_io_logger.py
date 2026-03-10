"""Tests for io_logger module."""

from __future__ import annotations


def test_should_log_node_mcp_level():
    from team_memory.io_logger import _should_log_node

    assert _should_log_node("mcp", "cache_check") is False
    assert _should_log_node("mcp", "search") is True


def test_should_log_node_pipeline_level():
    from team_memory.io_logger import _should_log_node

    assert _should_log_node("pipeline", "query_expansion") is True
    assert _should_log_node("pipeline", "cache_check") is False

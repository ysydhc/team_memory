"""Tests for context trimmer (token budget management)."""

import pytest

from team_memory.services.context_trimmer import (
    ContextTrimmer,
    estimate_tokens,
    result_to_text,
)
from team_memory.services.search_pipeline import SearchResultItem

# ======================== Token Estimation ========================


class TestEstimateTokens:
    def test_empty_string(self):
        assert estimate_tokens("") == 0

    def test_english_text(self):
        text = "Hello world"  # 11 chars -> ~2-3 tokens
        tokens = estimate_tokens(text)
        assert 2 <= tokens <= 4

    def test_chinese_text(self):
        text = "你好世界"  # 4 CJK chars -> ~4 tokens
        tokens = estimate_tokens(text)
        assert tokens == 4

    def test_mixed_text(self):
        text = "Hello 你好 World 世界"  # Mixed
        tokens = estimate_tokens(text)
        assert tokens > 0

    def test_long_text(self):
        text = "a" * 4000  # ~1000 tokens
        tokens = estimate_tokens(text)
        assert 900 <= tokens <= 1100


# ======================== result_to_text ========================


def _make_item(**kwargs) -> SearchResultItem:
    """Helper to create a SearchResultItem with default data."""
    data = {
        "title": kwargs.get("title", ""),
        "description": kwargs.get("description", ""),
        "solution": kwargs.get("solution", ""),
    }
    data.update(kwargs.get("extra", {}))
    return SearchResultItem(data=data, score=kwargs.get("score", 1.0))


class TestResultToText:
    def test_basic(self):
        item = _make_item(
            title="Test Title",
            description="Test Desc",
            solution="Test Solution",
        )
        text = result_to_text(item)
        assert "Test Title" in text
        assert "Test Desc" in text
        assert "Test Solution" in text

    def test_with_children(self):
        item = _make_item(
            title="Parent",
            extra={
                "children": [
                    {"title": "Child 1", "description": "Desc 1", "solution": "Sol 1"},
                ]
            },
        )
        text = result_to_text(item)
        assert "Child 1" in text


# ======================== ContextTrimmer ========================


class TestContextTrimmer:
    @pytest.mark.asyncio
    async def test_no_limit(self):
        """No token limit means all items pass through."""
        trimmer = ContextTrimmer(max_tokens=None)
        items = [_make_item(title=f"Item {i}") for i in range(10)]
        result = await trimmer.trim(items)
        assert len(result) == 10

    @pytest.mark.asyncio
    async def test_top_k_truncation(self):
        """top_k strategy should truncate to fit within budget."""
        trimmer = ContextTrimmer(max_tokens=50, trim_strategy="top_k")
        items = [
            _make_item(title="Short"),
            _make_item(title="A" * 200, description="B" * 200),  # Long
        ]
        result = await trimmer.trim(items)
        # First item should fit, second might not
        assert len(result) >= 1
        assert len(result) <= 2

    @pytest.mark.asyncio
    async def test_always_keeps_one(self):
        """Should always keep at least one result even if over budget."""
        trimmer = ContextTrimmer(max_tokens=1, trim_strategy="top_k")
        items = [_make_item(title="This is a long title " * 10)]
        result = await trimmer.trim(items)
        assert len(result) == 1

    @pytest.mark.asyncio
    async def test_empty_items(self):
        """Empty input should return empty output."""
        trimmer = ContextTrimmer(max_tokens=100, trim_strategy="top_k")
        result = await trimmer.trim([])
        assert result == []

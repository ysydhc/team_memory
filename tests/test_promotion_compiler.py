"""Tests for PromotionCompiler — compile L2 Experiences into structured Markdown.

Covers:
- Single Experience compiles to valid Markdown with frontmatter
- Single compile frontmatter contains promoted_from
- Multiple Experiences compile to a merged Markdown
- Multiple compile merges all tags (union, deduped)
- Multiple compile merges all ids into promoted_from
- Empty list raises ValueError
- Compile result contains "经验来源" section
- Frontmatter contains date field
"""

from __future__ import annotations

import re

import pytest

from team_memory.services.promotion_compiler import PromotionCompiler

# ============================================================
# Fixtures
# ============================================================


def _make_experience(
    exp_id: str = "exp-001",
    title: str = "Test Experience",
    description: str = "Some problem description",
    solution: str = "Some solution",
    tags: list[str] | None = None,
    created_at: str = "2026-04-20T10:00:00",
) -> dict:
    """Build a minimal experience dict for testing."""
    return {
        "id": exp_id,
        "title": title,
        "description": description,
        "solution": solution,
        "tags": tags or [],
        "created_at": created_at,
    }


@pytest.fixture
def compiler() -> PromotionCompiler:
    return PromotionCompiler()


# ============================================================
# Helper
# ============================================================


def _parse_frontmatter(md: str) -> dict:
    """Extract YAML-like frontmatter from a Markdown string."""
    match = re.match(r"^---\n(.*?)\n---", md, re.DOTALL)
    if not match:
        return {}
    fm_text = match.group(1)
    result: dict = {}
    for line in fm_text.splitlines():
        if ":" in line:
            key, _, value = line.partition(":")
            result[key.strip()] = value.strip()
    return result


# ============================================================
# Tests: Single Experience
# ============================================================


class TestCompileSingle:
    """Single-experience compilation tests."""

    @pytest.mark.asyncio
    async def test_single_produces_valid_markdown_with_frontmatter(
        self, compiler: PromotionCompiler
    ) -> None:
        """Single experience compiles to valid Markdown with frontmatter."""
        exp = _make_experience()
        result = await compiler.compile([exp])

        # Must start and end frontmatter
        assert result.startswith("---\n")
        assert "---\n\n## 问题描述" in result

    @pytest.mark.asyncio
    async def test_single_frontmatter_contains_promoted_from(
        self, compiler: PromotionCompiler
    ) -> None:
        """Single compile frontmatter must contain promoted_from."""
        exp = _make_experience(exp_id="abc-123")
        result = await compiler.compile([exp])
        fm = _parse_frontmatter(result)

        assert "promoted_from" in fm
        assert "abc-123" in fm["promoted_from"]

    @pytest.mark.asyncio
    async def test_single_uses_experience_title(self, compiler: PromotionCompiler) -> None:
        """Title in frontmatter comes from the experience."""
        exp = _make_experience(title="My Cool Fix")
        result = await compiler.compile([exp])
        fm = _parse_frontmatter(result)

        assert fm["title"] == "My Cool Fix"

    @pytest.mark.asyncio
    async def test_single_frontmatter_contains_date(
        self, compiler: PromotionCompiler
    ) -> None:
        """Frontmatter must contain a promoted_at date field."""
        exp = _make_experience()
        result = await compiler.compile([exp])
        fm = _parse_frontmatter(result)

        assert "promoted_at" in fm
        # Should be a date-like string (YYYY-MM-DD)
        assert re.match(r"\d{4}-\d{2}-\d{2}", fm["promoted_at"])

    @pytest.mark.asyncio
    async def test_single_contains_experience_source_section(
        self, compiler: PromotionCompiler
    ) -> None:
        """Compile result must contain '经验来源' section."""
        exp = _make_experience()
        result = await compiler.compile([exp])

        assert "## 经验来源" in result

    @pytest.mark.asyncio
    async def test_single_source_line_contains_date_and_title(
        self, compiler: PromotionCompiler
    ) -> None:
        """The source line should contain the experience date and title."""
        exp = _make_experience(
            title="Fix Memory Leak",
            created_at="2026-03-15T08:30:00",
        )
        result = await compiler.compile([exp])

        assert "2026-03-15" in result
        assert "Fix Memory Leak" in result


# ============================================================
# Tests: Multiple Experiences
# ============================================================


class TestCompileMulti:
    """Multi-experience compilation tests."""

    @pytest.mark.asyncio
    async def test_multi_produces_merged_markdown(
        self, compiler: PromotionCompiler
    ) -> None:
        """Multiple experiences compile to one merged Markdown."""
        exps = [
            _make_experience(
                exp_id="e1",
                title="Fix A",
                description="Problem A",
                solution="Solution A",
            ),
            _make_experience(
                exp_id="e2",
                title="Fix B",
                description="Problem B",
                solution="Solution B",
            ),
        ]
        result = await compiler.compile(exps, group_key="group-1")

        assert result.startswith("---\n")
        assert "## 问题描述" in result
        assert "## 解决方案" in result

    @pytest.mark.asyncio
    async def test_multi_merges_all_tags(self, compiler: PromotionCompiler) -> None:
        """Tags from all experiences should be merged (union)."""
        exps = [
            _make_experience(exp_id="e1", tags=["python", "async"]),
            _make_experience(exp_id="e2", tags=["async", "docker"]),
        ]
        result = await compiler.compile(exps, group_key="g1")
        fm = _parse_frontmatter(result)

        tags_str = fm["tags"]
        # All three unique tags should appear
        assert "python" in tags_str
        assert "async" in tags_str
        assert "docker" in tags_str

    @pytest.mark.asyncio
    async def test_multi_merges_all_ids_to_promoted_from(
        self, compiler: PromotionCompiler
    ) -> None:
        """All experience ids should appear in promoted_from."""
        exps = [
            _make_experience(exp_id="id-alpha"),
            _make_experience(exp_id="id-beta"),
        ]
        result = await compiler.compile(exps, group_key="g1")
        fm = _parse_frontmatter(result)

        assert "id-alpha" in fm["promoted_from"]
        assert "id-beta" in fm["promoted_from"]

    @pytest.mark.asyncio
    async def test_multi_title_indicates_count(self, compiler: PromotionCompiler) -> None:
        """Title should mention the number of experiences."""
        exps = [
            _make_experience(exp_id="e1"),
            _make_experience(exp_id="e2"),
            _make_experience(exp_id="e3"),
        ]
        result = await compiler.compile(exps, group_key="g1")
        fm = _parse_frontmatter(result)

        assert "3" in fm["title"]

    @pytest.mark.asyncio
    async def test_multi_problem_merges_descriptions(
        self, compiler: PromotionCompiler
    ) -> None:
        """Problem section should contain all descriptions."""
        exps = [
            _make_experience(exp_id="e1", description="Problem Alpha"),
            _make_experience(exp_id="e2", description="Problem Beta"),
        ]
        result = await compiler.compile(exps, group_key="g1")

        assert "Problem Alpha" in result
        assert "Problem Beta" in result

    @pytest.mark.asyncio
    async def test_multi_solution_merges_solutions(
        self, compiler: PromotionCompiler
    ) -> None:
        """Solution section should contain all solutions."""
        exps = [
            _make_experience(exp_id="e1", solution="Sol Alpha"),
            _make_experience(exp_id="e2", solution="Sol Beta"),
        ]
        result = await compiler.compile(exps, group_key="g1")

        assert "Sol Alpha" in result
        assert "Sol Beta" in result

    @pytest.mark.asyncio
    async def test_multi_source_lines(self, compiler: PromotionCompiler) -> None:
        """Source section should have one line per experience."""
        exps = [
            _make_experience(
                exp_id="e1",
                title="Fix A",
                created_at="2026-01-10T09:00:00",
            ),
            _make_experience(
                exp_id="e2",
                title="Fix B",
                created_at="2026-02-20T10:00:00",
            ),
        ]
        result = await compiler.compile(exps, group_key="g1")

        assert "2026-01-10" in result
        assert "Fix A" in result
        assert "2026-02-20" in result
        assert "Fix B" in result


# ============================================================
# Tests: Edge Cases
# ============================================================


class TestCompileEdgeCases:
    """Edge-case compilation tests."""

    @pytest.mark.asyncio
    async def test_empty_list_raises(self, compiler: PromotionCompiler) -> None:
        """Empty list should raise ValueError."""
        with pytest.raises(ValueError):
            await compiler.compile([])

    @pytest.mark.asyncio
    async def test_result_contains_source_section(
        self, compiler: PromotionCompiler
    ) -> None:
        """Compile result always contains '经验来源' section."""
        exp = _make_experience()
        result = await compiler.compile([exp])
        assert "## 经验来源" in result

    @pytest.mark.asyncio
    async def test_frontmatter_contains_date_field(
        self, compiler: PromotionCompiler
    ) -> None:
        """Frontmatter always contains promoted_at date field."""
        exp = _make_experience()
        result = await compiler.compile([exp])
        fm = _parse_frontmatter(result)

        assert "promoted_at" in fm

    @pytest.mark.asyncio
    async def test_frontmatter_source_field(self, compiler: PromotionCompiler) -> None:
        """Frontmatter contains source: tm-promotion."""
        exp = _make_experience()
        result = await compiler.compile([exp])
        fm = _parse_frontmatter(result)

        assert fm.get("source") == "tm-promotion"

    @pytest.mark.asyncio
    async def test_no_tags_produces_empty_brackets(
        self, compiler: PromotionCompiler
    ) -> None:
        """Experience with no tags should produce [] in frontmatter."""
        exp = _make_experience(tags=[])
        result = await compiler.compile([exp])
        fm = _parse_frontmatter(result)

        assert fm["tags"] == "[]"

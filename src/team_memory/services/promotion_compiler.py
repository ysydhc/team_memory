"""PromotionCompiler — compile L2 Experiences into structured Markdown.

Takes one or more Experience dicts and produces a single Markdown document
with YAML frontmatter following the PROMOTION_TEMPLATE format.
"""

from __future__ import annotations

from datetime import date
from typing import Any

PROMOTION_TEMPLATE = """\
---
title: {title}
tags: [{tags}]
source: tm-promotion
promoted_from: [{promoted_from}]
promoted_at: {date}
---

## 问题描述
{problem}

## 解决方案
{solution}

## 经验来源
{sources}
"""


class PromotionCompiler:
    """Compile one or more L2 Experience dicts into a structured Markdown document.

    For a single experience, fills the template directly.
    For multiple experiences, merges descriptions/solutions/tags and
    produces a consolidated document (LLM synthesis reserved for future use).
    """

    def __init__(self, llm_config: dict[str, Any] | None = None) -> None:
        self._llm_config = llm_config

    async def compile(
        self,
        experiences: list[dict],
        group_key: str | None = None,
    ) -> str:
        """Compile experiences into a structured Markdown string.

        Args:
            experiences: List of experience dicts. Must contain at least one.
            group_key: Optional grouping key for multi-experience compilation.

        Returns:
            A Markdown string with YAML frontmatter.

        Raises:
            ValueError: If experiences list is empty.
        """
        if not experiences:
            raise ValueError("Cannot compile an empty list of experiences")

        if len(experiences) == 1:
            return self._compile_single(experiences[0])

        return await self._compile_multi(experiences, group_key=group_key or "")

    # ------------------------------------------------------------------
    # Single-experience path
    # ------------------------------------------------------------------

    def _compile_single(self, exp: dict) -> str:
        """Compile a single experience into the promotion template."""
        title = exp["title"]
        tags = exp.get("tags", [])
        promoted_from = exp["id"]
        today = date.today().isoformat()
        problem = exp.get("description", "")
        solution = exp.get("solution", "")
        sources = f"- {exp['created_at'][:10]}: {exp['title']}"

        return PROMOTION_TEMPLATE.format(
            title=title,
            tags=", ".join(tags),
            promoted_from=promoted_from,
            date=today,
            problem=problem,
            solution=solution,
            sources=sources,
        )

    # ------------------------------------------------------------------
    # Multi-experience path
    # ------------------------------------------------------------------

    async def _compile_multi(
        self,
        exps: list[dict],
        group_key: str,
    ) -> str:
        """Compile multiple experiences into a merged promotion document.

        Currently uses simple concatenation. LLM-based synthesis is
        reserved for when llm_config is populated.
        """
        count = len(exps)
        title = f"从{count}条经验提炼的知识"

        # Merge tags as a union (preserving order, deduped)
        seen_tags: set[str] = set()
        merged_tags: list[str] = []
        for exp in exps:
            for tag in exp.get("tags", []):
                if tag not in seen_tags:
                    seen_tags.add(tag)
                    merged_tags.append(tag)

        # Merge all ids
        promoted_from = ", ".join(exp["id"] for exp in exps)

        today = date.today().isoformat()

        # Merge descriptions
        problem_parts: list[str] = []
        for exp in exps:
            desc = exp.get("description", "")
            if desc:
                problem_parts.append(desc)
        problem = "\n\n".join(problem_parts)

        # Merge solutions
        solution_parts: list[str] = []
        for exp in exps:
            sol = exp.get("solution", "")
            if sol:
                solution_parts.append(sol)
        solution = "\n\n".join(solution_parts)

        # Source lines — one per experience
        source_lines: list[str] = []
        for exp in exps:
            line = f"- {exp['created_at'][:10]}: {exp['title']}"
            source_lines.append(line)
        sources = "\n".join(source_lines)

        return PROMOTION_TEMPLATE.format(
            title=title,
            tags=", ".join(merged_tags),
            promoted_from=promoted_from,
            date=today,
            problem=problem,
            solution=solution,
            sources=sources,
        )

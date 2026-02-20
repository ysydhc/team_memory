"""PageIndex-Lite helpers for long-document tree extraction."""

from __future__ import annotations

import re

HEADING_RE = re.compile(r"^(#{1,6})\s+(.+)$")
NUMBERED_RE = re.compile(r"^(\d+(?:\.\d+){0,5})[\)\.\s-]+(.+)$")


class PageIndexLiteBuilder:
    """Build a lightweight tree index from markdown/plain-text content."""

    def __init__(
        self,
        *,
        min_doc_chars: int = 800,
        max_tree_depth: int = 4,
        max_nodes_per_doc: int = 40,
        max_node_chars: int = 1200,
    ):
        self._min_doc_chars = min_doc_chars
        self._max_tree_depth = max_tree_depth
        self._max_nodes_per_doc = max_nodes_per_doc
        self._max_node_chars = max_node_chars

    def is_long_document(self, content: str) -> bool:
        """Return True when content is long enough for tree indexing."""
        return len((content or "").strip()) >= self._min_doc_chars

    def build_nodes(self, content: str) -> list[dict]:
        """Build tree nodes from markdown/plain text."""
        text = (content or "").strip()
        if not text:
            return []

        lines = text.splitlines()
        headings: list[tuple[int, int, str]] = []

        for idx, raw in enumerate(lines):
            line = raw.strip()
            if not line:
                continue
            m = HEADING_RE.match(line)
            if m:
                depth = min(len(m.group(1)), self._max_tree_depth)
                headings.append((idx, depth, m.group(2).strip()))
                continue

            n = NUMBERED_RE.match(line)
            if n:
                depth = min(n.group(1).count(".") + 1, self._max_tree_depth)
                headings.append((idx, depth, n.group(2).strip()))

        if not headings:
            return [
                {
                    "path": "1",
                    "node_title": "全文",
                    "depth": 1,
                    "node_order": 0,
                    "content": self._clip(text),
                    "content_summary": self._summarize(text),
                    "char_count": len(text),
                    "is_leaf": True,
                }
            ]

        counters = [0] * (self._max_tree_depth + 1)
        prev_depth = 1
        nodes: list[dict] = []
        ordered_headings = headings[: self._max_nodes_per_doc]

        for order, (line_no, depth, title) in enumerate(ordered_headings):
            depth = max(1, min(depth, prev_depth + 1, self._max_tree_depth))
            counters[depth] += 1
            for i in range(depth + 1, len(counters)):
                counters[i] = 0
            path = ".".join(str(counters[i]) for i in range(1, depth + 1))
            prev_depth = depth

            next_line = (
                ordered_headings[order + 1][0]
                if order + 1 < len(ordered_headings)
                else len(lines)
            )
            block = "\n".join(lines[line_no + 1 : next_line]).strip()
            clipped = self._clip(block)
            nodes.append(
                {
                    "path": path,
                    "node_title": title or f"Section {order + 1}",
                    "depth": depth,
                    "node_order": order,
                    "content": clipped,
                    "content_summary": self._summarize(block),
                    "char_count": len(block),
                    "is_leaf": True,  # updated in second pass
                }
            )

        # Mark non-leaf nodes
        for idx, node in enumerate(nodes):
            prefix = f"{node['path']}."
            is_leaf = True
            for j in range(idx + 1, len(nodes)):
                if nodes[j]["path"].startswith(prefix):
                    is_leaf = False
                    break
            node["is_leaf"] = is_leaf

        return nodes

    def build_experience_document(
        self,
        *,
        problem: str,
        solution: str | None = None,
        root_cause: str | None = None,
        code_snippets: str | None = None,
    ) -> str:
        """Build a normalized long-document string from experience fields."""
        parts = ["# Problem", problem.strip()]
        if root_cause:
            parts.extend(["## Root Cause", root_cause.strip()])
        if solution:
            parts.extend(["## Solution", solution.strip()])
        if code_snippets:
            parts.extend(["## Code Snippets", code_snippets.strip()])
        return "\n\n".join(p for p in parts if p)

    def _clip(self, text: str) -> str:
        stripped = (text or "").strip()
        if len(stripped) <= self._max_node_chars:
            return stripped
        return f"{stripped[: self._max_node_chars]}..."

    @staticmethod
    def _summarize(text: str, max_chars: int = 220) -> str:
        stripped = re.sub(r"\s+", " ", (text or "").strip())
        if len(stripped) <= max_chars:
            return stripped
        return f"{stripped[:max_chars]}..."

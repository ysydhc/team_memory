"""Context trimmer — manages token budget for search results.

When the LLM has a limited context window, we need to ensure
the returned experiences don't overflow it. Two strategies:

1. top_k: Simply truncate results to fit within max_tokens.
   Fast and predictable, preserves original content.

2. summary: When results exceed max_tokens, use an LLM to
   generate summaries of the overflow portion.
   More expensive but retains more information.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from team_memory.config import LLMConfig
    from team_memory.services.search_pipeline import SearchResultItem

import httpx

logger = logging.getLogger("team_memory.context_trimmer")


def estimate_tokens(text: str) -> int:
    """Estimate token count from text length.

    A rough heuristic: ~4 characters per token for English,
    ~2 characters per token for Chinese. We use a blended estimate.
    This avoids requiring tiktoken as a dependency.
    """
    if not text:
        return 0
    # Count CJK characters
    cjk_count = sum(1 for c in text if "\u4e00" <= c <= "\u9fff")
    non_cjk_count = len(text) - cjk_count
    # CJK chars ≈ 1 token each, non-CJK ≈ 4 chars/token
    return cjk_count + non_cjk_count // 4


def result_to_text(item: "SearchResultItem") -> str:
    """Convert a search result item to a text representation for token counting."""
    data = item.data
    parts = []
    for key in ("title", "description", "solution", "root_cause", "code_snippets"):
        val = data.get(key)
        if val:
            parts.append(str(val))
    # Include children if grouped result
    children = data.get("children", [])
    if children:
        for child in children:
            for key in ("title", "description", "solution"):
                val = child.get(key)
                if val:
                    parts.append(str(val))
    matched_nodes = data.get("matched_nodes", [])
    if matched_nodes:
        for node in matched_nodes:
            title = node.get("node_title")
            summary = node.get("content_summary")
            if title:
                parts.append(str(title))
            if summary:
                parts.append(str(summary))
    return "\n".join(parts)


class ContextTrimmer:
    """Trims search results to fit within a token budget."""

    def __init__(
        self,
        max_tokens: int | None = None,
        trim_strategy: str = "top_k",
        summary_model: str | None = None,
        llm_config: "LLMConfig | None" = None,
    ):
        self._max_tokens = max_tokens
        self._strategy = trim_strategy
        self._summary_model = summary_model
        self._llm_config = llm_config

    async def trim(
        self, items: list["SearchResultItem"]
    ) -> list["SearchResultItem"]:
        """Trim results to fit within the token budget.

        If max_tokens is None, returns all items unchanged.

        Args:
            items: Sorted search results (best first).

        Returns:
            Trimmed list of results fitting within max_tokens.
        """
        if self._max_tokens is None or not items:
            return items

        if self._strategy == "summary":
            return await self._trim_with_summary(items)
        else:
            return self._trim_top_k(items)

    def _trim_top_k(
        self, items: list["SearchResultItem"]
    ) -> list["SearchResultItem"]:
        """Truncate results to fit within max_tokens.

        Keeps the highest-scored results that fit within budget.
        """
        budget = self._max_tokens
        result = []
        used = 0

        for item in items:
            text = result_to_text(item)
            tokens = estimate_tokens(text)

            if used + tokens > budget:
                # This item would exceed budget
                if not result:
                    # Always include at least one result
                    result.append(item)
                break

            result.append(item)
            used += tokens

        logger.debug(
            "top_k trim: %d/%d items, ~%d/%d tokens",
            len(result),
            len(items),
            used,
            budget,
        )
        return result

    async def _trim_with_summary(
        self, items: list["SearchResultItem"]
    ) -> list["SearchResultItem"]:
        """Keep top items that fit, summarize the rest.

        Items that fit within budget are kept as-is.
        Remaining items are summarized into a single "overflow summary"
        appended as a special result.
        """
        budget = self._max_tokens
        # Reserve 20% of budget for the summary
        content_budget = int(budget * 0.8)
        summary_budget = budget - content_budget

        kept = []
        overflow = []
        used = 0

        for item in items:
            text = result_to_text(item)
            tokens = estimate_tokens(text)

            if used + tokens <= content_budget:
                kept.append(item)
                used += tokens
            else:
                overflow.append(item)

        if not overflow:
            return kept

        # Summarize overflow items
        try:
            summary = await self._summarize_overflow(overflow, summary_budget)
            if summary:
                # Create a synthetic result for the summary
                from team_memory.services.search_pipeline import SearchResultItem

                summary_item = SearchResultItem(
                    data={
                        "title": f"其他 {len(overflow)} 条相关经验摘要",
                        "description": summary,
                        "solution": "",
                        "id": "summary",
                        "tags": ["auto-summary"],
                    },
                    score=0.0,
                    confidence="low",
                    source_type="summary",
                )
                kept.append(summary_item)
        except Exception as e:
            logger.warning("Summary generation failed: %s", e)
            # Fallback: just return the kept items without summary

        return kept

    async def _summarize_overflow(
        self,
        items: list["SearchResultItem"],
        max_tokens: int,
    ) -> str | None:
        """Use LLM to summarize overflow items."""
        if not self._llm_config:
            logger.warning("No LLM config for summary strategy, skipping")
            return None

        # Build text from overflow items
        texts = []
        for i, item in enumerate(items[:10]):  # Limit input
            data = item.data
            title = data.get("title", "")
            desc = data.get("description", "")[:200]
            sol = data.get("solution", "")[:200]
            texts.append(f"{i + 1}. {title}: {desc} -> {sol}")

        overflow_text = "\n".join(texts)

        prompt = f"""请将以下多条开发经验摘要合并为一段简洁的总结（不超过{max_tokens // 2}字）：

{overflow_text}

请用中文输出摘要，重点保留关键的问题描述和解决方案。"""

        model = self._summary_model or self._llm_config.model
        base_url = self._llm_config.base_url.rstrip("/")

        # Try OpenAI-compatible endpoint
        url = f"{base_url}/v1/chat/completions"
        payload = {
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.3,
            "max_tokens": max_tokens,
            "stream": False,
        }

        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                resp = await client.post(url, json=payload)
                if resp.status_code == 200:
                    data = resp.json()
                    return data["choices"][0]["message"]["content"]
        except Exception:
            pass

        # Fallback: Ollama native
        url = f"{base_url}/api/chat"
        payload = {
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "stream": False,
            "options": {"temperature": 0.3},
        }

        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                resp = await client.post(url, json=payload)
                if resp.status_code == 200:
                    data = resp.json()
                    return data.get("message", {}).get("content", "")
        except Exception as e:
            logger.warning("Summary LLM call failed: %s", e)

        return None

"""Ollama LLM reranker provider — uses a chat/completion API for relevance scoring.

This provider sends query-document pairs to an LLM via the OpenAI-compatible
chat API (supported by Ollama, vLLM, LocalAI, etc.) and asks it to rate
relevance on a 0-10 scale. Documents are scored in batches for efficiency.

Usage:
    Set reranker.provider = "ollama_llm" in config.yaml.
    The provider reuses llm.model and llm.base_url by default,
    so no extra model download is needed if Ollama is already running.
"""

from __future__ import annotations

import json
import logging
import re

import httpx

from team_memory.reranker.base import RerankerProvider, RerankResult

logger = logging.getLogger("team_memory.reranker.ollama_llm")

DEFAULT_PROMPT_TEMPLATE = """你是一个文档相关性评分专家。请对以下文档与查询的相关性进行打分。

查询: {query}

请对每个文档打分，分数范围 0-10（10 表示完全相关，0 表示完全无关）。
只返回 JSON 数组，格式为: [{{"index": 0, "score": 8}}, {{"index": 1, "score": 3}}]
不要返回任何其他内容。

文档列表:
{documents}"""


class OllamaLLMRerankerProvider(RerankerProvider):
    """LLM-based reranker using OpenAI-compatible chat API.

    Works with Ollama, vLLM, LocalAI, or any service exposing
    the /v1/chat/completions or /api/chat endpoint.
    """

    def __init__(
        self,
        model: str,
        base_url: str,
        top_k: int = 10,
        batch_size: int = 5,
        prompt_template: str | None = None,
    ):
        self._model = model
        self._base_url = base_url.rstrip("/")
        self._top_k = top_k
        self._batch_size = batch_size
        self._prompt_template = prompt_template or DEFAULT_PROMPT_TEMPLATE

    async def rank(
        self,
        query: str,
        documents: list[str],
        top_k: int | None = None,
    ) -> list[RerankResult]:
        """Rerank documents using LLM relevance scoring."""
        effective_top_k = top_k or self._top_k
        if not documents:
            return []

        # Score all documents in batches
        all_scores: list[tuple[int, float]] = []  # (original_index, score)

        for batch_start in range(0, len(documents), self._batch_size):
            batch_end = min(batch_start + self._batch_size, len(documents))
            batch_docs = documents[batch_start:batch_end]
            batch_indices = list(range(batch_start, batch_end))

            try:
                scores = await self._score_batch(query, batch_docs)
                for local_idx, score in enumerate(scores):
                    global_idx = batch_indices[local_idx]
                    all_scores.append((global_idx, score))
            except Exception as e:
                logger.warning(
                    "LLM reranking failed for batch %d-%d: %s",
                    batch_start,
                    batch_end,
                    str(e),
                )
                # Fallback: assign decreasing scores for this batch
                for local_idx in range(len(batch_docs)):
                    global_idx = batch_indices[local_idx]
                    fallback_score = 5.0 - local_idx * 0.1
                    all_scores.append((global_idx, fallback_score))

        # Sort by score descending and take top_k
        all_scores.sort(key=lambda x: x[1], reverse=True)
        results = []
        for orig_idx, score in all_scores[:effective_top_k]:
            results.append(
                RerankResult(
                    index=orig_idx,
                    score=score,
                    text=documents[orig_idx],
                )
            )
        return results

    async def _score_batch(
        self, query: str, documents: list[str]
    ) -> list[float]:
        """Score a batch of documents using LLM.

        Returns a list of scores (0-10) in the same order as the input docs.
        """
        # Build documents section
        doc_lines = []
        for i, doc in enumerate(documents):
            # Truncate very long documents to avoid token overflow
            truncated = doc[:1500] if len(doc) > 1500 else doc
            doc_lines.append(f"[文档 {i}]: {truncated}")
        documents_text = "\n\n".join(doc_lines)

        prompt = self._prompt_template.format(
            query=query, documents=documents_text
        )

        # Try OpenAI-compatible endpoint first, then Ollama native
        scores = await self._call_openai_compatible(prompt)
        if scores is None:
            scores = await self._call_ollama_native(prompt)
        if scores is None:
            # Final fallback: uniform scores
            return [5.0] * len(documents)

        # Ensure we have scores for all documents
        while len(scores) < len(documents):
            scores.append(5.0)

        return scores[: len(documents)]

    async def _call_openai_compatible(
        self, prompt: str
    ) -> list[float] | None:
        """Call OpenAI-compatible /v1/chat/completions endpoint."""
        url = f"{self._base_url}/v1/chat/completions"
        payload = {
            "model": self._model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.0,
            "stream": False,
        }

        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                resp = await client.post(url, json=payload)
                if resp.status_code != 200:
                    return None
                data = resp.json()
                content = data["choices"][0]["message"]["content"]
                return self._parse_scores(content)
        except Exception as e:
            logger.debug("OpenAI-compatible endpoint failed: %s", e)
            return None

    async def _call_ollama_native(self, prompt: str) -> list[float] | None:
        """Call Ollama native /api/chat endpoint as fallback."""
        url = f"{self._base_url}/api/chat"
        payload = {
            "model": self._model,
            "messages": [{"role": "user", "content": prompt}],
            "stream": False,
            "options": {"temperature": 0.0},
        }

        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                resp = await client.post(url, json=payload)
                if resp.status_code != 200:
                    return None
                data = resp.json()
                content = data.get("message", {}).get("content", "")
                return self._parse_scores(content)
        except Exception as e:
            logger.debug("Ollama native endpoint failed: %s", e)
            return None

    @staticmethod
    def _parse_scores(content: str) -> list[float] | None:
        """Parse LLM response to extract scores.

        Expected format: [{"index": 0, "score": 8}, {"index": 1, "score": 3}]
        Also handles variations like markdown code blocks.
        """
        # Strip markdown code block if present
        content = content.strip()
        if content.startswith("```"):
            lines = content.split("\n")
            # Remove first and last lines (```json and ```)
            content = "\n".join(
                line
                for line in lines
                if not line.strip().startswith("```")
            )
            content = content.strip()

        try:
            parsed = json.loads(content)
            if isinstance(parsed, list):
                # Sort by index to maintain original order
                scored = {}
                for item in parsed:
                    idx = item.get("index", 0)
                    score = float(item.get("score", 5.0))
                    # Clamp score to 0-10
                    score = max(0.0, min(10.0, score))
                    scored[idx] = score
                # Build ordered list
                max_idx = max(scored.keys()) if scored else 0
                return [scored.get(i, 5.0) for i in range(max_idx + 1)]
        except (json.JSONDecodeError, ValueError, TypeError):
            pass

        # Fallback: try to extract numbers with regex
        # Match patterns like "score": 8 or score: 8
        matches = re.findall(r'"?score"?\s*[:=]\s*(\d+(?:\.\d+)?)', content)
        if matches:
            return [max(0.0, min(10.0, float(s))) for s in matches]

        logger.warning("Failed to parse LLM reranking response: %s", content[:200])
        return None

    @property
    def provider_name(self) -> str:
        return f"ollama_llm({self._model})"

"""Jina Reranker API provider — uses Jina's cloud reranking service.

Jina offers a cost-effective reranking API ($0.02/1M tokens) that
supports multilingual documents. This provider sends documents to
the Jina API and returns scored results.

Usage:
    Set reranker.provider = "jina" in config.yaml.
    Set reranker.jina.api_key to your Jina API key.
"""

from __future__ import annotations

import logging

import httpx

from team_memory.reranker.base import RerankerProvider, RerankResult

logger = logging.getLogger("team_memory.reranker.jina")

JINA_RERANK_URL = "https://api.jina.ai/v1/rerank"


class JinaRerankerProvider(RerankerProvider):
    """Jina Reranker API provider."""

    def __init__(
        self,
        api_key: str,
        model: str = "jina-reranker-v2-base-multilingual",
        top_k: int = 10,
    ):
        if not api_key:
            raise ValueError(
                "Jina API key is required. Set reranker.jina.api_key in config.yaml"
            )
        self._api_key = api_key
        self._model = model
        self._top_k = top_k

    async def rank(
        self,
        query: str,
        documents: list[str],
        top_k: int | None = None,
    ) -> list[RerankResult]:
        """Rerank documents using Jina Reranker API."""
        effective_top_k = top_k or self._top_k
        if not documents:
            return []

        payload = {
            "model": self._model,
            "query": query,
            "documents": documents,
            "top_n": effective_top_k,
        }

        headers = {
            "Authorization": f"Bearer {self._api_key}",
            "Content-Type": "application/json",
        }

        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                resp = await client.post(
                    JINA_RERANK_URL,
                    json=payload,
                    headers=headers,
                )
                resp.raise_for_status()
                data = resp.json()

            results = []
            for item in data.get("results", []):
                idx = item.get("index", 0)
                score = float(item.get("relevance_score", 0.0))
                text = documents[idx] if idx < len(documents) else ""
                results.append(
                    RerankResult(index=idx, score=score, text=text)
                )

            # Already sorted by Jina API, but ensure it
            results.sort(key=lambda x: x.score, reverse=True)
            return results[:effective_top_k]

        except httpx.HTTPStatusError as e:
            logger.error("Jina API error: %s — %s", e.response.status_code, e.response.text)
            raise
        except Exception as e:
            logger.error("Jina reranking failed: %s", e)
            # Fallback: return in original order
            return [
                RerankResult(index=i, score=1.0 - i * 0.1, text=doc)
                for i, doc in enumerate(documents[:effective_top_k])
            ]

    @property
    def provider_name(self) -> str:
        return f"jina({self._model})"

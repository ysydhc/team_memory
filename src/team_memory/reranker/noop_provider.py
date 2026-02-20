"""No-op reranker provider — passthrough that preserves original ordering.

Used when reranker.provider = "none". Does not require any external
model or API. Simply returns documents in their original order with
scores mapped from their position (higher position = higher score).
"""

from __future__ import annotations

from team_memory.reranker.base import RerankerProvider, RerankResult


class NoopRerankerProvider(RerankerProvider):
    """Passthrough reranker that preserves original ordering.

    This is the default provider when no server-side reranker is configured.
    It assigns monotonically decreasing scores based on position so that
    downstream confidence labeling can still work consistently.
    """

    async def rank(
        self,
        query: str,
        documents: list[str],
        top_k: int = 10,
    ) -> list[RerankResult]:
        """Return documents in original order with position-based scores."""
        results = []
        n = len(documents)
        for i, doc in enumerate(documents[:top_k]):
            # Score from 1.0 (first) to ~0.5 (last), linearly decreasing
            score = 1.0 - (i / max(n, 1)) * 0.5
            results.append(RerankResult(index=i, score=score, text=doc))
        return results

    @property
    def provider_name(self) -> str:
        return "noop"

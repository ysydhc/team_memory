"""Abstract base class for reranker providers.

All reranker implementations must inherit from RerankerProvider
and implement the `rank` method. This design allows plugging in
any reranking backend (LLM-based, cross-encoder, API, etc.)
without changing the search pipeline code.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass


@dataclass
class RerankResult:
    """A single reranked document result.

    Attributes:
        index: Original index of the document in the input list.
        score: Relevance score assigned by the reranker (higher = more relevant).
        text: The document text that was scored.
    """

    index: int
    score: float
    text: str


class RerankerProvider(ABC):
    """Abstract reranker provider interface.

    All reranker providers must implement this interface. To add a custom
    reranker, create a new class that inherits from RerankerProvider and
    implement the `rank` method.

    Example:
        class MyCustomReranker(RerankerProvider):
            async def rank(self, query, documents, top_k=10):
                # Your custom reranking logic here
                ...
    """

    @abstractmethod
    async def rank(
        self,
        query: str,
        documents: list[str],
        top_k: int = 10,
    ) -> list[RerankResult]:
        """Rerank documents by relevance to the query.

        Args:
            query: The search query string.
            documents: List of document texts to rerank.
            top_k: Maximum number of results to return.

        Returns:
            List of RerankResult sorted by score descending,
            limited to top_k entries.
        """
        ...

    @property
    def provider_name(self) -> str:
        """Return the name of this reranker provider."""
        return self.__class__.__name__

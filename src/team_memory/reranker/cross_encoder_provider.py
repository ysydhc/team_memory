"""Cross-encoder reranker provider — uses sentence-transformers for scoring.

This provider loads a cross-encoder model (e.g. ms-marco-MiniLM-L6-v2)
locally and scores query-document pairs directly. Cross-encoders are
more accurate than bi-encoders for ranking because they see both texts
at once, but they are slower (can't pre-compute embeddings).

Requires: pip install sentence-transformers

Usage:
    Set reranker.provider = "cross_encoder" in config.yaml.
"""

from __future__ import annotations

import logging

from team_memory.reranker.base import RerankerProvider, RerankResult

logger = logging.getLogger("team_memory.reranker.cross_encoder")


class CrossEncoderRerankerProvider(RerankerProvider):
    """Cross-encoder reranker using sentence-transformers.

    Loads the model lazily on first use to avoid slow startup.
    """

    def __init__(
        self,
        model_name: str = "cross-encoder/ms-marco-MiniLM-L6-v2",
        device: str = "cpu",
        top_k: int = 10,
    ):
        self._model_name = model_name
        self._device = device
        self._top_k = top_k
        self._model = None  # Lazy loading

    def _load_model(self):
        """Lazy-load the cross-encoder model."""
        if self._model is not None:
            return

        try:
            from sentence_transformers import CrossEncoder

            logger.info(
                "Loading cross-encoder model: %s (device=%s)",
                self._model_name,
                self._device,
            )
            self._model = CrossEncoder(
                self._model_name, device=self._device
            )
            logger.info("Cross-encoder model loaded successfully")
        except ImportError:
            raise RuntimeError(
                "sentence-transformers is required for cross-encoder reranker. "
                "Install it with: pip install sentence-transformers"
            )

    async def rank(
        self,
        query: str,
        documents: list[str],
        top_k: int | None = None,
    ) -> list[RerankResult]:
        """Rerank documents using cross-encoder scoring."""
        effective_top_k = top_k or self._top_k
        if not documents:
            return []

        self._load_model()

        # Create query-document pairs for cross-encoder
        pairs = [(query, doc) for doc in documents]

        try:
            # Cross-encoder predicts relevance scores for all pairs
            scores = self._model.predict(pairs)

            # Combine with original indices
            scored = [
                (i, float(score), doc)
                for i, (score, doc) in enumerate(zip(scores, documents))
            ]

            # Sort by score descending
            scored.sort(key=lambda x: x[1], reverse=True)

            # Return top_k results
            results = []
            for idx, score, text in scored[:effective_top_k]:
                results.append(
                    RerankResult(index=idx, score=score, text=text)
                )
            return results

        except Exception as e:
            logger.error("Cross-encoder scoring failed: %s", e)
            # Fallback: return in original order
            return [
                RerankResult(index=i, score=1.0 - i * 0.1, text=doc)
                for i, doc in enumerate(documents[:effective_top_k])
            ]

    @property
    def provider_name(self) -> str:
        return f"cross_encoder({self._model_name})"

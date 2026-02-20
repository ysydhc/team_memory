"""Local embedding model provider (Phase 2).

Uses sentence-transformers with BAAI/bge-m3 for local inference.
Requires: pip install team_memory[local-embedding]
"""

from __future__ import annotations

from team_memory.embedding.base import EmbeddingProvider


class LocalEmbedding(EmbeddingProvider):
    """Local embedding using sentence-transformers.

    Requires the `local-embedding` extra to be installed:
        pip install team_memory[local-embedding]
    """

    def __init__(
        self,
        model_name: str = "BAAI/bge-m3",
        device: str = "cpu",
        dim: int = 1024,
    ):
        self._model_name = model_name
        self._device = device
        self._dimension = dim
        self._model = None

    def _load_model(self):
        """Lazy-load the model on first use."""
        if self._model is None:
            try:
                from sentence_transformers import SentenceTransformer
            except ImportError:
                raise ImportError(
                    "sentence-transformers is required for local embedding. "
                    "Install it with: pip install team_memory[local-embedding]"
                )
            self._model = SentenceTransformer(self._model_name, device=self._device)

    @property
    def dimension(self) -> int:
        return self._dimension

    async def encode(self, texts: list[str]) -> list[list[float]]:
        """Encode texts using the local model.

        Note: sentence-transformers is synchronous, so this runs in
        the default executor to avoid blocking the event loop.
        """
        import asyncio

        self._load_model()

        loop = asyncio.get_event_loop()
        embeddings = await loop.run_in_executor(
            None,
            lambda: self._model.encode(texts, normalize_embeddings=True).tolist(),
        )
        return embeddings

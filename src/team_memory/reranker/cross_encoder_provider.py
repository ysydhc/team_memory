"""Cross-encoder reranker provider — uses sentence-transformers for scoring.

This provider loads a cross-encoder model (e.g. ms-marco-MiniLM-L6-v2 or
BAAI/bge-reranker-v2-m3) locally and scores query-document pairs directly.

Features:
  - Automatic Apple Silicon MPS detection with graceful CPU fallback.
  - Process-wide singleton lazy loading (keyed by model_name + resolved_device)
    to avoid repeated model initialization across pipelines/requests.
  - Strict inference timeout protection & graceful degradation: if dependencies
    are missing, model loading fails, or inference exceeds ``timeout_seconds``,
    logs a warning and passes through the original document order without
    blocking the main search pipeline.

Requires: pip install sentence-transformers

Usage:
    Set reranker.provider = "cross_encoder" in config.yaml.
"""

from __future__ import annotations

import asyncio
import logging
import threading
import time
from typing import Any

from team_memory.reranker.base import RerankerProvider, RerankResult

logger = logging.getLogger("team_memory.reranker.cross_encoder")

_MODEL_CACHE: dict[tuple[str, str], Any] = {}
_MODEL_CACHE_LOCK = threading.Lock()


def clear_model_cache() -> None:
    """Clear the process-wide cross-encoder model singleton cache (for testing)."""
    with _MODEL_CACHE_LOCK:
        _MODEL_CACHE.clear()


def resolve_device(requested: str = "auto") -> str:
    """Resolve inference device, preferring Apple Silicon MPS over CPU.

    Args:
        requested: ``"auto"``, ``"mps"``, ``"cuda"``, or ``"cpu"``.

    Returns:
        Resolved torch device string (``"mps"``, ``"cuda"``, or ``"cpu"``).
    """
    req = (requested or "auto").strip().lower()
    if req == "cpu":
        return "cpu"

    try:
        import torch
    except Exception:
        if req != "auto":
            logger.warning(
                "Requested cross-encoder device '%s' but torch is unavailable; falling back to cpu",
                req,
            )
        return "cpu"

    mps_backend = getattr(getattr(torch, "backends", None), "mps", None)
    mps_available = bool(mps_backend is not None and mps_backend.is_available())
    cuda_mod = getattr(torch, "cuda", None)
    cuda_available = bool(cuda_mod is not None and cuda_mod.is_available())

    if req == "auto":
        if mps_available:
            return "mps"
        if cuda_available:
            return "cuda"
        return "cpu"

    if req == "mps":
        if mps_available:
            return "mps"
        logger.warning("Requested cross-encoder device 'mps' is unavailable; falling back to cpu")
        return "cpu"

    if req == "cuda" or req.startswith("cuda:"):
        if cuda_available:
            return req
        logger.warning("Requested cross-encoder device '%s' is unavailable; falling back to cpu", req)
        return "cpu"

    logger.warning("Unknown cross-encoder device '%s'; falling back to cpu", req)
    return "cpu"


class CrossEncoderRerankerProvider(RerankerProvider):
    """Cross-encoder reranker using sentence-transformers.

    Uses process-wide singleton lazy loading, Apple Silicon MPS auto-detection,
    and strict timeout protection with pass-through fallback.
    """

    def __init__(
        self,
        model_name: str = "cross-encoder/ms-marco-MiniLM-L6-v2",
        device: str = "auto",
        top_k: int = 10,
        timeout_seconds: float = 3.0,
    ):
        self._model_name = model_name
        self._requested_device = device
        self._device = resolve_device(device)
        self._top_k = top_k
        self._timeout_seconds = max(0.01, float(timeout_seconds))
        self._model: Any = None  # Lazy singleton handle

    @property
    def device(self) -> str:
        """Return the resolved inference device ('mps', 'cuda', or 'cpu')."""
        return self._device

    @property
    def timeout_seconds(self) -> float:
        """Return the inference timeout budget in seconds."""
        return self._timeout_seconds

    def _load_model(self) -> Any:
        """Lazy-load the cross-encoder model using a process-wide singleton cache."""
        if self._model is not None:
            return self._model

        cache_key = (self._model_name, self._device)
        cached = _MODEL_CACHE.get(cache_key)
        if cached is not None:
            self._model = cached
            return self._model

        with _MODEL_CACHE_LOCK:
            cached = _MODEL_CACHE.get(cache_key)
            if cached is not None:
                self._model = cached
                return self._model

            try:
                from sentence_transformers import CrossEncoder
            except ImportError as exc:
                raise ImportError(
                    "sentence-transformers is required for cross-encoder reranker. "
                    "Install it with: pip install sentence-transformers"
                ) from exc

            logger.info(
                "Loading cross-encoder model: %s (device=%s, requested=%s)",
                self._model_name,
                self._device,
                self._requested_device,
            )
            model = CrossEncoder(self._model_name, device=self._device)
            _MODEL_CACHE[cache_key] = model
            self._model = model
            logger.info("Cross-encoder model loaded successfully (device=%s)", self._device)
            return self._model

    @staticmethod
    def _fallback_results(
        documents: list[str],
        effective_top_k: int,
    ) -> list[RerankResult]:
        """Pass through documents in their original order with descending scores."""
        return [
            RerankResult(
                index=i,
                score=round(1.0 / (1.0 + i * 0.05), 4),
                text=doc,
            )
            for i, doc in enumerate(documents[:effective_top_k])
        ]

    async def rank(
        self,
        query: str,
        documents: list[str],
        top_k: int | None = None,
        document_metadata: list[dict] | None = None,
    ) -> list[RerankResult]:
        """Rerank documents using cross-encoder scoring with timeout & graceful fallback."""
        effective_top_k = top_k or self._top_k
        if not documents:
            return []

        deadline = time.monotonic() + self._timeout_seconds

        try:
            model = self._model
            if model is None:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise TimeoutError("Cross-encoder budget exhausted before model load")
                model = await asyncio.wait_for(
                    asyncio.to_thread(self._load_model),
                    timeout=remaining,
                )

            pairs = [(query, doc) for doc in documents]
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise TimeoutError("Cross-encoder budget exhausted before inference")

            scores = await asyncio.wait_for(
                asyncio.to_thread(model.predict, pairs),
                timeout=remaining,
            )

            meta = document_metadata or []
            exact_bonus = {"exact": 0.5, "contains": 0.2}

            scored: list[tuple[int, float, str]] = []
            for i, (raw_score, doc) in enumerate(zip(scores, documents)):
                score = float(raw_score)
                if i < len(meta):
                    em = meta[i].get("exact_title_match", "")
                    score += exact_bonus.get(em, 0.0)
                scored.append((i, score, doc))

            scored.sort(key=lambda x: x[1], reverse=True)
            return [
                RerankResult(index=idx, score=score, text=text)
                for idx, score, text in scored[:effective_top_k]
            ]

        except (TimeoutError, asyncio.TimeoutError):
            logger.warning(
                "Cross-encoder reranker timed out after %.2fs (model=%s, device=%s, docs=%d); "
                "passing through original order",
                self._timeout_seconds,
                self._model_name,
                self._device,
                len(documents),
            )
            return self._fallback_results(documents, effective_top_k)
        except ImportError as exc:
            logger.warning(
                "Cross-encoder reranker dependency missing (%s); passing through original order",
                exc,
            )
            return self._fallback_results(documents, effective_top_k)
        except Exception as exc:
            logger.warning(
                "Cross-encoder reranker failed (%s: %s); passing through original order",
                type(exc).__name__,
                exc,
            )
            return self._fallback_results(documents, effective_top_k)

    @property
    def provider_name(self) -> str:
        return f"cross_encoder({self._model_name}, device={self._device})"

"""Ranking, boost, adaptive filtering, CRAG verification, and reranking stage for SearchPipeline."""

from __future__ import annotations

import asyncio
import logging
import os

from team_memory.services.search.search_pipeline.models import SearchRequest, SearchResultItem

logger = logging.getLogger("team_memory.search_pipeline")


class SearchRankingMixin:
    """Mixin implementing exact/popularity boost, adaptive filter, CRAG verifier, rerank, and confidence."""

    def _apply_exact_match_boost(self, candidates: list[SearchResultItem], query: str) -> list[SearchResultItem]:
        """Boost score when query exactly matches or is contained in title."""
        from team_memory.services.search.scoring import apply_exact_match_boost

        return apply_exact_match_boost(candidates, query)

    def _apply_popularity_boost(self, candidates: list[SearchResultItem]) -> list[SearchResultItem]:
        """Boost score based on recall_count and used_count with time decay."""
        from team_memory.services.search.scoring import apply_popularity_boost

        cfg = self._search_config
        return apply_popularity_boost(
            candidates,
            popularity_recall_weight=cfg.popularity_recall_weight,
            popularity_used_weight=cfg.popularity_used_weight,
            popularity_decay_lambda=cfg.popularity_decay_lambda,
            popularity_age_floor_ratio=cfg.popularity_age_floor_ratio,
            popularity_scale=cfg.popularity_scale,
        )

    def _apply_adaptive_filter(self, candidates: list[SearchResultItem]) -> list[SearchResultItem]:
        """Apply adaptive score filtering using similarity scores."""
        from team_memory.services.search.scoring import apply_adaptive_filter

        return apply_adaptive_filter(
            candidates,
            min_confidence_ratio=self._search_config.min_confidence_ratio,
            score_gap_threshold=self._search_config.score_gap_threshold,
            adaptive_min_keep=getattr(self._search_config, "adaptive_min_keep", 3),
        )

    async def _crag_verify_candidates(
        self,
        query: str,
        candidates: list[SearchResultItem],
        *,
        timeout_seconds: float = 1.5,
    ) -> list[SearchResultItem]:
        """Optional lightweight CRAG verifier for borderline low-confidence hits.

        Filters out clearly irrelevant candidates within a strict timeout; degrades
        safely to returning the original candidates on timeout or any error.
        """
        if not candidates or self._llm_config is None:
            return candidates

        import json

        import httpx

        top_slice = candidates[:5]
        base_url = getattr(self._llm_config, "base_url", "https://gw.gburouter.com/v1").rstrip("/")
        if not base_url.endswith("/v1"):
            base_url = f"{base_url}/v1"
        model = getattr(self._llm_config, "model", "gbu-gemini-3.8-flash")
        api_key_env = getattr(self._llm_config, "api_key_env", "GATEWAY_API_KEY")
        api_key = os.environ.get(api_key_env, "") or getattr(self._llm_config, "api_key", "") or "none"

        items_payload = [
            {
                "idx": idx,
                "title": (c.data.get("title") or "")[:160],
                "snippet": (c.data.get("solution") or c.data.get("description") or "")[:240],
            }
            for idx, c in enumerate(top_slice)
        ]
        prompt = (
            "Given a search query and candidate memory snippets, return ONLY a JSON array "
            "of integer indices `idx` that are genuinely relevant to the query (e.g. [0, 2])."
        )
        payload = {
            "model": model,
            "messages": [
                {"role": "system", "content": prompt},
                {
                    "role": "user",
                    "content": json.dumps({"query": query[:300], "candidates": items_payload}, ensure_ascii=False),
                },
            ],
            "temperature": 0.0,
            "max_tokens": 128,
        }
        headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}

        async def _call() -> list[int]:
            async with httpx.AsyncClient(timeout=timeout_seconds) as client:
                resp = await client.post(f"{base_url}/chat/completions", json=payload, headers=headers)
                resp.raise_for_status()
                raw = resp.json()["choices"][0]["message"]["content"].strip()
                if "```" in raw:
                    raw = raw.split("```")[1]
                    if raw.startswith("json"):
                        raw = raw[4:]
                    raw = raw.strip()
                parsed = json.loads(raw)
                if not isinstance(parsed, list):
                    raise ValueError("CRAG verifier returned non-list")
                return [int(i) for i in parsed if isinstance(i, int) and 0 <= i < len(top_slice)]

        try:
            keep_indices = await asyncio.wait_for(_call(), timeout=timeout_seconds)
            keep_set = set(keep_indices)
            filtered = [top_slice[i] for i in range(len(top_slice)) if i in keep_set]
            # Never silently erase all results on a borderline query; keep top-1 tagged low_confidence
            return filtered if filtered else candidates
        except Exception as exc:
            logger.debug("CRAG lightweight verifier skipped/timed out (%s); keeping candidates", exc)
            return candidates

    def _build_rerank_document_text(self, item: SearchResultItem) -> str:
        """Concatenate title, problem (description), solution for reranker input."""
        d = item.data
        title = str(d.get("title") or "").strip()
        problem = str(d.get("description") or d.get("problem") or "").strip()
        solution = str(d.get("solution") or "").strip()
        parts = [p for p in (title, problem, solution) if p]
        text = "\n\n".join(parts) if parts else title
        max_c = self._rerank_max_document_chars
        if len(text) > max_c:
            text = text[:max_c] + "…"
        return text or "(empty)"

    async def _apply_rerank(
        self,
        request: SearchRequest,
        candidates: list[SearchResultItem],
    ) -> tuple[list[SearchResultItem], bool]:
        """Server-side rerank on experience candidates only. Returns (items, success)."""
        if not self._reranker or not candidates:
            return candidates, False

        query = (request.query or "").strip()
        documents = [self._build_rerank_document_text(c) for c in candidates]
        doc_meta: list[dict] = [{"exact_title_match": (c.data.get("exact_title_match") or "")} for c in candidates]
        top_k = len(documents)

        try:
            ranked = await self._reranker.rank(
                query,
                documents,
                top_k=top_k,
                document_metadata=doc_meta,
            )
        except Exception as e:
            logger.warning("Reranker failed, keeping pre-rerank order: %s", e, exc_info=True)
            return candidates, False

        if not ranked:
            return candidates, False

        seen: set[int] = set()
        new_items: list[SearchResultItem] = []
        for r in ranked:
            if r.index < 0 or r.index >= len(candidates) or r.index in seen:
                continue
            seen.add(r.index)
            item = candidates[r.index]
            item.score = float(r.score)
            item.data["rerank_score"] = round(float(r.score), 4)
            new_items.append(item)

        if len(new_items) != len(candidates):
            logger.warning(
                "Reranker returned incomplete index set (%d/%d); merging remainder",
                len(new_items),
                len(candidates),
            )
            for i, c in enumerate(candidates):
                if i not in seen:
                    new_items.append(c)

        return new_items, True

    @staticmethod
    def _label_confidence(
        candidates: list[SearchResultItem],
    ) -> list[SearchResultItem]:
        """Label each result with a confidence level based on its score."""
        from team_memory.services.search.scoring import label_confidence

        return label_confidence(candidates)

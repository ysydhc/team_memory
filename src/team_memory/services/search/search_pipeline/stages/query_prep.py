"""Query preparation, expansion, and candidate merging stage for SearchPipeline."""

from __future__ import annotations

import asyncio
import logging
import time
from typing import Any

from sqlalchemy.ext.asyncio import AsyncSession

from team_memory.services.search.search_pipeline.models import SearchRequest, SearchResultItem

logger = logging.getLogger("team_memory.search_pipeline")


async def _llm_expand_query(
    query: str,
    llm_config,
    timeout_seconds: float,
) -> str | None:
    """Call LLM to expand search query with extra keywords. Returns None on any failure.

    Delegates to :func:`team_memory.services.search.query_expansion.llm_expand_query`.
    """
    from team_memory.services.search.query_expansion import llm_expand_query

    return await llm_expand_query(query, llm_config, timeout_seconds)


class SearchQueryPrepMixin:
    """Mixin for query expansion, parallel graph/embedding/FTS prefetch, and candidate merge."""

    def _expand_query_synonyms(self, query: str) -> str:
        """Expand query with tag_synonyms (key<->value) for better recall."""
        from team_memory.services.search.query_expansion import expand_query_synonyms

        return expand_query_synonyms(query, self._tag_synonyms)

    def _needs_fallback_expansion(self, candidates: list[SearchResultItem], mode: str) -> bool:
        """Determine whether first-pass retrieval was too weak and needs LLM query expansion."""
        min_results = int(getattr(self._search_config, "query_expansion_fallback_min_results", 3))
        if len(candidates) < min_results:
            return True
        if mode == "fts":
            return float(candidates[0].score or 0.0) <= 0.0
        min_sim = float(getattr(self._search_config, "query_expansion_fallback_min_similarity", 0.65))
        top_sim = max((float(c.similarity or 0.0) for c in candidates), default=0.0)
        return top_sim < min_sim

    @staticmethod
    def _merge_candidates(
        primary: list[SearchResultItem],
        secondary: list[SearchResultItem],
    ) -> list[SearchResultItem]:
        """Merge and deduplicate two candidate lists, retaining the highest score per experience ID."""
        merged: dict[str, SearchResultItem] = {}
        order: list[str] = []
        for item in primary + secondary:
            cid = str(item.data.get("id") or item.data.get("group_id") or id(item))
            if cid not in merged:
                merged[cid] = item
                order.append(cid)
            elif float(item.score or 0.0) > float(merged[cid].score or 0.0):
                merged[cid] = item
        out = [merged[cid] for cid in order]
        out.sort(key=lambda x: float(x.score or 0.0), reverse=True)
        return out

    async def _run_parallel_prefetch(
        self,
        session: AsyncSession,
        repo: Any,
        request: SearchRequest,
        retrieval_query: str,
        *,
        enable_graph: bool,
        emb_timeout: float,
        over_fetch: int,
        should_prefetch_fts: bool,
    ) -> tuple[
        tuple[list[str], list[tuple[str, float]], int],
        tuple[list[float] | None, int, bool, str | None],
        tuple[list[SearchResultItem] | None, int],
        int,
    ]:
        """Run entity-graph expansion, query embedding, and optional FTS pre-fetch concurrently."""

        async def _run_graph_expand() -> tuple[list[str], list[tuple[str, float]], int]:
            g_t0 = time.monotonic()
            if not enable_graph:
                return [], [], 0
            ent_names: list[str] = []
            g_hits: list[tuple[str, float]] = []
            try:
                from team_memory.services.entity.entity_search import (
                    extract_entities_from_query,
                    find_weighted_experiences_by_entities,
                    get_cached_entity_lexicon,
                )

                db_lexicon: set[str] = set()
                if self._db_url and "test_db" not in self._db_url:
                    try:
                        db_lexicon = await get_cached_entity_lexicon(self._db_url)
                    except Exception:
                        db_lexicon = set()
                ent_names = extract_entities_from_query(retrieval_query, lexicon=db_lexicon or None)
                if ent_names and self._db_url:
                    projects = [request.project] if request.project else ["default"]
                    if request.fallback_projects:
                        for fp in request.fallback_projects:
                            if fp not in projects:
                                projects.append(fp)
                    # Embedding (HTTP) and FTS pre-fetch (separate session2 when db_url is set)
                    # do not touch `session`, so entity-graph CTE can safely reuse `session` here.
                    try:
                        reusable_session = (
                            session
                            if (isinstance(AsyncSession, type) and isinstance(session, AsyncSession))
                            else None
                        )
                    except TypeError:
                        reusable_session = None
                    graph_kwargs: dict[str, Any] = {
                        "db_url": self._db_url,
                        "entity_names": ent_names,
                        "projects": projects,
                        "max_depth": 2,
                    }
                    if reusable_session is not None:
                        graph_kwargs["session"] = reusable_session
                    g_hits = await asyncio.wait_for(
                        find_weighted_experiences_by_entities(**graph_kwargs),
                        timeout=1.5,
                    )
                    if g_hits:
                        logger.info(
                            "[GRAPH]  entity-graph found %d related experience(s) for query=%s",
                            len(g_hits),
                            retrieval_query[:50],
                        )
            except Exception:
                logger.debug("Entity graph traversal failed (non-critical)", exc_info=True)
            return ent_names, g_hits, int((time.monotonic() - g_t0) * 1000)

        async def _run_embedding() -> tuple[list[float] | None, int, bool, str | None]:
            if self._search_config.mode == "fts":
                return None, 0, False, None
            e_t0 = time.monotonic()
            try:
                vec = await self._cache.get_or_compute_embedding(
                    retrieval_query,
                    self._embedding,
                    timeout_seconds=emb_timeout,
                )
                return vec, int((time.monotonic() - e_t0) * 1000), False, None
            except asyncio.TimeoutError:
                elapsed = int((time.monotonic() - e_t0) * 1000)
                logger.warning(
                    "Embedding timed out after %.1fs (%dms), degrading to FTS+Graph",
                    emb_timeout,
                    elapsed,
                )
                return None, elapsed, True, "embedding_timeout"
            except Exception as e:
                elapsed = int((time.monotonic() - e_t0) * 1000)
                logger.warning("Embedding failed, degrading to FTS+Graph: %s", e, exc_info=True)
                return None, elapsed, True, f"embedding_error:{type(e).__name__}"

        async def _run_fts_prefetch() -> tuple[list[SearchResultItem] | None, int]:
            if not should_prefetch_fts:
                return None, 0
            f_t0 = time.monotonic()
            try:
                fts_items = await self._fts_search(repo, retrieval_query.strip(), over_fetch, request)
                return fts_items, int((time.monotonic() - f_t0) * 1000)
            except Exception as e:
                logger.warning("FTS pre-fetch failed (will retry in retrieve_and_fuse): %s", e)
                return None, int((time.monotonic() - f_t0) * 1000)

        parallel_begin = time.monotonic()
        graph_res, emb_res, fts_res = await asyncio.gather(
            _run_graph_expand(),
            _run_embedding(),
            _run_fts_prefetch(),
        )
        parallel_wall_ms = int((time.monotonic() - parallel_begin) * 1000)
        return graph_res, emb_res, fts_res, parallel_wall_ms

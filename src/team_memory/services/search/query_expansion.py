"""Query expansion utilities for the search pipeline.

Provides tag synonym expansion and LLM-based query expansion with caching.
"""

from __future__ import annotations

import asyncio
import logging
import time

logger = logging.getLogger("team_memory.search_pipeline")

# LLM query expansion cache: normalized query -> (expanded_text, monotonic_timestamp)
_expansion_cache: dict[str, tuple[str, float]] = {}
_EXPANSION_CACHE_TTL = 600  # 10 minutes

# Topic discovery cache: db_url -> (TopicResult, monotonic_timestamp)
_topic_cache: dict[str, tuple] = {}
_TOPIC_CACHE_TTL = 3600  # 1 hour


def expansion_cache_clear() -> None:
    """Clear the LLM query expansion cache."""
    _expansion_cache.clear()


def topic_cache_clear() -> None:
    """Clear the topic discovery cache."""
    _topic_cache.clear()


async def llm_expand_query(
    query: str,
    llm_config: object,
    timeout_seconds: float,
) -> str | None:
    """Call LLM to expand search query with extra keywords. Returns None on any failure.

    Results are cached for ``_EXPANSION_CACHE_TTL`` seconds keyed by the
    normalised (stripped + lowercased) query text so repeated identical
    searches skip the LLM round-trip.
    """
    if not query or not query.strip() or not llm_config:
        return None

    cache_key = query.strip().lower()

    # Check cache
    if cache_key in _expansion_cache:
        expanded, ts = _expansion_cache[cache_key]
        if time.monotonic() - ts < _EXPANSION_CACHE_TTL:
            logger.debug("LLM expansion cache hit for: %s", cache_key[:50])
            return expanded
        # Expired -- remove stale entry
        _expansion_cache.pop(cache_key, None)

    try:
        from team_memory.services.llm.llm_client import LLMClient

        client = LLMClient.from_config(llm_config)
        system = (
            "You are a search keyword expansion assistant. Given the user's search intent, "
            "output a single line of space-separated expanded search keywords only, "
            "no explanation, no newlines. Example: 'database connection timeout' -> "
            "'database connection timeout pool connect'. Output only the keyword line."
        )
        text = await asyncio.wait_for(
            client.chat(system, query.strip(), temperature=0.2, timeout=timeout_seconds),
            timeout=timeout_seconds + 1.0,
        )
        if not text:
            return None
        line = text.strip().split("\n")[0].strip()
        result = line if line else None

        # Cache successful result
        if result:
            _expansion_cache[cache_key] = (result, time.monotonic())

        return result
    except (asyncio.TimeoutError, Exception) as e:
        logger.debug("LLM query expansion failed (fallback to original): %s", e)
        return None


def _term_in_query(term: str, q_lower: str) -> bool:
    """Check if synonym term is present in lowercased query (handles CJK+ASCII mixed text)."""
    import re

    t = term.strip().lower()
    if not t:
        return False
    if t.isascii():
        return bool(re.search(rf"(?<![a-z0-9_]){re.escape(t)}(?![a-z0-9_])", q_lower))
    return t in q_lower


def expand_query_synonyms(query: str, tag_synonyms: dict[str, str]) -> str:
    """Expand query with tag_synonyms (key<->value) for better recall."""
    if not query or not tag_synonyms:
        return query.strip()
    q = query.strip()
    q_lower = q.lower()
    added: list[str] = []
    for key, value in tag_synonyms.items():
        if key and value and key != value:
            k_in = _term_in_query(key, q_lower)
            v_in = _term_in_query(value, q_lower)
            if k_in and not v_in:
                added.append(value)
            if v_in and not k_in:
                added.append(key)
    if not added:
        return q
    return q + " " + " ".join(dict.fromkeys(added))

"""Unit tests for per-user expansion (no DB)."""

from __future__ import annotations

from team_memory.services.search_pipeline import _expand_query_synonyms


def test_expand_query_synonyms_merges_terms():
    """_expand_query_synonyms adds synonym values when key in query."""
    out = _expand_query_synonyms("PG connection timeout", {"PG": "PostgreSQL"})
    assert "PostgreSQL" in out
    assert "PG" in out


def test_expand_query_synonyms_empty_dict_returns_unchanged():
    """_expand_query_synonyms with empty dict returns query as-is."""
    q = "some query"
    assert _expand_query_synonyms(q, {}) == q


def test_cache_make_key_includes_current_user():
    """Cache key differs when current_user is set (avoid cross-user pollution)."""
    from team_memory.services.cache import SearchCache

    k1 = SearchCache._make_key("query", None, None, None)
    k2 = SearchCache._make_key("query", None, None, "user-1")
    assert k1 != k2

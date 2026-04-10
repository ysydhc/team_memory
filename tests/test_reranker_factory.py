"""Reranker factory loads from Settings / RerankerConfig."""

from team_memory.config import RerankerConfig
from team_memory.reranker.factory import create_reranker


def test_create_reranker_imports_and_none_is_noop():
    p = create_reranker(RerankerConfig())
    assert p.provider_name == "noop"

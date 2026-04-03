"""Search, cache, vector, retrieval, and page-index configuration."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class RetrievalConfig(BaseModel):
    """Retrieval configuration."""

    max_tokens: int | None = None
    max_count: int = 20
    trim_strategy: Literal["top_k", "summary"] = "top_k"
    top_k_children: int = 3
    # Reserved for future ranking; MVP hybrid search does not read experiences.avg_rating.
    min_avg_rating: float = Field(
        default=0.0,
        description="Min avg feedback rating filter (0=off). Not used in MVP search yet.",
    )
    rating_weight: float = Field(
        default=0.0,
        description="Rating blend weight. Not used in MVP search yet.",
    )
    summary_model: str | None = Field(
        default=None,
        description="LLM model for summary trim; null = llm.model.",
    )


class PageIndexLiteConfig(BaseModel):
    """PageIndex-Lite configuration (used for Archive long-doc retrieval)."""

    enabled: bool = True
    only_long_docs: bool = True
    min_doc_chars: int = 800
    max_tree_depth: int = 4
    max_nodes_per_doc: int = 40
    max_node_chars: int = 1200
    tree_weight: float = 0.15
    min_node_score: float = 0.01
    include_matched_nodes: bool = True


class SearchConfig(BaseModel):
    """Search pipeline configuration."""

    mode: Literal["hybrid", "vector", "fts"] = "hybrid"
    rrf_k: int = 60
    vector_weight: float = 0.7
    fts_weight: float = 0.3
    adaptive_filter: bool = True
    score_gap_threshold: float = 0.15
    min_confidence_ratio: float = 0.6
    adaptive_min_keep: int = 3
    short_query_max_chars: int = 20
    min_similarity_short: float = 0.45
    query_expansion_enabled: bool = False
    query_expansion_timeout_seconds: float = 3.0


class CacheConfig(BaseModel):
    """Query result caching configuration."""

    enabled: bool = True
    backend: Literal["memory", "redis"] = "memory"
    redis_url: str = "redis://localhost:6379/0"
    ttl_seconds: int = 300
    max_size: int = 100
    embedding_cache_size: int = 200


class VectorConfig(BaseModel):
    """Vector index configuration."""

    index_type: Literal["ivfflat", "hnsw"] = "ivfflat"
    hnsw_m: int = 16
    hnsw_ef_construction: int = 200
    hnsw_ef_search: int = 100

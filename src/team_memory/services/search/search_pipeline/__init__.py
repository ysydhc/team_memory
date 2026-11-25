"""Unified search pipeline — orchestrates the complete retrieval process."""

from __future__ import annotations

from team_memory.services.search.query_expansion import (
    _EXPANSION_CACHE_TTL,
    _TOPIC_CACHE_TTL,
    _expansion_cache,
    _topic_cache,
    expansion_cache_clear,
    topic_cache_clear,
)
from team_memory.services.search.search_pipeline.models import (
    _SEARCH_DEBUG,
    SUPERSESSION_PENALTY,
    SearchPipelineResult,
    SearchRequest,
    SearchResultItem,
    _apply_bi_temporal_annotations,
    coerce_cached_search_pipeline_result,
)
from team_memory.services.search.search_pipeline.pipeline import SearchPipeline
from team_memory.services.search.search_pipeline.stages.query_prep import _llm_expand_query
from team_memory.storage.archive_repository import ArchiveRepository
from team_memory.storage.database import get_session
from team_memory.storage.repository import ExperienceRepository

# Backward-compatible aliases for the module-level cache helpers
_expansion_cache_clear = expansion_cache_clear
_topic_cache_clear = topic_cache_clear

__all__ = [
    "ArchiveRepository",
    "ExperienceRepository",
    "SUPERSESSION_PENALTY",
    "SearchPipeline",
    "SearchPipelineResult",
    "SearchRequest",
    "SearchResultItem",
    "_EXPANSION_CACHE_TTL",
    "_SEARCH_DEBUG",
    "_TOPIC_CACHE_TTL",
    "_apply_bi_temporal_annotations",
    "_expansion_cache",
    "_expansion_cache_clear",
    "_llm_expand_query",
    "_topic_cache",
    "_topic_cache_clear",
    "coerce_cached_search_pipeline_result",
    "expansion_cache_clear",
    "get_session",
    "topic_cache_clear",
]

"""Experience lifecycle configuration."""

from __future__ import annotations

from pydantic import BaseModel


class LifecycleConfig(BaseModel):
    """Experience lifecycle -- dedup on save & write-time mutation gate."""

    dedup_on_save: bool = True
    dedup_on_save_threshold: float = 0.90
    auto_resolve_mutations: bool = True
    mutation_similarity_threshold: float = 0.82
    mutation_top_k: int = 3
    mutation_llm_enabled: bool = True
    mutation_llm_timeout_seconds: float = 2.0

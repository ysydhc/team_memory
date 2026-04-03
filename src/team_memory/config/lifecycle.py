"""Experience lifecycle configuration."""

from __future__ import annotations

from pydantic import BaseModel


class LifecycleConfig(BaseModel):
    """Experience lifecycle -- dedup on save."""

    dedup_on_save: bool = True
    dedup_on_save_threshold: float = 0.90

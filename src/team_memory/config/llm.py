"""LLM and extraction configuration."""

from __future__ import annotations

from pydantic import BaseModel


class LLMConfig(BaseModel):
    """LLM configuration for document parsing and other AI tasks."""

    provider: str = ""  # openai | generic | ollama (must be configured)
    model: str = ""
    base_url: str = ""
    api_key: str = ""
    prompt_dir: str | None = None
    monthly_budget: float = 0.0


class EntityExtractionConfig(BaseModel):
    """Entity extraction LLM configuration (separate from main LLM)."""

    enabled: bool = True
    model: str = "gbu-gemini-3.8-flash"
    base_url: str = "https://gw.gburouter.com/v1"
    api_key_env: str = "GATEWAY_API_KEY"
    timeout: int = 30
    max_concurrent: int = 3


class ExtractionConfig(BaseModel):
    """Experience extraction quality gate and retry configuration."""

    quality_gate: int = 2
    max_retries: int = 1
    few_shot_examples: str | None = None

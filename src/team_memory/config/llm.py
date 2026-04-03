"""LLM and extraction configuration."""

from __future__ import annotations

from pydantic import BaseModel


class LLMConfig(BaseModel):
    """LLM configuration for document parsing and other AI tasks."""

    provider: str = "ollama"  # ollama | openai | generic
    model: str = "gpt-oss:120b-cloud"
    base_url: str = "http://localhost:11434"
    api_key: str = ""
    prompt_dir: str | None = None
    monthly_budget: float = 0.0


class ExtractionConfig(BaseModel):
    """Experience extraction quality gate and retry configuration."""

    quality_gate: int = 2
    max_retries: int = 1
    few_shot_examples: str | None = None

"""Aggregate Settings class and loading helpers."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Literal

import yaml
from pydantic import Field
from pydantic_settings import (
    BaseSettings,
    PydanticBaseSettingsSource,
    SettingsConfigDict,
)

from team_memory.config.auth import AuthConfig
from team_memory.config.database import DatabaseConfig
from team_memory.config.embedding import EmbeddingConfig
from team_memory.config.janitor import JanitorConfig
from team_memory.config.lifecycle import LifecycleConfig
from team_memory.config.llm import EntityExtractionConfig, ExtractionConfig, LLMConfig
from team_memory.config.logging_config import LoggingConfig
from team_memory.config.mcp import MCPConfig
from team_memory.config.reranker import RerankerConfig
from team_memory.config.search import (
    CacheConfig,
    PageIndexLiteConfig,
    RetrievalConfig,
    SearchConfig,
    VectorConfig,
)
from team_memory.config.web import UploadsConfig, WebConfig


class Settings(BaseSettings):
    """Application settings.

    Loads from ``config.{development|production}.yaml``, then env overrides.
    """

    database: DatabaseConfig = Field(default_factory=DatabaseConfig)
    embedding: EmbeddingConfig = Field(default_factory=EmbeddingConfig)
    llm: LLMConfig = Field(default_factory=LLMConfig)
    extraction: ExtractionConfig = Field(default_factory=ExtractionConfig)
    entity_extraction: EntityExtractionConfig = Field(
        default_factory=EntityExtractionConfig
    )
    auth: AuthConfig = Field(default_factory=AuthConfig)
    default_project: str = "default"
    retrieval: RetrievalConfig = Field(default_factory=RetrievalConfig)
    pageindex_lite: PageIndexLiteConfig = Field(default_factory=PageIndexLiteConfig)
    search: SearchConfig = Field(default_factory=SearchConfig)
    reranker: RerankerConfig = Field(default_factory=RerankerConfig)
    cache: CacheConfig = Field(default_factory=CacheConfig)
    vector: VectorConfig = Field(default_factory=VectorConfig)
    web: WebConfig = Field(default_factory=WebConfig)
    lifecycle: LifecycleConfig = Field(default_factory=LifecycleConfig)
    mcp: MCPConfig = Field(default_factory=MCPConfig)
    logging: LoggingConfig = Field(default_factory=LoggingConfig)
    janitor: JanitorConfig = Field(default_factory=JanitorConfig)
    tag_synonyms: dict[str, str] = Field(default_factory=dict)
    log_format: Literal["human", "json"] = "human"
    uploads: UploadsConfig = Field(default_factory=UploadsConfig)

    model_config = SettingsConfigDict(
        env_prefix="TEAM_MEMORY_",
        env_nested_delimiter="__",
    )

    @classmethod
    def settings_customise_sources(
        cls,
        settings_cls: type[BaseSettings],
        init_settings: PydanticBaseSettingsSource,
        env_settings: PydanticBaseSettingsSource,
        dotenv_settings: PydanticBaseSettingsSource,
        file_secret_settings: PydanticBaseSettingsSource,
    ) -> tuple[PydanticBaseSettingsSource, ...]:
        return (
            env_settings,
            init_settings,
            dotenv_settings,
            file_secret_settings,
        )


def _resolve_env_vars(value: str) -> str:
    """Resolve ${ENV_VAR} patterns in string values."""
    if not isinstance(value, str):
        return value
    if value.startswith("${") and value.endswith("}"):
        env_name = value[2:-1]
        return os.environ.get(env_name, "")
    return value


def _resolve_dict(d: dict) -> dict:
    """Recursively resolve environment variables in a dict."""
    result = {}
    for k, v in d.items():
        if isinstance(v, dict):
            result[k] = _resolve_dict(v)
        elif isinstance(v, str):
            result[k] = _resolve_env_vars(v)
        else:
            result[k] = v
    return result


def _load_dotenv_if_available() -> None:
    """Load .env into os.environ."""
    try:
        from dotenv import load_dotenv
    except ImportError:
        return
    for candidate in (
        Path.cwd() / ".env",
        Path(__file__).resolve().parent.parent.parent.parent / ".env",
    ):
        if candidate.exists():
            load_dotenv(candidate)
            return


def _normalize_env_name(raw: str) -> str:
    """Map legacy env names to development|production."""
    key = (raw or "development").strip().lower()
    aliases = {
        "dev": "development",
        "test": "development",
        "local": "development",
        "prod": "production",
    }
    key = aliases.get(key, key)
    return key if key in ("development", "production") else "development"


def load_settings(config_path: str | Path | None = None) -> Settings:
    """Load settings from a single YAML file, then apply environment overrides.

    Resolution order:
      1. ``config_path`` argument if provided and file exists
      2. ``TEAM_MEMORY_CONFIG_PATH`` if set and file exists
      3. ``config.{development|production}.yaml`` where env is ``TEAM_MEMORY_ENV``
         (default ``development``; aliases: test/local/dev -> development, prod -> production)
      4. Empty dict -> Pydantic defaults only when no matching file exists

    Environment variables (``TEAM_MEMORY_*``) always override YAML via pydantic-settings.
    """
    _load_dotenv_if_available()
    yaml_data: dict = {}

    if config_path is not None:
        path = Path(config_path)
        if path.exists():
            with open(path) as f:
                raw = yaml.safe_load(f) or {}
            yaml_data = _resolve_dict(raw)
    else:
        env_path = os.environ.get("TEAM_MEMORY_CONFIG_PATH")
        if env_path:
            path = Path(env_path)
            if path.exists():
                with open(path) as f:
                    raw = yaml.safe_load(f) or {}
                yaml_data = _resolve_dict(raw)
        else:
            env_name = _normalize_env_name(os.environ.get("TEAM_MEMORY_ENV", "development"))
            cfg_path = Path(f"config.{env_name}.yaml")
            if cfg_path.exists():
                with open(cfg_path) as f:
                    raw = yaml.safe_load(f) or {}
                yaml_data = _resolve_dict(raw)

    if os.environ.get("LOG_FORMAT"):
        v = os.environ["LOG_FORMAT"].lower()
        yaml_data["log_format"] = "json" if v == "json" else "human"

    return Settings(**yaml_data)


def _default_include_archives() -> bool:
    """Dev environments include archives by default; production does not."""
    return os.environ.get("TEAM_MEMORY_ENV", "development") != "production"


# Global settings singleton (lazy initialization)
_settings: Settings | None = None


def get_settings() -> Settings:
    """Get the global settings singleton."""
    global _settings  # noqa: PLW0603
    if _settings is None:
        _settings = load_settings()
    return _settings


def reset_settings() -> None:
    """Reset the global settings singleton. Useful for testing."""
    global _settings  # noqa: PLW0603
    _settings = None

"""Configuration management for team_memory.

Loads one YAML per environment (default ``config.development.yaml``;
``TEAM_MEMORY_ENV=production`` → ``config.production.yaml``), then
environment variables override (pydantic-settings).
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, Field
from pydantic_settings import (
    BaseSettings,
    PydanticBaseSettingsSource,
    SettingsConfigDict,
)


class DatabaseConfig(BaseModel):
    """Database connection configuration."""

    url: str = "postgresql+asyncpg://developer:devpass@localhost:5433/team_memory"


class OpenAIEmbeddingConfig(BaseModel):
    """OpenAI embedding API configuration."""

    api_key: str = ""
    model: str = "text-embedding-3-small"
    dimension: int = 1536


class OllamaEmbeddingConfig(BaseModel):
    """Ollama embedding configuration."""

    # Explicit :latest avoids ambiguity with Ollama tags (name is always tagged in /api/tags).
    model: str = "nomic-embed-text:latest"
    base_url: str = "http://localhost:11434"
    dimension: int = 768


class LocalEmbeddingConfig(BaseModel):
    """Local embedding model configuration."""

    model_name: str = "BAAI/bge-m3"
    device: str = "cpu"
    dimension: int = 1024


class GenericEmbeddingConfig(BaseModel):
    """Generic OpenAI-compatible embedding endpoint configuration."""

    base_url: str = "http://localhost:8080/v1"
    api_key: str = ""
    model: str = "text-embedding-3-small"
    dimension: int = 1536


class EmbeddingConfig(BaseModel):
    """Embedding service configuration."""

    provider: Literal["ollama", "openai", "local", "generic"] = "ollama"
    ollama: OllamaEmbeddingConfig = Field(default_factory=OllamaEmbeddingConfig)
    openai: OpenAIEmbeddingConfig = Field(default_factory=OpenAIEmbeddingConfig)
    local: LocalEmbeddingConfig = Field(default_factory=LocalEmbeddingConfig)
    generic: GenericEmbeddingConfig = Field(default_factory=GenericEmbeddingConfig)

    @property
    def dimension(self) -> int:
        """Return the dimension of the active embedding provider."""
        if self.provider == "ollama":
            return self.ollama.dimension
        if self.provider == "openai":
            return self.openai.dimension
        if self.provider == "generic":
            return self.generic.dimension
        return self.local.dimension


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


class AuthConfig(BaseModel):
    """Authentication configuration."""

    type: Literal["api_key", "db_api_key", "none"] = "db_api_key"
    api_key: str | None = None
    user: str = "admin"
    allow_anonymous_search: bool = False
    default_admin_password: str | None = None
    session_secret: str | None = None


class WebConfig(BaseModel):
    """Web server configuration."""

    host: str = "0.0.0.0"
    port: int = 9111
    ssl_keyfile: str | None = None
    ssl_certfile: str | None = None


class InstallableCatalogConfig(BaseModel):
    """Installable rules/prompts catalog configuration."""

    sources: list[Literal["local", "registry"]] = Field(default_factory=lambda: ["local"])
    local_base_dir: str = "docs/res"
    registry_manifest_url: str = ""
    target_rules_dir: str = ".cursor/rules"
    target_prompts_dir: str = ".cursor/prompts"
    target_skills_dir: str = ".cursor/skills"
    request_timeout_seconds: int = 8


class LifecycleConfig(BaseModel):
    """Experience lifecycle — dedup on save."""

    dedup_on_save: bool = True
    dedup_on_save_threshold: float = 0.90


class MCPConfig(BaseModel):
    """MCP server output control."""

    max_output_tokens: int = 4000
    truncate_solution_at: int = 2000
    profile_max_strings_per_side: int = 20


class LoggingConfig(BaseModel):
    """Logging configuration."""

    log_io_enabled: bool = False
    log_io_detail: str = "mcp"
    log_io_truncate: int = 300
    log_file_enabled: bool = False
    log_file_path: str = "logs/team_memory.log"
    log_file_backup_count: int = 5
    log_file_max_bytes: int = 10 * 1024 * 1024


class WebhookItemConfig(BaseModel):
    """A single webhook target."""

    url: str
    events: list[str] = Field(default_factory=list)
    secret: str = ""
    active: bool = True


class UploadsConfig(BaseModel):
    """Local multipart uploads for archive attachments (MVP disk storage)."""

    enabled: bool = True
    root_dir: str = "data/uploads"
    max_bytes: int = 52_428_800  # ~50 MiB
    # Empty list = allow any non-empty extension; non-empty = lowercase suffix whitelist
    allowed_extensions: list[str] = Field(default_factory=lambda: [".md", ".txt", ".json", ".pdf"])


class Settings(BaseSettings):
    """Application settings.

    Loads from ``config.{development|production}.yaml``, then env overrides.
    """

    database: DatabaseConfig = Field(default_factory=DatabaseConfig)
    embedding: EmbeddingConfig = Field(default_factory=EmbeddingConfig)
    llm: LLMConfig = Field(default_factory=LLMConfig)
    extraction: ExtractionConfig = Field(default_factory=ExtractionConfig)
    auth: AuthConfig = Field(default_factory=AuthConfig)
    default_project: str = "default"
    retrieval: RetrievalConfig = Field(default_factory=RetrievalConfig)
    pageindex_lite: PageIndexLiteConfig = Field(default_factory=PageIndexLiteConfig)
    search: SearchConfig = Field(default_factory=SearchConfig)
    cache: CacheConfig = Field(default_factory=CacheConfig)
    vector: VectorConfig = Field(default_factory=VectorConfig)
    web: WebConfig = Field(default_factory=WebConfig)
    lifecycle: LifecycleConfig = Field(default_factory=LifecycleConfig)
    mcp: MCPConfig = Field(default_factory=MCPConfig)
    logging: LoggingConfig = Field(default_factory=LoggingConfig)
    installable_catalog: InstallableCatalogConfig = Field(default_factory=InstallableCatalogConfig)
    webhooks: list[WebhookItemConfig] = Field(default_factory=list)
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
        Path(__file__).resolve().parent.parent.parent / ".env",
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
         (default ``development``; aliases: test/local/dev → development, prod → production)
      4. Empty dict → Pydantic defaults only when no matching file exists

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


# Global settings singleton (lazy initialization)
_settings: Settings | None = None


def get_settings() -> Settings:
    """Get the global settings singleton."""
    global _settings
    if _settings is None:
        _settings = load_settings()
    return _settings


def reset_settings() -> None:
    """Reset the global settings singleton. Useful for testing."""
    global _settings
    _settings = None

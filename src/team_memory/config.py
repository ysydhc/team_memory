"""Configuration management for team_memory.

Supports loading from config.yaml + environment variable overrides.
Environment variables take precedence over YAML values.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings, PydanticBaseSettingsSource, SettingsConfigDict


class DatabaseConfig(BaseModel):
    """Database connection configuration."""

    url: str = "postgresql+asyncpg://developer:devpass@localhost:5432/team_memory"


class OpenAIEmbeddingConfig(BaseModel):
    """OpenAI embedding API configuration."""

    api_key: str = ""
    model: str = "text-embedding-3-small"
    dimension: int = 1536


class OllamaEmbeddingConfig(BaseModel):
    """Ollama embedding configuration."""

    model: str = "nomic-embed-text"
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
    model: str = "gpt-oss:20b-cloud"
    base_url: str = "http://localhost:11434"
    api_key: str = ""  # For OpenAI/generic providers
    prompt_dir: str | None = None  # Custom prompt template directory
    monthly_budget: float = 0.0  # Monthly budget in USD (0 = unlimited)


class RetrievalConfig(BaseModel):
    """Retrieval and context trimming configuration."""

    max_tokens: int | None = None  # Token budget for returned content (None = no limit)
    max_count: int = 20  # Maximum number of experience groups returned
    trim_strategy: Literal["top_k", "summary"] = "top_k"
    top_k_children: int = 3  # Max children per group
    min_avg_rating: float = 0.0  # Minimum avg_rating filter (0 = no filter)
    rating_weight: float = 0.3  # Weight of rating in final score
    summary_model: str | None = None  # LLM model for summary strategy


class PageIndexLiteConfig(BaseModel):
    """PageIndex-Lite configuration for long-document tree retrieval."""

    enabled: bool = True
    only_long_docs: bool = True  # only build tree for long document content
    min_doc_chars: int = 800  # minimum content length to build tree
    max_tree_depth: int = 4  # keep only top N heading levels
    max_nodes_per_doc: int = 40  # hard cap for node count
    max_node_chars: int = 1200  # truncate very long node content
    tree_weight: float = 0.15  # score contribution in search pipeline
    min_node_score: float = 0.01  # minimum node match score in tree search
    include_matched_nodes: bool = True  # include matched nodes in API/MCP output


# ====================== Reranker Configuration ======================


class OllamaLLMRerankerConfig(BaseModel):
    """Ollama LLM reranker config — uses chat API for relevance scoring.

    If model/base_url are None, they fall back to the global LLMConfig values.
    """

    model: str | None = None  # None = reuse LLMConfig.model
    base_url: str | None = None  # None = reuse LLMConfig.base_url
    top_k: int = 10  # Keep top K after reranking
    batch_size: int = 5  # Documents per LLM call
    prompt_template: str | None = None  # Custom prompt (None = use built-in default)


class CrossEncoderRerankerConfig(BaseModel):
    """Local cross-encoder reranker config (sentence-transformers)."""

    model_name: str = "cross-encoder/ms-marco-MiniLM-L6-v2"
    device: str = "cpu"
    top_k: int = 10


class JinaRerankerConfig(BaseModel):
    """Jina Reranker API config."""

    api_key: str = ""
    model: str = "jina-reranker-v2-base-multilingual"
    top_k: int = 10


class RerankerConfig(BaseModel):
    """Reranker configuration — pluggable provider architecture.

    provider = "none": No server-side reranking, rely on client LLM to judge.
    provider = "ollama_llm": Use Ollama LLM (or any OpenAI-compatible API) for scoring.
    provider = "cross_encoder": Use local cross-encoder model (sentence-transformers).
    provider = "jina": Use Jina Reranker API.
    """

    provider: Literal["none", "ollama_llm", "cross_encoder", "jina"] = "none"
    ollama_llm: OllamaLLMRerankerConfig = Field(
        default_factory=OllamaLLMRerankerConfig
    )
    cross_encoder: CrossEncoderRerankerConfig = Field(
        default_factory=CrossEncoderRerankerConfig
    )
    jina: JinaRerankerConfig = Field(default_factory=JinaRerankerConfig)


# ====================== Search Configuration ======================


class SearchConfig(BaseModel):
    """Search pipeline configuration."""

    mode: Literal["hybrid", "vector", "fts"] = "hybrid"  # Search mode
    rrf_k: int = 60  # RRF constant (standard value)
    vector_weight: float = 0.7  # Weight for vector results in RRF
    fts_weight: float = 0.3  # Weight for FTS results in RRF
    adaptive_filter: bool = True  # Enable adaptive score filtering
    score_gap_threshold: float = 0.15  # Gap threshold for elbow detection
    min_confidence_ratio: float = 0.6  # Min ratio vs top-1 for dynamic threshold


# ====================== Cache Configuration ======================


class CacheConfig(BaseModel):
    """Query result caching configuration.

    backend = "memory": In-process LRU cache (default, no external deps).
    backend = "redis": Redis-backed cache (requires redis server).
    """

    enabled: bool = True  # Enable/disable caching
    backend: Literal["memory", "redis"] = "memory"  # Cache backend
    redis_url: str = "redis://localhost:6379/0"  # Redis connection URL
    ttl_seconds: int = 300  # Cache TTL (5 minutes)
    max_size: int = 100  # Max cached entries
    embedding_cache_size: int = 200  # Max cached embeddings


class VectorConfig(BaseModel):
    """Vector index configuration (P2-4)."""

    index_type: Literal["ivfflat", "hnsw"] = "ivfflat"
    # HNSW parameters (only used when index_type=hnsw)
    hnsw_m: int = 16
    hnsw_ef_construction: int = 200
    hnsw_ef_search: int = 100


class AuthConfig(BaseModel):
    """Authentication configuration.

    Supported types:
      - "db_api_key": Database-backed multi-user auth (recommended).
        Supports password login on Web and API Key for MCP.
      - "api_key": In-memory single/dual key auth (legacy).
      - "none": No auth (development only).
    """

    type: Literal["api_key", "db_api_key", "none"] = "db_api_key"
    api_key: str | None = None
    user: str = "admin"
    allow_anonymous_search: bool = False


class WebConfig(BaseModel):
    """Web server configuration."""

    host: str = "0.0.0.0"
    port: int = 9111


class InstallableCatalogConfig(BaseModel):
    """Installable rules/prompts catalog configuration."""

    sources: list[Literal["local", "registry"]] = Field(default_factory=lambda: ["local"])
    local_base_dir: str = ".debug/knowledge-pack"
    registry_manifest_url: str = ""
    target_rules_dir: str = ".cursor/rules"
    target_prompts_dir: str = ".cursor/prompts"
    target_skills_dir: str = ".cursor/skills"
    request_timeout_seconds: int = 8


class ReviewConfig(BaseModel):
    """Experience review workflow configuration."""

    enabled: bool = True  # Enable review workflow
    auto_publish_threshold: float = 0.0  # Auto-publish if avg_rating >= this (0 = disabled)
    require_review_for_ai: bool = True  # AI-created experiences require review


class MemoryConfig(BaseModel):
    """Memory compaction / summary configuration."""

    auto_summarize: bool = True  # Auto-generate summary for long experiences
    summary_threshold_tokens: int = 500  # Token threshold to trigger summarization
    summary_model: str = ""  # LLM model override for summarization (empty = use default)
    batch_size: int = 10  # Batch size for bulk summarization


class LifecycleConfig(BaseModel):
    """Experience lifecycle management configuration."""

    stale_months: int = 6  # Months of inactivity before marking as stale
    scan_interval_hours: int = 24  # Background scan interval in hours
    duplicate_threshold: float = 0.92  # Cosine similarity threshold for dedup
    dedup_on_save: bool = True  # Check for duplicates before saving
    dedup_on_save_threshold: float = 0.90  # Similarity threshold for save-time dedup


class MCPConfig(BaseModel):
    """MCP server output control.

    Controls how much data MCP tools return to the LLM client,
    preventing context window overflow.
    """

    max_output_tokens: int = 4000  # Max token budget for a single tool response
    truncate_solution_at: int = 2000  # Max chars per solution field before truncation
    include_code_snippets: bool = True  # Whether to include code_snippets in results


# ====================== Schema Configuration ======================


class StructuredFieldDef(BaseModel):
    """Definition of a custom structured field for an experience type."""

    name: str
    type: str = "text"  # text / list / bool
    label: str = ""
    required: bool = False


class ExperienceTypeDef(BaseModel):
    """Definition of an experience type (built-in or custom)."""

    id: str
    label: str = ""
    severity: bool = False  # Whether this type uses severity levels
    progress_states: list[str] = Field(default_factory=list)
    structured_fields: list[StructuredFieldDef] = Field(default_factory=list)


class CategoryDef(BaseModel):
    """Definition of a category."""

    id: str
    label: str = ""


class CustomSchemaConfig(BaseModel):
    """Schema customisation — preset + user-defined overrides.

    Progressive complexity:
      - preset alone: zero-config, pick one of the built-in packs
      - experience_types / categories / severity_levels: add or override items
    """

    preset: str = "software-dev"  # software-dev / data-engineering / devops / general
    experience_types: list[ExperienceTypeDef] = Field(default_factory=list)
    categories: list[CategoryDef] = Field(default_factory=list)
    severity_levels: list[str] = Field(default_factory=list)  # empty = inherit from preset


# ====================== AI Behavior Configuration ======================


class AIBehaviorConfig(BaseModel):
    """AI behaviour preferences — the friendly layer above raw prompt files.

    Users describe what they want in natural language; the system translates
    these into prompt constraints automatically.
    """

    output_language: str = "zh-CN"
    detail_level: str = "detailed"  # detailed / concise
    focus_areas: list[str] = Field(
        default_factory=lambda: ["root_cause", "solution", "code_snippets"]
    )
    custom_instructions: str = ""  # Free-form team instructions


# ====================== Webhook Configuration ======================


class WebhookItemConfig(BaseModel):
    """A single webhook target."""

    url: str
    events: list[str] = Field(default_factory=list)
    secret: str = ""
    active: bool = True


class Settings(BaseSettings):
    """Application settings.

    Loads from config.yaml, then overrides with environment variables.
    """

    database: DatabaseConfig = Field(default_factory=DatabaseConfig)
    embedding: EmbeddingConfig = Field(default_factory=EmbeddingConfig)
    llm: LLMConfig = Field(default_factory=LLMConfig)
    auth: AuthConfig = Field(default_factory=AuthConfig)
    default_project: str = "default"
    retrieval: RetrievalConfig = Field(default_factory=RetrievalConfig)
    pageindex_lite: PageIndexLiteConfig = Field(default_factory=PageIndexLiteConfig)
    reranker: RerankerConfig = Field(default_factory=RerankerConfig)
    search: SearchConfig = Field(default_factory=SearchConfig)
    cache: CacheConfig = Field(default_factory=CacheConfig)
    vector: VectorConfig = Field(default_factory=VectorConfig)
    web: WebConfig = Field(default_factory=WebConfig)
    lifecycle: LifecycleConfig = Field(default_factory=LifecycleConfig)
    review: ReviewConfig = Field(default_factory=ReviewConfig)
    memory: MemoryConfig = Field(default_factory=MemoryConfig)
    mcp: MCPConfig = Field(default_factory=MCPConfig)
    installable_catalog: InstallableCatalogConfig = Field(
        default_factory=InstallableCatalogConfig
    )
    custom_schema: CustomSchemaConfig = Field(default_factory=CustomSchemaConfig)
    ai_behavior: AIBehaviorConfig = Field(default_factory=AIBehaviorConfig)
    webhooks: list[WebhookItemConfig] = Field(default_factory=list)
    tag_synonyms: dict[str, str] = Field(default_factory=dict)  # P2-7: PG -> PostgreSQL

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
        """Let environment variables override YAML-provided init values."""
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


def _deep_merge(base: dict, override: dict) -> dict:
    """Deep merge override dict into base dict (override wins)."""
    result = base.copy()
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = value
    return result


def load_settings(config_path: str | Path | None = None) -> Settings:
    """Load settings from YAML config file(s) with environment variable overrides.

    Configuration layering (later layers override earlier ones):
      1. config.yaml          — full defaults (always loaded as base)
      2. config.local.yaml    — developer machine overrides (deep merged)
      3. config.{env}.yaml    — environment-specific overlay (deep merged)
      4. Environment variables — highest priority (pydantic-settings)

    Special behavior:
      - TEAM_MEMORY_CONFIG_PATH: explicit base config path (replaces step 1-2)
      - config.minimal.yaml is NOT auto-merged by default
      - TEAM_MEMORY_ENABLE_MINIMAL_OVERLAY=1 can explicitly enable minimal overlay
      - If config.yaml does not exist, config.minimal.yaml is used as fallback base

    Args:
        config_path: Path to the YAML config file. If None, auto-detects.

    Returns:
        Fully resolved Settings instance.
    """
    yaml_data: dict = {}

    if config_path is not None:
        # Explicit path provided — use it as the sole base
        path = Path(config_path)
        if path.exists():
            with open(path) as f:
                raw = yaml.safe_load(f) or {}
            yaml_data = _resolve_dict(raw)
    else:
        env_path = os.environ.get("TEAM_MEMORY_CONFIG_PATH")
        if env_path:
            # Explicit env var — use as sole base
            path = Path(env_path)
            if path.exists():
                with open(path) as f:
                    raw = yaml.safe_load(f) or {}
                yaml_data = _resolve_dict(raw)
        else:
            # Layered loading: base -> local.
            # config.minimal.yaml is treated as example/template by default.
            base_path = Path("config.yaml")
            if base_path.exists():
                with open(base_path) as f:
                    raw = yaml.safe_load(f) or {}
                yaml_data = _resolve_dict(raw)

            # Optional explicit overlay for config.minimal.yaml
            minimal_overlay_enabled = os.environ.get(
                "TEAM_MEMORY_ENABLE_MINIMAL_OVERLAY", ""
            ).lower() in {"1", "true", "yes", "on"}
            minimal_path = Path("config.minimal.yaml")
            if minimal_overlay_enabled and minimal_path.exists():
                with open(minimal_path) as f:
                    minimal_raw = yaml.safe_load(f) or {}
                minimal_data = _resolve_dict(minimal_raw)
                yaml_data = _deep_merge(yaml_data, minimal_data)
            elif not yaml_data and minimal_path.exists():
                # Bootstrap fallback when config.yaml is absent.
                with open(minimal_path) as f:
                    minimal_raw = yaml.safe_load(f) or {}
                yaml_data = _resolve_dict(minimal_raw)

            # Overlay config.local.yaml (developer overrides)
            local_path = Path("config.local.yaml")
            if local_path.exists():
                with open(local_path) as f:
                    local_raw = yaml.safe_load(f) or {}
                local_data = _resolve_dict(local_raw)
                yaml_data = _deep_merge(yaml_data, local_data)

    # Multi-environment overlay: load config.{TEAM_MEMORY_ENV}.yaml
    env_name = os.environ.get("TEAM_MEMORY_ENV", "")
    if env_name:
        env_config_path = Path(f"config.{env_name}.yaml")
        if env_config_path.exists():
            with open(env_config_path) as f:
                env_raw = yaml.safe_load(f) or {}
            env_data = _resolve_dict(env_raw)
            yaml_data = _deep_merge(yaml_data, env_data)

    # Environment variable overrides (e.g. TEAM_MEMORY_DATABASE__URL)
    # are handled by pydantic-settings automatically
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

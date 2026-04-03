"""Configuration management for team_memory.

Loads one YAML per environment (default ``config.development.yaml``;
``TEAM_MEMORY_ENV=production`` -> ``config.production.yaml``), then
environment variables override (pydantic-settings).

This package re-exports every public symbol so that existing imports
(``from team_memory.config import Settings, load_settings``) continue to work.
"""

from team_memory.config.auth import AuthConfig
from team_memory.config.database import DatabaseConfig
from team_memory.config.embedding import (
    EmbeddingConfig,
    GenericEmbeddingConfig,
    LocalEmbeddingConfig,
    OllamaEmbeddingConfig,
    OpenAIEmbeddingConfig,
)
from team_memory.config.lifecycle import LifecycleConfig
from team_memory.config.llm import ExtractionConfig, LLMConfig
from team_memory.config.logging_config import LoggingConfig
from team_memory.config.mcp import MCPConfig
from team_memory.config.search import (
    CacheConfig,
    PageIndexLiteConfig,
    RetrievalConfig,
    SearchConfig,
    VectorConfig,
)
from team_memory.config.settings import (
    Settings,
    _default_include_archives,
    _load_dotenv_if_available,
    _normalize_env_name,
    _resolve_dict,
    _resolve_env_vars,
    get_settings,
    load_settings,
    reset_settings,
)
from team_memory.config.web import UploadsConfig, WebConfig

__all__ = [
    # database
    "DatabaseConfig",
    # embedding
    "OpenAIEmbeddingConfig",
    "OllamaEmbeddingConfig",
    "LocalEmbeddingConfig",
    "GenericEmbeddingConfig",
    "EmbeddingConfig",
    # llm
    "LLMConfig",
    "ExtractionConfig",
    # search
    "RetrievalConfig",
    "PageIndexLiteConfig",
    "SearchConfig",
    "CacheConfig",
    "VectorConfig",
    # auth
    "AuthConfig",
    # web
    "WebConfig",
    "UploadsConfig",
    # mcp
    "MCPConfig",
    # lifecycle
    "LifecycleConfig",
    # logging
    "LoggingConfig",
    # settings
    "Settings",
    "load_settings",
    "get_settings",
    "reset_settings",
    "_default_include_archives",
    "_load_dotenv_if_available",
    "_normalize_env_name",
    "_resolve_env_vars",
    "_resolve_dict",
]

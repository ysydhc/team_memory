"""Unified client configuration for Team Memory.

Reads from (priority high→low):
  1. Environment variables
  2. Project .env (dotenv)
  3. ~/.config/tm/config.toml
  4. Code defaults

Used by CLI, MCP server, daemon, and hooks.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

try:
    import tomllib
except ModuleNotFoundError:
    try:
        import tomli as tomllib  # type: ignore[no-redef]
    except ModuleNotFoundError:
        tomllib = None  # type: ignore[assignment]

try:
    from dotenv import load_dotenv
except ImportError:
    load_dotenv = None  # type: ignore[assignment]

logger = logging.getLogger("team_memory.client_config")

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

_CONFIG_DIR = Path.home() / ".config" / "tm"
_CONFIG_FILE = _CONFIG_DIR / "config.toml"


def _find_project_root(start: str | None = None) -> Path | None:
    """Walk up to find project root (has .git or pyproject.toml)."""
    current = Path(start or os.getcwd()).resolve()
    for parent in [current] + list(current.parents):
        if (parent / ".git").exists() or (parent / "pyproject.toml").exists():
            return parent
    return None


def _load_dotenv() -> None:
    """Load .env from project root if dotenv is available."""
    if load_dotenv is None:
        return
    # Try project root .env first
    root = _find_project_root()
    if root and (root / ".env").exists():
        load_dotenv(root / ".env", override=False)
    # Also try CWD
    load_dotenv(override=False)


def _read_config_toml() -> dict[str, Any]:
    """Read ~/.config/tm/config.toml."""
    if tomllib is None:
        return {}
    if not _CONFIG_FILE.exists():
        return {}
    try:
        with open(_CONFIG_FILE, "rb") as f:
            return tomllib.load(f)
    except Exception as e:
        logger.warning("Failed to read %s: %s", _CONFIG_FILE, e)
        return {}


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass
class ServerConfig:
    """Remote server connection / 远程服务连接"""

    url: str = "http://localhost:9111"
    api_key: str = ""


@dataclass
class LLMConfig:
    """Client-side LLM configuration / 客户端 LLM 配置

    Used by: faithfulness_judge, refinement_worker,
    personal_memory_extractor, auto_dream, solution_backfill.
    """

    base_url: str = "http://localhost:4000/v1"
    api_key: str = ""
    api_key_env: str = ""  # Reference to env var name, e.g. "LITELLM_MASTER_KEY"
    model: str = "DeepSeek-V3"
    timeout: int = 30

    @property
    def resolved_api_key(self) -> str:
        """Resolve API key: env var reference > direct value."""
        if self.api_key_env:
            return os.environ.get(self.api_key_env, "")
        return self.api_key


@dataclass
class ConversationConfig:
    """Local conversation buffer / 本地对话缓存"""

    db_path: str = str(Path.home() / ".cache" / "tm-pipeline" / "drafts.db")


@dataclass
class WikiConfig:
    """LLMWiki settings / LLMWiki 设置"""

    enabled: bool = False
    db_path: str = str(Path.home() / ".cache" / "tm-pipeline" / "wiki.db")


@dataclass
class DaemonConfig:
    """Daemon process settings / Daemon 进程设置"""

    host: str = "127.0.0.1"
    port: int = 3901
    log_level: str = "INFO"


@dataclass
class ClientConfig:
    """Unified client configuration / 统一客户端配置

    All TM client components (CLI, MCP, daemon, hooks) read from this.
    """

    server: ServerConfig = field(default_factory=ServerConfig)
    llm: LLMConfig = field(default_factory=LLMConfig)
    conversation: ConversationConfig = field(default_factory=ConversationConfig)
    wiki: WikiConfig = field(default_factory=WikiConfig)
    daemon: DaemonConfig = field(default_factory=DaemonConfig)

    @classmethod
    def load(cls) -> ClientConfig:
        """Load config with full priority chain."""
        _load_dotenv()
        toml_data = _read_config_toml()

        config = cls()

        # --- [server] ---
        srv = toml_data.get("server", {})
        config.server.url = (
            os.environ.get("TM_SERVER_URL")
            or os.environ.get("TM_BASE_URL")
            or srv.get("url", config.server.url)
        )
        config.server.api_key = (
            os.environ.get("TEAM_MEMORY_API_KEY")
            or srv.get("api_key", "")
        )

        # --- [llm] ---
        llm = toml_data.get("llm", {})
        config.llm.base_url = (
            os.environ.get("TM_LLM_BASE_URL")
            or llm.get("base_url", config.llm.base_url)
        )
        config.llm.api_key = llm.get("api_key", "")
        config.llm.api_key_env = llm.get("api_key_env", "")
        config.llm.model = (
            os.environ.get("TM_LLM_MODEL")
            or llm.get("model", config.llm.model)
        )
        config.llm.timeout = llm.get("timeout", config.llm.timeout)

        # --- [conversation] ---
        conv = toml_data.get("conversation", {})
        config.conversation.db_path = (
            os.environ.get("TM_CONVERSATION_DB")
            or conv.get("db_path", config.conversation.db_path)
        )

        # --- [wiki] ---
        wiki = toml_data.get("wiki", {})
        config.wiki.enabled = wiki.get("enabled", config.wiki.enabled)
        config.wiki.db_path = (
            os.environ.get("TM_WIKI_DB")
            or wiki.get("db_path", config.wiki.db_path)
        )

        # --- [daemon] ---
        dmn = toml_data.get("daemon", {})
        config.daemon.host = (
            os.environ.get("TM_DAEMON_HOST")
            or dmn.get("host", config.daemon.host)
        )
        config.daemon.port = int(
            os.environ.get("TM_DAEMON_PORT", "")
            or dmn.get("port", config.daemon.port)
        )
        config.daemon.log_level = (
            os.environ.get("TEAM_MEMORY_LOG_LEVEL")
            or dmn.get("log_level", config.daemon.log_level)
        )

        return config

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dict (for display/debug). Masks API keys."""
        return {
            "server": {
                "url": self.server.url,
                "api_key": _mask(self.server.api_key),
            },
            "llm": {
                "base_url": self.llm.base_url,
                "api_key": _mask(self.llm.api_key),
                "api_key_env": self.llm.api_key_env or "(not set)",
                "resolved_api_key": _mask(self.llm.resolved_api_key),
                "model": self.llm.model,
                "timeout": self.llm.timeout,
            },
            "conversation": {
                "db_path": self.conversation.db_path,
            },
            "wiki": {
                "enabled": self.wiki.enabled,
                "db_path": self.wiki.db_path,
            },
            "daemon": {
                "host": self.daemon.host,
                "port": self.daemon.port,
                "log_level": self.daemon.log_level,
            },
        }


def _mask(value: str) -> str:
    """Mask sensitive value for display."""
    if not value or len(value) < 8:
        return "***" if value else "(not set)"
    return f"{value[:4]}...{value[-4:]}"


# ---------------------------------------------------------------------------
# Convenience accessors (backward compat with project_resolver.py)
# ---------------------------------------------------------------------------

_cached_config: ClientConfig | None = None


def get_client_config() -> ClientConfig:
    """Get or create cached ClientConfig."""
    global _cached_config
    if _cached_config is None:
        _cached_config = ClientConfig.load()
    return _cached_config


def get_server_url() -> str:
    return get_client_config().server.url


def get_api_key() -> str:
    return get_client_config().server.api_key

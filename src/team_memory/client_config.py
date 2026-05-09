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

    @classmethod
    def load_with_sources(cls) -> tuple[ClientConfig, dict[str, str]]:
        """Load config and return (config, sources_map).

        sources_map 的 key 是 "section.field" 格式，value 是来源说明:
          - "env: VAR_NAME"  — 环境变量 / environment variable
          - ".env file"      — .env 文件 / dotenv file
          - "config.toml"    — TOML 配置文件 / TOML config file
          - "default"        — 代码默认值 / code default

        Returns:
            (ClientConfig, dict[str, str]) — 配置对象和每个字段的来源 / config and source per field
        """
        # 先记录哪些环境变量已设置 / track which env vars are set
        env_keys = {
            "server.url": bool(os.environ.get("TM_SERVER_URL") or os.environ.get("TM_BASE_URL")),
            "server.api_key": bool(os.environ.get("TEAM_MEMORY_API_KEY")),
            "llm.base_url": bool(os.environ.get("TM_LLM_BASE_URL")),
            "llm.model": bool(os.environ.get("TM_LLM_MODEL")),
            "conversation.db_path": bool(os.environ.get("TM_CONVERSATION_DB")),
            "wiki.db_path": bool(os.environ.get("TM_WIKI_DB")),
            "daemon.host": bool(os.environ.get("TM_DAEMON_HOST")),
            "daemon.port": bool(os.environ.get("TM_DAEMON_PORT")),
            "daemon.log_level": bool(os.environ.get("TEAM_MEMORY_LOG_LEVEL")),
        }

        # 记录 .env 加载前后的差异 / detect dotenv-added vars
        env_before = dict(os.environ)
        _load_dotenv()
        env_after = dict(os.environ)

        # 检测哪些变量是 .env 文件引入的 / find which vars came from .env
        dotenv_vars: set[str] = set()
        for k, v in env_after.items():
            if k not in env_before or env_before[k] != v:
                dotenv_vars.add(k)

        toml_data = _read_config_toml()
        config = cls()

        sources: dict[str, str] = {}

        # --- [server] ---
        srv = toml_data.get("server", {})

        # server.url
        if os.environ.get("TM_SERVER_URL"):
            config.server.url = os.environ["TM_SERVER_URL"]
            sources["server.url"] = "env: TM_SERVER_URL"
        elif os.environ.get("TM_BASE_URL"):
            config.server.url = os.environ["TM_BASE_URL"]
            if "TM_BASE_URL" in dotenv_vars:
                sources["server.url"] = ".env file"
            elif env_keys["server.url"]:
                sources["server.url"] = "env: TM_BASE_URL"
            else:
                sources["server.url"] = ".env file"
        elif srv.get("url"):
            config.server.url = srv["url"]
            sources["server.url"] = "config.toml"
        else:
            sources["server.url"] = "default"

        # server.api_key
        if os.environ.get("TEAM_MEMORY_API_KEY"):
            config.server.api_key = os.environ["TEAM_MEMORY_API_KEY"]
            if "TEAM_MEMORY_API_KEY" in dotenv_vars:
                sources["server.api_key"] = ".env file"
            else:
                sources["server.api_key"] = "env: TEAM_MEMORY_API_KEY"
        elif srv.get("api_key"):
            config.server.api_key = srv["api_key"]
            sources["server.api_key"] = "config.toml"
        else:
            sources["server.api_key"] = "default"

        # --- [llm] ---
        llm = toml_data.get("llm", {})

        # llm.base_url
        if os.environ.get("TM_LLM_BASE_URL"):
            config.llm.base_url = os.environ["TM_LLM_BASE_URL"]
            sources["llm.base_url"] = "env: TM_LLM_BASE_URL"
        elif llm.get("base_url"):
            config.llm.base_url = llm["base_url"]
            sources["llm.base_url"] = "config.toml"
        else:
            sources["llm.base_url"] = "default"

        # llm.api_key
        if llm.get("api_key"):
            config.llm.api_key = llm["api_key"]
            sources["llm.api_key"] = "config.toml"
        else:
            sources["llm.api_key"] = "default"

        # llm.api_key_env
        if llm.get("api_key_env"):
            config.llm.api_key_env = llm["api_key_env"]
            sources["llm.api_key_env"] = "config.toml"
        else:
            sources["llm.api_key_env"] = "default"

        # llm.model
        if os.environ.get("TM_LLM_MODEL"):
            config.llm.model = os.environ["TM_LLM_MODEL"]
            sources["llm.model"] = "env: TM_LLM_MODEL"
        elif llm.get("model"):
            config.llm.model = llm["model"]
            sources["llm.model"] = "config.toml"
        else:
            sources["llm.model"] = "default"

        # llm.timeout
        if llm.get("timeout") is not None:
            config.llm.timeout = llm["timeout"]
            sources["llm.timeout"] = "config.toml"
        else:
            sources["llm.timeout"] = "default"

        # --- [conversation] ---
        conv = toml_data.get("conversation", {})
        if os.environ.get("TM_CONVERSATION_DB"):
            config.conversation.db_path = os.environ["TM_CONVERSATION_DB"]
            sources["conversation.db_path"] = "env: TM_CONVERSATION_DB"
        elif conv.get("db_path"):
            config.conversation.db_path = conv["db_path"]
            sources["conversation.db_path"] = "config.toml"
        else:
            sources["conversation.db_path"] = "default"

        # --- [wiki] ---
        wiki = toml_data.get("wiki", {})
        if wiki.get("enabled") is not None:
            config.wiki.enabled = wiki["enabled"]
            sources["wiki.enabled"] = "config.toml"
        else:
            sources["wiki.enabled"] = "default"

        if os.environ.get("TM_WIKI_DB"):
            config.wiki.db_path = os.environ["TM_WIKI_DB"]
            sources["wiki.db_path"] = "env: TM_WIKI_DB"
        elif wiki.get("db_path"):
            config.wiki.db_path = wiki["db_path"]
            sources["wiki.db_path"] = "config.toml"
        else:
            sources["wiki.db_path"] = "default"

        # --- [daemon] ---
        dmn = toml_data.get("daemon", {})

        if os.environ.get("TM_DAEMON_HOST"):
            config.daemon.host = os.environ["TM_DAEMON_HOST"]
            sources["daemon.host"] = "env: TM_DAEMON_HOST"
        elif dmn.get("host"):
            config.daemon.host = dmn["host"]
            sources["daemon.host"] = "config.toml"
        else:
            sources["daemon.host"] = "default"

        if os.environ.get("TM_DAEMON_PORT"):
            config.daemon.port = int(os.environ["TM_DAEMON_PORT"])
            sources["daemon.port"] = "env: TM_DAEMON_PORT"
        elif dmn.get("port") is not None:
            config.daemon.port = int(dmn["port"])
            sources["daemon.port"] = "config.toml"
        else:
            sources["daemon.port"] = "default"

        if os.environ.get("TEAM_MEMORY_LOG_LEVEL"):
            config.daemon.log_level = os.environ["TEAM_MEMORY_LOG_LEVEL"]
            sources["daemon.log_level"] = "env: TEAM_MEMORY_LOG_LEVEL"
        elif dmn.get("log_level"):
            config.daemon.log_level = dmn["log_level"]
            sources["daemon.log_level"] = "config.toml"
        else:
            sources["daemon.log_level"] = "default"

        return config, sources

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

"""TM Daemon 配置模块。

数据类定义：
- DaemonSettings: 服务监听地址
- TMSettings: TM 连接配置（local/remote）
- VaultConfig: Obsidian vault 配置
- ObsidianSettings: vault 列表
- DraftSettings: 草稿缓冲配置
- RetrievalSettings: 检索配置
- ProjectMapping: 项目名 → 路径模式
- DaemonConfig: 聚合配置

load_config(config_path): 从 YAML 文件加载配置，返回 DaemonConfig。
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

logger = logging.getLogger("daemon.config")


# ---------------------------------------------------------------------------
# 子配置数据类
# ---------------------------------------------------------------------------


@dataclass
class DaemonSettings:
    """Daemon 服务监听配置。"""

    host: str = "127.0.0.1"
    port: int = 3901


@dataclass
class TMSettings:
    """TM 连接配置。"""

    mode: str = "local"  # "local" | "remote"
    base_url: str = ""
    user: str = "daemon"


@dataclass
class VaultConfig:
    """单个 Obsidian vault 配置。"""

    path: str = ""
    project: str = ""
    exclude: list[str] = field(default_factory=list)


@dataclass
class ObsidianSettings:
    """Obsidian 设置。"""

    vaults: list[VaultConfig] = field(default_factory=list)


@dataclass
class DraftSettings:
    """草稿缓冲配置。"""

    max_age_minutes: int = 30
    db_path: str = ""


@dataclass
class RetrievalSettings:
    """检索配置。"""

    session_start_top_k: int = 3
    keyword_triggers: list[str] = field(default_factory=list)


@dataclass
class RefinementSettings:
    """LLM 精炼配置。"""

    enabled: bool = True
    provider: str = "litellm"  # "litellm" | "ollama"
    model: str = "qwen-plus"
    base_url: str = "http://localhost:4000/v1"
    api_key_env: str = "LITELLM_MASTER_KEY"
    timeout: int = 30
    scan_interval_seconds: int = 30
    max_input_chars: int = 6000
    max_retries: int = 1
    fallback_on_failure: bool = True


@dataclass
class EvaluationSettings:
    """评估配置：模糊匹配开关与阈值。"""
    fuzzy_match_enabled: bool = True
    fuzzy_match_threshold: float = 0.8


@dataclass
class WikiSettings:
    """Wiki 编译配置。"""
    enabled: bool = True
    wiki_root: str = ""  # 默认：项目根目录下的 wiki/
    db_path: str = ""  # 默认：<项目根>/.wiki/cache.db


@dataclass
class ProjectMapping:
    """项目名 → 路径模式映射。"""

    name: str = ""
    path_patterns: list[str] = field(default_factory=list)


@dataclass
class DaemonConfig:
    """Daemon 聚合配置。"""

    daemon: DaemonSettings = field(default_factory=DaemonSettings)
    tm: TMSettings = field(default_factory=TMSettings)
    obsidian: ObsidianSettings = field(default_factory=ObsidianSettings)
    draft: DraftSettings = field(default_factory=DraftSettings)
    retrieval: RetrievalSettings = field(default_factory=RetrievalSettings)
    refinement: RefinementSettings = field(default_factory=RefinementSettings)
    evaluation: EvaluationSettings = field(default_factory=EvaluationSettings)
    wiki: WikiSettings = field(default_factory=WikiSettings)
    projects: list[ProjectMapping] = field(default_factory=list)


# ---------------------------------------------------------------------------
# 配置加载
# ---------------------------------------------------------------------------


# 子配置类 → YAML 键名 → 字段名 的映射
_SUBCONFIG_KEYS = {
    "daemon": (DaemonSettings, None),
    "tm": (TMSettings, None),
    "obsidian": (ObsidianSettings, "vaults"),
    "draft": (DraftSettings, None),
    "retrieval": (RetrievalSettings, None),
    "refinement": (RefinementSettings, None),
    "evaluation": (EvaluationSettings, None),
    "wiki": (WikiSettings, None),
}


def _dict_to_dataclass(cls: type, data: dict[str, Any]) -> Any:
    """将字典转为 dataclass 实例，忽略多余键，嵌套处理 list[子dataclass]。"""
    if not isinstance(data, dict):
        return cls()

    field_names = {f.name for f in cls.__dataclass_fields__.values()}
    filtered = {k: v for k, v in data.items() if k in field_names}

    # 特殊处理：ObsidianSettings.vaults → list[VaultConfig]
    if cls is ObsidianSettings and "vaults" in filtered:
        vaults_data = filtered["vaults"]
        if isinstance(vaults_data, list):
            filtered["vaults"] = [
                _dict_to_dataclass(VaultConfig, v) if isinstance(v, dict) else VaultConfig()
                for v in vaults_data
            ]

    return cls(**filtered)


def load_config(config_path: str | None = None) -> DaemonConfig:
    """从 YAML 文件加载配置，返回 DaemonConfig。

    如果 config_path 为 None 或文件不存在，返回全默认配置。
    部分配置只覆盖指定字段，未指定的保持默认值。
    """
    if config_path is None:
        # 默认路径：scripts/hooks/config.yaml
        default_path = Path(__file__).resolve().parent.parent / "hooks" / "config.yaml"
        if default_path.exists():
            config_path = str(default_path)
        else:
            return DaemonConfig()

    if not os.path.isfile(config_path):
        logger.debug("Config file not found: %s, using defaults", config_path)
        return DaemonConfig()

    try:
        with open(config_path, "r", encoding="utf-8") as f:
            raw = yaml.safe_load(f)
    except Exception:
        logger.warning("Failed to load config from %s, using defaults", config_path, exc_info=True)
        return DaemonConfig()

    if not isinstance(raw, dict):
        return DaemonConfig()

    # 构建各子配置
    kwargs: dict[str, Any] = {}

    for key, (cls, _) in _SUBCONFIG_KEYS.items():
        sub_data = raw.get(key, {})
        kwargs[key] = _dict_to_dataclass(cls, sub_data)

    # projects 列表
    projects_data = raw.get("projects", [])
    if isinstance(projects_data, list):
        kwargs["projects"] = [
            _dict_to_dataclass(ProjectMapping, p) if isinstance(p, dict) else ProjectMapping()
            for p in projects_data
        ]
    else:
        kwargs["projects"] = []

    config = DaemonConfig(**kwargs)

    # 展开所有路径中的 ~
    config.draft.db_path = os.path.expanduser(config.draft.db_path) if config.draft.db_path else ""
    config.wiki.wiki_root = (
        os.path.expanduser(config.wiki.wiki_root) if config.wiki.wiki_root else ""
    )
    config.wiki.db_path = (
        os.path.expanduser(config.wiki.db_path) if config.wiki.db_path else ""
    )
    for vault in config.obsidian.vaults:
        vault.path = os.path.expanduser(vault.path) if vault.path else ""

    return config

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
    user: str = "admin"


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
    idle_timeout_minutes: int = 30  # sessions idle longer than this are auto-refined
    max_input_chars: int = 6000
    max_retries: int = 1
    fallback_on_failure: bool = True
    quality_gate_enabled: bool = True
    quality_min_score: int = 60
    dedup_similarity_threshold: float = 0.9  # cosine > this → merge into existing
    dedup_context_threshold: float = 0.7  # cosine in [0.7, 0.9) → include as LLM context


@dataclass
class AutoDreamSettings:
    """Auto Dream 后台合并/裁剪配置。"""

    enabled: bool = True
    min_hours_since_last: int = 24
    min_new_experiences: int = 5
    merge_similarity_threshold: float = 0.85
    prune_max_age_days: int = 30
    scan_interval_minutes: int = 60  # how often to check trigger conditions


@dataclass
class EvaluationSettings:
    """评估配置：模糊匹配开关与阈值。"""

    fuzzy_match_enabled: bool = True
    fuzzy_match_threshold: float = 0.18
    faithfulness_batch_threshold: int = 5


@dataclass
class WikiSettings:
    """Wiki 编译配置。"""

    enabled: bool = True
    wiki_root: str = ""  # 默认：项目根目录下的 wiki/
    db_path: str = ""  # 默认：<项目根>/.wiki/cache.db


@dataclass
class ArenaSettings:
    """Arena 评估系统配置。"""

    enabled: bool = True
    interval_minutes: int = 30
    converged_interval_minutes: int = 120
    rounds_per_epoch: int = 10
    sample_size: int = 3
    data_dir: str = ""  # empty = default (project_root/data/arena/)


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
    auto_dream: AutoDreamSettings = field(default_factory=AutoDreamSettings)
    wiki: WikiSettings = field(default_factory=WikiSettings)
    arena: ArenaSettings = field(default_factory=ArenaSettings)
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
    "auto_dream": (AutoDreamSettings, None),
    "wiki": (WikiSettings, None),
    "arena": (ArenaSettings, None),
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
                _dict_to_dataclass(VaultConfig, v) if isinstance(v, dict) else VaultConfig() for v in vaults_data
            ]

    return cls(**filtered)


def load_config(config_path: str | None = None) -> DaemonConfig:
    """从 YAML 文件 + ClientConfig 加载配置，返回 DaemonConfig。

    配置合并优先级 (高→低):
      1. YAML 文件中的显式配置
      2. ~/.config/tm/config.toml (ClientConfig)
      3. 环境变量
      4. 代码默认值
    """
    # 先从 ClientConfig 读取基础配置
    try:
        from team_memory.client_config import get_client_config

        client_cfg = get_client_config()
    except Exception:
        client_cfg = None

    if config_path is None:
        # 默认路径：team_memory/hooks/config.yaml
        # __file__ = daemon/runtime/config.py
        # .parent = runtime/ → .parent.parent = daemon/ → .parent.parent.parent = team_memory/
        default_path = Path(__file__).resolve().parent.parent.parent / "hooks" / "config.yaml"
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
            _dict_to_dataclass(ProjectMapping, p) if isinstance(p, dict) else ProjectMapping() for p in projects_data
        ]
    else:
        kwargs["projects"] = []

    config = DaemonConfig(**kwargs)

    # 展开所有路径中的 ~
    config.draft.db_path = os.path.expanduser(config.draft.db_path) if config.draft.db_path else ""
    config.wiki.wiki_root = os.path.expanduser(config.wiki.wiki_root) if config.wiki.wiki_root else ""
    config.wiki.db_path = os.path.expanduser(config.wiki.db_path) if config.wiki.db_path else ""
    for vault in config.obsidian.vaults:
        vault.path = os.path.expanduser(vault.path) if vault.path else ""

    # 从 ClientConfig 合并基础配置（仅当 YAML 未显式指定时）
    if client_cfg is not None:
        if not config.tm.base_url:
            config.tm.base_url = client_cfg.server.url
        if not config.tm.mode and client_cfg.server.url != "http://localhost:9111":
            config.tm.mode = "remote"
        if config.daemon.host == "127.0.0.1" and client_cfg.daemon.host != "127.0.0.1":
            config.daemon.host = client_cfg.daemon.host
        if config.daemon.port == 3901 and client_cfg.daemon.port != 3901:
            config.daemon.port = client_cfg.daemon.port
        if not config.draft.db_path and hasattr(client_cfg, "conversation") and client_cfg.conversation.db_path:
            config.draft.db_path = os.path.expanduser(client_cfg.conversation.db_path)
        if not config.wiki.db_path and hasattr(client_cfg, "wiki") and client_cfg.wiki.db_path:
            config.wiki.db_path = os.path.expanduser(client_cfg.wiki.db_path)

    return config

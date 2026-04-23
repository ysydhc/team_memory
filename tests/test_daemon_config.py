"""测试 Daemon Config 模块：DaemonConfig 数据类、load_config 函数。"""

from __future__ import annotations

import os
import tempfile
from dataclasses import dataclass, fields

import pytest
import yaml


class TestDaemonSettings:
    """DaemonSettings 数据类。"""

    def test_defaults(self):
        from daemon.config import DaemonSettings

        s = DaemonSettings()
        assert s.host == "127.0.0.1"
        assert s.port == 3901

    def test_custom_values(self):
        from daemon.config import DaemonSettings

        s = DaemonSettings(host="0.0.0.0", port=8080)
        assert s.host == "0.0.0.0"
        assert s.port == 8080


class TestTMSettings:
    """TMSettings 数据类。"""

    def test_defaults(self):
        from daemon.config import TMSettings

        s = TMSettings()
        assert s.mode == "local"
        assert s.base_url == ""
        assert s.user == "daemon"

    def test_custom_values(self):
        from daemon.config import TMSettings

        s = TMSettings(mode="remote", base_url="http://tm:3900", user="test")
        assert s.mode == "remote"
        assert s.base_url == "http://tm:3900"
        assert s.user == "test"


class TestVaultConfig:
    """VaultConfig 数据类。"""

    def test_defaults(self):
        from daemon.config import VaultConfig

        v = VaultConfig()
        assert v.path == ""
        assert v.project == ""
        assert v.exclude == []

    def test_custom_values(self):
        from daemon.config import VaultConfig

        v = VaultConfig(path="/vault", project="knowledge", exclude=[".obsidian"])
        assert v.path == "/vault"
        assert v.project == "knowledge"
        assert v.exclude == [".obsidian"]


class TestObsidianSettings:
    """ObsidianSettings 数据类。"""

    def test_defaults(self):
        from daemon.config import ObsidianSettings

        s = ObsidianSettings()
        assert s.vaults == []

    def test_custom_values(self):
        from daemon.config import ObsidianSettings, VaultConfig

        vaults = [VaultConfig(path="/v1", project="p1")]
        s = ObsidianSettings(vaults=vaults)
        assert len(s.vaults) == 1
        assert s.vaults[0].path == "/v1"


class TestDraftSettings:
    """DraftSettings 数据类。"""

    def test_defaults(self):
        from daemon.config import DraftSettings

        s = DraftSettings()
        assert s.max_age_minutes == 30
        assert s.db_path == ""

    def test_custom_values(self):
        from daemon.config import DraftSettings

        s = DraftSettings(max_age_minutes=60, db_path="/tmp/drafts.db")
        assert s.max_age_minutes == 60
        assert s.db_path == "/tmp/drafts.db"


class TestRetrievalSettings:
    """RetrievalSettings 数据类。"""

    def test_defaults(self):
        from daemon.config import RetrievalSettings

        s = RetrievalSettings()
        assert s.session_start_top_k == 3
        assert s.keyword_triggers == []

    def test_custom_values(self):
        from daemon.config import RetrievalSettings

        s = RetrievalSettings(session_start_top_k=5, keyword_triggers=["之前", "经验"])
        assert s.session_start_top_k == 5
        assert s.keyword_triggers == ["之前", "经验"]


class TestProjectMapping:
    """ProjectMapping 数据类。"""

    def test_defaults(self):
        from daemon.config import ProjectMapping

        p = ProjectMapping()
        assert p.name == ""
        assert p.path_patterns == []

    def test_custom_values(self):
        from daemon.config import ProjectMapping

        p = ProjectMapping(name="team_doc", path_patterns=["team_doc", "td"])
        assert p.name == "team_doc"
        assert p.path_patterns == ["team_doc", "td"]


class TestDaemonConfig:
    """DaemonConfig 聚合数据类。"""

    def test_defaults(self):
        from daemon.config import DaemonConfig, DaemonSettings, TMSettings

        cfg = DaemonConfig()
        assert isinstance(cfg.daemon, DaemonSettings)
        assert isinstance(cfg.tm, TMSettings)
        assert cfg.daemon.host == "127.0.0.1"
        assert cfg.tm.mode == "local"

    def test_all_subconfigs_present(self):
        from daemon.config import (
            DaemonConfig,
            DaemonSettings,
            DraftSettings,
            ObsidianSettings,
            ProjectMapping,
            RetrievalSettings,
            TMSettings,
        )

        cfg = DaemonConfig()
        assert isinstance(cfg.daemon, DaemonSettings)
        assert isinstance(cfg.tm, TMSettings)
        assert isinstance(cfg.obsidian, ObsidianSettings)
        assert isinstance(cfg.draft, DraftSettings)
        assert isinstance(cfg.retrieval, RetrievalSettings)
        assert isinstance(cfg.projects, list)


class TestLoadConfig:
    """load_config 函数测试。"""

    def test_load_from_yaml_file(self):
        from daemon.config import DaemonConfig, load_config

        config_data = {
            "daemon": {"host": "0.0.0.0", "port": 8080},
            "tm": {"mode": "remote", "base_url": "http://tm:3900", "user": "test"},
            "obsidian": {
                "vaults": [
                    {"path": "/my/vault", "project": "knowledge", "exclude": [".obsidian"]}
                ]
            },
            "draft": {"max_age_minutes": 60, "db_path": "/tmp/d.db"},
            "retrieval": {"session_start_top_k": 5, "keyword_triggers": ["之前"]},
            "projects": [
                {"name": "team_doc", "path_patterns": ["team_doc"]}
            ],
        }

        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".yaml", delete=False
        ) as f:
            yaml.dump(config_data, f)
            f.flush()
            cfg = load_config(f.name)

        os.unlink(f.name)

        assert isinstance(cfg, DaemonConfig)
        assert cfg.daemon.host == "0.0.0.0"
        assert cfg.daemon.port == 8080
        assert cfg.tm.mode == "remote"
        assert cfg.tm.base_url == "http://tm:3900"
        assert cfg.tm.user == "test"
        assert len(cfg.obsidian.vaults) == 1
        assert cfg.obsidian.vaults[0].path == "/my/vault"
        assert cfg.draft.max_age_minutes == 60
        assert cfg.retrieval.session_start_top_k == 5
        assert cfg.retrieval.keyword_triggers == ["之前"]
        assert len(cfg.projects) == 1
        assert cfg.projects[0].name == "team_doc"

    def test_load_missing_file_returns_defaults(self):
        """配置文件不存在时返回默认值。"""
        from daemon.config import DaemonConfig, load_config

        cfg = load_config("/nonexistent/config.yaml")
        assert isinstance(cfg, DaemonConfig)
        assert cfg.daemon.host == "127.0.0.1"
        assert cfg.daemon.port == 3901
        assert cfg.tm.mode == "local"

    def test_load_partial_config(self):
        """部分配置只覆盖指定字段，其余用默认值。"""
        from daemon.config import load_config

        config_data = {
            "daemon": {"port": 9999},
            "tm": {"mode": "remote"},
        }

        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".yaml", delete=False
        ) as f:
            yaml.dump(config_data, f)
            f.flush()
            cfg = load_config(f.name)

        os.unlink(f.name)

        # 覆盖的部分
        assert cfg.daemon.port == 9999
        assert cfg.tm.mode == "remote"
        # 未覆盖的部分保持默认
        assert cfg.daemon.host == "127.0.0.1"
        assert cfg.tm.base_url == ""
        assert cfg.tm.user == "daemon"

    def test_load_config_none_path_uses_default(self):
        """config_path=None 时使用默认路径（可能不存在），返回默认值。"""
        from daemon.config import DaemonConfig, load_config

        cfg = load_config(None)
        assert isinstance(cfg, DaemonConfig)

    def test_load_empty_yaml(self):
        """空 YAML 文件返回全默认配置。"""
        from daemon.config import load_config

        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".yaml", delete=False
        ) as f:
            f.write("")
            f.flush()
            cfg = load_config(f.name)

        os.unlink(f.name)

        assert cfg.daemon.host == "127.0.0.1"
        assert cfg.tm.mode == "local"

"""Tests for config show/init and setup subcommand enhancements.

测试 config show 值来源标注、config init 连接测试、setup hook 写入。
"""

from __future__ import annotations

import json
from io import StringIO
from pathlib import Path
from unittest.mock import MagicMock, patch

import httpx
import pytest

# ============================================================
# Helpers
# ============================================================


def _make_mock_response(
    *,
    status_code: int = 200,
    json_data: dict | None = None,
) -> MagicMock:
    resp = MagicMock(spec=httpx.Response)
    resp.status_code = status_code
    resp.json.return_value = json_data or {}
    resp.raise_for_status = MagicMock()
    resp.text = ""
    return resp


# ============================================================
# config show — value source annotations
# ============================================================


class TestConfigShowSources:
    """Test that 'config show' annotates each field with its source."""

    @pytest.fixture(autouse=True)
    def _mock_client_config(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Mock config.toml reading so tests only use env vars."""
        monkeypatch.setattr(
            "team_memory.client_config._read_config_toml",
            lambda: {},
        )
        monkeypatch.setattr(
            "team_memory.client_config._load_dotenv",
            lambda: None,
        )

    def test_show_with_env_vars(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Config show should annotate env-sourced values."""
        monkeypatch.setenv("TEAM_MEMORY_API_KEY", "test-key-12345678")
        monkeypatch.setenv("TM_BASE_URL", "http://custom:8080")

        from team_memory.cli import main

        captured = StringIO()
        with (
            patch(
                "sys.argv",
                ["tm-cli", "config", "show"],
            ),
            patch("sys.stdout", captured),
        ):
            main()

        output = captured.getvalue()
        # 应该包含来源标注 / should contain source annotations
        assert "env:" in output or ".env file" in output or "default" in output
        # 应该显示各 section / should display sections
        assert "[server]" in output
        assert "[llm]" in output
        assert "[daemon]" in output

    def test_show_with_defaults(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Config show should annotate default values."""
        monkeypatch.delenv("TEAM_MEMORY_API_KEY", raising=False)
        monkeypatch.delenv("TM_BASE_URL", raising=False)
        monkeypatch.delenv("TM_SERVER_URL", raising=False)
        monkeypatch.delenv("TM_LLM_BASE_URL", raising=False)
        monkeypatch.delenv("TM_LLM_MODEL", raising=False)

        from team_memory.cli import main

        captured = StringIO()
        with (
            patch(
                "sys.argv",
                ["tm-cli", "config", "show"],
            ),
            patch("sys.stdout", captured),
        ):
            main()

        output = captured.getvalue()
        # 默认值应该标注为 "default" / defaults should be annotated
        assert "default" in output
        assert "localhost:9111" in output

    def test_show_with_toml_source(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Config show should annotate TOML-sourced values."""
        monkeypatch.delenv("TEAM_MEMORY_API_KEY", raising=False)
        monkeypatch.delenv("TM_BASE_URL", raising=False)
        monkeypatch.delenv("TM_SERVER_URL", raising=False)

        monkeypatch.setattr(
            "team_memory.client_config._read_config_toml",
            lambda: {"server": {"url": "http://toml-server:9111", "api_key": "toml-key-12345678"}},
        )

        from team_memory.cli import main

        captured = StringIO()
        with (
            patch(
                "sys.argv",
                ["tm-cli", "config", "show"],
            ),
            patch("sys.stdout", captured),
        ):
            main()

        output = captured.getvalue()
        assert "config.toml" in output
        assert "toml-server" in output


# ============================================================
# config init — connection test
# ============================================================


class TestConfigInitConnectionTest:
    """Test that 'config init' tests the connection after writing config."""

    @pytest.fixture(autouse=True)
    def _mock_client_config(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Mock config.toml reading."""
        monkeypatch.setattr(
            "team_memory.client_config._read_config_toml",
            lambda: {},
        )
        monkeypatch.setattr(
            "team_memory.client_config._load_dotenv",
            lambda: None,
        )

    @patch("team_memory.cli.httpx")
    def test_init_connection_success(
        self, mock_httpx: MagicMock, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """Successful connection test should show checkmarks."""
        monkeypatch.delenv("TEAM_MEMORY_API_KEY", raising=False)

        # Mock health endpoint / 模拟 health 端点
        mock_httpx.get.return_value = _make_mock_response(
            status_code=200,
            json_data={"status": "ok"},
        )
        mock_httpx.ConnectError = httpx.ConnectError
        mock_httpx.TimeoutException = httpx.TimeoutException

        # Redirect config dir to temp / 重定向配置目录到临时目录
        monkeypatch.setattr("pathlib.Path.home", lambda: tmp_path)

        from team_memory.cli import main

        captured = StringIO()
        with (
            patch(
                "sys.argv",
                [
                    "tm-cli",
                    "config",
                    "init",
                    "--server-url",
                    "http://localhost:9111",
                    "--api-key",
                    "test-key-12345678",
                    "--non-interactive",
                ],
            ),
            patch("sys.stdout", captured),
        ):
            main()

        output = captured.getvalue()
        assert "Config written" in output
        assert "Testing connection" in output

    @patch("team_memory.cli.httpx")
    def test_init_connection_refused(
        self, mock_httpx: MagicMock, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """Connection refused should show error but not exit."""
        monkeypatch.delenv("TEAM_MEMORY_API_KEY", raising=False)

        mock_httpx.get.side_effect = httpx.ConnectError("refused")
        mock_httpx.ConnectError = httpx.ConnectError
        mock_httpx.TimeoutException = httpx.TimeoutException

        monkeypatch.setattr("pathlib.Path.home", lambda: tmp_path)

        from team_memory.cli import main

        captured = StringIO()
        with (
            patch(
                "sys.argv",
                [
                    "tm-cli",
                    "config",
                    "init",
                    "--server-url",
                    "http://localhost:9111",
                    "--non-interactive",
                ],
            ),
            patch("sys.stdout", captured),
        ):
            main()

        output = captured.getvalue()
        assert "Config written" in output
        assert "Cannot connect" in output or "❌" in output

    @patch("team_memory.cli.httpx")
    def test_init_auth_test_unauthorized(
        self, mock_httpx: MagicMock, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """401 auth response should show failure message."""
        monkeypatch.delenv("TEAM_MEMORY_API_KEY", raising=False)

        # First call: unauthenticated health check
        # Second call: authenticated health check returns 401
        unauth_resp = _make_mock_response(status_code=200, json_data={"status": "ok"})
        auth_resp = _make_mock_response(status_code=401, json_data={"error": "unauthorized"})

        mock_httpx.get.side_effect = [unauth_resp, auth_resp]
        mock_httpx.ConnectError = httpx.ConnectError
        mock_httpx.TimeoutException = httpx.TimeoutException

        monkeypatch.setattr("pathlib.Path.home", lambda: tmp_path)

        from team_memory.cli import main

        captured = StringIO()
        with (
            patch(
                "sys.argv",
                [
                    "tm-cli",
                    "config",
                    "init",
                    "--server-url",
                    "http://localhost:9111",
                    "--api-key",
                    "bad-key-12345678",
                    "--non-interactive",
                ],
            ),
            patch("sys.stdout", captured),
        ):
            main()

        output = captured.getvalue()
        assert "Config written" in output
        assert "401" in output or "Unauthorized" in output or "❌" in output


# ============================================================
# setup --check — enhanced status checking
# ============================================================


class TestSetupCheckEnhanced:
    """Test enhanced --check that validates hook entries, not just file existence."""

    @pytest.fixture(autouse=True)
    def _mock_client_config(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Mock config.toml reading."""
        monkeypatch.setattr(
            "team_memory.client_config._read_config_toml",
            lambda: {},
        )
        monkeypatch.setattr(
            "team_memory.client_config._load_dotenv",
            lambda: None,
        )

    def test_check_claude_code_with_hook(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """Claude Code settings.json with hook should show configured."""
        monkeypatch.setattr("pathlib.Path.home", lambda: tmp_path)

        settings_dir = tmp_path / ".claude"
        settings_dir.mkdir()
        settings = settings_dir / "settings.json"
        settings.write_text(json.dumps({
            "hooks": {
                "PostToolUse": [
                    {
                        "matcher": "mcp__team_memory__recall",
                        "command": "python -m team_memory.hooks.claude_post_tool_use",
                    }
                ]
            }
        }))

        from team_memory.cli import main

        captured = StringIO()
        with (
            patch("sys.argv", ["tm-cli", "setup", "--platform", "claude-code", "--check"]),
            patch("sys.stdout", captured),
        ):
            main()

        output = captured.getvalue()
        assert "hook configured" in output

    def test_check_claude_code_without_hook(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """Claude Code settings.json without hook should show warning."""
        monkeypatch.setattr("pathlib.Path.home", lambda: tmp_path)

        settings_dir = tmp_path / ".claude"
        settings_dir.mkdir()
        settings = settings_dir / "settings.json"
        settings.write_text(json.dumps({"hooks": {"PostToolUse": []}}))

        from team_memory.cli import main

        captured = StringIO()
        with (
            patch("sys.argv", ["tm-cli", "setup", "--platform", "claude-code", "--check"]),
            patch("sys.stdout", captured),
        ):
            main()

        output = captured.getvalue()
        assert "not found" in output

    def test_check_cursor_with_hook(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """Cursor hooks.json with hook should show configured."""
        hooks_dir = tmp_path / ".cursor"
        hooks_dir.mkdir()
        hooks_file = hooks_dir / "hooks.json"
        hooks_file.write_text(json.dumps({
            "hooks": {
                "afterMCPExecution": [
                    {
                        "matcher": "MCP:team_memory",
                        "hooks": [
                            {
                                "type": "command",
                                "command": "python -m team_memory.hooks.cursor_after_mcp",
                            }
                        ],
                    }
                ]
            }
        }))

        # Patch cwd to tmp_path / 设置工作目录
        monkeypatch.chdir(tmp_path)

        from team_memory.cli import main

        captured = StringIO()
        with (
            patch("sys.argv", ["tm-cli", "setup", "--platform", "cursor", "--check"]),
            patch("sys.stdout", captured),
        ):
            main()

        output = captured.getvalue()
        assert "hook configured" in output

    def test_check_cursor_without_hook(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """Cursor hooks.json without TM hook should show warning."""
        hooks_dir = tmp_path / ".cursor"
        hooks_dir.mkdir()
        hooks_file = hooks_dir / "hooks.json"
        hooks_file.write_text(json.dumps({"hooks": {}}))

        monkeypatch.chdir(tmp_path)

        from team_memory.cli import main

        captured = StringIO()
        with (
            patch("sys.argv", ["tm-cli", "setup", "--platform", "cursor", "--check"]),
            patch("sys.stdout", captured),
        ):
            main()

        output = captured.getvalue()
        assert "not found" in output

    def test_check_claude_code_no_file(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """Missing Claude Code settings.json should show error with hint."""
        monkeypatch.setattr("pathlib.Path.home", lambda: tmp_path)

        from team_memory.cli import main

        captured = StringIO()
        with (
            patch("sys.argv", ["tm-cli", "setup", "--platform", "claude-code", "--check"]),
            patch("sys.stdout", captured),
        ):
            main()

        output = captured.getvalue()
        assert "not found" in output
        assert "tm-cli setup" in output


# ============================================================
# setup — actual hook writing
# ============================================================


class TestSetupWrite:
    """Test that 'setup' (without --check) actually writes hook config."""

    @pytest.fixture(autouse=True)
    def _mock_client_config(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Mock config.toml reading."""
        monkeypatch.setattr(
            "team_memory.client_config._read_config_toml",
            lambda: {},
        )
        monkeypatch.setattr(
            "team_memory.client_config._load_dotenv",
            lambda: None,
        )

    def test_setup_claude_code_writes_hook(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """Setup claude-code should write PostToolUse hook to settings.json."""
        monkeypatch.setattr("pathlib.Path.home", lambda: tmp_path)

        # Create empty settings / 创建空设置文件
        settings_dir = tmp_path / ".claude"
        settings_dir.mkdir()
        settings = settings_dir / "settings.json"
        settings.write_text("{}")

        from team_memory.cli import main

        captured = StringIO()
        with (
            patch("sys.argv", ["tm-cli", "setup", "--platform", "claude-code"]),
            patch("sys.stdout", captured),
        ):
            main()

        # Verify file was written / 验证文件已写入
        data = json.loads(settings.read_text())
        assert "hooks" in data
        assert "PostToolUse" in data["hooks"]
        post_tool = data["hooks"]["PostToolUse"]
        assert any(h["matcher"] == "mcp__team_memory__recall" for h in post_tool)

    def test_setup_claude_code_idempotent(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """Running setup twice should not duplicate the hook entry."""
        monkeypatch.setattr("pathlib.Path.home", lambda: tmp_path)

        settings_dir = tmp_path / ".claude"
        settings_dir.mkdir()
        settings = settings_dir / "settings.json"
        settings.write_text("{}")

        from team_memory.cli import main

        for _ in range(2):
            with (
                patch("sys.argv", ["tm-cli", "setup", "--platform", "claude-code"]),
                patch("sys.stdout", StringIO()),
            ):
                main()

        data = json.loads(settings.read_text())
        post_tool = data["hooks"]["PostToolUse"]
        # 应该只有一个 TM 条目 / should have exactly one TM entry
        tm_entries = [h for h in post_tool if h["matcher"] == "mcp__team_memory__recall"]
        assert len(tm_entries) == 1

    def test_setup_cursor_writes_hook(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """Setup cursor should write afterMCPExecution hook to hooks.json."""
        monkeypatch.chdir(tmp_path)

        from team_memory.cli import main

        captured = StringIO()
        with (
            patch("sys.argv", ["tm-cli", "setup", "--platform", "cursor"]),
            patch("sys.stdout", captured),
        ):
            main()

        hooks_file = tmp_path / ".cursor" / "hooks.json"
        assert hooks_file.exists()
        data = json.loads(hooks_file.read_text())
        hooks_section = data.get("hooks", {})
        if isinstance(hooks_section, dict):
            after_mcp = hooks_section.get("afterMCPExecution", [])
            has_tm = any("team_memory" in entry.get("matcher", "") for entry in after_mcp if isinstance(entry, dict))
        else:
            has_tm = False
        assert has_tm

    def test_setup_cursor_idempotent(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """Running setup cursor twice should not duplicate the hook entry."""
        monkeypatch.chdir(tmp_path)

        from team_memory.cli import main

        for _ in range(2):
            with (
                patch("sys.argv", ["tm-cli", "setup", "--platform", "cursor"]),
                patch("sys.stdout", StringIO()),
            ):
                main()

        hooks_file = tmp_path / ".cursor" / "hooks.json"
        data = json.loads(hooks_file.read_text())
        hooks_section = data.get("hooks", {})
        if isinstance(hooks_section, dict):
            after_mcp = hooks_section.get("afterMCPExecution", [])
            tm_entries = [e for e in after_mcp if isinstance(e, dict) and "team_memory" in e.get("matcher", "")]
        else:
            tm_entries = []
        assert len(tm_entries) == 1

    def test_setup_hermes_prints_instructions(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Setup hermes should print manual instructions."""
        from team_memory.cli import main

        captured = StringIO()
        with (
            patch("sys.argv", ["tm-cli", "setup", "--platform", "hermes"]),
            patch("sys.stdout", captured),
        ):
            main()

        output = captured.getvalue()
        assert "Manual setup required" in output or "hermes" in output.lower()
        assert "config.yaml" in output


# ============================================================
# load_with_sources unit tests
# ============================================================


class TestLoadWithSources:
    """Unit tests for ClientConfig.load_with_sources()."""

    @pytest.fixture(autouse=True)
    def _mock_toml_and_dotenv(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Mock config.toml reading and dotenv loading."""
        monkeypatch.setattr(
            "team_memory.client_config._read_config_toml",
            lambda: {},
        )
        monkeypatch.setattr(
            "team_memory.client_config._load_dotenv",
            lambda: None,
        )

    def test_all_defaults(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """With no env vars or TOML, all sources should be 'default'."""
        monkeypatch.delenv("TEAM_MEMORY_API_KEY", raising=False)
        monkeypatch.delenv("TM_BASE_URL", raising=False)
        monkeypatch.delenv("TM_SERVER_URL", raising=False)
        monkeypatch.delenv("TM_LLM_BASE_URL", raising=False)
        monkeypatch.delenv("TM_LLM_MODEL", raising=False)

        from team_memory.client_config import ClientConfig

        cfg, sources = ClientConfig.load_with_sources()
        assert sources["server.url"] == "default"
        assert sources["server.api_key"] == "default"
        assert sources["llm.base_url"] == "default"
        assert sources["llm.model"] == "default"
        assert cfg.server.url == "http://localhost:9111"

    def test_env_source(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Values from env vars should be annotated as 'env: ...'."""
        monkeypatch.setenv("TEAM_MEMORY_API_KEY", "env-key-12345678")
        monkeypatch.setenv("TM_BASE_URL", "http://env-server:9111")

        from team_memory.client_config import ClientConfig

        cfg, sources = ClientConfig.load_with_sources()
        assert "env:" in sources["server.url"]
        assert "TM_BASE_URL" in sources["server.url"] or "TM_SERVER_URL" in sources["server.url"]
        assert "env:" in sources["server.api_key"]
        assert cfg.server.url == "http://env-server:9111"

    def test_toml_source(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Values from config.toml should be annotated as 'config.toml'."""
        monkeypatch.delenv("TEAM_MEMORY_API_KEY", raising=False)
        monkeypatch.delenv("TM_BASE_URL", raising=False)
        monkeypatch.delenv("TM_SERVER_URL", raising=False)

        monkeypatch.setattr(
            "team_memory.client_config._read_config_toml",
            lambda: {
                "server": {"url": "http://toml:9111", "api_key": "toml-key-12345678"},
                "llm": {"base_url": "http://llm-toml:4000/v1"},
            },
        )

        from team_memory.client_config import ClientConfig

        cfg, sources = ClientConfig.load_with_sources()
        assert sources["server.url"] == "config.toml"
        assert sources["server.api_key"] == "config.toml"
        assert sources["llm.base_url"] == "config.toml"
        assert cfg.server.url == "http://toml:9111"

    def test_env_overrides_toml(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Env vars should take priority over TOML values."""
        monkeypatch.setenv("TM_BASE_URL", "http://env:9111")

        monkeypatch.setattr(
            "team_memory.client_config._read_config_toml",
            lambda: {"server": {"url": "http://toml:9111"}},
        )

        from team_memory.client_config import ClientConfig

        cfg, sources = ClientConfig.load_with_sources()
        assert cfg.server.url == "http://env:9111"
        assert "env:" in sources["server.url"]

"""Tests for Phase 4 CLI subcommands: config check, config set, project init/show, setup --uninstall.

测试 Phase 4 CLI 子命令：config check、config set、project init/show、setup --uninstall。
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


def _mock_client_config(monkeypatch: pytest.MonkeyPatch) -> None:
    """Mock config.toml reading so tests only use env vars."""
    monkeypatch.setattr(
        "team_memory.client_config._read_config_toml",
        lambda: {},
    )
    monkeypatch.setattr(
        "team_memory.client_config._load_dotenv",
        lambda: None,
    )


# ============================================================
# config check — connectivity tests
# ============================================================


class TestConfigCheck:
    """Test 'config check' subcommand / 测试 config check 子命令。"""

    @pytest.fixture(autouse=True)
    def _setup(self, monkeypatch: pytest.MonkeyPatch) -> None:
        _mock_client_config(monkeypatch)

    @patch("team_memory.cli.httpx")
    def test_check_all_pass(
        self, mock_httpx: MagicMock, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """All connectivity checks pass / 所有连通性检查通过。"""
        monkeypatch.delenv("TEAM_MEMORY_API_KEY", raising=False)

        # Server health returns 200
        server_resp = _make_mock_response(status_code=200, json_data={"status": "ok"})
        # LLM returns 200
        llm_resp = _make_mock_response(status_code=200, json_data={"choices": []})
        # Ollama returns 200
        ollama_resp = _make_mock_response(
            status_code=200,
            json_data={"models": [{"name": "nomic-embed-text"}]},
        )

        mock_httpx.get.side_effect = [server_resp, ollama_resp]
        mock_httpx.post.return_value = llm_resp
        mock_httpx.ConnectError = httpx.ConnectError
        mock_httpx.TimeoutException = httpx.TimeoutException

        from team_memory.cli import main

        captured = StringIO()
        with (
            patch("sys.argv", ["tm-cli", "config", "check"]),
            patch("sys.stdout", captured),
        ):
            main()

        output = captured.getvalue()
        assert "Server reachable" in output
        assert "LLM reachable" in output
        assert "Ollama reachable" in output
        assert "All critical checks passed" in output

    @patch("team_memory.cli.httpx")
    def test_check_server_fails(
        self, mock_httpx: MagicMock, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Server unreachable causes exit code 1 / 服务器不可达导致退出码 1。"""
        monkeypatch.delenv("TEAM_MEMORY_API_KEY", raising=False)

        mock_httpx.get.side_effect = httpx.ConnectError("refused")
        mock_httpx.post.side_effect = httpx.ConnectError("refused")
        mock_httpx.ConnectError = httpx.ConnectError
        mock_httpx.TimeoutException = httpx.TimeoutException

        from team_memory.cli import main

        captured = StringIO()
        with (
            patch("sys.argv", ["tm-cli", "config", "check"]),
            patch("sys.stdout", captured),
            pytest.raises(SystemExit) as exc_info,
        ):
            main()

        assert exc_info.value.code == 1
        output = captured.getvalue()
        assert "Cannot connect" in output
        assert "Some critical checks failed" in output

    @patch("team_memory.cli.httpx")
    def test_check_llm_non_critical(
        self, mock_httpx: MagicMock, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """LLM failure is non-critical (no exit) / LLM 失败是非关键的（不退出）。"""
        monkeypatch.delenv("TEAM_MEMORY_API_KEY", raising=False)

        server_resp = _make_mock_response(status_code=200, json_data={"status": "ok"})
        ollama_resp = _make_mock_response(status_code=200, json_data={"models": []})

        mock_httpx.get.side_effect = [server_resp, ollama_resp]
        mock_httpx.post.side_effect = httpx.ConnectError("refused")
        mock_httpx.ConnectError = httpx.ConnectError
        mock_httpx.TimeoutException = httpx.TimeoutException

        from team_memory.cli import main

        captured = StringIO()
        with (
            patch("sys.argv", ["tm-cli", "config", "check"]),
            patch("sys.stdout", captured),
        ):
            main()

        output = captured.getvalue()
        assert "Cannot connect to LLM" in output
        # Should not exit 1 since server is OK
        assert "All critical checks passed" in output

    @patch("team_memory.cli.httpx")
    def test_check_ollama_non_critical(
        self, mock_httpx: MagicMock, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Ollama failure is non-critical / Ollama 失败是非关键的。"""
        monkeypatch.delenv("TEAM_MEMORY_API_KEY", raising=False)

        server_resp = _make_mock_response(status_code=200, json_data={"status": "ok"})
        llm_resp = _make_mock_response(status_code=200, json_data={"choices": []})

        mock_httpx.get.side_effect = [server_resp, httpx.ConnectError("refused")]
        mock_httpx.post.return_value = llm_resp
        mock_httpx.ConnectError = httpx.ConnectError
        mock_httpx.TimeoutException = httpx.TimeoutException

        from team_memory.cli import main

        captured = StringIO()
        with (
            patch("sys.argv", ["tm-cli", "config", "check"]),
            patch("sys.stdout", captured),
        ):
            main()

        output = captured.getvalue()
        assert "Ollama" in output
        assert "not critical" in output
        assert "All critical checks passed" in output


# ============================================================
# config set — write to config.toml
# ============================================================


class TestConfigSet:
    """Test 'config set KEY VALUE' subcommand / 测试 config set 子命令。"""

    @pytest.fixture(autouse=True)
    def _setup(self, monkeypatch: pytest.MonkeyPatch) -> None:
        _mock_client_config(monkeypatch)

    def test_set_string_value(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """Setting a string value writes to config.toml / 设置字符串值写入 config.toml。"""
        monkeypatch.setattr("pathlib.Path.home", lambda: tmp_path)

        from team_memory.cli import main

        with (
            patch("sys.argv", ["tm-cli", "config", "set", "server.url", "http://myserver:9111"]),
            patch("sys.stdout", StringIO()),
        ):
            main()

        config_file = tmp_path / ".config" / "tm" / "config.toml"
        assert config_file.exists()

        # Read back and verify / 读回并验证
        try:
            import tomllib
        except ModuleNotFoundError:
            import tomli as tomllib  # type: ignore[no-redef]

        with open(config_file, "rb") as f:
            data = tomllib.load(f)
        assert data["server"]["url"] == "http://myserver:9111"

    def test_set_integer_value(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """Setting an integer value converts type / 设置整数值会转换类型。"""
        monkeypatch.setattr("pathlib.Path.home", lambda: tmp_path)

        from team_memory.cli import main

        with (
            patch("sys.argv", ["tm-cli", "config", "set", "llm.timeout", "60"]),
            patch("sys.stdout", StringIO()),
        ):
            main()

        config_file = tmp_path / ".config" / "tm" / "config.toml"
        try:
            import tomllib
        except ModuleNotFoundError:
            import tomli as tomllib  # type: ignore[no-redef]

        with open(config_file, "rb") as f:
            data = tomllib.load(f)
        assert data["llm"]["timeout"] == 60

    def test_set_boolean_value(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """Setting a boolean value converts type / 设置布尔值会转换类型。"""
        monkeypatch.setattr("pathlib.Path.home", lambda: tmp_path)

        from team_memory.cli import main

        with (
            patch("sys.argv", ["tm-cli", "config", "set", "wiki.enabled", "true"]),
            patch("sys.stdout", StringIO()),
        ):
            main()

        config_file = tmp_path / ".config" / "tm" / "config.toml"
        try:
            import tomllib
        except ModuleNotFoundError:
            import tomli as tomllib  # type: ignore[no-redef]

        with open(config_file, "rb") as f:
            data = tomllib.load(f)
        assert data["wiki"]["enabled"] is True

    def test_set_preserves_existing(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """Setting a new key preserves existing config / 设置新键保留现有配置。"""
        monkeypatch.setattr("pathlib.Path.home", lambda: tmp_path)

        # Create initial config / 创建初始配置
        config_dir = tmp_path / ".config" / "tm"
        config_dir.mkdir(parents=True)
        config_file = config_dir / "config.toml"
        import tomli_w
        with open(config_file, "wb") as f:
            tomli_w.dump({"server": {"url": "http://old:9111"}}, f)

        from team_memory.cli import main

        with (
            patch("sys.argv", ["tm-cli", "config", "set", "server.api_key", "mykey123"]),
            patch("sys.stdout", StringIO()),
        ):
            main()

        try:
            import tomllib
        except ModuleNotFoundError:
            import tomli as tomllib  # type: ignore[no-redef]

        with open(config_file, "rb") as f:
            data = tomllib.load(f)
        # Old value preserved / 旧值保留
        assert data["server"]["url"] == "http://old:9111"
        # New value added / 新值添加
        assert data["server"]["api_key"] == "mykey123"

    def test_set_invalid_key_format(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """Key without dot notation causes error / 无点号表示法的键导致错误。"""
        monkeypatch.setattr("pathlib.Path.home", lambda: tmp_path)

        from team_memory.cli import main

        with (
            patch("sys.argv", ["tm-cli", "config", "set", "invalidkey", "value"]),
            pytest.raises(SystemExit) as exc_info,
        ):
            main()

        assert exc_info.value.code == 1

    def test_set_creates_config_if_missing(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """Config set creates config.toml if it doesn't exist / 配置 set 在文件不存在时创建它。"""
        monkeypatch.setattr("pathlib.Path.home", lambda: tmp_path)

        from team_memory.cli import main

        with (
            patch("sys.argv", ["tm-cli", "config", "set", "llm.model", "gpt-4"]),
            patch("sys.stdout", StringIO()),
        ):
            main()

        config_file = tmp_path / ".config" / "tm" / "config.toml"
        assert config_file.exists()

        try:
            import tomllib
        except ModuleNotFoundError:
            import tomli as tomllib  # type: ignore[no-redef]

        with open(config_file, "rb") as f:
            data = tomllib.load(f)
        assert data["llm"]["model"] == "gpt-4"


# ============================================================
# project init — create .tm.toml
# ============================================================


class TestProjectInit:
    """Test 'project init' subcommand / 测试 project init 子命令。"""

    @pytest.fixture(autouse=True)
    def _setup(self, monkeypatch: pytest.MonkeyPatch) -> None:
        _mock_client_config(monkeypatch)

    def test_init_creates_tm_toml(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """project init creates .tm.toml with detected name / project init 创建 .tm.toml。"""
        monkeypatch.chdir(tmp_path)

        # Mock git detection / 模拟 git 检测
        with patch("team_memory.cli._detect_project_name", return_value="my_project"):
            from team_memory.cli import main

            captured = StringIO()
            with (
                patch("sys.argv", ["tm-cli", "project", "init"]),
                patch("sys.stdout", captured),
            ):
                main()

        tm_file = tmp_path / ".tm.toml"
        assert tm_file.exists()

        try:
            import tomllib
        except ModuleNotFoundError:
            import tomli as tomllib  # type: ignore[no-redef]

        with open(tm_file, "rb") as f:
            data = tomllib.load(f)
        assert data["project"]["name"] == "my_project"
        assert data["project"]["aliases"] == []
        assert "Created .tm.toml" in captured.getvalue()

    def test_init_refuses_if_exists(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """project init refuses if .tm.toml already exists / .tm.toml 已存在时拒绝。"""
        monkeypatch.chdir(tmp_path)

        # Create existing .tm.toml / 创建已有的 .tm.toml
        (tmp_path / ".tm.toml").write_text('[project]\nname = "existing"\n')

        from team_memory.cli import main

        with (
            patch("sys.argv", ["tm-cli", "project", "init"]),
            pytest.raises(SystemExit) as exc_info,
        ):
            main()

        assert exc_info.value.code == 1

    def test_detect_project_name_from_git(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """_detect_project_name uses git repo name / _detect_project_name 使用 git 仓库名。"""
        monkeypatch.chdir(tmp_path)

        mock_result = MagicMock()
        mock_result.returncode = 0
        mock_result.stdout = "/path/to/my-awesome-project\n"

        with patch("subprocess.run", return_value=mock_result):
            from team_memory.cli import _detect_project_name
            name = _detect_project_name()
        assert name == "my_awesome_project"  # kebab → snake

    def test_detect_project_name_fallback_dir(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """_detect_project_name falls back to directory name / 回退到目录名。"""
        monkeypatch.chdir(tmp_path)

        with patch("subprocess.run", side_effect=FileNotFoundError("no git")):
            from team_memory.cli import _detect_project_name
            name = _detect_project_name()
        # Should be the tmp_path basename / 应该是 tmp_path 的基本名
        assert isinstance(name, str) and len(name) > 0


# ============================================================
# project show — display project info
# ============================================================


class TestProjectShow:
    """Test 'project show' subcommand / 测试 project show 子命令。"""

    @pytest.fixture(autouse=True)
    def _setup(self, monkeypatch: pytest.MonkeyPatch) -> None:
        _mock_client_config(monkeypatch)

    def test_show_with_tm_toml(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """project show displays info from .tm.toml / 显示 .tm.toml 中的信息。"""
        monkeypatch.chdir(tmp_path)

        # Create .tm.toml / 创建 .tm.toml
        import tomli_w
        tm_data = {"project": {"name": "test_project", "aliases": ["old_name"]}}
        with open(tmp_path / ".tm.toml", "wb") as _f:
            tomli_w.dump(tm_data, _f)

        from team_memory.cli import main

        captured = StringIO()
        with (
            patch("sys.argv", ["tm-cli", "project", "show"]),
            patch("sys.stdout", captured),
        ):
            main()

        output = captured.getvalue()
        assert "test_project" in output
        assert "old_name" in output
        assert ".tm.toml" in output

    def test_show_without_tm_toml(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """project show works without .tm.toml / 无 .tm.toml 也能工作。"""
        monkeypatch.chdir(tmp_path)

        from team_memory.cli import main

        captured = StringIO()
        with (
            patch("sys.argv", ["tm-cli", "project", "show"]),
            patch("sys.stdout", captured),
        ):
            main()

        output = captured.getvalue()
        assert "not found" in output
        assert "Inferred name" in output

    def test_show_displays_git_repo_name(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """project show displays git repo name / 显示 git 仓库名。"""
        monkeypatch.chdir(tmp_path)

        from team_memory.cli import main

        captured = StringIO()
        with (
            patch("sys.argv", ["tm-cli", "project", "show"]),
            patch("sys.stdout", captured),
            patch("team_memory.project_resolver._git_repo_name", return_value="my_repo"),
        ):
            main()

        output = captured.getvalue()
        assert "my_repo" in output


# ============================================================
# setup --uninstall — remove hooks
# ============================================================


class TestSetupUninstall:
    """Test 'setup --uninstall' removes TM hooks / 测试 setup --uninstall 移除 TM hooks。"""

    @pytest.fixture(autouse=True)
    def _setup(self, monkeypatch: pytest.MonkeyPatch) -> None:
        _mock_client_config(monkeypatch)

    def test_uninstall_claude_code(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """Uninstall removes TM hooks from Claude Code settings / 从 Claude Code 设置移除 TM hooks。"""
        monkeypatch.setattr("pathlib.Path.home", lambda: tmp_path)

        # Create settings with TM hook / 创建包含 TM hook 的设置
        settings_dir = tmp_path / ".claude"
        settings_dir.mkdir()
        settings = settings_dir / "settings.json"
        settings.write_text(json.dumps({
            "hooks": {
                "PostToolUse": [
                    {"matcher": "mcp__team_memory__recall", "command": "python hook.py"},
                    {"matcher": "other_hook", "command": "other.py"},
                ]
            }
        }))

        from team_memory.cli import main

        captured = StringIO()
        with (
            patch("sys.argv", ["tm-cli", "setup", "--platform", "claude-code", "--uninstall"]),
            patch("sys.stdout", captured),
        ):
            main()

        # Verify TM hook removed / 验证 TM hook 已移除
        data = json.loads(settings.read_text())
        post_tool = data["hooks"]["PostToolUse"]
        assert not any(h["matcher"] == "mcp__team_memory__recall" for h in post_tool)
        assert any(h["matcher"] == "other_hook" for h in post_tool)
        assert "Removed TM hooks" in captured.getvalue()

    def test_uninstall_cursor(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """Uninstall removes TM hooks from Cursor hooks.json / 从 Cursor hooks.json 移除 TM hooks。"""
        monkeypatch.chdir(tmp_path)

        # Create hooks.json with TM hook / 创建包含 TM hook 的 hooks.json
        hooks_dir = tmp_path / ".cursor"
        hooks_dir.mkdir()
        hooks_file = hooks_dir / "hooks.json"
        hooks_file.write_text(json.dumps({
            "hooks": {
                "afterMCPExecution": [
                    {"matcher": "MCP:team_memory", "hooks": [{"type": "command", "command": "hook.py"}]},
                    {"matcher": "MCP:other", "hooks": [{"type": "command", "command": "other.py"}]},
                ]
            }
        }))

        from team_memory.cli import main

        captured = StringIO()
        with (
            patch("sys.argv", ["tm-cli", "setup", "--platform", "cursor", "--uninstall"]),
            patch("sys.stdout", captured),
        ):
            main()

        data = json.loads(hooks_file.read_text())
        after_mcp = data["hooks"]["afterMCPExecution"]
        assert not any("team_memory" in e.get("matcher", "") for e in after_mcp)
        assert any("other" in e.get("matcher", "") for e in after_mcp)
        assert "Removed TM hooks" in captured.getvalue()

    def test_uninstall_hermes(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Uninstall hermes prints manual instructions / Hermes 卸载打印手动说明。"""
        from team_memory.cli import main

        captured = StringIO()
        with (
            patch("sys.argv", ["tm-cli", "setup", "--platform", "hermes", "--uninstall"]),
            patch("sys.stdout", captured),
        ):
            main()

        output = captured.getvalue()
        assert "Manual uninstall" in output
        assert "config.yaml" in output

    def test_uninstall_no_hooks_found(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """Uninstall with no TM hooks shows warning / 无 TM hooks 时显示警告。"""
        monkeypatch.setattr("pathlib.Path.home", lambda: tmp_path)

        settings_dir = tmp_path / ".claude"
        settings_dir.mkdir()
        settings = settings_dir / "settings.json"
        settings.write_text(json.dumps({
            "hooks": {"PostToolUse": [{"matcher": "other_hook", "command": "other.py"}]}
        }))

        from team_memory.cli import main

        captured = StringIO()
        with (
            patch("sys.argv", ["tm-cli", "setup", "--platform", "claude-code", "--uninstall"]),
            patch("sys.stdout", captured),
        ):
            main()

        output = captured.getvalue()
        assert "No TM hooks found" in output

    def test_uninstall_no_settings_file(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """Uninstall with no settings file shows warning / 无设置文件时显示警告。"""
        monkeypatch.setattr("pathlib.Path.home", lambda: tmp_path)

        from team_memory.cli import main

        captured = StringIO()
        with (
            patch("sys.argv", ["tm-cli", "setup", "--platform", "claude-code", "--uninstall"]),
            patch("sys.stdout", captured),
        ):
            main()

        output = captured.getvalue()
        assert "No settings.json found" in output

    def test_uninstall_all_platforms(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """Uninstall --platform all removes from all platforms / 全平台卸载。"""
        monkeypatch.setattr("pathlib.Path.home", lambda: tmp_path)
        monkeypatch.chdir(tmp_path)

        # Create both configs / 创建两个配置
        settings_dir = tmp_path / ".claude"
        settings_dir.mkdir()
        settings = settings_dir / "settings.json"
        settings.write_text(json.dumps({
            "hooks": {"PostToolUse": [{"matcher": "mcp__team_memory__recall", "command": "hook.py"}]}
        }))

        hooks_dir = tmp_path / ".cursor"
        hooks_dir.mkdir()
        hooks_file = hooks_dir / "hooks.json"
        hooks_file.write_text(json.dumps({
            "hooks": {"afterMCPExecution": [{"matcher": "MCP:team_memory", "hooks": []}]}
        }))

        from team_memory.cli import main

        captured = StringIO()
        with (
            patch("sys.argv", ["tm-cli", "setup", "--platform", "all", "--uninstall"]),
            patch("sys.stdout", captured),
        ):
            main()

        output = captured.getvalue()
        # Both platforms should be processed / 两个平台都应处理
        assert "[claude-code]" in output
        assert "[cursor]" in output
        assert "[hermes]" in output


# ============================================================
# setup (no --platform) — interactive wizard
# ============================================================


class TestSetupWizard:
    """Test setup without --platform runs wizard / 测试无 --platform 运行向导。"""

    @pytest.fixture(autouse=True)
    def _setup(self, monkeypatch: pytest.MonkeyPatch) -> None:
        _mock_client_config(monkeypatch)

    @patch("team_memory.cli.httpx")
    def test_wizard_non_interactive(
        self, mock_httpx: MagicMock, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """Non-interactive wizard runs config check + setup all platforms / 非交互式向导。"""
        monkeypatch.setattr("pathlib.Path.home", lambda: tmp_path)
        monkeypatch.delenv("TEAM_MEMORY_API_KEY", raising=False)

        server_resp = _make_mock_response(status_code=200, json_data={"status": "ok"})
        llm_resp = _make_mock_response(status_code=200, json_data={"choices": []})
        ollama_resp = _make_mock_response(status_code=200, json_data={"models": []})

        mock_httpx.get.side_effect = [server_resp, ollama_resp]
        mock_httpx.post.return_value = llm_resp
        mock_httpx.ConnectError = httpx.ConnectError
        mock_httpx.TimeoutException = httpx.TimeoutException

        from team_memory.cli import main

        captured = StringIO()
        with (
            patch("sys.argv", ["tm-cli", "setup", "--non-interactive"]),
            patch("sys.stdout", captured),
        ):
            main()

        output = captured.getvalue()
        # Should have run connectivity check / 应该运行了连通性检查
        assert "connectivity" in output.lower() or "Server" in output or "check" in output.lower()


# ============================================================
# T1-T2: _setup_cursor tests / 测试
# ============================================================

class TestSetupCursor:
    def test_setup_cursor_writes_all_four_hooks(self, tmp_path):
        """Verify _setup_cursor writes all 4 hook entries, not just afterMCPExecution."""
        from team_memory.cli import _setup_cursor

        # Set up temp .cursor directory
        cursor_dir = tmp_path / ".cursor"
        cursor_dir.mkdir()

        # Use tmp_path as cwd during setup
        import os
        old_cwd = os.getcwd()
        os.chdir(tmp_path)

        try:
            _setup_cursor(cfg=None, venv_python="/usr/bin/python3")

            # Read back hooks.json
            import json
            hooks_file = cursor_dir / "hooks.json"
            assert hooks_file.exists(), f"hooks.json not created at {hooks_file}"

            with open(hooks_file) as f:
                data = json.load(f)

            hooks_section = data.get("hooks", {})
            assert isinstance(hooks_section, dict), f"Expected dict, got {type(hooks_section)}"

            # Check all 4 hook events
            expected = ["sessionStart", "beforeSubmitPrompt", "afterAgentResponse", "afterMCPExecution"]
            for event in expected:
                entries = hooks_section.get(event, [])
                error_msg = f"Missing hook event '{event}' in hooks.json. Got: {list(hooks_section.keys())}"
                assert len(entries) > 0, error_msg
                assert isinstance(entries[0], dict), f"Entry for {event} is not dict"
                hooks_list = entries[0].get("hooks", [])
                assert len(hooks_list) > 0, f"No hooks in {event} entry"

        finally:
            os.chdir(old_cwd)

    def test_setup_cursor_no_duplicates_on_rerun(self, tmp_path):
        """Running setup twice should not create duplicate entries."""
        from team_memory.cli import _setup_cursor

        cursor_dir = tmp_path / ".cursor"
        cursor_dir.mkdir()

        import json
        import os
        old_cwd = os.getcwd()
        os.chdir(tmp_path)

        try:
            # Run setup twice
            _setup_cursor(cfg=None, venv_python="/usr/bin/python3")
            _setup_cursor(cfg=None, venv_python="/usr/bin/python3")

            # Read back
            with open(cursor_dir / "hooks.json") as f:
                data = json.load(f)

            # Each hook event should have exactly 1 entry
            for event in ["sessionStart", "beforeSubmitPrompt", "afterAgentResponse", "afterMCPExecution"]:
                entries = data["hooks"].get(event, [])
                assert len(entries) == 1, f"Expected 1 entry for {event}, got {len(entries)}: {entries}"

        finally:
            os.chdir(old_cwd)


# ============================================================
# T3-T4: _setup_hermes tests / 测试
# ============================================================

class TestSetupHermes:
    def test_setup_hermes_writes_yaml_hooks(self, tmp_path):
        """Verify _setup_hermes auto-writes config.yaml instead of just printing."""
        from team_memory.cli import _setup_hermes

        # Set up fake ~/.hermes directory
        hermes_dir = tmp_path / ".hermes"
        hermes_dir.mkdir()

        import yaml

        # Create existing config with some unrelated setting
        existing_config = {
            "model": {"default": "claude-sonnet-4.6"},
            "agent": {"max_turns": 90},
        }
        config_path = hermes_dir / "config.yaml"
        with open(config_path, "w") as f:
            yaml.dump(existing_config, f)

        # Mock Path.home() to return tmp_path
        from unittest.mock import patch

        with patch("pathlib.Path.home", return_value=tmp_path):
            _setup_hermes(cfg=None, venv_python="/usr/bin/python3")

        # Read back config.yaml
        with open(config_path) as f:
            result = yaml.safe_load(f)

        # Verify hooks section exists with all 3 hooks
        hooks = result.get("hooks", {})
        expected_hooks = ["UserPromptSubmit", "Stop", "PostToolUse"]
        for hook in expected_hooks:
            assert hook in hooks, f"Missing hook '{hook}' in config.yaml. Got: {list(hooks.keys())}"
            entries = hooks[hook]
            assert len(entries) > 0, f"No entries for hook '{hook}'"

        # Verify non-hooks config preserved
        assert result.get("model", {}).get("default") == "claude-sonnet-4.6"
        assert result.get("agent", {}).get("max_turns") == 90

    def test_setup_hermes_preserves_formatting(self, tmp_path):
        """Non-TM hooks in config.yaml should not be overwritten."""
        from team_memory.cli import _setup_hermes

        hermes_dir = tmp_path / ".hermes"
        hermes_dir.mkdir()

        import yaml

        original = {
            "model": {"default": "test-model"},
            "agent": {"max_turns": 42},
            "hooks": {
                "SomeOtherHook": [],
            },
        }
        config_path = hermes_dir / "config.yaml"
        with open(config_path, "w") as f:
            yaml.dump(original, f)

        from unittest.mock import patch
        with patch("pathlib.Path.home", return_value=tmp_path):
            _setup_hermes(cfg=None, venv_python="/usr/bin/python3")

        with open(config_path) as f:
            result = yaml.safe_load(f)

        # TM hooks added
        assert "UserPromptSubmit" in result.get("hooks", {})
        # Non-TM hooks preserved
        assert "SomeOtherHook" in result.get("hooks", {}), "Non-TM hook was removed!"

    def test_setup_hermes_no_duplicates(self, tmp_path):
        """Running setup twice should not duplicate hook entries."""
        from team_memory.cli import _setup_hermes

        hermes_dir = tmp_path / ".hermes"
        hermes_dir.mkdir()
        import yaml

        config_path = hermes_dir / "config.yaml"
        config_path.write_text("{}")

        from unittest.mock import patch
        with patch("pathlib.Path.home", return_value=tmp_path):
            _setup_hermes(cfg=None, venv_python="/usr/bin/python3")
            _setup_hermes(cfg=None, venv_python="/usr/bin/python3")

        with open(config_path) as f:
            result = yaml.safe_load(f)

        assert len(result["hooks"]["UserPromptSubmit"]) == 1, "Duplicate UserPromptSubmit!"
        assert len(result["hooks"]["Stop"]) == 1, "Duplicate Stop!"

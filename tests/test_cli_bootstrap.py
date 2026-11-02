"""Unit tests for source-free installation & bootstrap shell capabilities in cli.py."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from team_memory import cli, client_config


def test_resolve_alembic_paths_finds_repo_migrations() -> None:
    """_resolve_alembic_paths locates migrations/ and alembic.ini in repo or bundled dir."""
    ini_path, mig_dir = cli._resolve_alembic_paths()
    assert mig_dir.is_dir()
    assert (mig_dir / "env.py").is_file()
    assert ini_path.is_file()


def test_daemon_plist_template_exists_and_handles_missing_dotenv() -> None:
    """Bundled com.teammemory.daemon.plist loads ~/.config/tm/.env and replaces placeholders."""
    plist = Path(cli.__file__).resolve().parent / "daemon" / "com.teammemory.daemon.plist"
    assert plist.is_file()
    content = plist.read_text(encoding="utf-8")
    assert "__PYTHON_PATH__" in content
    assert "__REPO_ROOT__" in content
    assert "$HOME/.config/tm/.env" in content


def test_ensure_unified_api_key_writes_config_and_env(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """_ensure_unified_api_key writes matching API key to config.toml and .env."""
    cfg_dir = tmp_path / ".config" / "tm"
    cfg_file = cfg_dir / "config.toml"
    monkeypatch.setattr(client_config, "_CONFIG_DIR", cfg_dir)
    monkeypatch.setattr(client_config, "_CONFIG_FILE", cfg_file)
    monkeypatch.delenv("TEAM_MEMORY_API_KEY", raising=False)

    api_key = cli._ensure_unified_api_key(explicit_key="tm_test_unified_key")

    assert api_key == "tm_test_unified_key"
    assert cfg_file.is_file()
    env_file = cfg_dir / ".env"
    assert env_file.is_file()
    assert "tm_test_unified_key" in cfg_file.read_text(encoding="utf-8")
    assert "TEAM_MEMORY_API_KEY=tm_test_unified_key" in env_file.read_text(encoding="utf-8")


def test_run_db_migrations_invokes_alembic_upgrade() -> None:
    """run_db_migrations configures Alembic script_location and invokes command.upgrade."""
    with patch("alembic.command.upgrade") as mock_upgrade:
        cli.run_db_migrations("postgresql+asyncpg://developer:devpass@localhost:5433/team_memory")
        assert mock_upgrade.called
        cfg_arg, rev_arg = mock_upgrade.call_args[0]
        assert rev_arg == "head"
        script_loc = cfg_arg.get_main_option("script_location")
        assert script_loc and Path(script_loc).is_dir()


def test_cli_help_includes_daemon_server_and_all_in_one(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """CLI parser registers daemon, server, and setup --all-in-one."""
    monkeypatch.setattr("sys.argv", ["tm-cli", "setup", "--help"])
    with pytest.raises(SystemExit) as exc:
        cli.main()
    assert exc.value.code == 0
    out = capsys.readouterr().out
    assert "--all-in-one" in out

    monkeypatch.setattr("sys.argv", ["tm-cli", "daemon", "--help"])
    with pytest.raises(SystemExit) as exc2:
        cli.main()
    assert exc2.value.code == 0
    out2 = capsys.readouterr().out
    for subcmd in ("install", "uninstall", "start", "stop", "status"):
        assert subcmd in out2

    monkeypatch.setattr("sys.argv", ["tm-cli", "server", "--help"])
    with pytest.raises(SystemExit) as exc3:
        cli.main()
    assert exc3.value.code == 0
    out3 = capsys.readouterr().out
    for subcmd in ("up", "down", "migrate"):
        assert subcmd in out3


def test_cmd_setup_all_in_one_delegates_to_server_up_and_platform_setup(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """cmd_setup with --all-in-one calls cmd_server_up, cmd_daemon_install, and _setup_platform."""
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    mock_up = MagicMock()
    mock_daemon = MagicMock()
    mock_setup_plat = MagicMock()
    monkeypatch.setattr(cli, "cmd_server_up", mock_up)
    monkeypatch.setattr(cli, "cmd_daemon_install", mock_daemon)
    monkeypatch.setattr(cli, "_setup_platform", mock_setup_plat)

    args = MagicMock()
    args.check = False
    args.uninstall = False
    args.all_in_one = True
    args.platform = "cursor"
    args.scope = "global"
    args.target = "all"
    args.project_dir = None
    args.project = None
    args.non_interactive = True
    args.server_url = "http://localhost:9111"
    args.api_key = "tm_test_key"

    cli.cmd_setup(args)
    assert mock_up.called
    assert mock_daemon.called
    assert mock_setup_plat.called

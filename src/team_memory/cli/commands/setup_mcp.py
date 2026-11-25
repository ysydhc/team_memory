"""MCP server configuration and setup wizard commands for tm-cli."""

from __future__ import annotations

import argparse
import json
import sys
import tomllib
from pathlib import Path
from typing import Any

from team_memory.cli.commands.config_project import cmd_config_check
from team_memory.cli.http_client import _get_repo_root

ALL_SETUP_PLATFORMS = ["claude-code", "cursor", "antigravity", "windsurf", "codex", "hermes"]


def _resolve_effective_scope(platform: str, scope: str | None) -> str:
    """Resolve 'default'/None scope to platform default ('project' for cursor, 'global' for others)."""
    if scope in ("global", "project"):
        return scope
    return "project" if platform == "cursor" else "global"


def _resolve_mcp_config_path(platform: str, scope: str, project_dir: Path) -> Path | None:
    """Return the MCP config file path for a given platform and scope ('global' or 'project')."""
    eff = _resolve_effective_scope(platform, scope)
    if platform == "claude-code":
        return (Path.home() / ".claude.json") if eff == "global" else (project_dir / ".mcp.json")
    if platform == "cursor":
        return (Path.home() / ".cursor" / "mcp.json") if eff == "global" else (project_dir / ".cursor" / "mcp.json")
    if platform == "antigravity":
        return (
            (Path.home() / ".gemini" / "config" / "mcp_config.json")
            if eff == "global"
            else (project_dir / ".agents" / "mcp_config.json")
        )
    if platform == "windsurf":
        return (
            (Path.home() / ".codeium" / "windsurf" / "mcp_config.json")
            if eff == "global"
            else (project_dir / ".windsurf" / "mcp.json")
        )
    if platform == "codex":
        return (
            (Path.home() / ".codex" / "config.toml")
            if eff == "global"
            else (project_dir / ".codex" / "config.toml")
        )
    return None


def _build_mcp_server_entry(
    scope: str,
    project_dir: Path,
    project_override: str | None = None,
) -> dict[str, Any]:
    """Build the team_memory MCP server configuration entry."""
    repo_root = _get_repo_root()
    runner_script = repo_root / "scripts" / "run_mcp_with_dotenv.sh"
    if runner_script.exists():
        entry: dict[str, Any] = {
            "command": "/bin/bash",
            "args": [str(runner_script)],
        }
    else:
        entry = {
            "command": sys.executable,
            "args": ["-m", "team_memory.server"],
        }

    if scope == "project" or project_override:
        from team_memory.project_resolver import infer_project_name

        proj_name = project_override or infer_project_name(str(project_dir))
        if proj_name:
            entry["env"] = {"TEAM_MEMORY_PROJECT": proj_name}

    return entry


def _setup_mcp_for_platform(
    platform: str,
    scope: str = "default",
    project_dir: Path | None = None,
    project_override: str | None = None,
) -> None:
    """Write team_memory MCP server config for the given platform and scope."""
    proj_dir = (project_dir or Path.cwd()).resolve()
    eff_scope = _resolve_effective_scope(platform, scope)
    mcp_path = _resolve_mcp_config_path(platform, eff_scope, proj_dir)
    if mcp_path is None:
        return

    entry = _build_mcp_server_entry(eff_scope, proj_dir, project_override)
    mcp_path.parent.mkdir(parents=True, exist_ok=True)

    if platform == "codex":
        import tomli_w

        data: dict[str, Any] = {}
        if mcp_path.exists():
            try:
                with open(mcp_path, "rb") as f:
                    data = tomllib.load(f)
            except Exception:
                data = {}
        mcp_servers = data.setdefault("mcp_servers", {})
        if not isinstance(mcp_servers, dict):
            mcp_servers = {}
            data["mcp_servers"] = mcp_servers
        mcp_servers["team_memory"] = entry
        with open(mcp_path, "wb") as f:
            tomli_w.dump(data, f)
        print(f"[{platform}] ✅ MCP server 'team_memory' ({eff_scope}) written to {mcp_path}")
        return

    data_json: dict[str, Any] = {}
    if mcp_path.exists():
        try:
            with open(mcp_path, encoding="utf-8") as f:
                data_json = json.load(f)
        except Exception:
            data_json = {}

    mcp_servers = data_json.setdefault("mcpServers", {})
    if not isinstance(mcp_servers, dict):
        mcp_servers = {}
        data_json["mcpServers"] = mcp_servers
    mcp_servers["team_memory"] = entry

    with open(mcp_path, "w", encoding="utf-8") as f:
        json.dump(data_json, f, indent=2, ensure_ascii=False)
    print(f"[{platform}] ✅ MCP server 'team_memory' ({eff_scope}) written to {mcp_path}")


def _uninstall_mcp_for_platform(
    platform: str,
    scope: str = "default",
    project_dir: Path | None = None,
) -> bool:
    """Remove team_memory MCP server entry from the platform's MCP config file."""
    proj_dir = (project_dir or Path.cwd()).resolve()
    eff_scope = _resolve_effective_scope(platform, scope)
    mcp_path = _resolve_mcp_config_path(platform, eff_scope, proj_dir)
    if mcp_path is None or not mcp_path.exists():
        if platform in ("windsurf", "codex"):
            print(f"[{platform}] ⚠️  No MCP config found at {mcp_path}, nothing to uninstall")
        return False

    try:
        if platform == "codex":
            import tomli_w

            with open(mcp_path, "rb") as f:
                data = tomllib.load(f)
            mcp_servers = data.get("mcp_servers", {})
            if isinstance(mcp_servers, dict) and "team_memory" in mcp_servers:
                del mcp_servers["team_memory"]
                with open(mcp_path, "wb") as f:
                    tomli_w.dump(data, f)
                print(f"[{platform}] ✅ Removed TM MCP server from {mcp_path}")
                return True
            print(f"[{platform}] ⚠️  No TM MCP server found in {mcp_path}")
            return False

        with open(mcp_path, encoding="utf-8") as f:
            data_json = json.load(f)
        mcp_servers = data_json.get("mcpServers", {})
        if isinstance(mcp_servers, dict) and "team_memory" in mcp_servers:
            del mcp_servers["team_memory"]
            with open(mcp_path, "w", encoding="utf-8") as f:
                json.dump(data_json, f, indent=2, ensure_ascii=False)
            print(f"[{platform}] ✅ Removed TM MCP server from {mcp_path}")
            return True
        if platform == "windsurf":
            print(f"[{platform}] ⚠️  No TM MCP server found in {mcp_path}")
        return False
    except Exception as e:
        print(f"[{platform}] ❌ MCP uninstall error: {e}")
        return False


def _check_mcp_for_platform(
    platform: str,
    scope: str = "default",
    project_dir: Path | None = None,
) -> None:
    """Check whether team_memory MCP server is configured for the platform."""
    proj_dir = (project_dir or Path.cwd()).resolve()
    eff_scope = _resolve_effective_scope(platform, scope)
    mcp_path = _resolve_mcp_config_path(platform, eff_scope, proj_dir)
    if mcp_path is None:
        return
    if not mcp_path.exists():
        print(f"[{platform}] ❌ MCP ({eff_scope}: {mcp_path} not found)")
        return
    try:
        if platform == "codex":
            with open(mcp_path, "rb") as f:
                data = tomllib.load(f)
            ok = "team_memory" in (data.get("mcp_servers") or {})
        else:
            data_json = json.loads(mcp_path.read_text(encoding="utf-8"))
            ok = "team_memory" in (data_json.get("mcpServers") or {})
        print(f"[{platform}] {'✅' if ok else '❌'} MCP (team_memory @ {mcp_path})")
    except Exception as e:
        print(f"[{platform}] ❌ MCP parse error ({mcp_path}): {e}")


def cmd_setup(args: argparse.Namespace) -> None:
    """Configure AI client MCP & hooks / 一键配置 AI 客户端 MCP 与 hooks。"""
    from team_memory.cli.commands.daemon import (
        cmd_daemon_install as _default_daemon_install,
    )
    from team_memory.cli.commands.daemon import (
        cmd_server_up as _default_server_up,
    )
    from team_memory.cli.commands.setup_check_uninstall import (
        _check_platform as _default_check_platform,
    )
    from team_memory.cli.commands.setup_check_uninstall import (
        _uninstall_platform as _default_uninstall_platform,
    )
    from team_memory.cli.commands.setup_platforms import (
        _setup_platform as _default_setup_platform,
    )
    from team_memory.client_config import get_client_config

    cli_pkg = sys.modules.get("team_memory.cli")
    server_up_fn = getattr(cli_pkg, "cmd_server_up", _default_server_up)
    daemon_install_fn = getattr(cli_pkg, "cmd_daemon_install", _default_daemon_install)
    setup_platform_fn = getattr(cli_pkg, "_setup_platform", _default_setup_platform)
    uninstall_platform_fn = getattr(cli_pkg, "_uninstall_platform", _default_uninstall_platform)
    check_platform_fn = getattr(cli_pkg, "_check_platform", _default_check_platform)

    cfg = get_client_config()
    raw_scope = getattr(args, "scope", None) or "default"
    scopes = ["global", "project"] if raw_scope == "both" else [raw_scope]
    target = getattr(args, "target", None) or "all"
    project_dir = Path(args.project_dir).resolve() if getattr(args, "project_dir", None) else Path.cwd()
    project_override = getattr(args, "project", None)

    platforms = ALL_SETUP_PLATFORMS if args.platform == "all" else ([args.platform] if args.platform else [])

    # Handle uninstall / 处理卸载
    if args.uninstall:
        target_platforms = platforms or ALL_SETUP_PLATFORMS
        for sc in scopes:
            for platform in target_platforms:
                uninstall_platform_fn(platform, scope=sc, target=target, project_dir=project_dir)
        return

    # Handle check / 处理检查
    if args.check:
        target_platforms = platforms or ALL_SETUP_PLATFORMS
        for sc in scopes:
            for platform in target_platforms:
                check_platform_fn(platform, cfg, scope=sc, target=target, project_dir=project_dir)
        return

    # Handle --all-in-one: 1) server up (pgvector + migrations + web), 2) daemon install, 3) IDE setup
    if getattr(args, "all_in_one", False):
        print("═══ [1/3] Provisioning Database, Schema Migrations & Web Server (tm-cli server up) ═══")
        server_up_fn(args)
        print("\n═══ [2/3] Installing & Starting TM Daemon (tm-cli daemon install) ═══")
        daemon_install_fn(args)
        print("\n═══ [3/3] Mounting MCP & Hooks Across IDE Platforms ═══")
        cfg = get_client_config()
        if not platforms:
            platforms = ALL_SETUP_PLATFORMS
        for sc in scopes:
            for platform in platforms:
                setup_platform_fn(
                    platform,
                    cfg,
                    scope=sc,
                    target=target,
                    project_dir=project_dir,
                    project_override=project_override,
                )
        print("\n🎉 All-in-One Setup Complete! Run `tm-cli status` to inspect your runtime health.")
        return

    # No --platform: interactive wizard / 无 --platform：交互式向导
    if not args.platform:
        _setup_wizard(args, cfg)
        return

    # Specific platform / 指定平台
    for sc in scopes:
        for platform in platforms:
            setup_platform_fn(
                platform,
                cfg,
                scope=sc,
                target=target,
                project_dir=project_dir,
                project_override=project_override,
            )


def _setup_wizard(args: argparse.Namespace, cfg: Any) -> None:
    """Interactive setup wizard / 交互式配置向导。"""
    from team_memory.cli.commands.setup_platforms import _setup_platform

    non_interactive = getattr(args, "non_interactive", False)
    raw_scope = getattr(args, "scope", None) or "default"
    target = getattr(args, "target", None) or "all"
    project_dir = Path(args.project_dir).resolve() if getattr(args, "project_dir", None) else Path.cwd()
    project_override = getattr(args, "project", None)

    # Step a: Server URL / 步骤 a: 服务器 URL
    server_url = getattr(args, "server_url", None)
    if not non_interactive and not server_url:
        server_url = input("TM server URL [http://localhost:9111]: ").strip()
        if not server_url:
            server_url = "http://localhost:9111"

    # Step b: API key / 步骤 b: API 密钥
    api_key = getattr(args, "api_key", None)
    if not non_interactive and not api_key:
        api_key = input("TM API key: ").strip()

    # Step c: Run config check / 步骤 c: 运行配置检查
    print("\nRunning connectivity check / 运行连通性检查 ...")
    check_args = argparse.Namespace()
    cmd_config_check(check_args)

    # Step d: Ask which platforms / 步骤 d: 询问平台
    platforms = ALL_SETUP_PLATFORMS
    selected = []
    if non_interactive:
        selected = ["claude-code", "cursor", "hermes"]
    else:
        print("\nWhich platforms to configure? / 配置哪些平台？")
        for i, p in enumerate(platforms, 1):
            print(f"  {i}. {p}")
        choices = input("Select (comma-separated numbers, or 'all'): ").strip()
        if choices.lower() in ("all", "a", "*"):
            selected = platforms
        else:
            for c in choices.split(","):
                c = c.strip()
                if c.isdigit() and 1 <= int(c) <= len(platforms):
                    selected.append(platforms[int(c) - 1])

    # Step e: Setup each platform / 步骤 e: 配置各平台
    for platform in selected:
        print()
        _setup_platform(
            platform,
            cfg,
            scope=raw_scope,
            target=target,
            project_dir=project_dir,
            project_override=project_override,
        )

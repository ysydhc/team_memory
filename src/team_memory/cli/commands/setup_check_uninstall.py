"""Platform-specific hook uninstall and check helpers (Claude, Cursor, Hermes, Antigravity)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from team_memory.cli.commands.setup_mcp import (
    _check_mcp_for_platform,
    _resolve_effective_scope,
    _uninstall_mcp_for_platform,
)
from team_memory.cli.http_client import _pkg_root


def _uninstall_platform(
    platform: str,
    scope: str = "default",
    target: str = "all",
    project_dir: Path | None = None,
) -> None:
    """Remove TM MCP & hooks from platform config / 移除平台配置中的 TM MCP 与 hooks。"""
    if target in ("all", "mcp"):
        _uninstall_mcp_for_platform(platform, scope=scope, project_dir=project_dir)

    if target in ("all", "hooks"):
        if platform == "claude-code":
            _uninstall_claude_code(scope=scope, project_dir=project_dir)
        elif platform == "cursor":
            _uninstall_cursor(scope=scope, project_dir=project_dir)
        elif platform == "hermes":
            _uninstall_hermes()
        elif platform == "antigravity":
            _uninstall_antigravity(scope=scope, project_dir=project_dir)


def _uninstall_claude_code(scope: str = "default", project_dir: Path | None = None) -> None:
    """Remove TM hooks from Claude Code settings / 从 Claude Code 设置移除 TM hooks。"""
    proj_dir = (project_dir or Path.cwd()).resolve()
    eff_scope = _resolve_effective_scope("claude-code", scope)
    settings_path = (
        (Path.home() / ".claude" / "settings.json")
        if eff_scope == "global"
        else (proj_dir / ".claude" / "settings.local.json")
    )
    if not settings_path.exists():
        print("[claude-code] ⚠️  No settings.json found, nothing to uninstall")
        return

    try:
        with open(settings_path) as f:
            settings = json.load(f)

        hooks = settings.get("hooks", {})
        removed_any = False
        for event in ("SessionStart", "UserPromptSubmit", "Stop", "PostToolUse"):
            entries = hooks.get(event, [])
            if not isinstance(entries, list):
                continue
            filtered = [
                h
                for h in entries
                if h.get("matcher") != "mcp__team_memory__recall"
                and "team_memory" not in json.dumps(h)
            ]
            if len(filtered) != len(entries):
                removed_any = True
                hooks[event] = filtered

        if not removed_any:
            print("[claude-code] ⚠️  No TM hooks found to remove")
            return

        settings["hooks"] = hooks

        with open(settings_path, "w") as f:
            json.dump(settings, f, indent=2)

        print(f"[claude-code] ✅ Removed TM hooks from {settings_path}")
        print(f"[claude-code] ✅ 已从 {settings_path} 移除 TM hooks")
    except (json.JSONDecodeError, OSError) as e:
        print(f"[claude-code] ❌ Error: {e}")


def _uninstall_cursor(scope: str = "default", project_dir: Path | None = None) -> None:
    """Remove TM hooks from Cursor hooks.json / 从 Cursor hooks.json 移除 TM hooks。"""
    proj_dir = (project_dir or Path.cwd()).resolve()
    eff_scope = _resolve_effective_scope("cursor", scope)
    hooks_path = (
        (Path.home() / ".cursor" / "hooks.json")
        if eff_scope == "global"
        else (proj_dir / ".cursor" / "hooks.json")
    )
    if not hooks_path.exists():
        print("[cursor] ⚠️  No hooks.json found, nothing to uninstall")
        return

    try:
        with open(hooks_path) as f:
            hooks_data = json.load(f)

        hooks_section = hooks_data.get("hooks", {})
        if not isinstance(hooks_section, dict):
            print("[cursor] ⚠️  Unexpected hooks format, skipping")
            return

        removed_any = False
        for event in list(hooks_section.keys()):
            entries = hooks_section.get(event, [])
            if not isinstance(entries, list):
                continue
            filtered = [
                entry
                for entry in entries
                if not (
                    isinstance(entry, dict)
                    and ("team_memory" in entry.get("matcher", "") or "team_memory" in json.dumps(entry))
                )
            ]
            if len(filtered) != len(entries):
                removed_any = True
                hooks_section[event] = filtered

        if not removed_any:
            print("[cursor] ⚠️  No TM hooks found to remove")
            return

        hooks_data["hooks"] = hooks_section

        with open(hooks_path, "w") as f:
            json.dump(hooks_data, f, indent=2)

        print(f"[cursor] ✅ Removed TM hooks from {hooks_path}")
        print(f"[cursor] ✅ 已从 {hooks_path} 移除 TM hooks")
    except (json.JSONDecodeError, OSError) as e:
        print(f"[cursor] ❌ Error: {e}")


def _uninstall_hermes() -> None:
    """Print manual uninstall instructions for Hermes / 打印 Hermes 手动卸载说明。"""
    print("[hermes] ⚠️  Manual uninstall required / 需要手动卸载:")
    print("  Remove post_tool_call hook from ~/.hermes/config.yaml")
    print("  从 ~/.hermes/config.yaml 中移除 post_tool_call hook")


def _uninstall_antigravity(scope: str = "default", project_dir: Path | None = None) -> None:
    """Remove TM hooks from Antigravity hooks.json / 从 Antigravity 配置移除 TM hooks。"""
    proj_dir = (project_dir or Path.cwd()).resolve()
    eff_scope = _resolve_effective_scope("antigravity", scope)
    hooks_path = (
        (Path.home() / ".gemini" / "config" / "hooks.json")
        if eff_scope == "global"
        else (proj_dir / ".agents" / "hooks.json")
    )
    if not hooks_path.exists():
        print("[antigravity] ⚠️  No hooks.json found, nothing to uninstall")
        return

    try:
        with open(hooks_path) as f:
            hooks_data = json.load(f)

        hooks_section = hooks_data.get("hooks", {})
        if not isinstance(hooks_section, dict):
            print("[antigravity] ⚠️  Unexpected hooks format, skipping")
            return

        removed_any = False
        for event in list(hooks_section.keys()):
            entries = hooks_section.get(event, [])
            if not isinstance(entries, list):
                continue
            filtered = [entry for entry in entries if "team_memory" not in json.dumps(entry)]
            if len(filtered) != len(entries):
                removed_any = True
                if filtered:
                    hooks_section[event] = filtered
                else:
                    del hooks_section[event]

        if not removed_any:
            print("[antigravity] ⚠️  No TM hooks found to remove")
            return

        hooks_data["hooks"] = hooks_section
        with open(hooks_path, "w") as f:
            json.dump(hooks_data, f, indent=2)

        print(f"[antigravity] ✅ Removed TM hooks from {hooks_path}")
    except (json.JSONDecodeError, OSError) as e:
        print(f"[antigravity] ❌ Error: {e}")


def _check_platform(
    platform: str,
    cfg: Any,
    scope: str = "default",
    target: str = "all",
    project_dir: Path | None = None,
) -> None:
    """Check MCP & hook configuration status / 检查 MCP 与 hook 配置状态。"""
    if target in ("all", "mcp"):
        _check_mcp_for_platform(platform, scope=scope, project_dir=project_dir)

    if target in ("all", "hooks"):
        hook_dir = _pkg_root() / "hooks" / "platforms"
        if platform == "claude-code":
            _check_claude(hook_dir / "claude", scope=scope, project_dir=project_dir)
        elif platform == "cursor":
            _check_cursor(hook_dir / "cursor", scope=scope, project_dir=project_dir)
        elif platform == "hermes":
            _check_hermes(hook_dir / "hermes", scope=scope, project_dir=project_dir)
        elif platform == "antigravity":
            _check_antigravity(scope=scope, project_dir=project_dir)


def _check_antigravity(scope: str = "default", project_dir: Path | None = None) -> None:
    """Check Antigravity hooks in hooks.json / 检查 Antigravity 全部 hook。"""
    proj_dir = (project_dir or Path.cwd()).resolve()
    eff_scope = _resolve_effective_scope("antigravity", scope)
    hooks_file = (
        (Path.home() / ".gemini" / "config" / "hooks.json")
        if eff_scope == "global"
        else (proj_dir / ".agents" / "hooks.json")
    )
    if not hooks_file.exists():
        print(f"[antigravity] ❌ {hooks_file} not found")
        return

    try:
        data = json.loads(hooks_file.read_text())
    except Exception as e:
        print(f"[antigravity] ❌ parse error: {e}")
        return

    hooks = data.get("hooks", {})
    for event in ("SessionStart", "PreInvocation", "PostToolUse", "Stop"):
        entries = hooks.get(event, [])
        ok = any("team_memory.hooks.platforms.antigravity" in json.dumps(e) for e in entries)
        print(f"[antigravity] {'✅' if ok else '❌'} {event}")


def _check_claude(
    hook_dir: Path,
    scope: str = "default",
    project_dir: Path | None = None,
) -> None:
    """Check all Claude Code hooks / 检查 Claude Code 全部 hook。"""
    proj_dir = (project_dir or Path.cwd()).resolve()
    eff_scope = _resolve_effective_scope("claude-code", scope)
    settings = (
        (Path.home() / ".claude" / "settings.json")
        if eff_scope == "global"
        else (proj_dir / ".claude" / "settings.local.json")
    )
    if not settings.exists():
        print(f"[claude-code] ❌ {settings} not found")
        return

    try:
        data = json.loads(settings.read_text())
    except Exception as e:
        print(f"[claude-code] ❌ parse error: {e}")
        return

    hooks_map = {
        "SessionStart": "session_start.py",
        "UserPromptSubmit": "user_prompt_submit.py",
        "Stop": "stop.py",
        "PostToolUse": "post_tool_use.py",
    }
    for event, script in hooks_map.items():
        ok = _hook_ok(data, event, hook_dir / script)
        print(f"[claude-code] {'✅' if ok else '❌'} {event}")


def _check_cursor(
    hook_dir: Path,
    scope: str = "default",
    project_dir: Path | None = None,
) -> None:
    """Check all Cursor hooks / 检查 Cursor 全部 hook。"""
    proj_dir = (project_dir or Path.cwd()).resolve()
    eff_scope = _resolve_effective_scope("cursor", scope)
    hooks_file = (
        (Path.home() / ".cursor" / "hooks.json")
        if eff_scope == "global"
        else (proj_dir / ".cursor" / "hooks.json")
    )
    if not hooks_file.exists():
        print(f"[cursor] ❌ {hooks_file} not found")
        return

    try:
        data = json.loads(hooks_file.read_text())
    except Exception as e:
        print(f"[cursor] ❌ parse error: {e}")
        return

    hooks_map = {
        "sessionStart": "session_start.py",
        "beforeSubmitPrompt": "before_prompt.py",
        "afterAgentResponse": "after_response.py",
        "afterMCPExecution": "after_mcp.py",
    }
    for event, script in hooks_map.items():
        ok = _hook_ok(data, event, hook_dir / script)
        print(f"[cursor] {'✅' if ok else '❌'} {event}")


def _check_hermes(
    hook_dir: Path,
    scope: str = "default",
    project_dir: Path | None = None,
) -> None:
    """Check all Hermes hooks / 检查 Hermes 全部 hook。"""
    import yaml

    proj_dir = (project_dir or Path.cwd()).resolve()
    eff_scope = _resolve_effective_scope("hermes", scope)
    config_file = (
        (Path.home() / ".hermes" / "config.yaml")
        if eff_scope == "global"
        else (proj_dir / ".hermes" / "config.yaml")
    )
    if not config_file.exists():
        print(f"[hermes] ❌ {config_file} not found")
        return

    try:
        config = yaml.safe_load(config_file.read_text()) or {}
    except Exception as e:
        print(f"[hermes] ❌ parse error: {e}")
        return

    hooks_map = {
        "UserPromptSubmit": "user_prompt_submit.py",
        "Stop": "stop.py",
        "PostToolUse": "post_tool_call.py",
    }
    for event, script in hooks_map.items():
        ok = _hook_ok(config, event, hook_dir / script)
        print(f"[hermes] {'✅' if ok else '❌'} {event}")


def _hook_ok(data: dict, event: str, script_path: Path) -> bool:
    """Check if hook event exists in config and script exists on disk."""
    hooks = data.get("hooks", {})
    entries = hooks.get(event, [])
    return len(entries) > 0 and script_path.exists()

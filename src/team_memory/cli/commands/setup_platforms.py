"""Platform-specific hook installation helpers (Claude, Cursor, Hermes, Antigravity)."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

from team_memory.cli.commands.setup_mcp import (
    _resolve_effective_scope,
    _setup_mcp_for_platform,
)


def _setup_platform(
    platform: str,
    cfg: Any,
    scope: str = "default",
    target: str = "all",
    project_dir: Path | None = None,
    project_override: str | None = None,
) -> None:
    """Write MCP & hook configuration / 写入 MCP 与 hook 配置。"""
    venv_python = sys.executable

    if target in ("all", "mcp"):
        _setup_mcp_for_platform(
            platform,
            scope=scope,
            project_dir=project_dir,
            project_override=project_override,
        )

    if target in ("all", "hooks"):
        if platform == "claude-code":
            _setup_claude_code(cfg, venv_python, scope=scope, project_dir=project_dir)
        elif platform == "cursor":
            _setup_cursor(cfg, venv_python, scope=scope, project_dir=project_dir)
        elif platform == "hermes":
            _setup_hermes(cfg, venv_python, scope=scope, project_dir=project_dir)
        elif platform == "antigravity":
            _setup_antigravity(cfg, venv_python, scope=scope, project_dir=project_dir)
        elif platform == "codex":
            _setup_codex(cfg, venv_python, scope=scope, project_dir=project_dir)


def _setup_claude_code(
    cfg: Any,
    venv_python: str,
    scope: str = "default",
    project_dir: Path | None = None,
) -> None:
    """Configure Claude Code hooks (SessionStart, UserPromptSubmit, Stop, PostToolUse)."""
    proj_dir = (project_dir or Path.cwd()).resolve()
    eff_scope = _resolve_effective_scope("claude-code", scope)
    settings_path = (
        (Path.home() / ".claude" / "settings.json")
        if eff_scope == "global"
        else (proj_dir / ".claude" / "settings.local.json")
    )
    settings: dict[str, Any] = {}
    if settings_path.exists():
        with open(settings_path) as f:
            settings = json.load(f)

    hooks = settings.setdefault("hooks", {})

    hooks["SessionStart"] = [
        {
            "matcher": "",
            "hooks": [
                {
                    "type": "command",
                    "command": f"{venv_python} -m team_memory.hooks.platforms.claude.session_start",
                    "timeout": 10000,
                }
            ],
        }
    ]
    hooks["UserPromptSubmit"] = [
        {
            "matcher": "",
            "hooks": [
                {
                    "type": "command",
                    "command": f"{venv_python} -m team_memory.hooks.platforms.claude.user_prompt_submit",
                    "timeout": 10000,
                }
            ],
        }
    ]
    hooks["Stop"] = [
        {
            "matcher": "",
            "hooks": [
                {
                    "type": "command",
                    "command": f"{venv_python} -m team_memory.hooks.platforms.claude.stop",
                    "timeout": 10000,
                }
            ],
        }
    ]

    # PostToolUse hook (TM recall only; preserve existing entries like ruff)
    post_tool = hooks.get("PostToolUse", [])
    post_tool = [h for h in post_tool if "team_memory" not in json.dumps(h) and "scripts/hooks" not in json.dumps(h)]
    post_tool.append(
        {
            "matcher": "mcp__team_memory__recall|mcp__team_memory__memory_recall",
            "hooks": [
                {
                    "type": "command",
                    "command": f"{venv_python} -m team_memory.hooks.platforms.claude.post_tool_use",
                    "timeout": 10000,
                }
            ],
        }
    )
    hooks["PostToolUse"] = post_tool

    settings["hooks"] = hooks
    settings_path.parent.mkdir(parents=True, exist_ok=True)
    with open(settings_path, "w") as f:
        json.dump(settings, f, indent=2)

    print(f"[claude-code] ✅ All hooks configured in {settings_path}")


def _setup_cursor(
    cfg: Any,
    venv_python: str,
    scope: str = "default",
    project_dir: Path | None = None,
) -> None:
    """Configure Cursor hooks: sessionStart, beforeSubmitPrompt, afterAgentResponse, afterMCPExecution."""
    proj_dir = (project_dir or Path.cwd()).resolve()
    eff_scope = _resolve_effective_scope("cursor", scope)
    hooks_path = (
        (Path.home() / ".cursor" / "hooks.json")
        if eff_scope == "global"
        else (proj_dir / ".cursor" / "hooks.json")
    )
    hooks_data: dict[str, Any] = {}
    if hooks_path.exists():
        with open(hooks_path) as f:
            hooks_data = json.load(f)

    hooks_section = hooks_data.setdefault("hooks", {})
    if isinstance(hooks_section, list):
        hooks_section = {}
        hooks_data["hooks"] = hooks_section

    cursor_hooks = [
        ("sessionStart", "team_memory.hooks.platforms.cursor.session_start", ""),
        ("beforeSubmitPrompt", "team_memory.hooks.platforms.cursor.before_prompt", ""),
        ("afterAgentResponse", "team_memory.hooks.platforms.cursor.after_response", ""),
        ("afterMCPExecution", "team_memory.hooks.platforms.cursor.after_mcp", "MCP:team_memory"),
    ]
    for event, module_path, matcher in cursor_hooks:
        hooks_section[event] = [
            {
                "matcher": matcher,
                "hooks": [{"type": "command", "command": f"{venv_python} -m {module_path}"}],
            }
        ]

    hooks_path.parent.mkdir(parents=True, exist_ok=True)
    with open(hooks_path, "w") as f:
        json.dump(hooks_data, f, indent=2)
    print(f"[cursor] ✅ 4 hooks written to {hooks_path}")


def _setup_hermes(
    cfg: Any,
    venv_python: str,
    scope: str = "default",
    project_dir: Path | None = None,
) -> None:
    """Configure Hermes hooks: UserPromptSubmit, Stop, PostToolUse. Auto-writes config.yaml."""
    import yaml

    proj_dir = (project_dir or Path.cwd()).resolve()
    eff_scope = _resolve_effective_scope("hermes", scope)
    config_path = (
        (Path.home() / ".hermes" / "config.yaml")
        if eff_scope == "global"
        else (proj_dir / ".hermes" / "config.yaml")
    )
    config: dict[str, Any] = {}
    if config_path.exists():
        config = yaml.safe_load(config_path.read_text()) or {}

    existing_hooks = config.get("hooks", {})
    tm_hooks = {
        "UserPromptSubmit": [
            {
                "matcher": "",
                "hooks": [
                    {
                        "type": "command",
                        "command": f"{venv_python} -m team_memory.hooks.platforms.hermes.user_prompt_submit",
                        "timeout": 10000,
                    }
                ],
            }
        ],
        "Stop": [
            {
                "matcher": "",
                "hooks": [
                    {
                        "type": "command",
                        "command": f"{venv_python} -m team_memory.hooks.platforms.hermes.stop",
                        "timeout": 10000,
                    }
                ],
            }
        ],
        "PostToolUse": [
            {
                "matcher": "recall|memory_recall",
                "hooks": [
                    {
                        "type": "command",
                        "command": f"{venv_python} -m team_memory.hooks.platforms.hermes.post_tool_call",
                        "timeout": 10000,
                    }
                ],
            }
        ],
    }
    config["hooks"] = {**existing_hooks, **tm_hooks}

    config_path.parent.mkdir(parents=True, exist_ok=True)
    config_path.write_text(yaml.dump(config, allow_unicode=True, default_flow_style=False, sort_keys=False))
    print(f"[hermes] ✅ 3 hooks written to {config_path}")


def _setup_antigravity(
    cfg: Any,
    venv_python: str,
    scope: str = "default",
    project_dir: Path | None = None,
) -> None:
    """Configure Antigravity hooks in ~/.gemini/config/hooks.json (global) or .agents/hooks.json (project)."""
    proj_dir = (project_dir or Path.cwd()).resolve()
    eff_scope = _resolve_effective_scope("antigravity", scope)
    hooks_path = (
        (Path.home() / ".gemini" / "config" / "hooks.json")
        if eff_scope == "global"
        else (proj_dir / ".agents" / "hooks.json")
    )
    hooks_data: dict[str, Any] = {}
    if hooks_path.exists():
        try:
            with open(hooks_path) as f:
                hooks_data = json.load(f)
        except Exception:
            hooks_data = {}

    hooks_section = hooks_data.setdefault("hooks", {})
    if not isinstance(hooks_section, dict):
        hooks_section = {}
        hooks_data["hooks"] = hooks_section

    agy_hooks: list[tuple[str, str, str | None]] = [
        ("SessionStart", "team_memory.hooks.platforms.antigravity.session_start", None),
        ("PreInvocation", "team_memory.hooks.platforms.antigravity.pre_invocation", None),
        ("PostToolUse", "team_memory.hooks.platforms.antigravity.post_tool_use", "*"),
        ("Stop", "team_memory.hooks.platforms.antigravity.stop", None),
    ]
    for event, module_path, matcher in agy_hooks:
        existing = hooks_section.get(event, [])
        if not isinstance(existing, list):
            existing = []
        existing = [e for e in existing if "team_memory" not in json.dumps(e)]
        handler = {
            "type": "command",
            "command": f"{venv_python} -m {module_path}",
            "timeout": 10,
        }
        if matcher is not None:
            existing.append(
                {
                    "matcher": matcher,
                    "hooks": [handler],
                }
            )
        else:
            existing.append(handler)
        hooks_section[event] = existing

    hooks_path.parent.mkdir(parents=True, exist_ok=True)
    with open(hooks_path, "w") as f:
        json.dump(hooks_data, f, indent=2)
    print(f"[antigravity] ✅ 4 hooks written to {hooks_path}")


def _setup_codex(
    cfg: Any,
    venv_python: str,
    scope: str = "default",
    project_dir: Path | None = None,
) -> None:
    """Configure OpenAI Codex CLI hooks in ~/.codex/hooks.json (global) or .codex/hooks.json (project)."""
    proj_dir = (project_dir or Path.cwd()).resolve()
    eff_scope = _resolve_effective_scope("codex", scope)
    hooks_path = (
        (Path.home() / ".codex" / "hooks.json") if eff_scope == "global" else (proj_dir / ".codex" / "hooks.json")
    )
    hooks_data: dict[str, Any] = {}
    if hooks_path.exists():
        try:
            with open(hooks_path, encoding="utf-8") as f:
                hooks_data = json.load(f)
        except Exception:
            hooks_data = {}

    hooks_section = hooks_data.setdefault("hooks", {})
    if not isinstance(hooks_section, dict):
        hooks_section = {}
        hooks_data["hooks"] = hooks_section

    codex_hooks: list[tuple[str, str, str | None]] = [
        ("SessionStart", "team_memory.hooks.platforms.codex.session_start", None),
        ("UserPromptSubmit", "team_memory.hooks.platforms.codex.user_prompt_submit", None),
        ("PostToolUse", "team_memory.hooks.platforms.codex.post_tool_use", "*"),
        ("Stop", "team_memory.hooks.platforms.codex.stop", None),
    ]
    for event, module_path, matcher in codex_hooks:
        existing = hooks_section.get(event, [])
        if not isinstance(existing, list):
            existing = []
        # Remove previous TeamMemory hooks while preserving user/third-party hooks
        existing = [e for e in existing if "team_memory" not in json.dumps(e)]
        handler = {
            "type": "command",
            "command": f"{venv_python} -m {module_path}",
            "timeout": 10,
        }
        entry: dict[str, Any] = {"hooks": [handler]}
        if matcher is not None:
            entry["matcher"] = matcher
        existing.append(entry)
        hooks_section[event] = existing

    hooks_path.parent.mkdir(parents=True, exist_ok=True)
    with open(hooks_path, "w", encoding="utf-8") as f:
        json.dump(hooks_data, f, indent=2)
    print(f"[codex] ✅ 4 hooks written to {hooks_path}")

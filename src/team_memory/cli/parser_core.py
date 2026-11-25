"""Core subcommand parser registration (memory, archive, config, project, setup, import)."""

from __future__ import annotations

import argparse

from team_memory.cli.commands.config_project import (
    cmd_config_check,
    cmd_config_init,
    cmd_config_set,
    cmd_config_show,
    cmd_project_init,
    cmd_project_list,
    cmd_project_show,
)
from team_memory.cli.commands.memory import (
    cmd_archive,
    cmd_context,
    cmd_feedback,
    cmd_get_archive,
    cmd_import,
    cmd_recall,
    cmd_save,
    cmd_upload,
)
from team_memory.cli.commands.setup_mcp import cmd_setup


def register_core_subparsers(sub: argparse._SubParsersAction) -> None:
    """Register archive, upload, save, recall, context, feedback, config, project, setup, import."""
    # archive subcommand
    p_arch = sub.add_parser("archive", help="Create or update an archive")
    p_arch.add_argument("--title", required=True, help="Archive title")
    p_arch.add_argument("--solution-doc", default="", help="L2 solution document text")
    p_arch.add_argument("--content-type", default="session_archive", help="Content type")
    p_arch.add_argument("--value-summary", help="One-line value summary")
    p_arch.add_argument("--tags", help="Comma-separated tags")
    p_arch.add_argument("--overview", help="L1 overview text")
    p_arch.add_argument("--overview-file", help="Read L1 overview from file")
    p_arch.add_argument("--solution-file", help="Read L2 solution from file")
    p_arch.add_argument("--summary", help="Conversation summary")
    p_arch.add_argument("--linked-experience-ids", help="Comma-separated experience UUIDs")
    p_arch.add_argument("--project", help="Project scope (auto-inferred if omitted)")
    p_arch.add_argument("--scope", default="session", help="Archive scope")
    p_arch.add_argument("--scope-ref", help="Scope reference")
    p_arch.set_defaults(func=cmd_archive)

    # upload subcommand
    p_up = sub.add_parser("upload", help="Upload a file attachment to an archive")
    p_up.add_argument("--archive-id", required=True, help="Target archive UUID")
    p_up.add_argument("--file", required=True, help="File path to upload")
    p_up.add_argument("--kind", default="file", help="Attachment kind")
    p_up.add_argument("--snippet", help="Optional snippet/note")
    p_up.add_argument(
        "--project",
        help="Project scope (auto-inferred if omitted)",
    )
    p_up.set_defaults(func=cmd_upload)

    # save subcommand
    p_save = sub.add_parser("save", help="Save team knowledge (MCP save)")
    p_save.add_argument("--title", help="Title (direct mode)")
    p_save.add_argument("--problem", help="Problem (direct mode)")
    p_save.add_argument("--solution", help="Solution (direct mode)")
    p_save.add_argument(
        "--content",
        help="Raw text for LLM parse mode (prefer --title + --problem + --solution for fast direct save)",
    )
    p_save.add_argument("--tags", help="Comma-separated tags")
    p_save.add_argument(
        "--scope",
        choices=("project", "personal"),
        default="project",
        help="Scope (default: project)",
    )
    p_save.add_argument(
        "--experience-type",
        help="Experience type (e.g. general, feature, bugfix)",
    )
    p_save.add_argument("--project", help="Project scope (auto-inferred if omitted)")
    p_save.add_argument("--group-key", help="Group key for deduplication")
    p_save.set_defaults(func=cmd_save)

    # recall subcommand
    p_recall = sub.add_parser("recall", help="Search team knowledge (MCP recall)")
    p_recall.add_argument("--query", help="Exploratory search query")
    p_recall.add_argument("--problem", help="Problem-focused recall")
    p_recall.add_argument("--file-path", help="File path for suggestions")
    p_recall.add_argument("--language", help="Language hint")
    p_recall.add_argument("--framework", help="Framework hint")
    p_recall.add_argument("--tags", help="Comma-separated tags")
    p_recall.add_argument("--max-results", type=int, default=5, help="Max results (default: 5)")
    p_recall.add_argument("--project", help="Project scope (auto-inferred if omitted)")
    p_recall.add_argument(
        "--include-archives",
        action="store_true",
        help="Include archive previews in results",
    )
    p_recall.add_argument(
        "--include-user-profile",
        action="store_true",
        help="Attach user profile to response",
    )
    p_recall.set_defaults(func=cmd_recall)

    # context subcommand
    p_ctx = sub.add_parser("context", help="Profile + relevant knowledge (MCP context)")
    p_ctx.add_argument("--file-paths", help="Comma-separated file paths")
    p_ctx.add_argument("--task-description", help="Current task description")
    p_ctx.add_argument("--project", help="Project scope (auto-inferred if omitted)")
    p_ctx.set_defaults(func=cmd_context)

    # get-archive subcommand
    p_ga = sub.add_parser("get-archive", help="Fetch full archive L2 by id (MCP get_archive)")
    p_ga.add_argument("--id", required=True, dest="archive_id", help="Archive UUID")
    p_ga.add_argument("--project", help="Project scope (auto-inferred if omitted)")
    p_ga.set_defaults(func=cmd_get_archive)

    # feedback subcommand
    p_fb = sub.add_parser("feedback", help="Rate an experience (MCP feedback)")
    p_fb.add_argument("--experience-id", required=True, help="Experience UUID")
    p_fb.add_argument("--rating", type=int, required=True, help="Rating 1-5")
    p_fb.add_argument("--comment", help="Optional comment")
    p_fb.set_defaults(func=cmd_feedback)

    # ── config ──────────────────────────────────────────────────
    p_cfg = sub.add_parser("config", help="Manage client configuration / 管理客户端配置")
    cfg_sub = p_cfg.add_subparsers(dest="config_command")

    p_cfg_show = cfg_sub.add_parser("show", help="Show current config / 显示当前配置")
    p_cfg_show.set_defaults(func=cmd_config_show)

    p_cfg_check = cfg_sub.add_parser("check", help="Test connectivity / 测试连通性")
    p_cfg_check.set_defaults(func=cmd_config_check)

    p_cfg_set = cfg_sub.add_parser("set", help="Set a config key / 设置配置项")
    p_cfg_set.add_argument("key", help="Config key (dot notation, e.g. server.url)")
    p_cfg_set.add_argument("value", help="Config value")
    p_cfg_set.set_defaults(func=cmd_config_set)

    p_cfg_init = cfg_sub.add_parser("init", help="Initialize config interactively / 交互式初始化配置")
    p_cfg_init.add_argument("--server-url", help="TM server URL")
    p_cfg_init.add_argument("--api-key", help="TM API key")
    p_cfg_init.add_argument(
        "--non-interactive",
        action="store_true",
        help="Use flags only, no prompts / 仅使用标志，无提示",
    )
    p_cfg_init.set_defaults(func=cmd_config_init)

    # ── project ──────────────────────────────────────────────────
    p_proj = sub.add_parser("project", help="Manage project configuration / 管理项目配置")
    proj_sub = p_proj.add_subparsers(dest="project_command")

    p_proj_init = proj_sub.add_parser("init", help="Initialize .tm.toml / 初始化 .tm.toml")
    p_proj_init.set_defaults(func=cmd_project_init)

    p_proj_show = proj_sub.add_parser("show", help="Show project info / 显示项目信息")
    p_proj_show.set_defaults(func=cmd_project_show)

    p_proj_list = proj_sub.add_parser("list", help="List all projects / 列出所有项目")
    p_proj_list.set_defaults(func=cmd_project_list)

    # ── setup ──────────────────────────────────────────────────
    p_setup = sub.add_parser("setup", help="Configure AI client MCP & hooks / 配置 AI 客户端 MCP 与 hooks")
    p_setup.add_argument(
        "--platform",
        choices=["claude-code", "cursor", "antigravity", "windsurf", "codex", "hermes", "all"],
        help="Which platform to configure / 配置哪个平台 (omit for interactive wizard)",
    )
    p_setup.add_argument(
        "--scope",
        choices=["global", "project", "both"],
        default=None,
        help="Configuration scope: global (~/.xxx) or project (<cwd>/.xxx) / 配置范围：全局或单项目",
    )
    p_setup.add_argument(
        "--target",
        choices=["all", "mcp", "hooks"],
        default="all",
        help="Configure MCP server, hooks, or both (default: all) / 配置目标：mcp、hooks 或 all",
    )
    p_setup.add_argument(
        "--project-dir",
        help="Target project directory when --scope project/both (default: cwd) / 目标项目目录",
    )
    p_setup.add_argument(
        "--project",
        help="Override TEAM_MEMORY_PROJECT in project MCP config / 覆盖单项目 MCP 中的项目名",
    )
    p_setup.add_argument(
        "--check",
        action="store_true",
        help="Only check current status, don't modify / 仅检查状态",
    )
    p_setup.add_argument(
        "--uninstall",
        action="store_true",
        help="Remove TM MCP & hooks instead of adding / 移除而非添加 TM MCP 与 hooks",
    )
    p_setup.add_argument(
        "--non-interactive",
        action="store_true",
        help="Skip prompts, use defaults/flags only / 跳过提示",
    )
    p_setup.add_argument(
        "--server-url",
        help="TM server URL (for wizard) / 服务器 URL（向导用）",
    )
    p_setup.add_argument(
        "--api-key",
        help="TM API key (for wizard) / API 密钥（向导用）",
    )
    p_setup.add_argument(
        "--all-in-one",
        action="store_true",
        help="Bootstrap full local stack (Postgres + Migrate + Web + Daemon + IDE) / 免源码一键拉起本地全栈",
    )
    p_setup.set_defaults(func=cmd_setup)

    # ── import ──────────────────────────────────────────────────
    p_import = sub.add_parser("import", help="Import historical sessions (ChatGPT / Antigravity)")
    p_import.add_argument(
        "--source",
        required=True,
        choices=["chatgpt", "antigravity"],
        help="Session source platform",
    )
    p_import.add_argument(
        "--file",
        help="Path to conversations.json/.zip (ChatGPT) or brain dir (Antigravity)",
    )
    p_import.add_argument(
        "--project",
        help="Default project scope when auto-inference falls back",
    )
    p_import.add_argument(
        "--since",
        help="Only import sessions created since YYYY-MM-DD",
    )
    p_import.add_argument(
        "--dry-run",
        action="store_true",
        help="Preview session filtering and segmentation without writing",
    )
    p_import.set_defaults(func=cmd_import)

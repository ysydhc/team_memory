"""Operations subcommand parser registration (daemon, server, status, watch, dream, eval, ops)."""

from __future__ import annotations

import argparse
from pathlib import Path

from team_memory.cli.commands.admin import cmd_dream, cmd_status
from team_memory.cli.commands.daemon import (
    cmd_daemon_install,
    cmd_daemon_start,
    cmd_daemon_status,
    cmd_daemon_stop,
    cmd_daemon_uninstall,
    cmd_server_down,
    cmd_server_migrate,
    cmd_server_up,
)
from team_memory.cli.commands.eval import cmd_eval, cmd_ops_digest, cmd_verify_staleness


def register_ops_subparsers(sub: argparse._SubParsersAction) -> None:
    """Register daemon, server, status, recall-list, recall-trace, watch, dream, eval, ops-digest, verify-staleness."""
    # ── daemon ──────────────────────────────────────────────────
    p_daemon = sub.add_parser(
        "daemon",
        help="Manage local TM Daemon launchd service / 管理本地 TM Daemon 守护进程",
    )
    p_daemon.set_defaults(func=cmd_daemon_status)
    daemon_sub = p_daemon.add_subparsers(dest="daemon_command")

    p_daemon_install = daemon_sub.add_parser(
        "install",
        help="Install & start macOS launchd service (no source repo required) / 安装并启动 launchd 守护进程",
    )
    p_daemon_install.set_defaults(func=cmd_daemon_install)

    p_daemon_uninstall = daemon_sub.add_parser(
        "uninstall",
        help="Stop & remove macOS launchd service / 停止并卸载 launchd 守护进程",
    )
    p_daemon_uninstall.set_defaults(func=cmd_daemon_uninstall)

    p_daemon_start = daemon_sub.add_parser(
        "start",
        help="Start TM Daemon service / 启动守护进程",
    )
    p_daemon_start.set_defaults(func=cmd_daemon_start)

    p_daemon_stop = daemon_sub.add_parser(
        "stop",
        help="Stop TM Daemon service / 停止守护进程",
    )
    p_daemon_stop.set_defaults(func=cmd_daemon_stop)

    p_daemon_status = daemon_sub.add_parser(
        "status",
        help="Show TM Daemon launchd & HTTP health status / 查看守护进程状态",
    )
    p_daemon_status.set_defaults(func=cmd_daemon_status)

    # ── server ──────────────────────────────────────────────────
    p_server = sub.add_parser(
        "server",
        help="Manage local TM Server & pgvector Docker stack (no source repo required) / 免源码管理本地服务端与数据库",
    )
    p_server.set_defaults(func=lambda _a: p_server.print_help())
    server_sub = p_server.add_subparsers(dest="server_command")

    p_server_up = server_sub.add_parser(
        "up",
        help=(
            "Start PostgreSQL (native :5432 or docker :5433) + migrations + Web & Daemon / "
            "一键启动数据库、迁移、Web 与 Daemon"
        ),
    )
    p_server_up.add_argument(
        "--mode",
        choices=["auto", "native", "docker"],
        default="auto",
        help="Database backend mode: native (:5432 Homebrew PG) or docker (:5433 container)",
    )
    p_server_up.add_argument(
        "--no-web",
        action="store_true",
        help="Skip starting background Web server / 不启动后台 Web 服务",
    )
    p_server_up.add_argument(
        "--no-daemon",
        action="store_true",
        help="Skip installing/starting TM Daemon / 不启动 TM Daemon",
    )
    p_server_up.set_defaults(func=cmd_server_up)

    p_server_down = server_sub.add_parser(
        "down",
        help="Stop local Web server & pgvector container / 停止本地 Web 服务与数据库容器",
    )
    p_server_down.set_defaults(func=cmd_server_down)

    p_server_migrate = server_sub.add_parser(
        "migrate",
        help="Run bundled Alembic database migrations (upgrade head) / 执行内置数据库迁移",
    )
    p_server_migrate.add_argument(
        "--revision",
        default="head",
        help="Target Alembic revision (default: head)",
    )
    p_server_migrate.set_defaults(func=cmd_server_migrate)

    # ── status & watch ──────────────────────────────────────────
    p_status = sub.add_parser(
        "status",
        help="Show panoramic runtime status & recent recall pipeline traces / 全景查看 TM 运行状态与召回全链路追踪",
    )
    p_status.add_argument(
        "target",
        nargs="?",
        help="Optional recall index (e.g. 1) or Log ID to inspect recall trace / 输入召回序号或 Log ID",
    )
    p_status.add_argument(
        "-t",
        "--trace",
        help="Inspect full recall pipeline trace for index (1-N) or Log ID / 打印完整召回链路图",
    )
    p_status.add_argument(
        "-w",
        "--watch",
        action="store_true",
        help="Continuously refresh status in terminal / 持续刷新观察状态 (Ctrl+C 退出)",
    )
    p_status.add_argument(
        "-i",
        "--interval",
        type=float,
        default=2.0,
        help="Refresh interval in seconds when --watch is enabled (default: 2.0)",
    )
    p_status.add_argument(
        "--json",
        action="store_true",
        help="Output machine-readable JSON snapshot or trace",
    )
    p_status.add_argument(
        "-b",
        "--batch",
        action="store_true",
        help="Run in headless / batch mode without interactive terminal prompt (alias for --no-interactive)",
    )
    p_status.add_argument(
        "--no-interactive",
        action="store_true",
        help="Disable interactive recall selection prompt in TTY / 禁用终端交互式选择提示",
    )
    p_status.add_argument(
        "-n",
        "--limit",
        type=int,
        default=12,
        help="Number of recent recalls to display in list (default: 12, e.g. -n 30) / 展示最近召回条数",
    )
    p_status.add_argument(
        "--days",
        type=int,
        default=7,
        help="Time window in days for search/faithfulness telemetry (default: 7)",
    )
    p_status.add_argument(
        "--since",
        nargs="?",
        const="7d",
        default=None,
        help="Optional lookback window (e.g. 7d, 24h, YYYY-MM-DD)",
    )
    p_status.add_argument("--server-url", help="Override Core Server URL (default: http://localhost:9111)")
    p_status.add_argument("--daemon-url", help="Override TM Daemon URL (default: http://127.0.0.1:3901)")
    p_status.add_argument("--db-path", help="Override local SQLite drafts.db path")
    p_status.set_defaults(func=cmd_status)

    p_recall_list = sub.add_parser(
        "recall-list",
        help="List recent recalls & inspect pipeline traces (shortcut for status -n N) / 查看最近召回列表与链路",
    )
    p_recall_list.add_argument(
        "-n",
        "--limit",
        type=int,
        default=20,
        help="Number of recent recalls to display (default: 20)",
    )
    p_recall_list.add_argument(
        "--days",
        type=int,
        default=7,
        help="Time window in days (default: 7)",
    )
    p_recall_list.add_argument(
        "--since",
        nargs="?",
        const="7d",
        default=None,
        help="Optional lookback window (e.g. 7d, 24h, YYYY-MM-DD)",
    )
    p_recall_list.add_argument(
        "--json",
        action="store_true",
        help="Output JSON snapshot",
    )
    p_recall_list.add_argument(
        "-b",
        "--batch",
        action="store_true",
        help="Run in headless / batch mode without interactive terminal prompt (alias for --no-interactive)",
    )
    p_recall_list.add_argument(
        "--no-interactive",
        action="store_true",
        help="Disable interactive recall selection prompt",
    )
    p_recall_list.add_argument("--server-url", help="Override Core Server URL")
    p_recall_list.add_argument("--daemon-url", help="Override TM Daemon URL")
    p_recall_list.add_argument("--db-path", help="Override local SQLite drafts.db path")
    p_recall_list.set_defaults(func=cmd_status)

    p_recall_trace = sub.add_parser(
        "recall-trace",
        help="Inspect full recall pipeline trace for index (1-N) or Log ID (shortcut for status <target>)",
    )
    p_recall_trace.add_argument(
        "target",
        help="Recall index (1-N) or Log ID to inspect",
    )
    p_recall_trace.add_argument(
        "-n",
        "--limit",
        type=int,
        default=50,
        help="Lookback limit when resolving index (default: 50)",
    )
    p_recall_trace.add_argument(
        "--json",
        action="store_true",
        help="Output machine-readable JSON trace",
    )
    p_recall_trace.set_defaults(func=cmd_status)

    p_watch = sub.add_parser(
        "watch",
        help="Continuously monitor TM runtime status in terminal / 命令行实时持续监控面板 (status --watch 快捷方式)",
    )
    p_watch.add_argument(
        "-i",
        "--interval",
        type=float,
        default=2.0,
        help="Refresh interval in seconds (default: 2.0)",
    )
    p_watch.add_argument(
        "--json",
        action="store_true",
        help="Output JSON Lines on each refresh tick",
    )
    p_watch.add_argument(
        "-n",
        "--limit",
        type=int,
        default=12,
        help="Number of recent recalls to display in list (default: 12)",
    )
    p_watch.add_argument(
        "--days",
        type=int,
        default=7,
        help="Time window in days for search/faithfulness telemetry (default: 7)",
    )
    p_watch.add_argument("--server-url", help="Override Core Server URL")
    p_watch.add_argument("--daemon-url", help="Override TM Daemon URL")
    p_watch.add_argument("--db-path", help="Override local SQLite drafts.db path")
    p_watch.set_defaults(func=cmd_status, watch=True)

    # ── dream ───────────────────────────────────────────────────
    p_dream = sub.add_parser(
        "dream",
        help="Run AutoDream offline semantic consolidation & distillation / 手动触发 AutoDream 记忆整合",
    )
    p_dream.add_argument(
        "--dry-run",
        action="store_true",
        help="Preview clusters and LLM decisions without writing to DB or advancing watermark",
    )
    p_dream.add_argument(
        "--force",
        action="store_true",
        help="Bypass min_hours_since_last and min_new_experiences trigger gates",
    )
    p_dream.add_argument(
        "--project",
        help="Limit AutoDream scan to a specific project",
    )
    p_dream.add_argument(
        "--max-clusters",
        type=int,
        default=None,
        help="Override max_clusters_per_run cap for this run",
    )
    p_dream.add_argument(
        "--model",
        help="Override LLM model for this run (e.g. gbu-gemini-3.8-flash)",
    )
    p_dream.add_argument(
        "--json",
        action="store_true",
        help="Output machine-readable JSON result",
    )
    p_dream.set_defaults(func=cmd_dream)

    # ── eval ────────────────────────────────────────────────────
    p_eval = sub.add_parser(
        "eval",
        help="Run Golden Eval 3-way ablation benchmark (Vector vs Hybrid vs Full+Graph)",
    )
    p_eval.add_argument(
        "--golden-set",
        type=Path,
        default=None,
        help="Path to golden_set.json (default: evals/golden_set.json)",
    )
    p_eval.add_argument(
        "--history",
        type=Path,
        default=None,
        help="Path to history.jsonl (default: evals/history.jsonl)",
    )
    p_eval.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Evaluate only the first N queries",
    )
    p_eval.add_argument(
        "--no-save",
        action="store_true",
        help="Do not append result to evals/history.jsonl",
    )
    p_eval.add_argument(
        "--json",
        action="store_true",
        help="Output machine-readable JSON result",
    )
    p_eval.set_defaults(func=cmd_eval)

    # ── ops-digest ──────────────────────────────────────────────
    p_ops = sub.add_parser(
        "ops-digest",
        help="Exception-driven 5-subsystem health check & weekly digest",
    )
    p_ops.add_argument("--days", type=int, default=30, help="Lookback window: 30 or 180 (default: 30)")
    p_ops.add_argument(
        "--alert-only",
        action="store_true",
        help="Silent when all green; output/push only on yellow/red alerts",
    )
    p_ops.add_argument(
        "--weekly",
        action="store_true",
        help="Compact <= 10-line weekly health summary",
    )
    p_ops.add_argument(
        "--webhook",
        default="",
        help="Optional Feishu webhook URL (or env TEAM_MEMORY_FEISHU_WEBHOOK)",
    )
    p_ops.add_argument(
        "--json",
        action="store_true",
        help="Output raw observability JSON overview",
    )
    p_ops.set_defaults(func=cmd_ops_digest)

    # ── verify-staleness ────────────────────────────────────────
    p_stale = sub.add_parser(
        "verify-staleness",
        help="Verify code-grounded evidence_anchors against a repository and flag possibly_stale experiences",
    )
    p_stale.add_argument(
        "--project",
        default=None,
        help="Target project slug (auto-inferred from --repo-path or CWD if omitted)",
    )
    p_stale.add_argument(
        "--repo-path",
        default=".",
        help="Path to the Git repository root to verify anchors against (default: current directory)",
    )
    p_stale.add_argument(
        "--dry-run",
        action="store_true",
        help="Preview staleness transitions without modifying the database",
    )
    p_stale.add_argument(
        "--json",
        action="store_true",
        help="Output machine-readable JSON report",
    )
    p_stale.set_defaults(func=cmd_verify_staleness)

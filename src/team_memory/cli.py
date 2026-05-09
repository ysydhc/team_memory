"""tm-cli -- command-line interface for Team Memory.

Subcommands:
    archive      Create or update an archive (POST /api/v1/archives)
    upload       Upload a file attachment to an archive
    save         Save team knowledge (POST /api/v1/mcp/save)
    recall       Search team knowledge (POST /api/v1/mcp/recall)
    context      Profile + relevant knowledge (POST /api/v1/mcp/context)
    get-archive  Fetch full archive L2 by id (GET /api/v1/mcp/archive/{id})
    feedback     Rate an experience (POST /api/v1/mcp/feedback)

Usage:
    python -m team_memory.cli archive --title "..." --solution-doc "..." [options]
    python -m team_memory.cli upload --archive-id <id> --file <path> [options]
    python -m team_memory.cli save --title "..." --problem "..." --solution "..." [options]
    python -m team_memory.cli recall [--query "..."] [options]
    python -m team_memory.cli context [--file-paths "a.py,b.py"] [options]
    python -m team_memory.cli get-archive --id <uuid> [options]
    python -m team_memory.cli feedback --experience-id <uuid> --rating 5 [options]
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

import httpx

# ---------------------------------------------------------------------------
# Config resolution: env > .env file > default
# ---------------------------------------------------------------------------

def _load_dotenv() -> None:
    """Load .env from current directory or parents (best-effort)."""
    try:
        from dotenv import load_dotenv
        # Walk up from cwd to find .env
        load_dotenv(verbose=False)
    except ImportError:
        pass


# Load .env before any config reads
_load_dotenv()


def _get_base_url() -> str:
    try:
        from team_memory.client_config import ClientConfig
        return ClientConfig.load().server.url
    except Exception:
        return os.environ.get("TM_BASE_URL", "http://localhost:9111")


def _get_api_key() -> str:
    try:
        from team_memory.client_config import ClientConfig
        key = ClientConfig.load().server.api_key
    except Exception:
        key = os.environ.get("TEAM_MEMORY_API_KEY", "")
    if not key:
        print("Error: TEAM_MEMORY_API_KEY not set.", file=sys.stderr)
        print("  Set it via: export TEAM_MEMORY_API_KEY=***", file=sys.stderr)
        print("  Or add it to .env file or ~/.config/tm/config.toml", file=sys.stderr)
        sys.exit(1)
    return key


def _headers() -> dict[str, str]:
    return {
        "Authorization": f"Bearer {_get_api_key()}",
        "Content-Type": "application/json",
    }


def _get_project(cli_project: str | None) -> str | None:
    """Resolve project name: CLI arg > .tm.toml > git repo > dir name."""
    if cli_project:
        return cli_project
    try:
        from team_memory.project_resolver import infer_project_name
        return infer_project_name()
    except Exception:
        return None


# ---------------------------------------------------------------------------
# HTTP helpers
# ---------------------------------------------------------------------------

def _print_response_json(resp: httpx.Response) -> None:
    print(json.dumps(resp.json(), indent=2, ensure_ascii=False))


def _post_mcp_json(path: str, body: dict[str, Any], *, timeout: float = 60.0) -> None:
    url = f"{_get_base_url()}{path}"
    try:
        resp = httpx.post(url, json=body, headers=_headers(), timeout=timeout)
        resp.raise_for_status()
        _print_response_json(resp)
    except httpx.HTTPStatusError as e:
        print(f"Error {e.response.status_code}: {e.response.text}", file=sys.stderr)
        sys.exit(1)
    except httpx.ConnectError:
        print(f"Error: Cannot connect to {url}. Is the server running?", file=sys.stderr)
        sys.exit(1)


def _get_mcp_json(
    path: str, *, params: dict[str, str] | None = None, timeout: float = 30.0
) -> None:
    url = f"{_get_base_url()}{path}"
    try:
        resp = httpx.get(url, headers=_headers(), params=params or None, timeout=timeout)
        resp.raise_for_status()
        _print_response_json(resp)
    except httpx.HTTPStatusError as e:
        print(f"Error {e.response.status_code}: {e.response.text}", file=sys.stderr)
        sys.exit(1)
    except httpx.ConnectError:
        print(f"Error: Cannot connect to {url}. Is the server running?", file=sys.stderr)
        sys.exit(1)


# ---------------------------------------------------------------------------
# Subcommands
# ---------------------------------------------------------------------------

def cmd_save(args: argparse.Namespace) -> None:
    """POST /api/v1/mcp/save — direct fields or long-form content."""
    has_direct = bool(args.title or args.problem or args.solution)
    has_content = bool(args.content)
    if has_content and has_direct:
        print(
            "Error: --content cannot be used together with --title/--problem/--solution.",
            file=sys.stderr,
        )
        sys.exit(1)
    if not has_content and not (args.title and args.problem and args.solution):
        print(
            "Error: use direct mode (--title, --problem, --solution) or --content.",
            file=sys.stderr,
        )
        sys.exit(1)

    body: dict[str, Any] = {}
    if has_content:
        body["content"] = args.content
    else:
        body["title"] = args.title
        body["problem"] = args.problem
        body["solution"] = args.solution
    if args.tags:
        body["tags"] = [t.strip() for t in args.tags.split(",") if t.strip()]
    if args.scope:
        body["scope"] = args.scope
    if args.experience_type:
        body["experience_type"] = args.experience_type
    project = _get_project(args.project)
    if project:
        body["project"] = project
    if args.group_key:
        body["group_key"] = args.group_key

    _post_mcp_json("/api/v1/mcp/save", body, timeout=120.0)


def cmd_recall(args: argparse.Namespace) -> None:
    """POST /api/v1/mcp/recall."""
    body: dict[str, Any] = {}
    if args.query is not None:
        body["query"] = args.query
    if args.problem is not None:
        body["problem"] = args.problem
    if args.file_path is not None:
        body["file_path"] = args.file_path
    if args.language is not None:
        body["language"] = args.language
    if args.framework is not None:
        body["framework"] = args.framework
    if args.tags:
        body["tags"] = [t.strip() for t in args.tags.split(",") if t.strip()]
    body["max_results"] = args.max_results
    project = _get_project(args.project)
    if project:
        body["project"] = project
    if args.include_archives:
        body["include_archives"] = True
    if args.include_user_profile:
        body["include_user_profile"] = True

    _post_mcp_json("/api/v1/mcp/recall", body, timeout=120.0)


def cmd_context(args: argparse.Namespace) -> None:
    """POST /api/v1/mcp/context."""
    body: dict[str, Any] = {}
    if args.file_paths:
        body["file_paths"] = [p.strip() for p in args.file_paths.split(",") if p.strip()]
    if args.task_description is not None:
        body["task_description"] = args.task_description
    project = _get_project(args.project)
    if project:
        body["project"] = project

    _post_mcp_json("/api/v1/mcp/context", body, timeout=120.0)


def cmd_get_archive(args: argparse.Namespace) -> None:
    """GET /api/v1/mcp/archive/{archive_id}."""
    archive_id = args.archive_id
    params: dict[str, str] = {}
    project = _get_project(args.project)
    if project:
        params["project"] = project
    path = f"/api/v1/mcp/archive/{archive_id}"
    _get_mcp_json(path, params=params or None)


def cmd_feedback(args: argparse.Namespace) -> None:
    """POST /api/v1/mcp/feedback."""
    body: dict[str, Any] = {
        "experience_id": args.experience_id,
        "rating": args.rating,
    }
    if args.comment is not None:
        body["comment"] = args.comment

    _post_mcp_json("/api/v1/mcp/feedback", body)


# ---------------------------------------------------------------------------
# config / setup commands
# ---------------------------------------------------------------------------


def cmd_config_show(_args: argparse.Namespace) -> None:
    """Show current effective configuration with source annotations / 显示当前生效配置及值来源。"""
    from team_memory.client_config import ClientConfig

    cfg, sources = ClientConfig.load_with_sources()
    data = cfg.to_dict()

    # 按分组输出，每个字段标注来源 / output by section with source annotation
    sections = [
        ("server", ["url", "api_key"]),
        ("llm", ["base_url", "api_key", "api_key_env", "resolved_api_key", "model", "timeout"]),
        ("conversation", ["db_path"]),
        ("wiki", ["enabled", "db_path"]),
        ("daemon", ["host", "port", "log_level"]),
    ]

    for section_name, fields in sections:
        print(f"[{section_name}]")
        section_data = data.get(section_name, {})
        for field_name in fields:
            value = section_data.get(field_name, "")
            source_key = f"{section_name}.{field_name}"
            # resolved_api_key 没有 source / no source for derived fields
            source = sources.get(source_key, "derived")
            print(f"  {field_name} = {value}  ({source})")
        print()


def cmd_config_init(args: argparse.Namespace) -> None:
    """Initialize ~/.config/tm/config.toml / 初始化客户端配置。"""
    import tomli_w

    config_dir = Path.home() / ".config" / "tm"
    config_file = config_dir / "config.toml"

    # Gather values / 收集配置值
    server_url = args.server_url
    api_key = args.api_key

    if not args.non_interactive:
        if not server_url:
            server_url = input("TM server URL [http://localhost:9111]: ").strip()
            if not server_url:
                server_url = "http://localhost:9111"
        if not api_key:
            api_key = input("TM API key: ").strip()

    if not api_key and not args.non_interactive:
        print("Warning: No API key provided. Set TEAM_MEMORY_API_KEY env var or edit config.")

    # Build config dict / 构建配置字典
    config_data: dict[str, Any] = {
        "server": {
            "url": server_url,
        },
    }
    if api_key:
        config_data["server"]["api_key"] = api_key

    # Write / 写入文件
    config_dir.mkdir(parents=True, exist_ok=True)
    with open(config_file, "wb") as f:
        tomli_w.dump(config_data, f)

    print(f"Config written to {config_file}")

    # 连接测试 / connection test
    _test_connection(server_url, api_key)


def _test_connection(server_url: str, api_key: str | None) -> None:
    """Test HTTP connection and auth to TM server / 测试与服务器的连接和认证。"""
    print(f"\nTesting connection to {server_url} ...")

    # 测试 HTTP 连接 / test HTTP reachability
    try:
        resp = httpx.get(f"{server_url}/health", timeout=5.0)
    except httpx.ConnectError:
        print(f"  ❌ Cannot connect to {server_url}. Is the server running?")
        return
    except httpx.TimeoutException:
        print(f"  ❌ Connection timed out to {server_url}.")
        return
    except Exception as e:
        print(f"  ❌ Connection error: {e}")
        return

    # 测试 API key 认证 / test API key authentication
    if resp.status_code == 200:
        print("  ✅ Server is reachable (HTTP 200)")
    else:
        print(f"  ⚠️  Server responded with status {resp.status_code}")

    if not api_key:
        print("  ⚠️  No API key configured — skipping auth test")
        return

    try:
        auth_resp = httpx.get(
            f"{server_url}/health",
            headers={"Authorization": f"Bearer {api_key}"},
            timeout=5.0,
        )
        if auth_resp.status_code == 200:
            print("  ✅ API key authentication successful")
        elif auth_resp.status_code == 401:
            print("  ❌ API key authentication failed (401 Unauthorized)")
        else:
            print(f"  ⚠️  Auth test returned status {auth_resp.status_code}")
    except Exception as e:
        print(f"  ❌ Auth test error: {e}")


def cmd_setup(args: argparse.Namespace) -> None:
    """Configure AI client hooks / 一键配置 AI 客户端 hooks。"""
    platforms = (
        ["claude-code", "cursor", "hermes"] if args.platform == "all" else [args.platform]
    )
    from team_memory.client_config import get_client_config

    cfg = get_client_config()

    for platform in platforms:
        if args.check:
            _check_platform(platform, cfg)
        else:
            _setup_platform(platform, cfg)


def _check_platform(platform: str, cfg: Any) -> None:
    """Check hook configuration status / 检查 hook 配置状态。"""
    if platform == "claude-code":
        settings = Path.home() / ".claude" / "settings.json"
        if settings.exists():
            # 检查 hook 条目是否存在 / check if hook entry exists
            try:
                with open(settings) as f:
                    data = json.load(f)
                hooks = data.get("hooks", {})
                post_tool = hooks.get("PostToolUse", [])
                has_tm = any(
                    h.get("matcher") == "mcp__team_memory__recall"
                    for h in post_tool
                )
                if has_tm:
                    print(f"[claude-code] ✅ {settings} exists, PostToolUse hook configured")
                else:
                    print(f"[claude-code] ⚠️  {settings} exists but PostToolUse hook not found")
                    print("  Run: tm-cli setup --platform claude-code")
            except (json.JSONDecodeError, OSError) as e:
                print(f"[claude-code] ⚠️  {settings} exists but cannot parse: {e}")
        else:
            print(f"[claude-code] ❌ {settings} not found")
            print("  Run: tm-cli setup --platform claude-code")

    elif platform == "cursor":
        hooks_file = Path.cwd() / ".cursor" / "hooks.json"
        if hooks_file.exists():
            # 检查 hook 条目是否存在 / check if hook entry exists
            try:
                with open(hooks_file) as f:
                    data = json.load(f)
                # Cursor format: {"hooks": {"afterMCPExecution":
                # [{"matcher": "...", "hooks": [...]}]}}
                hooks_section = data.get("hooks", {})
                if isinstance(hooks_section, dict):
                    after_mcp = hooks_section.get("afterMCPExecution", [])
                    has_tm = any(
                        "team_memory" in entry.get("matcher", "")
                        for entry in after_mcp
                        if isinstance(entry, dict)
                    )
                else:
                    has_tm = False
                if has_tm:
                    print(f"[cursor] ✅ {hooks_file} exists, afterMCPExecution hook configured")
                else:
                    print(f"[cursor] ⚠️  {hooks_file} exists but afterMCPExecution hook not found")
                    print("  Run: tm-cli setup --platform cursor")
            except (json.JSONDecodeError, OSError) as e:
                print(f"[cursor] ⚠️  {hooks_file} exists but cannot parse: {e}")
        else:
            print(f"[cursor] ❌ {hooks_file} not found")
            print("  Run: tm-cli setup --platform cursor")

    elif platform == "hermes":
        config_file = Path.home() / ".hermes" / "config.yaml"
        if config_file.exists():
            print(f"[hermes] ✅ {config_file} exists")
            print("  Verify manually that post_tool_call hook is configured")
        else:
            print(f"[hermes] ❌ {config_file} not found")
            print("  Run: tm-cli setup --platform hermes for instructions")


def _setup_platform(platform: str, cfg: Any) -> None:
    """Write hook configuration / 写入 hook 配置。"""
    venv_python = sys.executable

    if platform == "claude-code":
        _setup_claude_code(cfg, venv_python)
    elif platform == "cursor":
        _setup_cursor(cfg, venv_python)
    elif platform == "hermes":
        _setup_hermes(cfg, venv_python)


def _setup_claude_code(cfg: Any, venv_python: str) -> None:
    """Configure Claude Code PostToolUse hook."""
    import json

    settings_path = Path.home() / ".claude" / "settings.json"
    settings: dict[str, Any] = {}
    if settings_path.exists():
        with open(settings_path) as f:
            settings = json.load(f)

    hook_entry = {
        "matcher": "mcp__team_memory__recall",
        "command": f"{venv_python} -m team_memory.hooks.claude_post_tool_use",
    }

    hooks = settings.setdefault("hooks", {})
    post_tool = hooks.setdefault("PostToolUse", [])
    # Remove existing TM entry if any
    post_tool = [h for h in post_tool if h.get("matcher") != "mcp__team_memory__recall"]
    post_tool.append(hook_entry)
    hooks["PostToolUse"] = post_tool
    settings["hooks"] = hooks

    settings_path.parent.mkdir(parents=True, exist_ok=True)
    with open(settings_path, "w") as f:
        json.dump(settings, f, indent=2)

    print(f"[claude-code] ✅ PostToolUse hook added to {settings_path}")


def _setup_cursor(cfg: Any, venv_python: str) -> None:
    """Configure Cursor afterMCPExecution hook.

    Cursor hooks.json format:
    {"hooks": {"afterMCPExecution":
     [{"matcher": "...", "hooks": [{"type": "command", "command": "..."}]}]}}
    """
    import json

    hooks_path = Path.cwd() / ".cursor" / "hooks.json"
    hooks_data: dict[str, Any] = {}
    if hooks_path.exists():
        with open(hooks_path) as f:
            hooks_data = json.load(f)

    # Ensure nested dict structure / 确保嵌套 dict 结构
    hooks_section = hooks_data.setdefault("hooks", {})
    if isinstance(hooks_section, list):
        # Migrate old format → new format / 迁移旧格式到新格式
        hooks_section = {}
        hooks_data["hooks"] = hooks_section

    after_mcp = hooks_section.setdefault("afterMCPExecution", [])

    # Remove existing TM entry / 移除已有的 TM 条目
    after_mcp = [
        entry for entry in after_mcp
        if not (isinstance(entry, dict) and "team_memory" in entry.get("matcher", ""))
    ]

    # Add new entry in Cursor format / 用 Cursor 格式添加新条目
    after_mcp.append({
        "matcher": "MCP:team_memory",
        "hooks": [
            {
                "type": "command",
                "command": f"{venv_python} -m team_memory.hooks.cursor_after_mcp",
            }
        ],
    })
    hooks_section["afterMCPExecution"] = after_mcp

    hooks_path.parent.mkdir(parents=True, exist_ok=True)
    with open(hooks_path, "w") as f:
        json.dump(hooks_data, f, indent=2)

    print(f"[cursor] ✅ afterMCPExecution hook added to {hooks_path}")


def _setup_hermes(cfg: Any, venv_python: str) -> None:
    """Configure Hermes post_tool_call hook."""
    print("[hermes] ⚠️  Manual setup required:")
    print("  Add to ~/.hermes/config.yaml:")
    print("    hooks:")
    print(f'      post_tool_call: "{venv_python} -m team_memory.hooks.hermes_post_tool_call"')
    print("    hook_matchers:")
    print('      post_tool_call: "recall|memory_recall"')


def cmd_archive(args: argparse.Namespace) -> None:
    """Create or update an archive via POST /api/v1/archives."""
    body: dict = {
        "title": args.title,
        "solution_doc": args.solution_doc,
    }
    if args.content_type:
        body["content_type"] = args.content_type
    if args.value_summary:
        body["value_summary"] = args.value_summary
    if args.tags:
        body["tags"] = [t.strip() for t in args.tags.split(",") if t.strip()]
    if args.overview_file:
        body["overview"] = Path(args.overview_file).read_text(encoding="utf-8")
    elif args.overview:
        body["overview"] = args.overview
    if args.solution_file:
        body["solution_doc"] = Path(args.solution_file).read_text(encoding="utf-8")
    if args.summary:
        body["conversation_summary"] = args.summary
    if args.linked_experience_ids:
        body["linked_experience_ids"] = [
            s.strip() for s in args.linked_experience_ids.split(",") if s.strip()
        ]
    project = _get_project(args.project)
    if project:
        body["project"] = project
    if args.scope:
        body["scope"] = args.scope
    if args.scope_ref:
        body["scope_ref"] = args.scope_ref

    url = f"{_get_base_url()}/api/v1/archives"
    try:
        resp = httpx.post(url, json=body, headers=_headers(), timeout=30.0)
        resp.raise_for_status()
        data = resp.json()
        item = data.get("item") if isinstance(data.get("item"), dict) else data
        action = item.get("action", "unknown")
        archive_id = item.get("archive_id", "?")
        print(f"Archive {action}: {archive_id}")
        if action == "updated" and item.get("previous_updated_at"):
            print(f"  (previous version from {item['previous_updated_at']})")
    except httpx.HTTPStatusError as e:
        print(f"Error {e.response.status_code}: {e.response.text}", file=sys.stderr)
        sys.exit(1)
    except httpx.ConnectError:
        print(f"Error: Cannot connect to {url}. Is the server running?", file=sys.stderr)
        sys.exit(1)


def cmd_upload(args: argparse.Namespace) -> None:
    """Upload a file attachment to an archive."""
    file_path = Path(args.file)
    if not file_path.is_file():
        print(f"Error: File not found: {file_path}", file=sys.stderr)
        sys.exit(1)

    url = f"{_get_base_url()}/api/v1/archives/{args.archive_id}/attachments/upload"
    params: dict[str, str] = {}
    project = _get_project(getattr(args, "project", None))
    if project:
        params["project"] = project
    headers = {"Authorization": f"Bearer {_get_api_key()}"}
    files = {"file": (file_path.name, file_path.open("rb"), "application/octet-stream")}
    data = {"kind": args.kind or "file"}
    if args.snippet:
        data["note"] = args.snippet

    try:
        resp = httpx.post(
            url,
            headers=headers,
            files=files,
            data=data,
            params=params or None,
            timeout=60.0,
        )
        resp.raise_for_status()
        result = resp.json()
        print(f"Uploaded: {result.get('id', '?')}")
        if result.get("download_api_path"):
            print(f"  Download: {_get_base_url()}{result['download_api_path']}")
    except httpx.HTTPStatusError as e:
        print(f"Error {e.response.status_code}: {e.response.text}", file=sys.stderr)
        sys.exit(1)
    except httpx.ConnectError:
        print(f"Error: Cannot connect to {url}. Is the server running?", file=sys.stderr)
        sys.exit(1)


# ---------------------------------------------------------------------------
# Argument parser
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        prog="tm-cli",
        description="Team Memory CLI — archives, uploads, and MCP HTTP helpers",
    )
    sub = parser.add_subparsers(dest="command", required=True)

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
        help="Long text for LLM parse mode (mutually exclusive with title/problem/solution)",
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
    p_cfg = sub.add_parser("config", help="Manage client configuration")
    cfg_sub = p_cfg.add_subparsers(dest="config_command")

    p_cfg_show = cfg_sub.add_parser("show", help="Show current config")
    p_cfg_show.set_defaults(func=cmd_config_show)

    p_cfg_init = cfg_sub.add_parser("init", help="Initialize config interactively")
    p_cfg_init.add_argument("--server-url", help="TM server URL")
    p_cfg_init.add_argument("--api-key", help="TM API key")
    p_cfg_init.add_argument(
        "--non-interactive",
        action="store_true",
        help="Use flags only, no prompts",
    )
    p_cfg_init.set_defaults(func=cmd_config_init)

    # ── setup ──────────────────────────────────────────────────
    p_setup = sub.add_parser("setup", help="Configure AI client hooks")
    p_setup.add_argument(
        "--platform",
        required=True,
        choices=["claude-code", "cursor", "hermes", "all"],
        help="Which platform to configure",
    )
    p_setup.add_argument(
        "--check",
        action="store_true",
        help="Only check current status, don't modify",
    )
    p_setup.set_defaults(func=cmd_setup)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()

"""tm-cli -- command-line interface for Team Memory.

Subcommands:
    archive      Create or update an archive (POST /api/v1/archives)
    upload       Upload a file attachment to an archive
    save         Save team knowledge (POST /api/v1/mcp/save)
    recall       Search team knowledge (POST /api/v1/mcp/recall)
    context      Profile + relevant knowledge (POST /api/v1/mcp/context)
    get-archive  Fetch full archive L2 by id (GET /api/v1/mcp/archive/{id})
    feedback     Rate an experience (POST /api/v1/mcp/feedback)
    config       Manage client configuration (show/check/set/init)
    project      Manage project configuration (init/show/list)
    setup        Configure AI client hooks (with --uninstall support)

Usage:
    python -m team_memory.cli archive --title "..." --solution-doc "..." [options]
    python -m team_memory.cli upload --archive-id <id> --file <path> [options]
    python -m team_memory.cli save --title "..." --problem "..." --solution "..." [options]
    python -m team_memory.cli recall [--query "..."] [options]
    python -m team_memory.cli context [--file-paths "a.py,b.py"] [options]
    python -m team_memory.cli get-archive --id <uuid> [options]
    python -m team_memory.cli feedback --experience-id <uuid> --rating 5 [options]
    python -m team_memory.cli config show
    python -m team_memory.cli config check
    python -m team_memory.cli config set server.url http://localhost:9111
    python -m team_memory.cli project init
    python -m team_memory.cli project show
    python -m team_memory.cli setup [--platform ...] [--uninstall]
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


def cmd_config_check(_args: argparse.Namespace) -> None:
    """Test connectivity for server, LLM, and Ollama/embedding.

    测试服务器、LLM 和 Ollama/embedding 的连通性。
    Returns exit code 1 if any critical check fails.
    """
    from team_memory.client_config import ClientConfig

    cfg = ClientConfig.load()
    any_fail = False

    # ── 1. Server health / 服务器健康检查 ──
    print("1. Server connectivity / 服务器连通性 ...")
    try:
        resp = httpx.get(f"{cfg.server.url}/health", timeout=5.0)
        if resp.status_code == 200:
            print("   ✅ Server reachable (HTTP 200)")
        else:
            print(f"   ⚠️  Server responded with status {resp.status_code}")
            any_fail = True
    except httpx.ConnectError:
        print(f"   ❌ Cannot connect to {cfg.server.url}")
        any_fail = True
    except httpx.TimeoutException:
        print(f"   ❌ Connection timed out to {cfg.server.url}")
        any_fail = True
    except Exception as e:
        print(f"   ❌ Error: {e}")
        any_fail = True

    # ── 2. LLM connectivity / LLM 连通性 ──
    print("2. LLM connectivity / LLM 连通性 ...")
    llm_key = cfg.llm.resolved_api_key
    llm_headers: dict[str, str] = {"Content-Type": "application/json"}
    if llm_key:
        llm_headers["Authorization"] = f"Bearer {llm_key}"
    try:
        # Minimal payload / 最小化请求体
        llm_resp = httpx.post(
            f"{cfg.llm.base_url}/chat/completions",
            headers=llm_headers,
            json={
                "model": cfg.llm.model,
                "messages": [{"role": "user", "content": "hi"}],
                "max_tokens": 1,
            },
            timeout=cfg.llm.timeout,
        )
        if llm_resp.status_code in (200, 201):
            print("   ✅ LLM reachable (chat/completions responded)")
        else:
            print(f"   ⚠️  LLM responded with status {llm_resp.status_code}")
            # Non-critical: LLM failure is warning, not fatal
    except httpx.ConnectError:
        print(f"   ❌ Cannot connect to LLM at {cfg.llm.base_url}")
    except httpx.TimeoutException:
        print(f"   ❌ Connection timed out to LLM at {cfg.llm.base_url}")
    except Exception as e:
        print(f"   ⚠️  LLM check error: {e}")

    # ── 3. Ollama / embedding / Ollama 嵌入模型 ──
    print("3. Ollama/embedding connectivity / Ollama 嵌入连通性 ...")
    ollama_url = "http://localhost:11434"
    try:
        ollama_resp = httpx.get(f"{ollama_url}/api/tags", timeout=5.0)
        if ollama_resp.status_code == 200:
            data = ollama_resp.json()
            models = [m.get("name", "?") for m in data.get("models", [])]
            print(f"   ✅ Ollama reachable ({len(models)} model(s) loaded)")
            if models:
                for m in models[:5]:
                    print(f"      - {m}")
                if len(models) > 5:
                    print(f"      ... and {len(models) - 5} more")
        else:
            print(f"   ⚠️  Ollama responded with status {ollama_resp.status_code}")
    except httpx.ConnectError:
        print(f"   ⚠️  Cannot connect to Ollama at {ollama_url} (not critical)")
    except httpx.TimeoutException:
        print("   ⚠️  Connection timed out to Ollama (not critical)")
    except Exception as e:
        print(f"   ⚠️  Ollama check error: {e} (not critical)")

    if any_fail:
        print("\n❌ Some critical checks failed / 部分关键检查未通过")
        sys.exit(1)
    else:
        print("\n✅ All critical checks passed / 所有关键检查通过")


def cmd_config_set(args: argparse.Namespace) -> None:
    """Set a config key in ~/.config/tm/config.toml.

    使用点号表示法设置配置项，例如: config set server.url http://localhost:9111
    """
    import tomli_w

    try:
        import tomllib
    except ModuleNotFoundError:
        import tomli as tomllib  # type: ignore[no-redef]

    key = args.key  # e.g. "server.url"
    value = args.value

    # Parse dot notation / 解析点号表示法
    parts = key.split(".")
    if len(parts) < 2:
        print(
            "Error: Key must be in section.field format,"
            " e.g. server.url / 键必须为 section.field 格式",
            file=sys.stderr,
        )
        sys.exit(1)

    # Read existing config / 读取现有配置
    config_dir = Path.home() / ".config" / "tm"
    config_file = config_dir / "config.toml"
    config_data: dict[str, Any] = {}
    if config_file.exists():
        try:
            with open(config_file, "rb") as f:
                config_data = tomllib.load(f)
        except Exception as e:
            print(
                f"Warning: Cannot read existing config: {e}"
                " / 警告：无法读取现有配置",
                file=sys.stderr,
            )

    # Try to convert value to appropriate type / 尝试类型转换
    converted_value: Any = value
    if value.lower() == "true":
        converted_value = True
    elif value.lower() == "false":
        converted_value = False
    else:
        try:
            converted_value = int(value)
        except ValueError:
            try:
                converted_value = float(value)
            except ValueError:
                pass  # Keep as string / 保持字符串

    # Set nested key / 设置嵌套键
    current = config_data
    for part in parts[:-1]:
        if part not in current or not isinstance(current[part], dict):
            current[part] = {}
        current = current[part]
    current[parts[-1]] = converted_value

    # Write back / 写回文件
    config_dir.mkdir(parents=True, exist_ok=True)
    with open(config_file, "wb") as f:
        tomli_w.dump(config_data, f)

    print(f"Set {key} = {converted_value!r} in {config_file}")
    print(f"已设置 {key} = {converted_value!r} 于 {config_file}")


def cmd_project_init(args: argparse.Namespace) -> None:
    """Generate .tm.toml in current directory / 在当前目录生成 .tm.toml。

    Auto-detect project name from git repo name.
    If .tm.toml already exists, print warning and exit.
    """
    import tomli_w

    tm_path = Path.cwd() / ".tm.toml"
    if tm_path.exists():
        print(
            f"Warning: .tm.toml already exists at {tm_path}"
            " / 警告：.tm.toml 已存在",
            file=sys.stderr,
        )
        sys.exit(1)

    # Auto-detect project name / 自动检测项目名称
    detected_name = _detect_project_name()

    # Write .tm.toml / 写入 .tm.toml
    project_data: dict[str, Any] = {
        "project": {
            "name": detected_name,
            "aliases": [],
        }
    }

    with open(tm_path, "wb") as f:
        tomli_w.dump(project_data, f)

    print(f"Created .tm.toml with project name: {detected_name}")
    print(f"已创建 .tm.toml，项目名称: {detected_name}")


def _detect_project_name() -> str:
    """Detect project name from git repo / 从 git 仓库检测项目名称。"""
    import subprocess

    try:
        result = subprocess.run(
            ["git", "rev-parse", "--show-toplevel"],
            capture_output=True,
            text=True,
            timeout=3,
        )
        if result.returncode == 0 and result.stdout.strip():
            return os.path.basename(result.stdout.strip()).replace("-", "_")
    except (FileNotFoundError, subprocess.TimeoutExpired, OSError):
        pass
    # Fallback: directory name / 回退：目录名
    return os.path.basename(os.getcwd()).replace("-", "_")


def cmd_project_show(_args: argparse.Namespace) -> None:
    """Print current project name and aliases / 显示当前项目名称和别名。"""
    from team_memory.project_resolver import _git_repo_name, read_tm_toml

    cwd = os.getcwd()
    tm_path = Path.cwd() / ".tm.toml"

    print(f"Project info / 项目信息 (cwd: {cwd})")
    print()

    # .tm.toml info / .tm.toml 信息
    if tm_path.exists():
        tm_config = read_tm_toml(cwd)
        if tm_config:
            print(f"  .tm.toml path: {tm_path}")
            print(f"  Project name:  {tm_config.get('name', '(not set)')}")
            aliases = tm_config.get("aliases", [])
            print(f"  Aliases:       {aliases if aliases else '(none)'}")
        else:
            print(f"  .tm.toml path: {tm_path} (invalid or no [project].name)")
    else:
        print("  .tm.toml:      (not found)")

    # Git repo name / Git 仓库名
    git_name = _git_repo_name(cwd)
    print(f"  Git repo name: {git_name or '(not a git repo)'}")

    # Directory name / 目录名
    dir_name = os.path.basename(os.path.abspath(cwd))
    print(f"  Directory name: {dir_name}")

    # Inferred name / 推断名称
    from team_memory.project_resolver import infer_project_name
    inferred = infer_project_name(cwd)
    print(f"  Inferred name: {inferred}")


def cmd_project_list(args: argparse.Namespace) -> None:
    """List all projects with experience count / 列出所有项目及经验数量。"""
    headers = _headers()
    base_url = _get_base_url()

    # Try dedicated projects endpoint / 尝试专用的项目端点
    try:
        resp = httpx.get(f"{base_url}/api/v1/projects", headers=headers, timeout=10.0)
        if resp.status_code == 200:
            data = resp.json()
            projects = data.get("projects", data) if isinstance(data, dict) else data
            if isinstance(projects, list):
                print(f"Projects ({len(projects)} found) / 项目（共 {len(projects)} 个）:")
                for p in projects:
                    name = p if isinstance(p, str) else p.get("name", p.get("project", "?"))
                    print(f"  {name}")
                return
    except (httpx.HTTPStatusError, httpx.ConnectError):
        pass

    # Fallback: query experiences and extract unique projects / 回退：查询经验并提取唯一项目
    try:
        resp = httpx.get(
            f"{base_url}/api/v1/experiences",
            headers=headers,
            params={"limit": "100"},
            timeout=10.0,
        )
        if resp.status_code == 200:
            data = resp.json()
            project_counts: dict[str, int] = {}
            items = (data if isinstance(data, list)
                else data.get("items", data.get("experiences", [])))
            for item in items:
                proj = item.get("project", "default")
                project_counts[proj] = project_counts.get(proj, 0) + 1
            if project_counts:
                n = len(project_counts)
                print(f"Projects ({n} found) / 项目（共 {n} 个）:")
                for name, count in sorted(project_counts.items()):
                    print(f"  {name}: {count} experiences")
            else:
                print("No projects found / 未找到项目")
        else:
            print(f"Error: API returned status {resp.status_code}", file=sys.stderr)
            sys.exit(1)
    except httpx.ConnectError:
        print(f"Error: Cannot connect to {base_url}. Is the server running?", file=sys.stderr)
        sys.exit(1)

    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


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
    from team_memory.client_config import get_client_config

    cfg = get_client_config()

    # Handle uninstall / 处理卸载
    if args.uninstall:
        platforms = (
            ["claude-code", "cursor", "hermes"] if args.platform == "all" else [args.platform]
        )
        for platform in platforms:
            _uninstall_platform(platform)
        return

    # Handle check / 处理检查
    if args.check:
        platforms = (
            ["claude-code", "cursor", "hermes"] if args.platform == "all" else [args.platform]
        )
        for platform in platforms:
            _check_platform(platform, cfg)
        return

    # No --platform: interactive wizard / 无 --platform：交互式向导
    if not args.platform:
        _setup_wizard(args, cfg)
        return

    # Specific platform / 指定平台
    platforms = (
        ["claude-code", "cursor", "hermes"] if args.platform == "all" else [args.platform]
    )
    for platform in platforms:
        _setup_platform(platform, cfg)


def _setup_wizard(args: argparse.Namespace, cfg: Any) -> None:
    """Interactive setup wizard / 交互式配置向导。"""
    non_interactive = getattr(args, "non_interactive", False)

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
    # Create a mock args for config check
    check_args = argparse.Namespace()
    cmd_config_check(check_args)

    # Step d: Ask which platforms / 步骤 d: 询问平台
    platforms = ["claude-code", "cursor", "hermes"]
    selected = []
    if non_interactive:
        selected = platforms
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
        _setup_platform(platform, cfg)


def _uninstall_platform(platform: str) -> None:
    """Remove TM hooks from platform config / 移除平台配置中的 TM hooks。"""
    if platform == "claude-code":
        _uninstall_claude_code()
    elif platform == "cursor":
        _uninstall_cursor()
    elif platform == "hermes":
        _uninstall_hermes()


def _uninstall_claude_code() -> None:
    """Remove TM hooks from Claude Code settings / 从 Claude Code 设置移除 TM hooks。"""
    settings_path = Path.home() / ".claude" / "settings.json"
    if not settings_path.exists():
        print("[claude-code] ⚠️  No settings.json found, nothing to uninstall")
        return

    try:
        with open(settings_path) as f:
            settings = json.load(f)

        hooks = settings.get("hooks", {})
        post_tool = hooks.get("PostToolUse", [])
        original_len = len(post_tool)
        post_tool = [h for h in post_tool if h.get("matcher") != "mcp__team_memory__recall"]

        if len(post_tool) == original_len:
            print("[claude-code] ⚠️  No TM hooks found to remove")
            return

        hooks["PostToolUse"] = post_tool
        settings["hooks"] = hooks

        with open(settings_path, "w") as f:
            json.dump(settings, f, indent=2)

        print(f"[claude-code] ✅ Removed TM hooks from {settings_path}")
        print(f"[claude-code] ✅ 已从 {settings_path} 移除 TM hooks")
    except (json.JSONDecodeError, OSError) as e:
        print(f"[claude-code] ❌ Error: {e}")


def _uninstall_cursor() -> None:
    """Remove TM hooks from Cursor hooks.json / 从 Cursor hooks.json 移除 TM hooks。"""
    hooks_path = Path.cwd() / ".cursor" / "hooks.json"
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

        after_mcp = hooks_section.get("afterMCPExecution", [])
        original_len = len(after_mcp)
        after_mcp = [
            entry for entry in after_mcp
            if not (isinstance(entry, dict) and "team_memory" in entry.get("matcher", ""))
        ]

        if len(after_mcp) == original_len:
            print("[cursor] ⚠️  No TM hooks found to remove")
            return

        hooks_section["afterMCPExecution"] = after_mcp
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


def _check_platform(platform: str, cfg: Any) -> None:
    """Check hook configuration status / 检查 hook 配置状态。"""
    hook_dir = Path(__file__).resolve().parent.parent / "hooks"

    if platform == "claude-code":
        _check_claude(hook_dir)
    elif platform == "cursor":
        _check_cursor(hook_dir)
    elif platform == "hermes":
        _check_hermes(hook_dir)


def _check_claude(hook_dir: Path) -> None:
    """Check all Claude Code hooks / 检查 Claude Code 全部 hook。"""
    settings = Path.home() / ".claude" / "settings.json"
    if not settings.exists():
        print(f"[claude-code] ❌ {settings} not found")
        return

    try:
        data = json.loads(settings.read_text())
    except Exception as e:
        print(f"[claude-code] ❌ parse error: {e}")
        return

    hooks_map = {
        "SessionStart": "claude_session_start.py",
        "UserPromptSubmit": "claude_user_prompt_submit.py",
        "Stop": "claude_stop.py",
        "PostToolUse": "claude_post_tool.py",
    }
    for event, script in hooks_map.items():
        ok = _hook_ok(data, event, hook_dir / script)
        print(f"[claude-code] {'✅' if ok else '❌'} {event}")


def _check_cursor(hook_dir: Path) -> None:
    """Check all Cursor hooks / 检查 Cursor 全部 hook。"""
    hooks_file = Path.cwd() / ".cursor" / "hooks.json"
    if not hooks_file.exists():
        print(f"[cursor] ❌ {hooks_file} not found")
        return

    try:
        data = json.loads(hooks_file.read_text())
    except Exception as e:
        print(f"[cursor] ❌ parse error: {e}")
        return

    hooks_map = {
        "sessionStart": "cursor_session_start.py",
        "beforeSubmitPrompt": "cursor_before_prompt.py",
        "afterAgentResponse": "cursor_after_response.py",
        "afterMCPExecution": "cursor_after_mcp.py",
    }
    for event, script in hooks_map.items():
        ok = _hook_ok(data, event, hook_dir / script)
        print(f"[cursor] {'✅' if ok else '❌'} {event}")


def _check_hermes(hook_dir: Path) -> None:
    """Check all Hermes hooks / 检查 Hermes 全部 hook。"""
    import yaml

    config_file = Path.home() / ".hermes" / "config.yaml"
    if not config_file.exists():
        print(f"[hermes] ❌ {config_file} not found")
        return

    try:
        config = yaml.safe_load(config_file.read_text()) or {}
    except Exception as e:
        print(f"[hermes] ❌ parse error: {e}")
        return

    hooks_map = {
        "UserPromptSubmit": "hermes_user_prompt_submit.py",
        "Stop": "hermes_stop.py",
        "PostToolUse": "hermes_post_tool_call.py",
    }
    for event, script in hooks_map.items():
        ok = _hook_ok(config, event, hook_dir / script)
        print(f"[hermes] {'✅' if ok else '❌'} {event}")


def _hook_ok(data: dict, event: str, script_path: Path) -> bool:
    """Check if hook event exists in config and script exists on disk."""
    hooks = data.get("hooks", {})
    entries = hooks.get(event, [])
    return len(entries) > 0 and script_path.exists()


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
    """Configure Claude Code hooks (SessionStart, UserPromptSubmit, Stop, PostToolUse).

    配置 Claude Code 全部 hook 事件。
    """
    import json

    settings_path = Path.home() / ".claude" / "settings.json"
    settings: dict[str, Any] = {}
    if settings_path.exists():
        with open(settings_path) as f:
            settings = json.load(f)

    hooks = settings.setdefault("hooks", {})

    # SessionStart hook
    hooks["SessionStart"] = [
        {
            "matcher": "",
            "hooks": [{
                "type": "command",
                "command": f"{venv_python} -m team_memory.hooks.claude_session_start",
                "timeout": 10000,
            }],
        }
    ]

    # UserPromptSubmit hook
    hooks["UserPromptSubmit"] = [
        {
            "matcher": "",
            "hooks": [{
                "type": "command",
                "command": f"{venv_python} -m team_memory.hooks.claude_user_prompt_submit",
                "timeout": 10000,
            }],
        }
    ]

    # Stop hook
    hooks["Stop"] = [
        {
            "matcher": "",
            "hooks": [{
                "type": "command",
                "command": f"{venv_python} -m team_memory.hooks.claude_stop",
                "timeout": 10000,
            }],
        }
    ]

    # PostToolUse hook (TM recall only; preserve existing entries like ruff)
    post_tool = hooks.get("PostToolUse", [])
    # Remove old TM PostToolUse entries
    post_tool = [
        h for h in post_tool
        if "team_memory" not in json.dumps(h) and "scripts/hooks" not in json.dumps(h)
    ]
    post_tool.append({
        "matcher": "mcp__team_memory__recall|mcp__team_memory__memory_recall",
        "hooks": [{
            "type": "command",
            "command": f"{venv_python} -m team_memory.hooks.claude_post_tool_use",
            "timeout": 10000,
        }],
    })
    hooks["PostToolUse"] = post_tool

    settings["hooks"] = hooks
    settings_path.parent.mkdir(parents=True, exist_ok=True)
    with open(settings_path, "w") as f:
        json.dump(settings, f, indent=2)

    print(f"[claude-code] ✅ All hooks configured in {settings_path}")


def _setup_cursor(cfg: Any, venv_python: str) -> None:
    """Configure Cursor hooks: sessionStart, beforeSubmitPrompt, afterAgentResponse, afterMCPExecution."""
    import json

    hooks_path = Path.cwd() / ".cursor" / "hooks.json"
    hooks_data: dict[str, Any] = {}
    if hooks_path.exists():
        with open(hooks_path) as f:
            hooks_data = json.load(f)

    hooks_section = hooks_data.setdefault("hooks", {})
    if isinstance(hooks_section, list):
        hooks_section = {}
        hooks_data["hooks"] = hooks_section

    cursor_hooks = [
        ("sessionStart", "cursor_session_start", ""),
        ("beforeSubmitPrompt", "cursor_before_prompt", ""),
        ("afterAgentResponse", "cursor_after_response", ""),
        ("afterMCPExecution", "cursor_after_mcp", "MCP:team_memory"),
    ]
    for event, script, matcher in cursor_hooks:
        hooks_section[event] = [
            {"matcher": matcher, "hooks": [{"type": "command",
             "command": f"{venv_python} -m team_memory.hooks.{script}"}]}
        ]

    hooks_path.parent.mkdir(parents=True, exist_ok=True)
    with open(hooks_path, "w") as f:
        json.dump(hooks_data, f, indent=2)
    print(f"[cursor] ✅ 4 hooks written to {hooks_path}")


def _setup_hermes(cfg: Any, venv_python: str) -> None:
    """Configure Hermes hooks: UserPromptSubmit, Stop, PostToolUse. Auto-writes config.yaml."""
    import yaml

    config_path = Path.home() / ".hermes" / "config.yaml"
    config: dict[str, Any] = {}
    if config_path.exists():
        config = yaml.safe_load(config_path.read_text()) or {}

    existing_hooks = config.get("hooks", {})
    tm_hooks = {
        "UserPromptSubmit": [{"matcher": "", "hooks": [{"type": "command",
            "command": f"{venv_python} -m team_memory.hooks.hermes_user_prompt_submit",
            "timeout": 10000}]}],
        "Stop": [{"matcher": "", "hooks": [{"type": "command",
            "command": f"{venv_python} -m team_memory.hooks.hermes_stop",
            "timeout": 10000}]}],
        "PostToolUse": [{"matcher": "recall|memory_recall", "hooks": [{"type": "command",
            "command": f"{venv_python} -m team_memory.hooks.hermes_post_tool_call",
            "timeout": 10000}]}],
    }
    config["hooks"] = {**existing_hooks, **tm_hooks}

    config_path.parent.mkdir(parents=True, exist_ok=True)
    config_path.write_text(yaml.dump(config, allow_unicode=True,
                                      default_flow_style=False, sort_keys=False))
    print(f"[hermes] ✅ 3 hooks written to {config_path}")


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

def _get_queue_db_path() -> str:
    """Resolve the drafts.db path for queue access / 解析队列数据库路径。"""
    import os
    env_path = os.environ.get("TM_PIPELINE_DB_PATH")
    if env_path:
        return env_path
    from pathlib import Path
    default = Path.home() / ".cache" / "tm-pipeline" / "drafts.db"
    return str(default)


def cmd_queue_list(args: Any) -> None:
    """List pending retry operations / 列出待重试操作。"""
    from team_memory.daemon.pending_queue import PendingQueue
    db_path = _get_queue_db_path()
    try:
        queue = PendingQueue(db_path=db_path)
    except Exception as e:
        print(f"Error opening queue: {e}")
        return

    ops = queue.list_ops(op_type=getattr(args, "op_type", None))
    if not ops:
        print("Queue is empty")
        return

    print(f"{'ID':<14} {'Type':<12} {'Session':<14} {'Retries':<8} {'Status':<10} {'Next Retry'}")
    print("-" * 80)
    for op in ops:
        next_retry = (op.get("next_retry_at") or "")[:19]
        print(f"{op['id']:<14} {op['op_type']:<12} {(op.get('session_id') or '')[:12]:<14} "
              f"{op['retries']:<8} {op['status']:<10} {next_retry}")


def cmd_queue_stats(args: Any) -> None:
    """Print queue statistics / 打印队列统计。"""
    from team_memory.daemon.pending_queue import PendingQueue
    db_path = _get_queue_db_path()
    try:
        queue = PendingQueue(db_path=db_path)
    except Exception as e:
        print(f"Error opening queue: {e}")
        return

    s = queue.stats()
    print(f"Queue: {s['pending']} pending, {s['retrying']} retrying, {s['failed']} failed")


def cmd_queue_retry(args: Any) -> None:
    """Force immediate retry of a queued operation / 强制立即重试。"""
    from team_memory.daemon.pending_queue import PendingQueue
    db_path = _get_queue_db_path()
    try:
        queue = PendingQueue(db_path=db_path)
    except Exception as e:
        print(f"Error opening queue: {e}")
        return

    from datetime import datetime, timezone
    now = datetime.now(timezone.utc).isoformat()
    queue._db.execute(
        "UPDATE pending_ops SET next_retry_at = ?, status = 'pending' WHERE id = ?",
        (now, args.op_id),
    )
    queue._db.commit()
    print(f"Reset retry timer for {args.op_id} — will be picked up on next tick")


def cmd_queue_clear(args: Any) -> None:
    """Clear queue entries / 清空队列条目。"""
    from team_memory.daemon.pending_queue import PendingQueue
    db_path = _get_queue_db_path()
    try:
        queue = PendingQueue(db_path=db_path)
    except Exception as e:
        print(f"Error opening queue: {e}")
        return

    if getattr(args, "op_id", None):
        queue._db.execute("DELETE FROM pending_ops WHERE id = ?", (args.op_id,))
        queue._db.commit()
        print(f"Cleared {args.op_id}")
    elif getattr(args, "clear_all", None):
        count = queue._db.execute("SELECT COUNT(*) FROM pending_ops").fetchone()[0]
        queue._db.execute("DELETE FROM pending_ops")
        queue._db.commit()
        print(f"Cleared {count} entries")
    else:
        print("Use --id <id> or --all")

def cmd_metrics_show(args: Any) -> None:
    """Show current metrics / 显示当前指标."""
    from team_memory.daemon.metrics_collector import MetricsCollector
    mc = MetricsCollector(db_path=_get_queue_db_path())
    rows = mc.query(name=getattr(args, "name", None))
    if not rows:
        print("No metrics recorded yet")
        return
    print(f"{'Name':<30} {'Labels':<30} {'Value':<10}")
    print("-" * 72)
    for r in rows:
        labels = r.get("labels", "{}")
        if isinstance(labels, str):
            import json as _json
            try:
                labels = _json.loads(labels)
            except Exception:
                pass
        if isinstance(labels, dict):
            label_str = ",".join(f"{k}={v}" for k, v in sorted(labels.items()))
        else:
            label_str = str(labels)
        print(f"{r['name']:<30} {label_str[:28]:<30} {r['value']:<10.2f}")


def cmd_metrics_summary(args: Any) -> None:
    """Print aggregated metrics summary / 打印聚合指标摘要."""
    from team_memory.daemon.metrics_collector import MetricsCollector
    mc = MetricsCollector(db_path=_get_queue_db_path())
    s = mc.summary(hours=getattr(args, "hours", 24))
    if not s:
        print("No metrics data in time window")
        return
    for name, labels_dict in sorted(s.items()):
        print(f"{name}:")
        for label_key, val in sorted(labels_dict.items()):
            print(f"  {label_key}: {val:.2f}")

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

    p_cfg_init = cfg_sub.add_parser(
        "init", help="Initialize config interactively / 交互式初始化配置")
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
    p_setup = sub.add_parser("setup", help="Configure AI client hooks / 配置 AI 客户端 hooks")
    p_setup.add_argument(
        "--platform",
        choices=["claude-code", "cursor", "hermes", "all"],
        help="Which platform to configure / 配置哪个平台 (omit for interactive wizard)",
    )
    p_setup.add_argument(
        "--check",
        action="store_true",
        help="Only check current status, don't modify / 仅检查状态",
    )
    p_setup.add_argument(
        "--uninstall",
        action="store_true",
        help="Remove TM hooks instead of adding / 移除而非添加 TM hooks",
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
    p_setup.set_defaults(func=cmd_setup)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()

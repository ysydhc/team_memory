"""Client config and project management subcommands for tm-cli."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Any

from team_memory.cli.http_client import _get_base_url, _headers, _httpx


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

    hx = _httpx()
    cfg = ClientConfig.load()
    any_fail = False

    # ── 1. Server health / 服务器健康检查 ──
    print("1. Server connectivity / 服务器连通性 ...")
    try:
        resp = hx.get(f"{cfg.server.url}/health", timeout=5.0)
        if resp.status_code == 200:
            print("   ✅ Server reachable (HTTP 200)")
        else:
            print(f"   ⚠️  Server responded with status {resp.status_code}")
            any_fail = True
    except hx.ConnectError:
        print(f"   ❌ Cannot connect to {cfg.server.url}")
        any_fail = True
    except hx.TimeoutException:
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
        llm_resp = hx.post(
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
    except hx.ConnectError:
        print(f"   ❌ Cannot connect to LLM at {cfg.llm.base_url}")
    except hx.TimeoutException:
        print(f"   ❌ Connection timed out to LLM at {cfg.llm.base_url}")
    except Exception as e:
        print(f"   ⚠️  LLM check error: {e}")

    # ── 3. Ollama / embedding / Ollama 嵌入模型 ──
    print("3. Ollama/embedding connectivity / Ollama 嵌入连通性 ...")
    ollama_url = "http://localhost:11434"
    try:
        ollama_resp = hx.get(f"{ollama_url}/api/tags", timeout=5.0)
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
    except hx.ConnectError:
        print(f"   ⚠️  Cannot connect to Ollama at {ollama_url} (not critical)")
    except hx.TimeoutException:
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
            "Error: Key must be in section.field format, e.g. server.url / 键必须为 section.field 格式",
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
                f"Warning: Cannot read existing config: {e} / 警告：无法读取现有配置",
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


def cmd_project_init(args: argparse.Namespace) -> None:
    """Generate .tm.toml in current directory / 在当前目录生成 .tm.toml。

    Auto-detect project name from git repo name.
    If .tm.toml already exists, print warning and exit.
    """
    import tomli_w

    tm_path = Path.cwd() / ".tm.toml"
    if tm_path.exists():
        print(
            f"Warning: .tm.toml already exists at {tm_path} / 警告：.tm.toml 已存在",
            file=sys.stderr,
        )
        sys.exit(1)

    # Auto-detect project name / 自动检测项目名称 (patch-aware via team_memory.cli)
    pkg = sys.modules.get("team_memory.cli")
    detect_fn = getattr(pkg, "_detect_project_name", _detect_project_name)
    detected_name = detect_fn()

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


def cmd_project_show(_args: argparse.Namespace) -> None:
    """Print current project name and aliases / 显示当前项目名称和别名。"""
    from team_memory.project_resolver import _git_repo_name, infer_project_name, read_tm_toml

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
    inferred = infer_project_name(cwd)
    print(f"  Inferred name: {inferred}")


def cmd_project_list(args: argparse.Namespace) -> None:
    """List all projects with experience count / 列出所有项目及经验数量。"""
    hx = _httpx()
    headers = _headers()
    base_url = _get_base_url()

    # Try dedicated projects endpoint / 尝试专用的项目端点
    try:
        resp = hx.get(f"{base_url}/api/v1/projects", headers=headers, timeout=10.0)
        if resp.status_code == 200:
            data = resp.json()
            projects = data.get("projects", data) if isinstance(data, dict) else data
            if isinstance(projects, list):
                print(f"Projects ({len(projects)} found) / 项目（共 {len(projects)} 个）:")
                for p in projects:
                    name = p if isinstance(p, str) else p.get("name", p.get("project", "?"))
                    print(f"  {name}")
                return
    except (hx.HTTPStatusError, hx.ConnectError):
        pass

    # Fallback: query experiences and extract unique projects / 回退：查询经验并提取唯一项目
    try:
        resp = hx.get(
            f"{base_url}/api/v1/experiences",
            headers=headers,
            params={"limit": "100"},
            timeout=10.0,
        )
        if resp.status_code == 200:
            data = resp.json()
            project_counts: dict[str, int] = {}
            items = data if isinstance(data, list) else data.get("items", data.get("experiences", []))
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
    except hx.ConnectError:
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
    hx = _httpx()
    print(f"\nTesting connection to {server_url} ...")

    # 测试 HTTP 连接 / test HTTP reachability
    try:
        resp = hx.get(f"{server_url}/health", timeout=5.0)
    except hx.ConnectError:
        print(f"  ❌ Cannot connect to {server_url}. Is the server running?")
        return
    except hx.TimeoutException:
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
        auth_resp = hx.get(
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

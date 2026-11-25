"""Shared HTTP helpers, dotenv/config resolution, and package path utilities for tm-cli."""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any

import httpx as _default_httpx


def _httpx():
    """Resolve httpx from team_memory.cli for unittest.mock.patch compatibility."""
    pkg = sys.modules.get("team_memory.cli")
    return getattr(pkg, "httpx", _default_httpx)


def _pkg_root() -> Path:
    """Return the installed `team_memory` package directory (src/team_memory)."""
    import team_memory

    return Path(team_memory.__file__).resolve().parent


def _get_repo_root() -> Path:
    """Resolve the team_doc repository root directory."""
    return _pkg_root().parent.parent


def _load_dotenv() -> None:
    """Load .env from current directory or parents (best-effort)."""
    try:
        from dotenv import load_dotenv

        # Walk up from cwd to find .env
        load_dotenv(verbose=False)
    except ImportError:
        pass


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


def _get_queue_db_path() -> str:
    """Resolve the drafts.db path for queue access / 解析队列数据库路径。"""
    env_path = os.environ.get("TM_PIPELINE_DB_PATH")
    if env_path:
        return env_path
    default = Path.home() / ".cache" / "tm-pipeline" / "drafts.db"
    return str(default)


def _print_response_json(resp: Any) -> None:
    print(json.dumps(resp.json(), indent=2, ensure_ascii=False))


def _post_mcp_json(path: str, body: dict[str, Any], *, timeout: float = 60.0) -> None:
    hx = _httpx()
    url = f"{_get_base_url()}{path}"
    try:
        resp = hx.post(url, json=body, headers=_headers(), timeout=timeout)
        resp.raise_for_status()
        _print_response_json(resp)
    except hx.HTTPStatusError as e:
        print(f"Error {e.response.status_code}: {e.response.text}", file=sys.stderr)
        sys.exit(1)
    except hx.ConnectError:
        print(f"Error: Cannot connect to {url}. Is the server running?", file=sys.stderr)
        sys.exit(1)


def _get_mcp_json(path: str, *, params: dict[str, str] | None = None, timeout: float = 30.0) -> None:
    hx = _httpx()
    url = f"{_get_base_url()}{path}"
    try:
        resp = hx.get(url, headers=_headers(), params=params or None, timeout=timeout)
        resp.raise_for_status()
        _print_response_json(resp)
    except hx.HTTPStatusError as e:
        print(f"Error {e.response.status_code}: {e.response.text}", file=sys.stderr)
        sys.exit(1)
    except hx.ConnectError:
        print(f"Error: Cannot connect to {url}. Is the server running?", file=sys.stderr)
        sys.exit(1)

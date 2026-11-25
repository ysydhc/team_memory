"""Daemon, server provisioning, PostgreSQL backend, and Alembic migration commands for tm-cli."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

from team_memory.cli.http_client import _get_repo_root, _httpx, _pkg_root


def _resolve_alembic_paths() -> tuple[Path, Path]:
    """Resolve (alembic_ini_path, migrations_dir) from repo root or bundled wheel."""
    repo_root = _get_repo_root()
    repo_ini = repo_root / "alembic.ini"
    repo_mig = repo_root / "migrations"
    if repo_ini.exists() and repo_mig.is_dir():
        return repo_ini, repo_mig

    pkg_dir = _pkg_root()
    bundled_ini = pkg_dir / "_bundled_alembic.ini"
    bundled_mig = pkg_dir / "_bundled_migrations"
    if bundled_ini.exists() and bundled_mig.is_dir():
        return bundled_ini, bundled_mig

    raise FileNotFoundError(
        "Could not locate Alembic migrations in repo root or bundled package "
        f"(checked {repo_mig} and {bundled_mig})"
    )


def run_db_migrations(db_url: str | None = None) -> None:
    """Run Alembic `upgrade head` programmatically using repo or wheel-bundled migrations."""
    from alembic import command
    from alembic.config import Config

    ini_path, mig_dir = _resolve_alembic_paths()
    alembic_cfg = Config(str(ini_path))
    alembic_cfg.set_main_option("script_location", str(mig_dir))
    resolved_url = (
        db_url
        or os.environ.get("TEAM_MEMORY_DATABASE__URL")
        or os.environ.get("TEAM_MEMORY_DB_URL")
        or "postgresql+asyncpg://developer:devpass@localhost:5432/team_memory"
    )
    os.environ["TEAM_MEMORY_DATABASE__URL"] = resolved_url
    alembic_cfg.set_main_option("sqlalchemy.url", resolved_url)
    command.upgrade(alembic_cfg, "head")
    print(f"  ✅ Database schema upgraded to head (migrations: {mig_dir})")


def _resolve_runtime_workdir() -> Path:
    """Return repo root when inside git checkout, else ~/.config/tm for source-free installs."""
    repo_root = _get_repo_root()
    if (repo_root / "pyproject.toml").exists():
        return repo_root
    cfg_dir = Path.home() / ".config" / "tm"
    cfg_dir.mkdir(parents=True, exist_ok=True)
    return cfg_dir


def cmd_daemon_install(_args: argparse.Namespace) -> None:
    """Install and start TM Daemon via macOS launchd (works with or without source repo)."""
    import subprocess

    plist_src = _pkg_root() / "daemon" / "com.teammemory.daemon.plist"
    if not plist_src.exists():
        print(f"Error: plist template not found at {plist_src}", file=sys.stderr)
        sys.exit(1)

    launch_agents = Path.home() / "Library" / "LaunchAgents"
    launch_agents.mkdir(parents=True, exist_ok=True)
    plist_dst = launch_agents / "com.teammemory.daemon.plist"

    work_dir = _resolve_runtime_workdir()
    content = (
        plist_src.read_text(encoding="utf-8")
        .replace("__PYTHON_PATH__", sys.executable)
        .replace("__REPO_ROOT__", str(work_dir))
    )
    plist_dst.write_text(content, encoding="utf-8")

    subprocess.run(["launchctl", "unload", str(plist_dst)], capture_output=True, check=False)
    res = subprocess.run(["launchctl", "load", str(plist_dst)], capture_output=True, text=True, check=False)
    if res.returncode == 0:
        print(f"  ✅ TM Daemon installed & started via launchd ({plist_dst})")
        print(f"     python={sys.executable} | workdir={work_dir}")
    else:
        print(f"  ⚠️  Plist written to {plist_dst}, launchctl load returned: {res.stderr.strip()}")


def cmd_daemon_uninstall(_args: argparse.Namespace) -> None:
    """Stop and remove TM Daemon launchd service."""
    import subprocess

    plist_dst = Path.home() / "Library" / "LaunchAgents" / "com.teammemory.daemon.plist"
    if plist_dst.exists():
        subprocess.run(["launchctl", "unload", str(plist_dst)], capture_output=True, check=False)
        plist_dst.unlink(missing_ok=True)
        print(f"  ✅ TM Daemon uninstalled ({plist_dst} removed)")
    else:
        print("  ℹ TM Daemon launchd plist not found, nothing to uninstall.")


def cmd_daemon_start(_args: argparse.Namespace) -> None:
    """Start TM Daemon service."""
    import subprocess

    plist_dst = Path.home() / "Library" / "LaunchAgents" / "com.teammemory.daemon.plist"
    if not plist_dst.exists():
        cmd_daemon_install(_args)
        return
    subprocess.run(["launchctl", "load", str(plist_dst)], capture_output=True, check=False)
    print("  ✅ TM Daemon start requested via launchctl.")


def cmd_daemon_stop(_args: argparse.Namespace) -> None:
    """Stop TM Daemon service."""
    import subprocess

    plist_dst = Path.home() / "Library" / "LaunchAgents" / "com.teammemory.daemon.plist"
    if plist_dst.exists():
        subprocess.run(["launchctl", "unload", str(plist_dst)], capture_output=True, check=False)
        print("  ✅ TM Daemon stopped via launchctl.")
    else:
        print("  ℹ TM Daemon launchd plist not installed.")


def cmd_daemon_status(_args: argparse.Namespace) -> None:
    """Check TM Daemon HTTP health (:3901/status)."""
    from team_memory.client_config import ClientConfig

    hx = _httpx()
    cfg = ClientConfig.load()
    url = f"http://{cfg.daemon.host}:{cfg.daemon.port}/status"
    try:
        resp = hx.get(url, timeout=2.5)
        if resp.status_code == 200:
            data = resp.json()
            print(f"  ✅ TM Daemon RUNNING at {url} (mode={data.get('mode', '?')})")
        else:
            print(f"  ⚠️  TM Daemon returned HTTP {resp.status_code} at {url}")
    except Exception as exc:
        print(f"  ❌ TM Daemon unreachable at {url} ({exc})")


def _ensure_unified_api_key(explicit_key: str | None = None) -> str:
    """Ensure TEAM_MEMORY_API_KEY exists in ~/.config/tm/config.toml and ~/.config/tm/.env."""
    import secrets

    import tomli_w

    from team_memory.client_config import _CONFIG_DIR, _CONFIG_FILE, _read_config_toml

    _CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    toml_data = _read_config_toml()
    srv_sec = toml_data.setdefault("server", {})
    existing_key = (
        explicit_key
        or os.environ.get("TEAM_MEMORY_API_KEY")
        or srv_sec.get("api_key")
        or ""
    )
    if existing_key.startswith("${"):
        existing_key = ""
    api_key = existing_key.strip() or secrets.token_hex(32)

    srv_sec.setdefault("url", "http://localhost:9111")
    srv_sec["api_key"] = api_key
    with open(_CONFIG_FILE, "wb") as f:
        tomli_w.dump(toml_data, f)

    env_file = _CONFIG_DIR / ".env"
    env_lines: list[str] = []
    if env_file.exists():
        env_lines = [
            ln
            for ln in env_file.read_text(encoding="utf-8").splitlines()
            if not ln.startswith("TEAM_MEMORY_API_KEY=")
        ]
    env_lines.append(f"TEAM_MEMORY_API_KEY={api_key}")
    env_file.write_text("\n".join(env_lines) + "\n", encoding="utf-8")
    os.environ["TEAM_MEMORY_API_KEY"] = api_key
    return api_key


def _set_env_db_mode(mode: str, port: int) -> str:
    """Persist TEAM_MEMORY_DB_MODE and TEAM_MEMORY_DATABASE__URL in ~/.config/tm/.env and repo .env."""
    from team_memory.client_config import _CONFIG_DIR

    db_url = f"postgresql+asyncpg://developer:devpass@localhost:{port}/team_memory"
    targets = [_CONFIG_DIR / ".env"]
    repo_root = _get_repo_root()
    if (repo_root / ".env").exists():
        targets.append(repo_root / ".env")

    for env_file in targets:
        lines: list[str] = []
        if env_file.exists():
            lines = [
                ln
                for ln in env_file.read_text(encoding="utf-8").splitlines()
                if not ln.startswith("TEAM_MEMORY_DB_MODE=")
                and not ln.startswith("TEAM_MEMORY_DATABASE__URL=")
                and not ln.startswith("TEAM_MEMORY_DB_URL=")
            ]
        lines.append(f"TEAM_MEMORY_DB_MODE={mode}")
        lines.append(f'TEAM_MEMORY_DATABASE__URL="{db_url}"')
        lines.append(f'TEAM_MEMORY_DB_URL="{db_url}"')
        env_file.write_text("\n".join(lines) + "\n", encoding="utf-8")

    os.environ["TEAM_MEMORY_DB_MODE"] = mode
    os.environ["TEAM_MEMORY_DATABASE__URL"] = db_url
    os.environ["TEAM_MEMORY_DB_URL"] = db_url
    return db_url


def _ensure_postgres_backend(mode: str = "auto") -> str:
    """Ensure PostgreSQL + pgvector is ready in 'native' (:5432) or 'docker' (:5433) mode."""
    import asyncio
    import shutil
    import socket
    import subprocess
    import time

    def _port_open(p: int) -> bool:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.settimeout(1.0)
            return s.connect_ex(("127.0.0.1", p)) == 0

    resolved_mode = (mode or "auto").strip().lower()
    if resolved_mode == "auto":
        env_mode = os.environ.get("TEAM_MEMORY_DB_MODE", "").strip().lower()
        if env_mode in ("native", "docker"):
            resolved_mode = env_mode
        elif _port_open(5432) or Path("/opt/homebrew/opt/postgresql@16/bin/postgres").exists():
            resolved_mode = "native"
        else:
            resolved_mode = "docker"

    if resolved_mode == "native":
        port = 5432
        if not _port_open(port):
            print("  ⚡ Starting native Homebrew PostgreSQL 16 on 127.0.0.1:5432...")
            if shutil.which("brew"):
                subprocess.run(["brew", "services", "start", "postgresql@16"], check=False)
            for _ in range(15):
                if _port_open(port):
                    break
                time.sleep(1.0)
        else:
            print("  ✅ Native PostgreSQL already listening on 127.0.0.1:5432 (zero Docker overhead)")

        # Stop redundant Docker containers if running to free memory
        repo_root = _get_repo_root()
        if shutil.which("docker"):
            subprocess.run(
                ["docker", "compose", "stop", "postgres", "redis", "ollama"],
                cwd=str(repo_root) if (repo_root / "docker-compose.yml").exists() else None,
                capture_output=True,
                check=False,
            )
    else:
        port = 5433
        if _port_open(port):
            print(f"  ✅ Docker PostgreSQL already listening on 127.0.0.1:{port}")
        else:
            print(f"  🐳 Starting Docker PostgreSQL (pgvector/pgvector:pg16) on port {port}...")
            repo_root = _get_repo_root()
            if (repo_root / "docker-compose.yml").exists():
                subprocess.run(
                    ["docker", "compose", "up", "-d", "postgres"],
                    cwd=str(repo_root),
                    check=False,
                )
            else:
                inspect_res = subprocess.run(
                    ["docker", "inspect", "tm-postgres"],
                    capture_output=True,
                    check=False,
                )
                if inspect_res.returncode == 0:
                    subprocess.run(["docker", "start", "tm-postgres"], check=False)
                else:
                    subprocess.run(
                        [
                            "docker",
                            "run",
                            "-d",
                            "--name",
                            "tm-postgres",
                            "--restart",
                            "unless-stopped",
                            "-e",
                            "POSTGRES_DB=team_memory",
                            "-e",
                            "POSTGRES_USER=developer",
                            "-e",
                            "POSTGRES_PASSWORD=devpass",
                            "-p",
                            f"{port}:5432",
                            "-v",
                            "tm_pgdata:/var/lib/postgresql/data",
                            "pgvector/pgvector:pg16",
                        ],
                        check=True,
                    )
            for _ in range(20):
                if _port_open(port):
                    break
                time.sleep(1.0)

    db_url = _set_env_db_mode(resolved_mode, port)

    async def _ensure_db() -> None:
        import asyncpg

        for _attempt in range(10):
            try:
                conn = await asyncpg.connect(
                    f"postgresql://developer:devpass@127.0.0.1:{port}/postgres",
                    timeout=3.0,
                )
                try:
                    exists = await conn.fetchval(
                        "SELECT 1 FROM pg_database WHERE datname='team_memory'"
                    )
                    if not exists:
                        await conn.execute("CREATE DATABASE team_memory;")
                finally:
                    await conn.close()
                return
            except Exception:
                await asyncio.sleep(1.0)

    asyncio.run(_ensure_db())
    print(f"  ✅ [{resolved_mode.upper()}] PostgreSQL + database 'team_memory' ready on 127.0.0.1:{port}")
    return db_url


def _ensure_web_service_up(port: int = 9111) -> None:
    """Ensure team-memory-web (:9111) is running and healthy (via launchd on macOS)."""
    import subprocess
    import time

    hx = _httpx()
    url = f"http://127.0.0.1:{port}/health"
    try:
        r = hx.get(url, timeout=2.0)
        if r.status_code == 200:
            print(f"  ✅ Web/MCP Server already running at http://localhost:{port}")
            return
    except Exception:
        pass

    work_dir = _resolve_runtime_workdir()
    launch_agents = Path.home() / "Library" / "LaunchAgents"
    if launch_agents.exists() or sys.platform == "darwin":
        launch_agents.mkdir(parents=True, exist_ok=True)
        plist_dst = launch_agents / "com.teammemory.web.plist"
        web_cmd = (
            f'cd "{work_dir}" &amp;&amp; set -a; '
            '[ -f "$HOME/.config/tm/.env" ] &amp;&amp; source "$HOME/.config/tm/.env"; '
            f'[ -f .env ] &amp;&amp; source .env; set +a; exec "{sys.executable}" -m team_memory.web.app'
        )
        plist_xml = f"""<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>Label</key>
    <string>com.teammemory.web</string>
    <key>ProgramArguments</key>
    <array>
        <string>/bin/bash</string>
        <string>-c</string>
        <string>{web_cmd}</string>
    </array>
    <key>WorkingDirectory</key>
    <string>{work_dir}</string>
    <key>RunAtLoad</key>
    <true/>
    <key>KeepAlive</key>
    <true/>
    <key>StandardOutPath</key>
    <string>/tmp/tm-web.log</string>
    <key>StandardErrorPath</key>
    <string>/tmp/tm-web.err</string>
</dict>
</plist>
"""
        plist_dst.write_text(plist_xml, encoding="utf-8")
        subprocess.run(["launchctl", "unload", str(plist_dst)], capture_output=True, check=False)
        subprocess.run(["launchctl", "load", str(plist_dst)], capture_output=True, check=False)
    else:
        subprocess.Popen(
            [sys.executable, "-m", "team_memory.web.app"],
            cwd=str(work_dir),
            stdout=open("/tmp/tm-web.log", "a"),
            stderr=open("/tmp/tm-web.err", "a"),
            start_new_session=True,
        )

    for _ in range(15):
        try:
            r = hx.get(url, timeout=1.5)
            if r.status_code == 200:
                print(f"  ✅ Web/MCP Server started and healthy at http://localhost:{port}")
                return
        except Exception:
            time.sleep(1.0)
    print(f"  ⚠️  Web Server starting in background (check /tmp/tm-web.log if :{port} is not yet ready)")


def cmd_server_up(args: argparse.Namespace) -> None:
    """Provision PostgreSQL (native :5432 or docker :5433), run migrations, unify API key, and start Web (:9111)."""
    api_key = _ensure_unified_api_key(getattr(args, "api_key", None))
    print(f"  🔑 Unified API Key ready in ~/.config/tm/config.toml & ~/.config/tm/.env ({api_key[:8]}...)")
    mode = getattr(args, "mode", "auto") or "auto"
    db_url = _ensure_postgres_backend(mode=mode)
    run_db_migrations(getattr(args, "db_url", None) or db_url)
    if not getattr(args, "skip_web", False) and not getattr(args, "no_web", False):
        _ensure_web_service_up(port=9111)


def cmd_server_down(args: argparse.Namespace) -> None:
    """Stop Web service launchd agent and optionally stop tm-postgres container."""
    import subprocess

    plist_dst = Path.home() / "Library" / "LaunchAgents" / "com.teammemory.web.plist"
    if plist_dst.exists():
        subprocess.run(["launchctl", "unload", str(plist_dst)], capture_output=True, check=False)
        plist_dst.unlink(missing_ok=True)
        print("  ✅ Stopped and removed com.teammemory.web launchd service")
    if getattr(args, "include_db", False):
        subprocess.run(["docker", "stop", "tm-postgres"], capture_output=True, check=False)
        print("  ✅ Stopped tm-postgres Docker container")


def cmd_server_migrate(args: argparse.Namespace) -> None:
    """Run bundled or repo Alembic database migrations (`upgrade head`)."""
    run_db_migrations(getattr(args, "db_url", None))

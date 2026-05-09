#!/usr/bin/env python3
"""Lightweight daemon auto-reloader.

Watches src/team_memory/daemon/*.py for changes. When a file is modified,
stops and restarts the TM Daemon via launchd.

Usage:
    python -m team_memory.daemon.reloader [--interval 5]
    
Or run in background:
    python -m team_memory.daemon.reloader --interval 5 &
"""
import os
import subprocess
import sys
import time
from pathlib import Path

DAEMON_PLIST = "com.teammemory.daemon"
DAEMON_DIR = Path(__file__).parent
DEFAULT_INTERVAL = 3  # seconds


def get_mtime_map(d: Path) -> dict[str, float]:
    """Get modification times of all .py files in directory."""
    return {
        str(f): f.stat().st_mtime
        for f in d.glob("*.py")
        if f.is_file()
    }


def restart_daemon() -> bool:
    """Stop and start the daemon via launchctl."""
    try:
        # Stop
        subprocess.run(
            ["launchctl", "unload", str(Path.home() / "Library/LaunchAgents" / f"{DAEMON_PLIST}.plist")],
            capture_output=True,
            timeout=5,
        )
        time.sleep(1)

        # Start
        result = subprocess.run(
            ["launchctl", "load", str(Path.home() / "Library/LaunchAgents" / f"{DAEMON_PLIST}.plist")],
            capture_output=True,
            timeout=5,
        )
        if result.returncode == 0:
            print("[reloader] Daemon restarted successfully")
            return True
        else:
            print(f"[reloader] Daemon start failed: {result.stderr.decode()}")
            return False
    except Exception as e:
        print(f"[reloader] Restart failed: {e}")
        return False


def main():
    interval = int(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_INTERVAL

    print(f"[reloader] Watching {DAEMON_DIR} every {interval}s")
    prev = get_mtime_map(DAEMON_DIR)

    while True:
        time.sleep(interval)
        curr = get_mtime_map(DAEMON_DIR)

        changed = []
        for f, t in curr.items():
            if f not in prev or prev[f] != t:
                changed.append(os.path.basename(f))

        if changed:
            print(f"[reloader] Detected changes: {', '.join(changed)}")
            restart_daemon()
            time.sleep(2)  # Wait for daemon to start
            prev = get_mtime_map(DAEMON_DIR)
        else:
            prev = curr


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n[reloader] Stopped")

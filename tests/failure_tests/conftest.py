"""Failure injection tests — validates system resilience against faults.

故障注入测试 — 验证系统在故障下的弹性。
"""

from __future__ import annotations

import sqlite3
import subprocess
import time
from pathlib import Path

import httpx
import pytest

DAEMON_URL = "http://localhost:3901"
WAIT_TIMEOUT = 30


def _daemon_ready() -> bool:
    """Check if daemon /status returns 200 / 检查 daemon 是否就绪。"""
    try:
        r = httpx.get(f"{DAEMON_URL}/status", timeout=2)
        return r.status_code == 200
    except Exception:
        return False


def _cleanup_test_data() -> None:
    """Remove test sessions from SQLite + PG / 清理测试数据。"""
    db_path = Path.home() / ".cache" / "tm-pipeline" / "drafts.db"
    try:
        db = sqlite3.connect(str(db_path))
        db.execute("DELETE FROM sessions WHERE id LIKE 'fail-%'")
        db.execute("DELETE FROM messages WHERE session_id LIKE 'fail-%'")
        db.commit()
    except Exception:
        pass


@pytest.fixture(scope="module")
def daemon():
    """Start daemon once per module. Cleanup test data on teardown.

    启动 daemon（module scope），所有测试复用。teardown 清理测试数据。
    """
    # Start daemon
    subprocess.run(["make", "daemon-start"], capture_output=True, text=True)

    # Wait for ready
    for _ in range(WAIT_TIMEOUT):
        if _daemon_ready():
            break
        time.sleep(1)
    else:
        pytest.skip(f"Daemon did not start within {WAIT_TIMEOUT}s")

    yield DAEMON_URL

    # Cleanup
    _cleanup_test_data()

    # Stop daemon
    subprocess.run(["make", "daemon-stop"], capture_output=True, text=True)


@pytest.fixture
def unique_session() -> str:
    """Generate unique session ID for test isolation / 生成唯一 session ID。"""
    return f"fail-{int(time.time() * 1000)}"

"""Kill daemon mid-refining — 验证 refining 状态 session 重启后恢复。"""

from __future__ import annotations

import subprocess
import time

import httpx
import pytest


@pytest.mark.failure_injection
class TestKillRefining:
    def test_refining_session_recovered_on_restart(self, daemon: str, unique_session: str) -> None:
        """Kill daemon while session is refining → restart → session recovered."""
        sid = unique_session

        # 1. Send 3 rounds to trigger convergence
        for i in range(3):
            httpx.post(
                f"{daemon}/hooks/before_prompt",
                json={
                    "conversation_id": sid,
                    "prompt": f"Kill test {i + 1}",
                    "workspace_roots": ["/Users/yeshouyou/Work/agent/team_doc"],
                },
                timeout=10,
            )
            httpx.post(
                f"{daemon}/hooks/after_response",
                json={
                    "conversation_id": sid,
                    "response_text": f"Response {i + 1}",
                    "workspace_roots": ["/Users/yeshouyou/Work/agent/team_doc"],
                },
                timeout=10,
            )

        # 2. Wait for _tick to pick up and mark as refining
        time.sleep(35)

        # 3. Kill daemon
        subprocess.run(["make", "daemon-stop"], capture_output=True)

        # 4. Restart daemon
        subprocess.run(["make", "daemon-start"], capture_output=True)
        for _ in range(30):
            try:
                r = httpx.get(f"{daemon}/status", timeout=2)
                if r.status_code == 200:
                    break
            except Exception:
                pass
            time.sleep(1)

        # 5. Wait for recovery tick
        time.sleep(40)

        # 6. Verify session was recovered
        import sqlite3
        from pathlib import Path

        db = sqlite3.connect(str(Path.home() / ".cache" / "tm-pipeline" / "drafts.db"))
        row = db.execute("SELECT status FROM sessions WHERE id = ?", (sid,)).fetchone()
        assert row is not None, f"Session {sid} disappeared after restart"
        # Valid post-recovery states per NoStrandedRefiningSession invariant:
        # either transitioned out of 'refining', or re-entered 'refining' on the
        # recovery tick with an active retry queued in pending_ops (e.g. no LLM in CI).
        if row[0] == "refining":
            queued = db.execute(
                "SELECT id FROM pending_ops WHERE session_id = ? AND status != 'failed'",
                (sid,),
            ).fetchone()
            assert queued is not None, "Session stranded in 'refining' without an active pending_ops entry"
        else:
            assert row[0] in ("needs_refinement", "published", "publishing"), f"Unexpected status: {row[0]}"

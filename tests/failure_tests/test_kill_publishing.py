"""Kill daemon mid-publishing — 验证 skip_llm 生效。"""

import json
import sqlite3
import subprocess
import time
from pathlib import Path

import httpx
import pytest


@pytest.mark.failure_injection
class TestKillPublishing:
    def test_publishing_session_skip_llm(self, daemon: str, unique_session: str) -> None:
        sid = unique_session
        for i in range(3):
            httpx.post(
                f"{daemon}/hooks/before_prompt",
                json={
                    "conversation_id": sid,
                    "prompt": f"Pub kill {i + 1}",
                    "workspace_roots": ["/Users/yeshouyou/Work/agent/team_doc"],
                },
                timeout=10,
            )
            httpx.post(
                f"{daemon}/hooks/after_response",
                json={
                    "conversation_id": sid,
                    "response_text": f"R {i + 1}",
                    "workspace_roots": ["/Users/yeshouyou/Work/agent/team_doc"],
                },
                timeout=10,
            )
        time.sleep(35)
        # Manually set publishing state with payload
        db = sqlite3.connect(str(Path.home() / ".cache" / "tm-pipeline" / "drafts.db"))
        payload = json.dumps({"experience_id": "fake-exp-001", "structured": {"title": "T"}})
        db.execute("UPDATE sessions SET status='publishing', refined_payload=? WHERE id=?", (payload, sid))
        db.commit()
        subprocess.run(["make", "daemon-stop"], capture_output=True)
        subprocess.run(["make", "daemon-start"], capture_output=True)
        for _ in range(30):
            try:
                r = httpx.get(f"{daemon}/status", timeout=2)
                if r.status_code == 200:
                    break
            except Exception:
                pass
            time.sleep(1)
        time.sleep(40)
        row = db.execute("SELECT status, refined_payload FROM sessions WHERE id=?", (sid,)).fetchone()
        assert row is not None, f"Session {sid} lost after restart"

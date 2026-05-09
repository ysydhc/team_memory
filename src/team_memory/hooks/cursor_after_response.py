"""Cursor afterAgentResponse hook — forwards to TM Daemon.

Maps Cursor's field names to what the daemon expects.
Cursor sends different field names than Hermes/Claude Code,
so we normalize the payload here.

Cursor afterAgentResponse stdin format:
  {"conversation_id": "...", "agent_response": "...", ...}

Daemon expects:
  {"conversation_id": "...", "response_text": "...", "workspace_roots": [...]}

Cursor 字段映射: agent_response → response_text
"""
import json
import os
import sys

import httpx

DAEMON_URL = "http://127.0.0.1:3901"


def main() -> None:
    try:
        input_data = json.load(sys.stdin)
    except Exception:
        print(json.dumps({"action": "error", "message": "invalid input"}))
        return

    # Map Cursor fields to daemon fields / 映射 Cursor 字段到 daemon 字段
    response_text = (
        input_data.get("response_text", "")
        or input_data.get("agent_response", "")
        or input_data.get("content", "")
        or input_data.get("response", "")
        or input_data.get("prompt", "")
        or ""
    )
    conversation_id = input_data.get("conversation_id", "") or input_data.get("session_id", "")
    workspace_roots = input_data.get("workspace_roots", [])
    if not workspace_roots:
        cwd = input_data.get("cwd", "") or input_data.get("workspace", "")
        if cwd:
            workspace_roots = [cwd]
        else:
            workspace_roots = [os.getcwd()]

    payload = {
        "conversation_id": conversation_id,
        "response_text": response_text,
        "workspace_roots": workspace_roots,
        "model": input_data.get("model", ""),
    }

    try:
        resp = httpx.post(f"{DAEMON_URL}/hooks/after_response", json=payload, timeout=10.0)
        print(resp.text)
    except httpx.ConnectError:
        print(json.dumps({"action": "ok", "message": "daemon not running"}))
    except Exception as e:
        print(json.dumps({"action": "error", "message": str(e)}))


if __name__ == "__main__":
    main()

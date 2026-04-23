# Phase 2: Thin Hooks + 启动脚本

**Goal:** 将 hook 脚本改薄（只转发到 daemon），添加 launchd 自动启动。

## Task 2-1: Thin Hook 脚本

**Files:**
- Modify: `scripts/hooks/cursor_after_response.py` — 改薄
- Modify: `scripts/hooks/cursor_session_start.py` — 改薄
- Modify: `scripts/hooks/cursor_before_prompt.py` — 改薄
- Modify: `scripts/hooks/claude_session_start.py` — 改薄
- Test: `tests/test_thin_hooks.py`

所有 hook 脚本统一模式：

```python
# scripts/hooks/cursor_after_response.py (thin version)
"""Cursor afterAgentResponse hook — forwards to TM Daemon."""
import json
import sys
import httpx

DAEMON_URL = "http://127.0.0.1:3901"

def main():
    try:
        input_data = json.load(sys.stdin)
    except Exception:
        print(json.dumps({"action": "error", "message": "invalid input"}))
        return

    try:
        resp = httpx.post(f"{DAEMON_URL}/hooks/after_response", json=input_data, timeout=10.0)
        print(resp.text)
    except httpx.ConnectError:
        # Daemon not running — silently skip
        print(json.dumps({"action": "ok", "message": "daemon not running"}))
    except Exception as e:
        print(json.dumps({"action": "error", "message": str(e)}))

if __name__ == "__main__":
    main()
```

同样的模式应用于其他 hook 脚本，只是 endpoint 不同：
- `cursor_session_start.py` → `/hooks/session_start`
- `cursor_before_prompt.py` → `/hooks/before_prompt`
- `claude_session_start.py` → `/hooks/session_start`

关键点：
- daemon 没跑时静默失败，不影响 Agent 正常工作
- 超时 10s，防止 daemon 卡住阻塞 Agent

**Step 2: Update .cursor/hooks.json**

command 改成 `.venv/bin/python`，确保能 import httpx：

```json
{
  "hooks": {
    "afterAgentResponse": [{
      "matcher": "",
      "hooks": [{"type": "command", "command": "/Users/yeshouyou/Work/agent/team_doc/.venv/bin/python /Users/yeshouyou/Work/agent/team_doc/scripts/hooks/cursor_after_response.py"}]
    }],
    "sessionStart": [{
      "matcher": "",
      "hooks": [{"type": "command", "command": "/Users/yeshouyou/Work/agent/team_doc/.venv/bin/python /Users/yeshouyou/Work/agent/team_doc/scripts/hooks/cursor_session_start.py"}]
    }],
    "beforeSubmitPrompt": [{
      "matcher": "",
      "hooks": [{"type": "command", "command": "/Users/yeshouyou/Work/agent/team_doc/.venv/bin/python /Users/yeshouyou/Work/agent/team_doc/scripts/hooks/cursor_before_prompt.py"}]
    }]
  }
}
```

**Step 3: Run tests + commit**

---

## Task 2-2: Hermes Pipeline 改薄

**Files:**
- Modify: `scripts/hooks/hermes_pipeline.py` — 调 daemon HTTP API

Hermes 不走 Cursor/Claude Code hooks，而是通过 hermes_pipeline skill 在每个 turn 调用。改为 HTTP 调 daemon：

```python
# scripts/hooks/hermes_pipeline.py (thin version)
"""Hermes memory pipeline — calls TM Daemon HTTP API."""
from __future__ import annotations

import httpx

DAEMON_URL = "http://127.0.0.1:3901"


async def on_turn_start(context: dict) -> dict:
    """Called at the start of each Hermes turn. Inject relevant memories."""
    try:
        resp = httpx.post(
            f"{DAEMON_URL}/hooks/session_start",
            json=context, timeout=10.0,
        )
        return resp.json()
    except Exception:
        return {"additional_context": ""}


async def on_turn_end(context: dict) -> dict:
    """Called after Hermes responds. Accumulate draft."""
    try:
        resp = httpx.post(
            f"{DAEMON_URL}/hooks/after_response",
            json=context, timeout=10.0,
        )
        return resp.json()
    except Exception:
        return {"action": "ok"}


async def on_session_end(context: dict) -> dict:
    """Called when Hermes session ends. Flush pending drafts."""
    try:
        resp = httpx.post(
            f"{DAEMON_URL}/hooks/session_end",
            json=context, timeout=10.0,
        )
        return resp.json()
    except Exception:
        return {"action": "ok"}
```

**Step 2: Run tests + commit**

---

## Task 2-3: Daemon 启动脚本 + launchd

**Files:**
- Create: `scripts/daemon/start.sh` — 启动 daemon
- Create: `com.teammemory.daemon.plist` — launchd 配置
- Modify: `Makefile` — 添加 daemon 启动/停止命令

```bash
# scripts/daemon/start.sh
#!/usr/bin/env bash
set -euo pipefail
REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$REPO_ROOT"
exec .venv/bin/python -m daemon
```

```xml
<!-- ~/Library/LaunchAgents/com.teammemory.daemon.plist -->
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>Label</key>
    <string>com.teammemory.daemon</string>
    <key>ProgramArguments</key>
    <array>
        <string>/Users/yeshouyou/Work/agent/team_doc/.venv/bin/python</string>
        <string>-m</string>
        <string>daemon</string>
    </array>
    <key>WorkingDirectory</key>
    <string>/Users/yeshouyou/Work/agent/team_doc</string>
    <key>RunAtLoad</key>
    <true/>
    <key>KeepAlive</key>
    <true/>
    <key>StandardOutPath</key>
    <string>/tmp/tm-daemon.log</string>
    <key>StandardErrorPath</key>
    <string>/tmp/tm-daemon.err</string>
</dict>
</plist>
```

Makefile additions:
```makefile
daemon-start:    ## Start TM Daemon (launchd)
	launchctl load ~/Library/LaunchAgents/com.teammemory.daemon.plist

daemon-stop:     ## Stop TM Daemon
	launchctl unload ~/Library/LaunchAgents/com.teammemory.daemon.plist

daemon-run:      ## Run TM Daemon in foreground (for testing)
	.venv/bin/python -m daemon
```

**Step 2: Run tests + commit**

---

## Task 2-4: MarkdownIndexer 迁移

**Files:**
- Copy: `scripts/hooks/markdown_indexer.py` → `scripts/daemon/markdown_indexer.py`
- Minor: 移除对 scripts.hooks 的 import 路径依赖

**Step 3: Run tests + commit**

---

## Task 2-5: 端到端验证

1. 启动 daemon: `make daemon-run`
2. 模拟 hook 调用: `curl -X POST http://localhost:3901/hooks/after_response -d '{"conversation_id":"test","prompt":"hello","workspace_roots":["/Users/yeshouyou/Work/agent/team_doc"]}'`
3. 检查 TM 数据库是否有新增 draft
4. 模拟 session_start: `curl -X POST http://localhost:3901/hooks/session_start -d '{"workspace_roots":["/Users/yeshouyou/Work/agent/team_doc"]}'`
5. 检查返回的 additional_context
6. 修改 Obsidian vault 中的 .md 文件，检查 daemon 日志是否触发索引

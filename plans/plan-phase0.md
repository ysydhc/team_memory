# 阶段 0：基础准备

## 项目背景

Agent 共享记忆系统的核心驱动力是 Hooks——Cursor 和 Claude Code 的生命周期钩子。
Hooks 是系统级的，Agent 无法跳过，解决了"Agent 不主动调 MCP"的根本问题。

本阶段目标：**在 TM 外部搭建 Hook 脚本框架，不修改任何 TM 代码**。
验证 Hooks 能否自动触发、能否拿到对话数据、能否调通 TM 的 MCP。

## 预计改动

| 文件 | 操作 | 说明 |
|------|------|------|
| `~/.cursor/hooks.json` | 新建 | Cursor Hooks 配置 |
| `~/.claude/hooks.json` | 新建 | Claude Code Hooks 配置 |
| `scripts/hooks/` | 新建目录 | Hook 脚本目录 |
| `scripts/hooks/common.py` | 新建 | 共享工具函数 |
| `scripts/hooks/cursor_after_response.py` | 新建 | Cursor afterAgentResponse Hook |
| `scripts/hooks/cursor_session_start.py` | 新建 | Cursor sessionStart Hook |
| `scripts/hooks/cursor_before_prompt.py` | 新建 | Cursor beforeSubmitPrompt Hook |
| `scripts/hooks/claude_session_start.py` | 新建 | Claude Code sessionStart Hook |
| `scripts/hooks/draft_buffer.py` | 新建 | 草稿缓冲区（本地 SQLite） |
| `scripts/hooks/config.yaml` | 新建 | 管线配置 |

## 架构图

### Hook 脚本工作流（新架构）

```
┌──────────────────────────────────────────────────────────────────┐
│                    Cursor / Claude Code                          │
│                                                                  │
│  sessionStart ──→ hook脚本 ──→ MCP memory_context ──→ 注入上下文 │
│                                                                  │
│  beforeSubmitPrompt ──→ hook脚本 ──→ 解析意图                    │
│                           │                                      │
│                           ├── 需要检索 → MCP memory_recall       │
│                           └── 不需要 → 跳过                      │
│                                                                  │
│  afterAgentResponse ──→ hook脚本 ──→ 更新草稿缓冲区             │
│                           │                                      │
│                           ├── 收敛信号? → MCP memory_draft_save  │
│                           └── 未收敛 → 等待                      │
│                                                                  │
│  sessionEnd ──→ hook脚本 ──→ 强制提炼草稿 → MCP memory_draft_save│
└──────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌──────────────────────────────────────────────────────────────────┐
│                    本地草稿缓冲区（SQLite）                       │
│                                                                  │
│  drafts 表：                                                     │
│    id | project | content | status | created_at | updated_at     │
│                                                                  │
│  · 管线写入时先存本地草稿                                        │
│  · 收敛后调 MCP 写入 TM                                         │
│  · 防止 Hook 脚本崩溃丢失数据                                    │
└──────────────────────────────────────────────────────────────────┘
```

### Cursor hooks.json 结构

```json
{
  "hooks": {
    "sessionStart": [
      {
        "matcher": "",
        "hooks": [
          {
            "type": "command",
            "command": "python3 ~/.cursor/hooks/cursor_session_start.py"
          }
        ]
      }
    ],
    "afterAgentResponse": [
      {
        "matcher": "",
        "hooks": [
          {
            "type": "command",
            "command": "python3 ~/.cursor/hooks/cursor_after_response.py"
          }
        ]
      }
    ],
    "beforeSubmitPrompt": [
      {
        "matcher": "",
        "hooks": [
          {
            "type": "command",
            "command": "python3 ~/.cursor/hooks/cursor_before_prompt.py"
          }
        ]
      }
    ]
  }
}
```

## 子任务拆分

---

### 任务 0-1：创建 Hook 脚本目录和共享工具

**描述**：创建 `scripts/hooks/` 目录，实现共享工具模块 `common.py`。

**TDD 流程**：
1. 写测试：`tests/test_hook_common.py`
   - 测试 `parse_hook_input()`：能解析 stdin 的 JSON 数据
   - 测试 `call_mcp_tool()`：能构造正确的 MCP 请求（mock HTTP）
   - 测试 `load_config()`：能读取 config.yaml
   - 测试 `get_project_from_path()`：能从工作目录推断 project 名
2. 实现 `scripts/hooks/common.py`
3. 验证测试通过

**涉及文件**：
- `scripts/hooks/common.py`（新建）
- `scripts/hooks/config.yaml`（新建）
- `tests/test_hook_common.py`（新建）

**common.py 伪代码**：
```python
def parse_hook_input() -> dict:
    """从 stdin 读取 Hook 输入的 JSON"""

async def call_mcp_tool(tool_name: str, arguments: dict) -> dict:
    """通过 HTTP 调 TM 的 MCP 端点（localhost:3900）"""

def load_config() -> dict:
    """加载 scripts/hooks/config.yaml"""

def get_project_from_path(workspace_root: str) -> str:
    """从工作目录路径推断 TM project 名
    规则：路径中包含 team_doc → 'team_doc'
          路径中包含 ad_learning → 'ad_learning'
          默认 → 'default'
    """
```

**config.yaml 伪代码**：
```yaml
tm:
  base_url: "http://localhost:3900"
  
projects:
  team_doc:
    path_patterns: ["team_doc"]
  ad_learning:
    path_patterns: ["ad_learning"]
  ai_learning:
    path_patterns: ["ai_learning"]

retrieval:
  session_start_top_k: 3
  keyword_triggers: ["之前", "上次", "经验", "踩坑", "遇到过"]
  
draft:
  max_age_minutes: 30
  db_path: "~/.cache/tm-pipeline/drafts.db"
```

**验收标准**：
- [ ] `parse_hook_input()` 能正确解析 `{"conversation_id": "xxx", "prompt": "hello", "workspace_roots": ["/path"]}` 格式的 JSON
- [ ] `call_mcp_tool("memory_recall", {"query": "test"})` 能构造正确的 HTTP 请求
- [ ] `load_config()` 能读取 config.yaml 中的 tm.base_url
- [ ] `get_project_from_path("/Users/yeshouyou/Work/ad_learning")` 返回 "ad_learning"
- [ ] 所有测试通过：`make test PYTEST_ARGS='tests/test_hook_common.py'`

**Subagent 执行方式**：delegate_task，toolsets=['terminal', 'file']

---

### 任务 0-2：实现本地草稿缓冲区

**描述**：实现 `draft_buffer.py`，用本地 SQLite 存储草稿，防止 Hook 崩溃导致数据丢失。

**TDD 流程**：
1. 写测试：`tests/test_hook_draft_buffer.py`
   - 测试 `create_draft()`：能创建草稿
   - 测试 `update_draft()`：能更新草稿内容（新覆盖旧）
   - 测试 `get_pending_drafts()`：能获取未提炼的草稿
   - 测试 `mark_published()`：能标记草稿为已发布
   - 测试 `get_older_than()`：能获取超过 30 分钟的草稿
2. 实现 `scripts/hooks/draft_buffer.py`
3. 验证测试通过

**涉及文件**：
- `scripts/hooks/draft_buffer.py`（新建）
- `tests/test_hook_draft_buffer.py`（新建）

**draft_buffer.py 伪代码**：
```python
class DraftBuffer:
    def __init__(self, db_path: str):
        """初始化 SQLite 连接，创建 drafts 表"""
        # CREATE TABLE IF NOT EXISTS drafts (
        #   id TEXT PRIMARY KEY,
        #   project TEXT NOT NULL,
        #   conversation_id TEXT,
        #   content TEXT NOT NULL,       -- 草稿内容（JSON）
        #   status TEXT DEFAULT 'pending', -- pending / published / discarded
        #   source TEXT DEFAULT 'pipeline',
        #   created_at TIMESTAMP,
        #   updated_at TIMESTAMP
        # )
    
    def create_draft(self, project, conversation_id, content) -> str:
        """创建新草稿，返回 draft id"""
    
    def update_draft(self, draft_id, content):
        """更新草稿内容（新事实覆盖旧事实）"""
    
    def get_pending_drafts(self, project=None) -> list[dict]:
        """获取所有未提炼的草稿"""
    
    def get_older_than(self, minutes: int) -> list[dict]:
        """获取超过 N 分钟的草稿（用于时间兜底）"""
    
    def mark_published(self, draft_id):
        """标记草稿为已发布"""
```

**验收标准**：
- [ ] 创建草稿后 `get_pending_drafts()` 能返回该草稿
- [ ] `update_draft()` 后草稿内容已更新
- [ ] `get_older_than(30)` 只返回超过 30 分钟的草稿
- [ ] `mark_published()` 后草稿不再出现在 `get_pending_drafts()` 中
- [ ] SQLite 文件在 `~/.cache/tm-pipeline/drafts.db`
- [ ] 所有测试通过

**Subagent 执行方式**：delegate_task，toolsets=['terminal', 'file']

---

### 任务 0-3：实现 Cursor afterAgentResponse Hook

**描述**：实现 Cursor 的 afterAgentResponse Hook 脚本，验证能收到 Agent 回复数据。

**TDD 流程**：
1. 写测试：`tests/test_hook_cursor_after_response.py`
   - 测试脚本能从 stdin 读取 afterAgentResponse 的 JSON
   - 测试脚本能提取 Agent 回复内容
   - 测试脚本能判断收敛信号（关键词匹配）
   - 测试脚本能更新草稿缓冲区
2. 实现 `scripts/hooks/cursor_after_response.py`
3. 验证测试通过
4. 手动验证：在 Cursor 中触发 Hook，查看日志

**涉及文件**：
- `scripts/hooks/cursor_after_response.py`（新建）
- `tests/test_hook_cursor_after_response.py`（新建）

**脚本伪代码**：
```python
#!/usr/bin/env python3
"""Cursor afterAgentResponse Hook — 采集 Agent 回复，更新草稿缓冲区"""

import sys, json, asyncio
from common import parse_hook_input, call_mcp_tool, get_project_from_path, load_config
from draft_buffer import DraftBuffer

CONVERGENCE_SIGNALS = ["解决了", "问题修复", "完成了", "先这样", "搞定", "test passed"]

def detect_convergence(agent_response: str) -> bool:
    """检测 Agent 回复中的收敛信号"""
    return any(sig in agent_response for sig in CONVERGENCE_SIGNALS)

async def main():
    data = parse_hook_input()
    agent_response = data.get("prompt", "")  # afterAgentResponse 的 prompt 字段
    workspace = data.get("workspace_roots", [""])[0]
    project = get_project_from_path(workspace)
    conversation_id = data.get("conversation_id", "unknown")
    
    config = load_config()
    buf = DraftBuffer(config["draft"]["db_path"])
    
    # 更新或创建草稿
    # ...
    
    # 检测收敛信号
    if detect_convergence(agent_response):
        # 触发提炼 → 调 MCP 写入 TM
        pending = buf.get_pending_drafts(project)
        for draft in pending:
            await call_mcp_tool("memory_save", {
                "title": draft["content"][:50],
                "content": draft["content"],
                "project": project,
                "scope": "project",
            })
            buf.mark_published(draft["id"])
    
    # 输出（Hooks 可以返回 additional_context）
    print(json.dumps({"status": "ok"}))

if __name__ == "__main__":
    asyncio.run(main())
```

**验收标准**：
- [ ] 脚本能解析 stdin 中的 afterAgentResponse JSON
- [ ] `detect_convergence("问题解决了，TUN 模式生效")` 返回 True
- [ ] `detect_convergence("我试试 bridge 模式")` 返回 False
- [ ] 收敛时草稿被标记为 published
- [ ] 非收敛时草稿留在 pending 状态
- [ ] 所有测试通过

**Subagent 执行方式**：delegate_task，toolsets=['terminal', 'file']

---

### 任务 0-4：实现 Cursor sessionStart + beforeSubmitPrompt Hook

**描述**：实现会话启动时的自动检索和用户输入时的按需检索。

**TDD 流程**：
1. 写测试：`tests/test_hook_cursor_session_start.py` 和 `tests/test_hook_cursor_before_prompt.py`
   - sessionStart：能调 MCP memory_context 并返回 additional_context
   - beforeSubmitPrompt：能解析用户输入，判断是否需要检索
   - 关键词触发："之前"/"上次"/"经验" → 触发检索
   - 非关键词 → 跳过
2. 实现两个脚本
3. 验证测试通过

**涉及文件**：
- `scripts/hooks/cursor_session_start.py`（新建）
- `scripts/hooks/cursor_before_prompt.py`（新建）
- `tests/test_hook_cursor_session_start.py`（新建）
- `tests/test_hook_cursor_before_prompt.py`（新建）

**cursor_session_start.py 伪代码**：
```python
async def main():
    data = parse_hook_input()
    workspace = data.get("workspace_roots", [""])[0]
    project = get_project_from_path(workspace)
    
    # 调 TM 检索项目级记忆
    result = await call_mcp_tool("memory_context", {
        "project": project,
    })
    
    # Hook 可以返回 additional_context 注入到 Agent
    output = {"additionalContext": result}
    print(json.dumps(output))
```

**cursor_before_prompt.py 伪代码**：
```python
KEYWORD_TRIGGERS = ["之前", "上次", "经验", "踩坑", "遇到过", "以前"]

def should_retrieve(user_prompt: str) -> bool:
    """判断用户输入是否需要检索记忆"""
    return any(kw in user_prompt for kw in KEYWORD_TRIGGERS)

async def main():
    data = parse_hook_input()
    user_prompt = data.get("prompt", "")
    
    if not should_retrieve(user_prompt):
        print(json.dumps({}))
        return
    
    workspace = data.get("workspace_roots", [""])[0]
    project = get_project_from_path(workspace)
    
    result = await call_mcp_tool("memory_recall", {
        "query": user_prompt,
        "max_results": 3,
        "project": project,
    })
    
    output = {"additionalContext": result}
    print(json.dumps(output))
```

**验收标准**：
- [ ] sessionStart Hook 能调通 MCP memory_context
- [ ] beforeSubmitPrompt 对 "之前遇到过的 Docker 问题" 返回检索结果
- [ ] beforeSubmitPrompt 对 "帮我写个函数" 返回空（不触发检索）
- [ ] 所有测试通过

**Subagent 执行方式**：delegate_task，toolsets=['terminal', 'file']

---

### 任务 0-5：配置 Cursor 和 Claude Code 的 hooks.json

**描述**：配置两端的 hooks.json，完成端到端手动验证。

**TDD 流程**：
1. 写 `~/.cursor/hooks.json`
2. 写 `~/.claude/hooks.json`
3. 手动验证：在 Cursor 中发一条消息，检查 Hook 日志
4. 手动验证：在 Claude Code 中启动会话，检查 Hook 日志

**涉及文件**：
- `~/.cursor/hooks.json`（新建）
- `~/.claude/hooks.json`（新建）

**Cursor hooks.json 内容**：
```json
{
  "hooks": {
    "sessionStart": [
      {
        "matcher": "",
        "hooks": [{
          "type": "command",
          "command": "python3 /Users/yeshouyou/Work/agent/team_doc/scripts/hooks/cursor_session_start.py"
        }]
      }
    ],
    "afterAgentResponse": [
      {
        "matcher": "",
        "hooks": [{
          "type": "command",
          "command": "python3 /Users/yeshouyou/Work/agent/team_doc/scripts/hooks/cursor_after_response.py"
        }]
      }
    ],
    "beforeSubmitPrompt": [
      {
        "matcher": "",
        "hooks": [{
          "type": "command",
          "command": "python3 /Users/yeshouyou/Work/agent/team_doc/scripts/hooks/cursor_before_prompt.py"
        }]
      }
    ]
  }
}
```

**Claude Code hooks.json 内容**：
```json
{
  "hooks": {
    "sessionStart": [
      {
        "matcher": "",
        "hooks": [{
          "type": "command",
          "command": "python3 /Users/yeshouyou/Work/agent/team_doc/scripts/hooks/claude_session_start.py"
        }]
      }
    ]
  }
}
```

**验收标准**：
- [ ] Cursor 中发送一条消息后，`~/.cache/tm-pipeline/` 下有 Hook 日志
- [ ] Claude Code 启动会话后，有 Hook 日志
- [ ] Hook 触发不阻塞 Agent 正常对话
- [ ] Hook 脚本报错时，Agent 对话不受影响（静默失败）

**Subagent 执行方式**：delegate_task，toolsets=['terminal', 'file']

---

## 阶段 0 人工验收条目

完成所有子任务后，人工验收以下条目：

- [ ] 在 Cursor 中打开一个项目目录，发一条消息，检查 Hook 日志是否出现
- [ ] 在 Cursor 中说"之前遇到的 Docker 问题"，检查是否有检索行为
- [ ] 在 Cursor 中说"帮我写个函数"，检查是否没有检索行为
- [ ] 在 Cursor 中调试一个问题直到解决，检查草稿是否被写入 TM
- [ ] 在 Claude Code 中启动会话，检查 sessionStart Hook 是否触发
- [ ] Hook 脚本不会影响 Agent 的正常对话速度

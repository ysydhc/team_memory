# 阶段 2：管线实现

## 项目背景

阶段 1 已经让 TM 支持"管线模式"的写入（memory_draft_save / memory_draft_publish）和检索（[mem:xxx] 标记）。
阶段 0 已经搭建了 Hook 脚本框架和草稿缓冲区的基础结构。
本阶段要把两者连接起来：Hook 脚本通过调 TM 的 MCP 工具，跑通完整的草稿→写入流程。

核心流程：
1. Hook 采集 Agent 对话 → 更新本地草稿缓冲区
2. 收敛信号触发 → 调 TM memory_draft_publish
3. 30 分钟兜底 → 强制提炼草稿
4. 会话结束 → 强制提炼草稿
5. beforeSubmitPrompt → 按需检索 TM

## 预计改动

| 文件 | 操作 | 说明 |
|------|------|------|
| `scripts/hooks/cursor_after_response.py` | 修改 | 接入 TM MCP，实现完整草稿逻辑 |
| `scripts/hooks/cursor_session_start.py` | 修改 | 接入 TM memory_context |
| `scripts/hooks/cursor_before_prompt.py` | 修改 | 接入 TM memory_recall |
| `scripts/hooks/claude_session_start.py` | 修改 | 接入 TM memory_context |
| `scripts/hooks/convergence_detector.py` | 新建 | 收敛信号检测（关键词 + 工具调用模式） |
| `scripts/hooks/draft_refiner.py` | 新建 | 草稿提炼逻辑（调 LLM 分类痕迹/知识） |
| `scripts/hooks/timer_daemon.py` | 新建 | 30 分钟兜底定时器 |
| `scripts/hooks/common.py` | 修改 | 新增 TM MCP 调用方法 |

## 架构图

### 旧架构（阶段 0）：Hook 只打印日志

```
Cursor afterAgentResponse
    ↓
Hook 脚本打印日志
    ↓
（草稿存在本地 SQLite，不写入 TM）
```

### 新架构（阶段 2）：Hook 跑通完整管线

```
┌─────────────────────────────────────────────────────────────┐
│  Cursor 对话                                                 │
│                                                              │
│  afterAgentResponse ──→ 收集 Agent 回复                      │
│       ↓                                                      │
│  更新本地草稿缓冲区                                           │
│       │                                                      │
│       ├── 收敛信号？                                          │
│       │     ├── Yes → draft_refiner → TM memory_draft_publish│
│       │     └── No  → 等待                                   │
│       │                                                      │
│       ├── 30 分钟到了？                                       │
│       │     └── Yes → draft_refiner → TM memory_draft_publish│
│       │                                                      │
│       └── 会话结束？                                          │
│             └── Yes → draft_refiner → TM memory_draft_publish│
│                                                              │
│  beforeSubmitPrompt ──→ 解析用户输入                          │
│       ↓                                                      │
│  关键词匹配？                                                 │
│     ├── Yes → TM memory_recall → additionalContext           │
│     └── No  → 跳过                                           │
│                                                              │
│  sessionStart ──→ TM memory_context → additionalContext      │
└─────────────────────────────────────────────────────────────┘
```

### 草稿提炼流程

```
草稿缓冲区中的 pending 草稿
    ↓
draft_refiner.py
    ↓
LLM 分类：这条草稿是"痕迹"还是"知识"？
    ├── 痕迹 → TM memory_draft_publish（draft → published Experience）
    └── 知识 → 写 Markdown 文件（阶段 4 完善，现阶段先也走 TM）
    ↓
本地草稿标记为 published
```

## 子任务拆分

---

### 任务 2-1：实现收敛信号检测器

**描述**：从 Agent 回复和工具调用序列中检测任务是否收敛。

**TDD 流程**：
1. 写测试：`tests/test_hook_convergence.py`
   - 显式信号："解决了"/"问题修复了"/"先这样" → True
   - 工具调用模式：连续 terminal → git commit → True
   - 话题切换：新项目路径出现 → True
   - 调试中："试试看" / "我查一下" → False
   - 长时间无更新 → False（由 timer_daemon 处理）
2. 实现 `scripts/hooks/convergence_detector.py`
3. 验证测试通过

**伪代码**：
```python
EXPLICIT_SIGNALS = ["解决了", "问题修复", "完成了", "先这样", "搞定", "test passed", "已确认"]

TOOL_PATTERNS = {
    "debug_converge": [
        # 连续 shell 命令后跟 git commit
        {"last_tools": ["terminal", "terminal"], "final_tool": "terminal", 
         "final_cmd_pattern": r"git commit|pytest|test"},
    ],
}

class ConvergenceDetector:
    def check_explicit(self, agent_response: str) -> bool:
        """检查显式收敛信号"""
        return any(sig in agent_response for sig in EXPLICIT_SIGNALS)
    
    def check_tool_pattern(self, recent_tools: list[dict]) -> bool:
        """检查工具调用模式是否表明任务完成"""
        # 最近 3 次工具调用中有 git commit / pytest 等
        pass
    
    def check_topic_shift(self, current_path: str, previous_path: str) -> bool:
        """检查是否切换了项目/话题"""
        return current_path != previous_path
```

**验收标准**：
- [ ] `check_explicit("问题解决了，TUN 模式生效")` → True
- [ ] `check_explicit("我试试 bridge 模式")` → False
- [ ] `check_tool_pattern([{"tool": "terminal", "cmd": "git commit -m 'fix'"}])` → True
- [ ] `check_topic_shift("/Work/ad_learning", "/Work/ai_learning")` → True
- [ ] 所有测试通过

**Subagent 执行方式**：delegate_task，toolsets=['terminal', 'file']

---

### 任务 2-2：实现草稿提炼器

**描述**：草稿收敛后，调 LLM 分类痕迹/知识，然后调 TM MCP 写入。

**TDD 流程**：
1. 写测试：`tests/test_hook_draft_refiner.py`
   - 痕迹类草稿 → 调 memory_draft_publish
   - 知识类草稿 → 先走 TM（后续走 Markdown）
   - 空草稿 → 丢弃
   - 被推翻的草稿 → 只保留最终结论
2. 实现 `scripts/hooks/draft_refiner.py`
3. 验证测试通过

**伪代码**：
```python
class DraftRefiner:
    def __init__(self, mcp_base_url: str, llm_config: dict | None = None):
        self._mcp_url = mcp_base_url
        self._llm_config = llm_config
    
    async def refine(self, draft: dict) -> dict:
        """提炼一条草稿，返回 {action: 'publish'|'discard', category: 'trace'|'knowledge'}"""
        content = draft.get("content", "")
        if not content.strip():
            return {"action": "discard"}
        
        # LLM 分类（可选，没有 LLM 配置时默认为 trace）
        category = await self._classify(content)
        
        if category == "trace":
            return {"action": "publish", "category": "trace"}
        else:
            # 阶段 2 先都走 TM，阶段 4 再区分走 Markdown
            return {"action": "publish", "category": "knowledge"}
    
    async def _classify(self, content: str) -> str:
        """LLM 分类：trace or knowledge"""
        if not self._llm_config:
            return "trace"  # 默认
        # prompt: 判断以下内容是"操作痕迹"（不值得人看）还是"知识"（值得人学）
        pass
    
    async def publish_to_tm(self, draft: dict, refined_content: str):
        """调 TM MCP memory_draft_publish"""
        pass
```

**验收标准**：
- [ ] 空草稿被 discard
- [ ] 有内容的草稿默认分类为 trace
- [ ] refine 后调 memory_draft_publish 成功
- [ ] 所有测试通过

**Subagent 执行方式**：delegate_task，toolsets=['terminal', 'file']

---

### 任务 2-3：实现 30 分钟兜底定时器

**描述**：后台守护进程，每分钟检查草稿缓冲区，超过 30 分钟的草稿强制提炼。

**TDD 流程**：
1. 写测试：`tests/test_hook_timer_daemon.py`
   - 超过 30 分钟的草稿被标记为待提炼
   - 不到 30 分钟的草稿不受影响
   - 守护进程能启动和停止
2. 实现 `scripts/hooks/timer_daemon.py`
3. 验证测试通过

**伪代码**：
```python
class TimerDaemon:
    def __init__(self, draft_buffer: DraftBuffer, refiner: DraftRefiner, 
                 check_interval: int = 60, max_age_minutes: int = 30):
        self._buffer = draft_buffer
        self._refiner = refiner
        self._interval = check_interval
        self._max_age = max_age_minutes
        self._running = False
    
    async def run(self):
        """主循环"""
        self._running = True
        while self._running:
            expired = self._buffer.get_older_than(self._max_age)
            for draft in expired:
                result = await self._refiner.refine(draft)
                if result["action"] == "publish":
                    await self._refiner.publish_to_tm(draft, draft["content"])
                    self._buffer.mark_published(draft["id"])
            await asyncio.sleep(self._interval)
    
    def stop(self):
        self._running = False
```

**验收标准**：
- [ ] 30 分钟前的草稿被自动提炼
- [ ] 新草稿不受影响
- [ ] 守护进程可启动/停止
- [ ] 所有测试通过

**Subagent 执行方式**：delegate_task，toolsets=['terminal', 'file']

---

### 任务 2-4：改造 afterAgentResponse Hook 脚本

**描述**：把阶段 0 的简单脚本改造为完整的草稿管线——更新草稿 + 检测收敛 + 提炼。

**TDD 流程**：
1. 写测试：`tests/test_hook_cursor_after_response_v2.py`
   - Agent 回复"我试试 bridge"→ 草稿留在 pending
   - Agent 回复"问题解决了"→ 收敛触发，草稿被提炼
   - Agent 回复后又发新消息推翻 → 旧草稿更新
2. 修改 `scripts/hooks/cursor_after_response.py`
3. 验证测试通过

**脚本伪代码**：
```python
async def main():
    data = parse_hook_input()
    agent_response = data.get("prompt", "")
    workspace = data.get("workspace_roots", [""])[0]
    project = get_project_from_path(workspace)
    conversation_id = data.get("conversation_id", "unknown")
    
    config = load_config()
    buf = DraftBuffer(config["draft"]["db_path"])
    detector = ConvergenceDetector()
    refiner = DraftRefiner(config["tm"]["base_url"])
    
    # 更新草稿
    pending = buf.get_pending_drafts(project)
    if pending:
        # 更新已有草稿
        buf.update_draft(pending[0]["id"], agent_response)
    else:
        # 创建新草稿
        buf.create_draft(project, conversation_id, agent_response)
    
    # 检测收敛
    if detector.check_explicit(agent_response):
        for draft in buf.get_pending_drafts(project):
            result = await refiner.refine(draft)
            if result["action"] == "publish":
                await refiner.publish_to_tm(draft, draft["content"])
                buf.mark_published(draft["id"])
```

**验收标准**：
- [ ] 非收敛回复：草稿留在 pending
- [ ] 收敛回复：草稿被提炼并写入 TM
- [ ] 多次回复：草稿内容被更新（新覆盖旧）
- [ ] 所有测试通过

**Subagent 执行方式**：delegate_task，toolsets=['terminal', 'file']

---

### 任务 2-5：改造 sessionStart + beforeSubmitPrompt Hook

**描述**：让两个检索 Hook 真正调通 TM 的 MCP。

**TDD 流程**：
1. 写测试：`tests/test_hook_cursor_search_v2.py`
   - sessionStart 调 memory_context 成功，返回 additionalContext
   - beforeSubmitPrompt 对 "之前的问题" 触发 memory_recall
   - beforeSubmitPrompt 对 "写个函数" 不触发
2. 修改两个脚本
3. 验证测试通过

**验收标准**：
- [ ] sessionStart 能调通 memory_context 并返回结果
- [ ] beforeSubmitPrompt 关键词触发 memory_recall
- [ ] 非关键词不触发检索
- [ ] 所有测试通过

**Subagent 执行方式**：delegate_task，toolsets=['terminal', 'file']

---

### 任务 2-6：端到端集成验证

**描述**：在 Cursor 中进行真实对话，验证完整管线。

**测试流程**：
1. 启动 TM 服务
2. 启动 timer_daemon
3. 在 Cursor 中进行一段调试对话
4. 验证：
   - sessionStart 注入了项目记忆
   - afterAgentResponse 更新了草稿
   - 收敛后草稿被写入 TM
   - 下次对话能检索到之前写入的记忆

**验收标准**：
- [ ] 完整流程：对话 → 草稿 → 收敛 → TM 写入 → 检索命中
- [ ] 30 分钟兜底：未收敛的草稿被强制提炼
- [ ] beforeSubmitPrompt 关键词触发检索
- [ ] 非关键词不浪费 MCP 调用

**Subagent 执行方式**：delegate_task，toolsets=['terminal', 'file']

---

## 阶段 2 人工验收条目

- [ ] 在 Cursor 中进行一段调试对话直到解决，检查 TM 中是否有新 Experience
- [ ] 在 Cursor 中说"之前遇到过的 X 问题"，检查是否有检索结果注入
- [ ] 在 Cursor 中说普通的话（如"帮我写个函数"），检查没有检索行为
- [ ] 等待 30 分钟后检查未收敛的草稿是否被自动提炼
- [ ] 在 Claude Code 中启动会话，检查 sessionStart Hook 是否注入记忆
- [ ] 检查 TM Web UI 中新增的 Experience 的 source 字段为 "pipeline"

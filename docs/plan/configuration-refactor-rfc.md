# TM 配置与安装重构方案 (RFC)
# Team Memory Configuration & Installation Refactor

## 目标 / Goals

1. 新手 `pip install` 后 1 分钟内能用
2. 配置只分两层：服务端 + 客户端
3. 三平台 hooks 一键配置
4. 代码符合 Python 包规范

---

## Phase 1: 代码重组 — scripts/ → src/team_memory/

### 1.1 目录重组

```
当前:                              目标:
scripts/                           src/team_memory/
  daemon/                            daemon/
    app.py                            app.py
    config.py                         config.py
    pipeline.py                       pipeline.py
    ...                               ...
  hooks/                             hooks/
    claude_*.py                       claude_*.py
    cursor_*.py                       cursor_*.py
    hermes_*.py                       hermes_*.py
    common.py                         _common.py  (内部)
    shared.py                         → 删除，用 daemon/tm_sink
    tm_hook.py                        → 删除，用 cli.py
    draft_buffer.py                   → 删除，用 daemon/draft_buffer
    draft_refiner.py                  → 删除，用 daemon/draft_refiner
    convergence_detector.py           → 删除，用 daemon/convergence_detector
    markdown_indexer.py               → 删除，用 daemon/markdown_indexer
  monitoring/                        monitoring/
    alerting.py                       alerting.py
    ...                               ...
```

### 1.2 修复 import 路径

```python
# Before (不规范，依赖 PYTHONPATH)
from daemon.config import DaemonConfig
from hooks.common import get_daemon_url

# After (标准 Python 包)
from team_memory.daemon.config import DaemonConfig
from team_memory.hooks._common import get_daemon_url
```

### 1.3 消除代码重复

| 重复模块 | 保留 | 删除 | 说明 |
|---------|------|------|------|
| hooks/draft_buffer.py | daemon/draft_buffer.py | hooks/ 版本 | daemon 版更完整 |
| hooks/draft_refiner.py | daemon/draft_refiner.py | hooks/ 版本 | daemon 版用 ConversationBuffer |
| hooks/convergence_detector.py | daemon/convergence_detector.py | hooks/ 版本 | daemon 版更完整 |
| hooks/markdown_indexer.py | 合并到 daemon/ | hooks/ 版本 | 只保留一份 |
| hooks/shared.py (TMClient) | daemon/tm_sink.py | hooks/shared.py | 统一为 TMSink |
| hooks/common.py | hooks/_common.py | 保留但改名 | 只保留 HTTP helper |
| hooks/tm_hook.py | cli.py | hooks/tm_hook.py | CLI 已覆盖功能 |

### 1.4 Hooks 简化为纯 HTTP 调用

```
当前 hooks 的问题:
  hooks/ 里包含了业务逻辑 (draft_buffer, convergence_detector 等)
  → hooks 应该只是 "调 daemon API" 的薄层
  → 业务逻辑全部在 daemon 里

目标 hooks 结构:
  hooks/claude_post_tool_use.py  → POST /hooks/mcp_callback
  hooks/claude_user_prompt_submit.py → POST /hooks/before_prompt
  hooks/claude_stop.py           → POST /hooks/after_response
  hooks/claude_session_start.py  → POST /hooks/session_start
  hooks/cursor_after_mcp.py      → POST /hooks/mcp_callback
  hooks/cursor_after_response.py → POST /hooks/after_response
  hooks/cursor_before_prompt.py  → POST /hooks/before_prompt
  hooks/cursor_session_start.py  → POST /hooks/session_start
  hooks/hermes_post_tool_call.py → POST /hooks/mcp_callback
  hooks/hermes_pipeline.py       → POST /hooks/before_prompt + /hooks/after_response
  hooks/_common.py               → get_daemon_url(), http_post()
```

### 1.5 pyproject.toml 调整

```toml
[tool.hatch.build.targets.wheel]
packages = ["src/team_memory"]

[project.scripts]
tm-cli = "team_memory.cli:main"
tm-daemon = "team_memory.daemon.__main__:main"

[project.optional-dependencies]
# 服务端: Web + DB + Embedding
server = [
    "fastapi>=0.100",
    "uvicorn>=0.23",
    "sqlalchemy[asyncio]>=2.0",
    "asyncpg>=0.28",
    "alembic>=1.12",
    "sentence-transformers>=2.2",
]
# Daemon: 本地评估 + 对话缓冲
daemon = [
    "fastapi>=0.100",
    "uvicorn>=0.23",
    "aiosqlite>=0.19",
    "litellm>=1.0",
]
# 全部
all = ["team_memory[server,daemon]"]

[project.dependencies]
# 核心依赖 — CLI + MCP client 只需要这些
httpx>=0.25
pydantic>=2.0
python-dotenv>=1.0
toml>=0.10
```

---

## Phase 2: 配置统一

### 2.1 服务端配置 (部署者)

```bash
# .env 或环境变量 — 部署一次就完事
DATABASE_URL=postgresql+asyncpg://user:pass@host:5432/team_memory
REDIS_URL=redis://host:6379/0
EMBEDDING_PROVIDER=ollama
EMBEDDING_MODEL=qwen3-embedding:0.6b
EMBEDDING_DIM=1024
AUTH_SECRET=xxx
LITELLM_URL=http://localhost:4000/v1
LITELLM_KEY=xxx
```

### 2.2 客户端配置 (使用者)

```toml
# ~/.config/tm/config.toml — 全局配置
[server]
url = "http://localhost:9111"         # 远程 TM 服务地址
api_key = "sk-xxx"                    # 或从环境变量 TEAM_MEMORY_API_KEY

[llm]                                 # 客户端 LLM 操作 (faithfulness, refinement, etc.)
base_url = "http://localhost:4000/v1"
api_key_env = "LITELLM_MASTER_KEY"    # 引用环境变量，不直接写密钥
model = "DeepSeek-V3"
timeout = 30

[conversation]
db_path = "~/.cache/tm-pipeline/drafts.db"

[wiki]
enabled = true
db_path = "~/.cache/tm-pipeline/wiki.db"

[daemon]
host = "127.0.0.1"
port = 3901
```

### 2.3 项目级配置 (可选)

```toml
# .tm.toml — 项目根目录
[project]
name = "team_doc"           # 覆盖自动推断
aliases = ["team_memory"]   # 历史别名
```

### 2.4 配置优先级

```
环境变量 > .tm.toml > ~/.config/tm/config.toml > 代码默认值
```

### 2.5 配置读取 API

```python
# team_memory/config.py — 统一配置读取
from team_memory.project_resolver import get_server_url, get_api_key

class ClientConfig:
    """Read from ~/.config/tm/config.toml + env + .env"""

    @property
    def server_url(self) -> str: ...
    @property
    def api_key(self) -> str: ...
    @property
    def llm_base_url(self) -> str: ...
    @property
    def llm_model(self) -> str: ...
    ...

# 所有客户端组件统一使用
config = ClientConfig()
sink = RemoteTMSink(base_url=config.server_url, api_key=config.api_key)
```

---

## Phase 3: Hooks 一键配置

### 3.1 tm-cli setup 命令

```bash
# 交互式配置全部平台
tm-cli setup

# 或指定平台
tm-cli setup --platform claude-code
tm-cli setup --platform cursor
tm-cli setup --platform hermes

# 或配置全部
tm-cli setup --platform all
```

### 3.2 各平台自动配置逻辑

**Claude Code:**
```
读取 ~/.claude/settings.json
添加 PostToolUse hook (matcher: mcp__*__recall)
添加 PreToolUse hook (before_prompt)
添加 Notification hook (session_start)
添加 Stop hook (after_response)
写入 hooks.command = "python -m team_memory.hooks.claude_post_tool_use"
```

**Cursor:**
```
读取/创建 .cursor/hooks.json
添加 afterMCPExecution hook (matcher: MCP:team_memory)
添加 beforeSubmitPrompt hook
添加 afterAgentResponse hook
写入 command = "python -m team_memory.hooks.cursor_after_mcp"
```

**Hermes:**
```
读取 ~/.hermes/config.yaml
添加 post_tool_call hook (matcher: recall|memory_recall)
添加 pre_llm_call hook (before_prompt)
写入 command = "python -m team_memory.hooks.hermes_post_tool_call"
```

### 3.3 tm-cli setup 交互流程

```
$ tm-cli setup

  TM Setup Wizard / TM 配置向导
  ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

  [1/4] Server configuration / 服务配置
    TM server URL [http://localhost:9111]:
    API key: ****

  [2/4] LLM configuration / LLM 配置 (for faithfulness evaluation)
    LLM base URL [http://localhost:4000/v1]:
    LLM model [DeepSeek-V3]:
    LLM API key env var [LITELLM_MASTER_KEY]:

  [3/4] Platform hooks / 平台钩子
    Install hooks for:
      [x] Claude Code
      [x] Cursor
      [x] Hermes

  [4/4] Project / 项目
    Auto-detected: team_doc (from git repo)
    Project name [team_doc]:
    Aliases (comma-separated) []: team_memory

  ✓ Configuration saved to ~/.config/tm/config.toml
  ✓ Claude Code hooks installed in ~/.claude/settings.json
  ✓ Cursor hooks installed in .cursor/hooks.json
  ✓ Hermes hooks installed in ~/.hermes/config.yaml
  ✓ Project config saved to .tm.toml
```

### 3.4 tm-cli setup --check

```bash
# 验证当前配置状态
$ tm-cli setup --check

  TM Configuration Status / TM 配置状态
  ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

  Server:      ✓ http://localhost:9111 (reachable)
  API Key:     ✓ configured
  LLM:         ✓ DeepSeek-V3 @ http://localhost:4000/v1
  Daemon:      ✓ running on port 3901

  Hooks:
    Claude Code:  ✓ PostToolUse + Stop configured
    Cursor:       ✓ afterMCPExecution configured
    Hermes:       ✗ post_tool_call NOT configured (run: tm-cli setup --platform hermes)

  Project:
    .tm.toml:     ✓ name=team_doc, aliases=[team_memory]
    Auto-infer:   team_doc (from git repo)
```

---

## Phase 4: CLI 子命令完善

### 4.1 config 子命令

```bash
tm-cli config init       # 交互式生成 ~/.config/tm/config.toml
tm-cli config show       # 显示当前生效配置 (优先级合并后)
tm-cli config check      # 验证配置是否完整可达
tm-cli config set key val  # 设置单个配置项
```

### 4.2 project 子命令

```bash
tm-cli project init      # 在当前目录生成 .tm.toml
tm-cli project show      # 显示当前项目名 + aliases
tm-cli project list      # 列出 DB 中所有项目
tm-cli project migrate   # 合并历史别名数据
```

### 4.3 setup 子命令

```bash
tm-cli setup             # 完整配置向导 (server + llm + hooks + project)
tm-cli setup --check     # 验证当前配置
tm-cli setup --platform claude-code  # 只配置指定平台
tm-cli setup --uninstall            # 卸载 hooks
```

### 4.4 daemon 子命令

```bash
tm-daemon start          # 启动 daemon (前台)
tm-daemon start --detach # 后台启动
tm-daemon stop           # 停止 daemon
tm-daemon status         # 显示 daemon 状态
tm-daemon logs           # 查看日志
```

---

## Phase 5: MCP Server 远程模式

### 5.1 当前问题

```
MCP Server (server.py) 直接调 memory_operations → 直连 PG
→ 用户必须本地跑 PG 才能用 MCP
→ pip install 后用不了
```

### 5.2 解决方案：Remote MCP 模式

```python
# server.py 启动时检查
TM_SERVER_URL = os.environ.get("TM_SERVER_URL", "")

if TM_SERVER_URL:
    # Remote mode: 所有 tool 调用走 HTTP API
    from team_memory.remote_client import RemoteTMClient
    _tm = RemoteTMClient(TM_SERVER_URL, api_key=...)
else:
    # Local mode: 直接调 memory_operations (兼容现有部署)
    from team_memory.services import memory_operations
```

### 5.3 RemoteTMClient

```python
class RemoteTMClient:
    """HTTP client that mirrors memory_operations API."""

    async def op_recall(self, user, query, project, ...):
        resp = await httpx.post(f"{self.base_url}/api/v1/mcp/recall", ...)
        return resp.json()

    async def op_save(self, user, title, ...):
        resp = await httpx.post(f"{self.base_url}/api/v1/mcp/save", ...)
        return resp.json()
```

---

## 实施路径

| Phase | 优先级 | 改动量 | 依赖 |
|-------|--------|--------|------|
| Phase 1: 代码重组 | P1 | 大 | 无 |
| Phase 2: 配置统一 | P1 | 中 | Phase 1 |
| Phase 3: Hooks 一键配置 | P1 | 中 | Phase 1 + 2 |
| Phase 4: CLI 子命令 | P2 | 小 | Phase 2 |
| Phase 5: MCP 远程模式 | P2 | 中 | Phase 1 |

建议先做 Phase 1 (代码重组)，因为其他所有改动都依赖它。
Phase 1 完成后，Phase 2+3+4 可以并行。

---

## 开放问题

1. **api_key 存储方式**: config.toml 明文 vs api_key_env 引用环境变量？
   → 建议: 默认 api_key_env，高级用户可以 api_key 明文

2. **hooks 是独立的 Python 进程**: 配置里写 `python -m team_memory.hooks.xxx`
   → 需要 team_memory 已 pip install
   → 或者 hooks 是独立脚本 (用 urllib，不依赖 httpx)
   → 建议: 用 `python -m` 方式，pip install 后就有

3. **daemon 是否应该是 pip install 的一部分**:
   → 建议: 是，作为 optional dependency
   → pip install team_memory[daemon] 才装 daemon 相关依赖
   → tm-daemon 命令在 [daemon] 安装后才可用

4. **迁移策略**: 现有用户 PYTHONPATH=src:scripts 的用法
   → Phase 1 完成后 import 路径变了
   → 需要 changelog 说明
   → 保留一段时间兼容期 (PYTHONPATH 仍可用)

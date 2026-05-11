# CLAUDE.md

本文面向 Claude Code、Cursor 及在本仓库工作的其他 Agent。**本仓库主入口**。

## 速查命令

```bash
make help           # 列出可用命令
make setup          # 首次安装（Docker + 依赖 + 数据库迁移）
make dev            # 启动全部服务（Docker + Web）
make web            # 仅启动 Web（http://localhost:9111）
make mcp            # MCP（需仓库根 .env；run_mcp_with_dotenv → team_memory.server）
make mcp-verify     # 校验 6 个 memory_* 工具注册（pytest 快速单测）
make test           # 运行全部测试
make lint           # Ruff 代码检查
make verify         # 标准验收：lint + 全量测试
make harness-check  # Harness 门禁：import 检查 + ruff + lint-js + doc-harness 配置
make sync-agent-artifacts  # 由 agents/shared 生成 agents、prompts、skills（Claude 侧为 /slash skill）

tm-cli --help    # CLI 兼容层（MCP 工具的 Shell 等价）
tm-cli recall --query "..." # 搜索团队知识

# 单测 / 筛选
pytest tests/test_server.py::TestLiteToolRegistration::test_exactly_six_tools -v
pytest -k "search" -v
```

## Agent / 流程提示词（SSOT）

- **源**：`agents/shared/bodies/`、`agents/shared/prompts/`、`agents/manifest.yaml`。
- **同步**：`make sync-agent-artifacts` → `.claude/agents/`、`.cursor/agents/`、`.cursor/prompts/`、`.claude/skills/*/SKILL.md`。
- **与 Cursor 对齐**：`.cursor/prompts` 下的流程说明与 **同名 skill**（`/plan-eval-multi-agent-review` 等）同源。

## 架构概览

**TeamMemory** — 基于 MCP 的团队经验库，为 AI 提供跨会话记忆。

```
┌─────────────────────────────────────────┐
│  MCP Server / CLI (server.py, cli.py)   │  ← AI 入口，memory_* 工具
│  Web Routes (web/routes/)               │  ← HTTP API 入口
├─────────────────────────────────────────┤
│  Services (services/)                   │  ← 业务逻辑
│  Auth / Embedding / Reranker            │  ← 认证、向量、重排
├─────────────────────────────────────────┤
│  Storage (storage/)                     │  ← 数据访问（repository + DB）
├─────────────────────────────────────────┤
│  Models (storage/models.py, schemas.py) │  ← ORM + Pydantic
├─────────────────────────────────────────┤
│  Infrastructure (PostgreSQL + pgvector) │
└─────────────────────────────────────────┘
```

**依赖方向（仅自上而下）**：Server/Web → Services → Storage → Models。禁止反向 import。

### 核心领域

- **Experience**：title/problem/solution/tags/score/status；状态流 draft → review → published/rejected
- **Search Pipeline**：向量检索 + 全文检索 → RRF 融合 → 重排 → Token 裁剪 → Top-N
- **MCP Tools**：`server.py` 内 `memory_*` 工具；只能调用 Services 层

## 开发指南

新进仓库或要做跨模块改动时，建议先读 [README.md](README.md)，再按需查 [docs/README.md](docs/README.md)。

### MCP 工具开发

1. 在 [README.md](README.md) 与 [docs/guide/mcp-server.md](docs/guide/mcp-server.md) 保持叙述一致；实现以 [src/team_memory/server.py](src/team_memory/server.py) 为准
2. `@mcp.tool` + `@track_usage` 装饰 `async def memory_xxx(...) -> str`
3. 在 `tests/test_server.py` 补充测试
4. `make verify` 验收

### 代码约定

- **仅异步**：DB 操作 `async/await`
- **类型注解**：函数签名须标注
- **校验**：外部输入经 Pydantic 校验
- **日志**：项目 logger，禁止裸 `print()`
- **密钥**：禁止硬编码；分层与 import 见 `scripts/harness_import_check.py`

### 测试

覆盖率：MCP ≥ 90%，Services ≥ 80%。运行见 README「开发」与 `make test`。

### 数据库迁移

```bash
alembic revision --autogenerate -m "description"
alembic upgrade head
```

每次模型变更须包含迁移文件。

## 完成标准

- [ ] `make lint` 零报错
- [ ] `make test` 全绿（新功能须有测试）
- [ ] 若修改 model，须有 Alembic 迁移
- [ ] 若新增 MCP 工具，已更新 [README.md](README.md) / [docs/guide/mcp-server.md](docs/guide/mcp-server.md) 且测试覆盖
- [ ] 无硬编码密钥、无裸 `print()`
- [ ] Web 改动通过 `make lint-js`
- [ ] 若变更架构分层、MCP 对外契约或用户可见文档，已同步 [README.md](README.md)、[AGENTS.md](AGENTS.md) 与 [docs/](docs/) 下相关篇目

## 导航

| 主题 | 文件 |
|------|------|
| 人类指南与 MCP | [README.md](README.md) |
| 分层 / import 门禁 | `scripts/harness_import_check.py`（`LAYER_MAP`）；设计文档索引 [docs/README.md](docs/README.md) |
| Import 分层 | `scripts/harness_import_check.py` |
| MCP 实现 | [src/team_memory/server.py](src/team_memory/server.py) |
| CLI 兼容层 | [src/team_memory/cli.py](src/team_memory/cli.py)、[docs/guide/mcp-server.md](docs/guide/mcp-server.md) |
| MCP 运维与配置 | [docs/guide/mcp-server.md](docs/guide/mcp-server.md)、[docs/decision/mcp-lite-default.md](docs/decision/mcp-lite-default.md) |
| 设计文档 | [docs/](docs/) |
| Harness 框架 | [.harness/docs/harness-spec.md](.harness/docs/harness-spec.md) |
| Agent 入口（本仓库） | [AGENTS.md](AGENTS.md) |

## 调试

```bash
TEAM_MEMORY_LOG_LEVEL=DEBUG make web       # DEBUG 日志
TEAM_MEMORY_MCP_DEBUG=1 make mcp           # MCP IO 日志
docker compose exec postgres psql -U developer -d team_memory  # DB 调试
```

## 常见坑

| 问题 | 解决 |
|------|------|
| `make verify` import 报错 | 检查 import 方向，Storage 不能 import Services |
| 测试连不上数据库 | `docker compose up -d postgres` |
| Embedding 失败 | `ollama pull nomic-embed-text` |
| 端口 9111 被占用 | `make release-9111` |

## 不确定时

1. 查 `docs/` 对应规范
2. 找代码库同类实现
3. **停下询问，不要臆测**

<!-- SPECKIT START -->
For additional context about technologies to be used, project structure,
shell commands, and other important information, read the current plan
<!-- SPECKIT END -->

# CLAUDE.md

Behavioral guidelines to reduce common LLM coding mistakes. Merge with project-specific instructions as needed.

**Tradeoff:** These guidelines bias toward caution over speed. For trivial tasks, use judgment.

## 1. Think Before Coding

**Don't assume. Don't hide confusion. Surface tradeoffs.**

Before implementing:
- State your assumptions explicitly. If uncertain, ask.
- If multiple interpretations exist, present them - don't pick silently.
- If a simpler approach exists, say so. Push back when warranted.
- If something is unclear, stop. Name what's confusing. Ask.

## 2. Simplicity First

**Minimum code that solves the problem. Nothing speculative.**

- No features beyond what was asked.
- No abstractions for single-use code.
- No "flexibility" or "configurability" that wasn't requested.
- No error handling for impossible scenarios.
- If you write 200 lines and it could be 50, rewrite it.

Ask yourself: "Would a senior engineer say this is overcomplicated?" If yes, simplify.

## 3. Surgical Changes

**Touch only what you must. Clean up only your own mess.**

When editing existing code:
- Don't "improve" adjacent code, comments, or formatting.
- Don't refactor things that aren't broken.
- Match existing style, even if you'd do it differently.
- If you notice unrelated dead code, mention it - don't delete it.

When your changes create orphans:
- Remove imports/variables/functions that YOUR changes made unused.
- Don't remove pre-existing dead code unless asked.

The test: Every changed line should trace directly to the user's request.

## 4. Goal-Driven Execution

**Define success criteria. Loop until verified.**

Transform tasks into verifiable goals:
- "Add validation" → "Write tests for invalid inputs, then make them pass"
- "Fix the bug" → "Write a test that reproduces it, then make it pass"
- "Refactor X" → "Ensure tests pass before and after"

For multi-step tasks, state a brief plan:
```
1. [Step] → verify: [check]
2. [Step] → verify: [check]
3. [Step] → verify: [check]
```

Strong success criteria let you loop independently. Weak criteria ("make it work") require constant clarification.

---

**These guidelines are working if:** fewer unnecessary changes in diffs, fewer rewrites due to overcomplication, and clarifying questions come before implementation rather than after mistakes.

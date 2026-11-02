# AGENTS.md

本文面向 Cursor 及在本仓库工作的其他 Agent。**本仓库主入口**。

> Claude Code 用户请阅读 [CLAUDE.md](CLAUDE.md)。

## 速查命令

```bash
make help           # 列出可用命令
make setup          # 首次安装（Docker + 依赖 + 数据库迁移）
make dev            # 启动全部服务（Docker + Web）
make web            # 仅启动 Web（http://localhost:9111）
make mcp            # MCP（需仓库根 .env；run_mcp_with_dotenv → team_memory.server）
make test           # 运行全部测试
make lint           # Ruff 代码检查
make verify         # 标准验收：lint + 全量测试
make harness-check  # Harness 门禁：import 检查 + ruff + lint-js + doc-harness 配置
make sync-agent-artifacts  # 由 agents/shared 生成 .claude/.cursor agents、.cursor prompts、.claude skills

tm-cli --help    # CLI 兼容层（MCP 工具的 Shell 等价）
tm-cli status               # 全景查看 TM 四大平面运行状态（Core/Daemon/精炼漏斗/图谱）
tm-cli status --watch       # 终端实时持续刷新监控面板（等价于 tm-cli watch，Ctrl+C 退出）
tm-cli recall --query "..." # 搜索团队知识
tm-cli config show          # 显示当前生效配置及值来源
tm-cli config init          # 交互式生成 ~/.config/tm/config.toml
tm-cli setup --platform all # 一键配置平台 Hook

# 单测 / 筛选
pytest tests/test_server.py::TestLiteToolRegistration::test_exactly_six_tools -v
pytest -k "search" -v
```

## Agent / 流程提示词（SSOT）

- **正文与元数据**：`agents/shared/bodies/`（各 subagent）、`agents/shared/prompts/`（流程说明）；清单 `agents/manifest.yaml`。
- **生成产物**：修改后执行 `make sync-agent-artifacts`，会写入 `.claude/agents/`、`.cursor/agents/`、`.cursor/prompts/`，以及 **Claude Code** 侧与 Cursor prompt 对位的 **Skill**（`.claude/skills/<name>/SKILL.md`，可用 `/name` 调用）。
- **对照**：Cursor 的 `.cursor/prompts/*.md` 与 Claude 的同名 skill **同源**，按需在一端编辑 shared 后重新同步。

## 架构概览

**TeamMemory** — 基于 MCP 的团队经验库，为 AI 提供跨会话记忆。

```
┌──────────────────────────────────────────────┐
│  MCP Server / CLI (server.py, cli.py)        │  ← AI 入口，memory_* 工具
│  Web Routes (web/routes/)                    │  ← HTTP API 入口
├──────────────────────────────────────────────┤
│  Services (services/)                        │  ← 业务逻辑
│    search/   memory/   llm/   entity/        │
│    archive/  maintenance/ analysis/ infra/   │
│  Auth / Embedding (+ factory) / Reranker     │  ← 认证、向量、重排
├──────────────────────────────────────────────┤
│  Storage (storage/)                          │  ← 数据访问（repository + DB）
├──────────────────────────────────────────────┤
│  Models (storage/models.py, schemas.py)      │  ← ORM + Pydantic
├──────────────────────────────────────────────┤
│  Infrastructure (PostgreSQL + pgvector)      │
└──────────────────────────────────────────────┘

客户端组件（PYTHONPATH 只需 src/）:
┌──────────────────────────────────────────────┐
│  client_config.py                            │  ← ClientConfig（~/.config/tm/config.toml）
│  project_resolver.py                         │  ← 项目名零配置推断 + 别名注册
│  daemon/                                     │  ← TM Daemon（tm-daemon 入口）
│    runtime/ buffers/ refinement/ backfill/   │
│    quality/ wiki/ observe/ integration/      │
│  hooks/                                      │  ← 平台 Hook
│    core/ platforms/{claude,cursor,hermes}/   │
│    integrations/                             │
│  cli.py                                      │  ← tm-cli + config/setup 子命令
└──────────────────────────────────────────────┘
```

**依赖方向（仅自上而下）**：Server/Web → Services → Storage → Models。禁止反向 import。

**配置两层 / Two Config Layers**：
- **服务端**：`config.development.yaml` / `config.production.yaml`（Pydantic Settings + YAML）
- **客户端**：`~/.config/tm/config.toml`（ClientConfig + TOML），优先级：环境变量 > `.env` > `config.toml` > 代码默认值

**项目名推断 / Project Name Inference**：`.tm.toml` > git 仓库名 > 目录名。别名注册：`project_resolver.register_aliases()`。

### 代码目录结构规范

**新增文件必须放入正确的子包**，禁止在 `hooks/`、`daemon/`、`services/` 根目录直接创建业务模块。

| 包 | 子包 | 职责 |
|----|------|------|
| `hooks/` | `core/` | 共享基础设施（daemon_http、stdin、mcp_callback、output） |
| | `platforms/{claude,cursor,hermes}/` | 各平台 Hook 脚本 |
| | `integrations/` | 跨平台集成（git_watcher、weekly_report 等） |
| `daemon/` | `runtime/` | 启动、配置、热重载、健康检查 |
| | `buffers/` | 会话/草稿/响应缓冲、待处理队列 |
| | `refinement/` | 处理管线、草稿精炼 |
| | `backfill/` | 向量/实体/方案回填、实体去重 |
| | `quality/` | 忠实度判断、搜索评估、收敛检测 |
| | `wiki/` | Wiki 编译、Markdown 索引 |
| | `observe/` | 指标采集、搜索日志、统计、文件监控 |
| | `integration/` | TM Sink、自动梦境、个人记忆提取 |
| `services/` | `search/` | search_pipeline、search_orchestrator、cache |
| | `memory/` | memory_operations、experience、personal_memory |
| | `llm/` | llm_client、llm_parser、llm_provider |
| | `entity/` | entity_extractor、entity_search |
| | `archive/` | ArchiveService（`__init__.py` re-export） |
| | `maintenance/` | janitor、janitor_scheduler、promotion_compiler |
| | `analysis/` | evaluation、contradiction_detector、pattern_extractor |
| | `infra/` | event_bus、task_runner、tokenizer、intent_router |
| `embedding/` | `factory.py` | create_embedding_provider()（从 bootstrap 迁出） |

**禁止事项**：
- 禁止在子包父目录创建平铺模块（如 `daemon/xxx.py`，应为 `daemon/<subpkg>/xxx.py`）
- 禁止创建 shim / re-export 兼容文件，保持项目简洁
- 禁止跨层 import（Storage ↛ Services；Services ↛ Server/Web）
- import 路径必须完整：`from team_memory.daemon.runtime.config import ...`，不允许 `from team_memory.daemon.config import ...`

### 核心领域

- **Experience**：title/problem/solution/tags/score/status；状态流 draft → review → published/rejected
- **Search Pipeline**：向量检索 + 全文检索 → RRF 融合 → 重排 → Token 裁剪 → Top-N
- **MCP Tools**：`server.py` 内 `memory_*` 工具；只能调用 Services 层

## 开发指南

新进仓库或要做跨模块改动时，建议先读 [README.md](README.md)。

### MCP 工具开发

1. 在 [README.md](README.md)（MCP 工具表）保持叙述一致；契约以 [src/team_memory/server.py](src/team_memory/server.py) 为准
2. `@mcp.tool` + `@track_usage` 装饰 `async def memory_xxx(...) -> str`
3. 在 `tests/test_server.py` 补充测试
4. `make verify` 验收

### 代码约定

- **仅异步**：DB 操作 `async/await`
- **类型注解**：函数签名须标注
- **校验**：外部输入经 Pydantic 校验
- **日志**：项目 logger，禁止裸 `print()`
- **密钥**：禁止硬编码；分层与 import 方向以 `make harness-check` / `scripts/harness_import_check.py` 为准

### 测试

覆盖率：MCP ≥ 90%，Services ≥ 80%。运行方式见 README「开发」与 `make test`。

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
- [ ] 若新增 MCP 工具，已更新 [README.md](README.md) 且 `tests/test_server.py` 覆盖
- [ ] 无硬编码密钥、无裸 `print()`
- [ ] Web 改动通过 `make lint-js`
- [ ] 若变更架构分层、MCP 对外契约或用户可见文档，已同步更新 [README.md](README.md) 与本文件
- [ ] 若新增配置项，已在 `ClientConfig`（`client_config.py`）中添加，并在 `load_with_sources()` 中标注来源
- [ ] 若修改 daemon/hooks 代码，确认 import 路径为 `from team_memory.daemon/...` 而非旧 `from daemon/...`

## 导航

| 主题 | 文件 |
|------|------|
| 人类指南与 MCP 说明 | [README.md](README.md) |
| 分层 / import 门禁 | `scripts/harness_import_check.py`（`LAYER_MAP`） |
| Import 分层（L0–L3） | `scripts/harness_import_check.py`（`LAYER_MAP`） |
| MCP 实现 | [src/team_memory/server.py](src/team_memory/server.py) |
| CLI 兼容层 + config/setup | [src/team_memory/cli.py](src/team_memory/cli.py) |
| 统一客户端配置 | [src/team_memory/client_config.py](src/team_memory/client_config.py)（ClientConfig） |
| 项目名零配置推断 | [src/team_memory/project_resolver.py](src/team_memory/project_resolver.py) |
| TM Daemon | [src/team_memory/daemon/](src/team_memory/daemon/)（`tm-daemon` 入口） |
| 平台 Hook | [src/team_memory/hooks/](src/team_memory/hooks/)（Claude/Cursor/Hermes） |

## 调试

```bash
TEAM_MEMORY_LOG_LEVEL=DEBUG make web       # DEBUG 日志
TEAM_MEMORY_MCP_DEBUG=1 make mcp           # MCP IO 日志
docker compose exec postgres psql -U developer -d team_memory  # DB 调试
```

## 常见坑

| 问题 | 解决 |
|------|------|
| `make verify` import 报错 | 检查 import 方向，Storage 不能 import Services；PYTHONPATH 只需 `src/`，不再需要 `scripts/` |
| 测试连不上数据库 | `docker compose up -d postgres` |
| Embedding 失败 | `ollama pull qwen3-embedding:0.6b`；Ollama 重启后 `/health` 会自动重试初始化 |
| 端口 9111 被占用 | `make release-9111` |
| 找不到 daemon/hooks 模块 | 已迁入 `src/team_memory/daemon/` 和 `src/team_memory/hooks/`，使用 `from team_memory.daemon import ...` |
| 项目名不对 | 检查 `.tm.toml` 或 git 仓库名；`tm-cli config show` 查看来源 |

## 不确定时

1. 找代码库同类实现
2. **停下询问，不要臆测**

# Agent Workflow & Skills (Superpowers)
This project uses the **Superpowers** framework for AI agent execution.
- **DO NOT** use Speckit or generate `spec.md`, `plan.md`, or `tasks.md` files.
- **DO NOT** create `.specify` or `specs` directories.
- **DO** use `/brainstorming` for planning and `/subagent-driven-development` or `/executing-plans` for execution.
- Keep the project clean. Do not leave process artifacts in the repository.

## LLM Behavioral Guidelines


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


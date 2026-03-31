# AGENTS.md

本文面向 Cursor 及在本仓库工作的其他 Agent。**本仓库主入口**。

> Claude Code 用户请阅读 [CLAUDE.md](CLAUDE.md)。

## 速查命令

```bash
make help           # 列出可用命令
make setup          # 首次安装（Docker + 依赖 + 数据库迁移）
make dev            # 启动全部服务（Docker + Web）
make web            # 仅启动 Web（http://localhost:9111）
make mcp            # Lite MCP（memory_*）；遗留 tm_* 用 make mcp-full
make test           # 运行全部测试
make lint           # Ruff 代码检查
make verify         # 标准验收：lint + 全量测试
make harness-check  # Harness 门禁：import 检查 + ruff + lint-js

# 单测 / 筛选
pytest tests/test_server.py::test_tm_search -v
pytest -k "search" -v
```

## 架构概览

**TeamMemory** — 基于 MCP 的团队经验库，为 AI 提供跨会话记忆。

```
┌─────────────────────────────────────────┐
│  MCP Server (server.py)                 │  ← AI 入口，tm_* 工具
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
- **MCP Tools**：`server.py` 内 `tm_*` 命名空间；只能调用 Services 层

## 开发指南

### MCP 工具开发

1. 在 [docs/mcp-patterns.md](docs/mcp-patterns.md) 登记
2. `@mcp.tool` + `@track_usage` 装饰 `async def tm_xxx(...) -> str`
3. 在 `tests/test_server.py` 补充测试
4. `make verify` 验收

### 代码约定

- **仅异步**：DB 操作 `async/await`
- **类型注解**：函数签名须标注
- **校验**：外部输入经 Pydantic 校验
- **日志**：项目 logger，禁止裸 `print()`
- **密钥**：禁止硬编码

详见 [docs/conventions.md](docs/conventions.md)

### 测试

覆盖率：MCP ≥ 90%，Services ≥ 80%。详见 [docs/testing.md](docs/testing.md)

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
- [ ] 若新增 MCP 工具，已在 mcp-patterns.md 登记
- [ ] 无硬编码密钥、无裸 `print()`
- [ ] Web 改动通过 `make lint-js`

## 导航

| 主题 | 文件 |
|------|------|
| 架构约束与分层 | [docs/architecture.md](docs/architecture.md) · [project-extension](docs/design-docs/harness/project-extension.md) |
| MCP 工具开发 | [docs/mcp-patterns.md](docs/mcp-patterns.md) |
| Python 约定 | [docs/conventions.md](docs/conventions.md) |
| 测试 | [docs/testing.md](docs/testing.md) |
| 安全 | [docs/security.md](docs/security.md) |
| 设计文档 | [docs/design-docs/](docs/design-docs/) |
| Harness 框架 | [.harness/docs/harness-spec.md](.harness/docs/harness-spec.md) |
| Harness 配置 | [.harness/harness-config.yaml](.harness/harness-config.yaml) |
| 入门 | [docs/getting-started.md](docs/getting-started.md) |

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

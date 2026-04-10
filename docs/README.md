# 文档索引

本目录按用途拆为四类，便于人类与 Agent 快速定位。

**请先读**：[../README.md](../README.md) · [../AGENTS.md](../AGENTS.md) — 表结构与领域模型以 [`src/team_memory/storage/models.py`](../src/team_memory/storage/models.py) 与 Alembic `migrations/versions/` 为准。

---

## 路径约定（Markdown 链接）

| 源位置 | 目标 | 路径示例 |
|--------|------|----------|
| `.cursor/rules/` | `docs/` 下某篇 | `../docs/guide/mcp-server.md` |
| `docs/*/` | 仓库根文件 | `../../README.md` |
| `docs/*/` | `example/` | `../../example/...` |
| `docs/*/` | `src/` | `../../src/team_memory/...` |

---

## 索引

### 架构与门禁（入口仍在仓库根 / scripts）

| 文档 | 说明 |
|------|------|
| `scripts/harness_import_check.py` | **L0～L3** 模块映射（`LAYER_MAP`） |
| [AGENTS.md](../AGENTS.md) | 分层约定、MCP 与完成标准 |
| [.harness/docs/harness-spec.md](../.harness/docs/harness-spec.md) | Harness 框架 |
| [doc-harness.project.yaml](../doc-harness.project.yaml) | 文档门禁配置 |
| [agents/manifest.yaml](../agents/manifest.yaml) + `agents/shared/` | Subagent / 流程 SSOT；`make sync-agent-artifacts` 生成 `.claude` / `.cursor` 下 agents、prompts、skills |

### `decision/` — 技术决断 / ADR

| 文档 | 说明 |
|------|------|
| [database-design-decisions](decision/database-design-decisions.md) | 为何用 PG / UUID / ARRAY / 软删等 |
| [auth-api-key-design](decision/auth-api-key-design.md) | API Key、hash、预注册与 `api_keys` 表 |
| [mcp-lite-default](decision/mcp-lite-default.md) | 决策：仅 `memory_*` 单一 MCP 入口 |

### `cmd/` — 可复制命令

| 文档 | 说明 |
|------|------|
| [database-operations](cmd/database-operations.md) | Docker PG、psql、迁移 |
| [migrate-fts](cmd/migrate-fts.md) | 存量补全 `experiences.fts` |

### `guide/` — 场景操作说明与示例

| 文档 | 说明 |
|------|------|
| [quick-start](guide/quick-start.md) | 首次初始化与日常启动 |
| [web-server](guide/web-server.md) | Web 端口、环境变量、界面能力概览 |
| [mcp-server](guide/mcp-server.md) | MCP：`.env`、包装脚本、`mcp.json` |
| [mcp-pypi-local](guide/mcp-pypi-local.md) | PyPI 包装包本机跑 MCP |
| [env.team-memory.example](../example/env.team-memory.example) | `.env` 模板 |
| [cursor-mcp-team-memory.example.json](../example/cursor-mcp-team-memory.example.json) | Cursor / Claude `mcp.json` 模板 |

### `ops/` — 应急、排障、runbook

| 文档 | 说明 |
|------|------|
| [troubleshooting](ops/troubleshooting.md) | 常见问题 |
| [alembic-multiple-heads-fix](ops/alembic-multiple-heads-fix.md) | Alembic 多 head 修复 |
| [runbook](ops/runbook.md) | 回滚、迁移 revision 链、备份、健康检查 |

**档案馆设计历史稿**：Team Memory 标题 **`【docs-plans】mcp-archive-api-redesign`**；现行约定以 [mcp-server](guide/mcp-server.md) 与 `server.py` 为准。

**exec-plan 导出 manifest**：`scripts/remediate_tm_exec_plan_archives.py export-sources` 默认写出仓库根 `data/TM-ARCHIVE-MANIFEST.md`（由导出生成，**不一定**在克隆后的工作区里；默认路径见该脚本内 `_DEFAULT_MANIFEST`）。

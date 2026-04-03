# Design Docs

设计文档存放目录。用于架构设计、技术方案、决策记录等。

**Agent / 贡献者请先读**：[../../README.md](../../README.md) · [../../AGENTS.md](../../AGENTS.md)（再按需查阅本目录索引与具体文档）。

---

## 路径约定（Markdown 链接）

Markdown 链接 `[text](path)` 按**相对当前文件**解析。跨目录引用须使用正确相对路径：

| 源位置 | 目标 | 路径示例 |
|--------|------|----------|
| `.cursor/rules/` | `docs/design-docs/` | `../../docs/design-docs/xxx.md` |
| `.cursor/rules/` | `docs/exec-plans/` | `../../docs/exec-plans/xxx.md` |
| `.cursor/rules/` | `docs/`（根级） | `../../docs/xxx.md` |
| `docs/design-docs/` | 同目录 | `xxx.md` 或 `./xxx.md` |
| `docs/design-docs/` | `.cursor/rules/` | `../../.cursor/rules/xxx.mdc` |
| `docs/exec-plans/` | `docs/design-docs/` | `../design-docs/xxx.md` |
| `docs/` | 同目录 | `xxx.md` |

---

## 索引

### 架构、Harness 与文档门禁

| 文档 | 说明 |
|------|------|
| [架构设计图.md](../架构设计图.md) | Mermaid 分层与数据流 |
| `scripts/harness_import_check.py` | **L0～L3** 模块映射（`LAYER_MAP`），与 `make harness-check` 一致 |
| [AGENTS.md](../../AGENTS.md) | 分层约定、MCP 开发与完成标准 |
| [.harness/docs/harness-spec.md](../../.harness/docs/harness-spec.md) | 可移植 Harness 框架总览（编排见 `.harness/orchestration/`） |
| [doc-health skill](../../.claude/skills/doc-health/SKILL.md) | 设计文档维护约定与 rule_id（配合 `doc-harness.project.yaml`） |
| [doc-admin-organize](../../.cursor/agents/doc-admin-organize.md) | 文档整理 Agent |
| [doc-harness.project.yaml](../../doc-harness.project.yaml) | 文档门禁项目配置 |

### backend-architecture/ — 存储与检索架构

项目架构设计、模块划分、关键流程。设计决策见 `database-design-decisions`。

| 文档 | 说明 |
|------|------|
| [database-schema](backend-architecture/database-schema.md) | 数据库表结构 |
| [search-query-flow](backend-architecture/search-query-flow.md) | 搜索查询流程 |
| [experience-save-flow](backend-architecture/experience-save-flow.md) | 经验存入流程 |
| [auth-api-key-design](backend-architecture/auth-api-key-design.md) | API Key 与认证设计 |
| [database-design-decisions](backend-architecture/database-design-decisions.md) | 数据库设计决策 |

### tech-concepts/ — 技术概念

技术概念说明（向量、pgvector、Reranker、RRF 等），不涉及具体实现。

| 文档 | 说明 |
|------|------|
| [embedding-vector](tech-concepts/embedding-vector.md) | 向量与 Embedding |
| [pgvector-fts](tech-concepts/pgvector-fts.md) | pgvector 与全文搜索 |
| [reranker](tech-concepts/reranker.md) | Reranker 精排 |
| [rrf-hybrid-search](tech-concepts/rrf-hybrid-search.md) | RRF 混合检索 |

### ops/ — 运维操作

操作类文档：启动、配置、迁移、备份、故障排查。

| 文档 | 说明 |
|------|------|
| [quick-start](ops/quick-start.md) | 快速启动 |
| [database-operations](ops/database-operations.md) | 数据库操作 |
| [web-server](ops/web-server.md) | Web 服务 |
| [mcp-server](ops/mcp-server.md) | MCP 服务 |
| [troubleshooting](ops/troubleshooting.md) | 故障排查 |
| [migrate-fts](ops/migrate-fts.md) | FTS 迁移 |
| [alembic-multiple-heads-fix](ops/alembic-multiple-heads-fix.md) | Alembic 多 head 修复 |
| [ci-cd](ops/ci-cd.md) | CI/CD |

### 根级

| 文档 | 说明 |
|------|------|
| [logging-format](logging-format.md) | JSON 行日志规范、生产/CI 切换（Phase 4） |
| [agent-memory-projects-survey](agent-memory-projects-survey.md) | Agent 记忆类产品调研 |
| [experience-commit-binding-survey](experience-commit-binding-survey.md) | 经验与提交绑定等调研 |
| [archive-attachment-to-experience](archive-attachment-to-experience.md) | 档案馆与经验关联方案 |

---

## 相关

- **backend-architecture** ↔ **tech-concepts** ↔ **ops**：架构、概念、运维三者相互引用，详见各文档内链接。
# Design Docs

设计文档存放目录。用于架构设计、技术方案、决策记录等。

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

### harness/ — Harness 规范

| 文档 | 说明 |
|------|------|
| [harness-spec](harness/harness-spec.md) | 通用 Harness 规范（核心原则、Plan 执行、反馈回路、流程图） |
| [project-extension](harness/project-extension.md) | team_doc 项目扩展（架构分层、质量门禁、收尾清单、tm 边界） |
| [doc-maintenance-guide](harness/doc-maintenance-guide.md) | 文档维护规范（第一章）+ 扫描设计（第二章） |
| [plan-document-structure](harness/plan-document-structure.md) | Plan 文档结构规范（调研/计划/复盘）+ 扫描规则 |
| [doc-admin-organize](../../.cursor/agents/doc-admin-organize.md) | 文档整理 Agent（按规范实际整理） |

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

### extraction/ — 经验提取

| 文档 | 说明 |
|------|------|
| [extraction-build-guide](extraction/extraction-build-guide.md) | 经验提取能力构建、状态文件、触发条件、防抖 |

### 根级

| 文档 | 说明 |
|------|------|
| [logging-format](logging-format.md) | JSON 行日志规范、生产/CI 切换（Phase 4） |

---

## 相关

- **backend-architecture** ↔ **tech-concepts** ↔ **ops**：架构、概念、运维三者相互引用，详见各文档内链接。
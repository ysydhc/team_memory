# .debug/docs 与 .debug/tm 迁移计划

> 按 harness-engineering 规则，将 .debug 文档迁移至 docs/design-docs/ 与 docs/exec-plans/。  
> **已确认**：2025-03 用户确认 6 点 + 补充规则。

---

## 一、迁移规则（依据 harness-engineering）

| 类型 | 目标路径 | 说明 |
|------|----------|------|
| **设计文档** | `docs/design-docs/` | 架构、方案、决策、技术设计 |
| **执行计划**（进行中） | `docs/exec-plans/` | 未完成的 plan、implementation plan |
| **执行计划**（已完成） | `docs/exec-plans/completed/` | 已完成的 plan、执行报告 |
| **归档** | `docs/exec-plans/archive/` 或 `docs/design-docs/archive/` | deprecated、old、已过时 |

---

## 二、已确认项（用户 2025-03）

| # | 确认点 | 结论 |
|---|--------|------|
| 1 | tm/ops | **迁入** design-docs（不合并进 GETTING-STARTED） |
| 2 | governance 治理报告/整治/优化计划 | **全部删除** |
| 3 | analysis 三份研究 | **全部删除** |
| 4 | architecture-viz 综合方案 | **全部删除**（前置调研报告，本次不保留） |
| 5 | topics 下 .plan.md 及 GitNexus 代码可视化方案 | **仅保留**「与 GitNexus 配合设计代码可视化功能」的方案 → 迁入 exec-plans；**其余全部删除** |
| 6 | deprecated 目录 | **全部删除**（不迁移） |

### 补充规则（目录结构）

- **不新建** `tm-architecture`、`tm-tech-concepts`、`tm-ops` 等子目录。
- **简化核心目录**：目标是通过一个文件（如 README）能看清文档结构。
- **仅以下情况可建子目录**：
  - 非常底层的文档；
  - 有统一主题（如某个 plan 的 plan 文件、执行流程、plan 复盘等强相关内容）。

---

## 三、.debug/tm 迁移清单（20 个文件）

**原则**：不建 tm-architecture、tm-tech-concepts、tm-ops 子目录；全部迁入 `docs/design-docs/` 根级，用文件名区分主题。

| 源文件 | 目标 |
|--------|------|
| tm/architecture/README.md | design-docs/tm-architecture-readme.md |
| tm/architecture/01-database-schema.md | design-docs/tm-database-schema.md |
| tm/architecture/02-search-query-flow.md | design-docs/tm-search-query-flow.md |
| tm/architecture/03-experience-save-flow.md | design-docs/tm-experience-save-flow.md |
| tm/architecture/03-auth-api-key-design.md | design-docs/tm-auth-api-key-design.md |
| tm/architecture/thinking/01-database-design-decisions.md | design-docs/tm-database-design-decisions.md |
| tm/tech-concepts/README.md | design-docs/tm-tech-concepts-readme.md |
| tm/tech-concepts/01-embedding-vector.md | design-docs/tm-embedding-vector.md |
| tm/tech-concepts/02-pgvector-fts.md | design-docs/tm-pgvector-fts.md |
| tm/tech-concepts/03-reranker.md | design-docs/tm-reranker.md |
| tm/tech-concepts/04-rrf-hybrid-search.md | design-docs/tm-rrf-hybrid-search.md |
| tm/ops/README.md | design-docs/tm-ops-readme.md |
| tm/ops/01-quick-start.md | design-docs/tm-quick-start.md |
| tm/ops/02-database-operations.md | design-docs/tm-database-operations.md |
| tm/ops/03-web-server.md | design-docs/tm-web-server.md |
| tm/ops/04-mcp-server.md | design-docs/tm-mcp-server.md |
| tm/ops/05-troubleshooting.md | design-docs/tm-troubleshooting.md |
| tm/ops/06-migrate-fts.md | design-docs/tm-migrate-fts.md |
| tm/ops/07-alembic-multiple-heads-fix.md | design-docs/tm-alembic-multiple-heads-fix.md |
| tm/ops/08-ci-cd.md | design-docs/tm-ci-cd.md |

---

## 四、.debug/docs 迁移清单（按确认执行）

### 4.1 保留并迁入 design-docs

| 源文件 | 目标 |
|--------|------|
| docs/governance/文档维护规范建议.md | design-docs/doc-maintenance-guide.md |
| docs/ci-harness-check-commands.md | design-docs/harness-check-commands.md |

### 4.2 保留并迁入 exec-plans（GitNexus + 代码可视化）

| 源文件 | 目标 |
|--------|------|
| topics/architecture/architecture-viz-requirements-spec.md | exec-plans/wait/code-arch-viz-gitnexus/code-arch-viz-gitnexus-requirements-spec.md |
| topics/architecture/architecture-viz-task-group.md | exec-plans/wait/code-arch-viz-gitnexus/code-arch-viz-gitnexus-task-group.md |

### 4.3 全部删除（不迁移）

| 类别 | 路径/文件 |
|------|-----------|
| governance 治理报告/整治/优化计划 | governance/文档治理报告-2025-03.md、文档整治-执行全流程.md、文档优化执行计划-2025-03.md |
| analysis 三份研究 | analysis/ai-memory-agent-persistence-research.md、ai-agent-memory-trigger-research.md、ai-agent-memory-research-2024-2025.md |
| architecture-viz 综合方案 | architecture-viz/代码架构可视化-综合方案.md |
| topics 下除 4.2 外全部 | 21 个 .plan.md + experience-retrieval、personal-memory、web-ux、tm-evaluation、workflow、planning、ops、docs-standards、misc、architecture（除 requirements-spec、task-group） |
| harness_rules_audit | docs/harness_rules_audit.md |
| deprecated 目录 | deprecated/ 下全部内容 |
| plan/ 认证设计 | plan/2025-03-05-auth-password-first-design*.md |

---

## 五、待确认项

无。

---

## 六、实施顺序建议

1. 迁移 .debug/tm → docs/design-docs/（扁平，tm- 前缀）
2. 迁移 .debug/docs 保留项（4.1、4.2）
3. 删除 .debug 中确认删除的全部内容
4. 更新 docs/design-docs/README.md 索引（一个文件看清结构）
5. 更新 AGENTS.md 知识库导航（若有变更）
6. 运行 harness-check / doc-gardening 校验

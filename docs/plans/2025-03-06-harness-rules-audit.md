# Harness Rules 审计基线

**生成时间**: 2025-03-06

## 规则全量清单（.cursor/rules/ 共 9 个）

### alwaysApply 规则（6 个）

| 规则文件 | 行数 | 用途 |
|----------|------|------|
| tm-doc-maintenance.mdc | 74 | 文档维护：结构变更联动、数据准确性、plan 归档、deprecated 不引用 |
| tm-core.mdc | 68 | 核心 MCP、工作流、预检、任务管理、选活/完成/group_completed |
| tm-extraction-retrieval.mdc | 156 | 经验提取触发、检索触发、分层存储、质量原则 |
| team_memory.mdc | 32 | 团队经验库索引，指向各场景化规则 |
| tm-commit-push-checklist.mdc | 30 | 提交与推送收口：Web 第一节、Git 第二节 |
| team_memory-codified-shortcuts.mdc | 53 | 固化结论快路径、预检引用、回退条件 |
| **小计** | **413** | |

### 场景化规则（3 个，按 globs 触发）

| 规则文件 | 行数 | 触发条件 | 用途 |
|----------|------|----------|------|
| tm-web.mdc | 28 | web/**、web/static/**、web/routes/** | Web 改动验收、端到端验证、Smoke Test |
| tm-plan.mdc | 28 | .cursor/plans/**、*plan*.md | Plan 执行、多角色审视、阶段验证、PageIndex-Lite |
| tm-quality.mdc | 27 | src/**、tests/**、.debug/** | Ruff、经验质量标准、文档同步 |
| **小计** | **83** | |

| **合计** | **496** | 9 个规则 |

## 用途与冗余分析

- **tm-core** 与 **team_memory-codified-shortcuts**：部分内容重叠（固化结论引用），可考虑 tm-core 引用 shortcuts 而非重复
- **tm-extraction-retrieval**（156 行）：最长，可评估是否拆分为「触发」与「检索」两个规则
- **tm-doc-maintenance**：与 harness 的 doc 结构有交集，Phase 1 完成后需同步更新
- **tm-web** 与 **tm-commit-push-checklist**：Web 收口细节在 tm-web，checklist 第一节引用 tm-web，职责清晰，无冗余
- **tm-quality** 与 **tm-core**：Ruff 在两者均有提及，tm-core 侧重任务收口，tm-quality 侧重触发条件与详细规则，可接受

## 精简建议

1. **tm-core ↔ team_memory-codified-shortcuts**：合并或引用，减少固化结论重复
2. **tm-extraction-retrieval**：评估拆分为「提取触发」与「检索触发」，降低单文件复杂度
3. **tm-doc-maintenance**：Phase 1 完成后同步 harness doc 结构
4. 场景化规则（tm-web、tm-plan、tm-quality）保持独立，触发条件明确，不建议合并

## 备注

- 本审计为 Phase 0 基线，Phase 1 将新增 harness-engineering.mdc
- 精简建议待 Phase 1 完成后评估，避免过早合并影响现有流程

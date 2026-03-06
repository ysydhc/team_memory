# Harness Rules 审计基线

**生成时间**: 2025-03-06

## alwaysApply 规则统计

| 规则文件 | 行数 | 说明 |
|----------|------|------|
| tm-doc-maintenance.mdc | 74 | 文档维护 |
| tm-core.mdc | 68 | 核心 MCP、工作流、任务管理 |
| tm-extraction-retrieval.mdc | 156 | 经验提取与检索 |
| team_memory.mdc | 32 | 团队经验库索引 |
| tm-commit-push-checklist.mdc | 30 | 提交与推送收口 |
| team_memory-codified-shortcuts.mdc | 53 | 固化结论快路径 |
| **合计** | **413** | 6 个 alwaysApply 规则 |

## 可合并/精简候选

- **tm-core** 与 **team_memory-codified-shortcuts**：部分内容重叠（固化结论引用），可考虑 tm-core 引用 shortcuts 而非重复
- **tm-extraction-retrieval**（156 行）：最长，可评估是否拆分为「触发」与「检索」两个规则
- **tm-doc-maintenance**：与 harness 的 doc 结构有交集，Phase 1 完成后需同步更新

## 备注

- 本审计为 Phase 0 基线，Phase 1 将新增 harness-engineering.mdc
- 精简建议待 Phase 1 完成后评估，避免过早合并影响现有流程

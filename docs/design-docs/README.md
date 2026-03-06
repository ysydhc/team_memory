# Design Docs

设计文档存放目录。用于架构设计、技术方案、决策记录等。

---

## 索引

| 文档 | 说明 |
|------|------|
| [architecture-layers](architecture-layers.md) | 分层定义、import 约束、harness-check |
| [feedback-loop](feedback-loop.md) | Agent 出错时沉淀、规则更新 |
| [harness-workflow-execution](harness-workflow-execution.md) | Plan 执行模式（Subagent-Driven）、执行记录、摸底、通知、自检 |
| [harness-vs-tm-boundary](harness-vs-tm-boundary.md) | 纯 Harness 与 tm 叠加的分离说明 |
| [human-decision-points](human-decision-points.md) | 需用户确认的节点 |
| [plan-self-review-checklist](plan-self-review-checklist.md) | Task 完成后自审 |
| [phase1-closure-checklist](phase1-closure-checklist.md) | 文档迁移收尾清单 |
| [subagent-workflow](subagent-workflow.md) | 两阶段评审、逐步引入 |
| [logging-format](logging-format.md) | JSON 行日志规范、生产/CI 切换（Phase 4） |
| [doc-gardening](doc-gardening.md) | 文档健康度扫描、断裂链接、deprecated 引用（Phase 4） |

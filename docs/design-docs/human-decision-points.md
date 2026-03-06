# 人类决策点约定

> Plan 执行中，以下节点需人类确认后再继续。纯 Harness：通过「主 Agent 暂停并汇报，等用户回复后再继续」实现。

---

## 一、决策点清单

| 节点 | 需确认内容 | 确认后动作 |
|------|------------|------------|
| **摸底完成后** | 若摸底产出需人工解读（如违规较多），是否继续 | 用户确认后进入 Task 1 |
| **Phase 完成** | Phase 0 或 Phase 1 全部 Task 完成，是否进入下一 Phase | 用户确认后进入下一 Phase |
| **迁移前** | 文档/路径迁移范围、目标路径，是否执行 | 用户确认后执行 mv/cp |
| **AGENTS.md 改造后** | 改造后的结构、链接是否接受 | 用户确认后提交 |
| **高影响变更** | 涉及多模块、架构、破坏性变更时 | 用户确认后再提交 |

---

## 二、实现方式

- **纯 Harness**：主 Agent 在节点处暂停，在回复中写明「请确认：xxx，确认后回复“继续”」；收到用户肯定回复后再执行下一步。
- **到达决策点时**：更新 execute、调用 `./scripts/notify_plan_status.sh`、回复中写明「Plan 执行记录已更新：xxx，需确认：xxx」。
- **不依赖**：Cursor 原生 approval、tm_task、任何 MCP。

---

## 三、引用

- [harness-engineering](../../.cursor/rules/harness-engineering.mdc)
- [harness-workflow-execution](./harness-workflow-execution.md)
- [plan-self-review-checklist](./plan-self-review-checklist.md)

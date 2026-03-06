# Plan 执行自审清单（step-self-review）

> 每个 Plan Task 完成后，主 Agent 须按本清单自审，再进入下一 Task。纯 Harness 流程，不依赖 tm_task。

---

## 一、自审项

| 项 | 检查内容 | 通过标准 |
|----|----------|----------|
| **step-0 已执行** | Task 1 开始前，摸底（或等效）是否已完成 | 已产出基线报告或已确认 Plan 已加载 |
| **execute 已更新** | 本 Task 完成后，execute 文档是否已追加 | 有对应日志条目 |
| **设计符合 Plan** | 实现是否与 Plan 中该 Task 的要求一致 | 无遗漏、无多余 |
| **代码整洁** | 命名、结构、无过度设计 | 符合项目既有风格 |
| **测试覆盖** | 若 Task 涉及代码，是否有对应测试 | 关键路径有测试 |
| **文档同步** | 若涉及结构变更，README/索引是否更新 | 无断裂引用 |

---

## 二、使用方式

- **Subagent-Driven Development**：子 Agent 完成 Task 后，主 Agent 按本清单自审，再派发 spec-reviewer（若启用）。
- **纯 Harness**：主 Agent 完成 Task 后，自审本清单，通过后再进入下一 Task。
- **与 step-verify 区分**：step-verify 为 tm 工作流中的验收步骤；本清单为 Plan 执行的通用自审，不依赖 tm_task。

---

## 三、引用

- [harness-engineering](.cursor/rules/harness-engineering.mdc)
- [harness-workflow-execution](harness-workflow-execution.md)
- [subagent-workflow](subagent-workflow.md)

# Harness 方法论跟进实施计划

> **前置**：Phase 0～5 已完成
> **来源**：方法论对照未完成项（系统通知、notify 记录、文档加载、断点恢复）
> **修订**：已根据 plan-multi-agent-review 报告完成 P1/P2 修复；已根据 plan-evaluator 报告完成细节优化
> **执行模式**：Subagent-Driven（每个 Task 派发 implementer subagent，主 Agent 验收）

**Goal:** 完成方法论层面 4 项跟进（系统通知显式化、notify 调用记录、Agent 自动找文档、断点恢复显式化），并将 CI/tm/能力限制等沉淀为后续待执行文档。

**Architecture:** 规则与设计文档调整为主，无代码改动。规则（harness-plan-execution）为细则来源，设计文档（harness-workflow-execution）为流程与格式约定。

---

## 一、任务拆分（含评审修复）

| Task | 内容 | 产出位置 | 产出 |
|------|------|----------|------|
| 1 | 系统通知调用时机显式化 | harness-plan-execution.mdc | 4 个时机（Plan 开始、人类决策点、中断、完成）、best-effort 约定 |
| 2 | execute 日志增加 notify 字段 | harness-workflow-execution.md 3.2/3.3 | 日志条目含 `- **通知**：已调用 notify_plan_status.sh（{时机}）`；3.3 追加规则 |
| 3 | Agent 自动找文档清单强制化 | harness-plan-execution.mdc + harness-workflow-execution.md 5.3 | 必须加载 4 文档（见下）、未加载不得进入 step-0、显式列出「已加载文档」 |
| 4 | 断点恢复步骤显式化 | harness-workflow-execution.md 3.5 | 触发条件、步骤、与 step-0/4 文档加载衔接、5.2 交叉引用 |
| 5 | 后续计划文档 | docs/exec-plans/harness-follow-up-backlog.md | CI/tm/能力限制等后续项，表格含编号、内容、现状、难度、依赖 |

**Task 3 必须加载的 4 文档**：harness-workflow-execution、subagent-workflow、feedback-loop、human-decision-points。

---

## 二、执行顺序与依赖

| 顺序 | 说明 |
|------|------|
| Task 1～4 | **可并行**（涉及不同文件：harness-plan-execution、harness-workflow-execution） |
| Task 5 | 在 Task 1～4 完成后执行（或可并行，无技术依赖） |
| 隐含依赖 | Task 4 的 3.5 断点恢复中「按需重新加载 4 文档」与 Task 3 的文档加载约定衔接；Task 4 与 step-0 的衔接在 3.5 中写明 |

---

## 三、人类决策点

本计划为规则与文档调整，**无强制人类决策点**。若执行中遇到歧义（如文档路径变更、notify 脚本不可用），主 Agent 可暂停并请示用户确认。

---

## 四、成功指标与验收标准

| 指标 | 目标 | 可操作验收 |
|------|------|------------|
| 通知时机 | harness-plan-execution 中 4 个时机明确 | 规则中列出 Plan 开始、人类决策点、中断、Plan 完成 4 个时机，且每个时机有调用示例（标题+正文格式） |
| notify 记录 | harness-workflow-execution 3.2/3.3 含 notify 字段约定 | 3.2 文档格式示例含 `- **通知**：...`；3.3 追加规则含「调用 notify 后须在同条日志记录」 |
| 文档加载 | 执行前必须加载清单明确，未加载则中断 | harness-plan-execution 与 5.3 中列出 4 文档、约定「未加载不得进入 step-0」「须显式列出已加载文档」 |
| 断点恢复 | harness-workflow-execution 有 3.5 断点恢复步骤 | 3.5 含触发条件、步骤（加载 execute→重载 4 文档→step-0 是否重跑→继续 Task）、注意；5.2 有「详见 3.5」交叉引用 |
| 后续清单 | harness-follow-up-backlog.md 存在且内容完整 | 文件存在；含 CI/tm/能力限制三类；表格有编号、内容、现状、难度、依赖 |

---

## 五、风险与缓解

| 风险 | 缓解 |
|------|------|
| notify 脚本不可用 | 6.4 best-effort 约定：失败仅记录到 execute，不中断 Plan |
| 文档路径变更 | Plan 中已写明产出位置；执行时若路径变更，主 Agent 可请示用户 |

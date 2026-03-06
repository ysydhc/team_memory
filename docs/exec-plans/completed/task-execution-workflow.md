# 任务执行工作流

> **流程定义（单源）**：[task-execution-workflow.yaml](task-execution-workflow.yaml)。Agent 执行时读取该 YAML 并按步骤 ID 顺序执行。

---

## 元信息

- **工作流名称**：task-execution-workflow（任务执行）
- **说明**：从「选活」到「完成并沉淀」、再到「组级复盘」的 TM 任务执行流程；支持执行前校验与冷启动、任务完成后由 subagent 做验收（人验证或模拟人验证）。

---

## YAML 设计思路

- **单源**：流程的步骤、动作、验收标准、分支均定义在 YAML 中；无独立引擎，由 Agent 解析后按 `steps` 顺序执行。
- **结构**：
  - `meta`：工作流 id、name、description。
  - `steps`：每步含 `id`、`name`、`action`、`acceptance_criteria`、`checkpoint`（是否打断点）、`optional`（是否可选）；**流转**用 `allowed_next`（下一步列表）或 `when`（条件分支，如 `group_completed → step-retro`、`else → step-claim`）。
- **约束**：某步若只允许一个下一步（如 step-complete 仅允许 step-verify），在 YAML 中写 `allowed_next: [step-verify]`，Agent 不得跳步。
- **条件分支**：带 `when` 的步骤按条件选下一跳；无 `when` 时用 `allowed_next`。执行时根据上下文（如 step-complete 返回的 `group_completed`）决定走 step-retro 或 step-claim。执行以 TM 任务状态与 tm_message 为准，不使用状态文件。

设计说明、运行机制、如何编辑与优缺点见 [task-execution-workflow-design.md](../task-execution-workflow-design.md)。

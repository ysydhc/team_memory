# 工作流优化 — 执行计划（单文件任务管理）

> 基于「工作流问题梳理 + 过期内容清理建议 + 三份借鉴方案」整理出的可执行步骤，**本文件为当前轮次的任务清单**，按阶段顺序执行，完成后可勾选。

---

## 背景与目标（摘要）

- **现状问题**：覆盖范围有限、step ID 与文档不一致、可观测性弱、无步骤门控、扩展性未在 YAML 中声明。
- **优化方向**：单源与一致性 → 步骤门控与可观测 → 审计格式与断点恢复 → YAML/任务扩展性。
- **三方案可落地点**：步骤门控（进入下一步前调 tm_workflow_next_step）、Web 按 step 聚合与组进度视图、步骤审计格式统一（可选 JSON 行）、YAML step 可选元数据（timeout_hint/retry/idempotent）、验收自检写进 tm_message。

---

## 阶段〇：前置确认（不勾选，仅决策）

- [ ] **决策**：step ID 是否在本轮统一为**英文语义 id**（如 `step-coldstart`、`step-claim`、`step-execute` 等，避免数字编号便于增删步骤）并统一格式（如 step-&lt;kebab-case&gt;）？若否，后续任务中「step ID 统一」相关项可跳过。
- [ ] **决策**：是否在本轮执行「步骤门控」（进入下一步前必须调用 tm_workflow_next_step）写入 Rules？若否，对应阶段可延后。

---

## 阶段一：文档清理与过期内容处理

**目标**：移除或收缩与「以 TM + tm_message 为准」矛盾的表述，统一步骤编号引用。

| 序号 | 任务 | 状态 | 说明 |
|------|------|------|------|
| 1.1 | 处理 `workflow-scheme.md` | ✅ | 已删除；执行以 TM 任务状态与 tm_message 为准，不依赖该文件。 |
| 1.2 | 处理 `workflow-writing-alternatives.md` | ✅ | 已移入 `.cursor/plans/workflows/expired/`，后续优化不再考虑该文件内容。 |
| 1.3 | 修订 `task-execution-workflow-design.md` | ✅ | 已泛化路径表述，删除过期信息，与「以 TM + tm_message 为准」一致。 |
| 1.4 | 清理 `workflow-improvement-checklist.md` | ✅ | 已删除状态文件相关条目，设计说明见 task-execution-workflow-design.md。 |
| 1.5 | 归档已完成任务详情文档 | ✅ | 两篇任务详情已移入 `workflows/archive/` 并加归档注。 |

---

## 阶段二：单源与一致性（step ID 与命名）

**前提**：阶段〇确认采用英文语义 step ID 与统一格式。

| 序号 | 任务 | 状态 | 说明 |
|------|------|------|------|
| 2.1 | 统一 `task-execution-workflow.yaml` 的 step id | ✅ | 已改为英文语义 id：step-coldstart、step-claim、step-execute、step-complete、step-verify、step-retro；allowed_next/when 已同步。 |
| 2.2 | 统一 `workflow-optimization-workflow.yaml` 的 step id | ✅ | 已改为 step-coldstart、step-claim、step-execute、step-complete、step-retro；拼写与空格已修正。 |
| 2.3 | 同步更新设计文档与 Rules 中的 step 引用 | ✅ | 已更新 task-execution-workflow-design.md、tm-core.mdc、execute-task-workflow-by-match.md、workflow-improvement-checklist、task-execution-workflow.md、workflow-guide、workflow-optimization-tasks、task-sediment-extract 等。 |
| 2.4 | 同步 workflow_oracle 与解析逻辑 | ✅ | _last_step_from_messages 已支持英文语义 step id（step-[a-z][a-z0-9-]*）及兼容旧 step-\d；server.py 中 tm_workflow_next_step 描述已更新。 |

---

## 阶段三：步骤门控与 Rules 强化

**目标**：在「无独立引擎」前提下，通过 Rules 与入口 Prompt 提高「必须执行 step-verify」「必须调 tm_save_group 后再选活」及「下一步由 MCP 返回」的遵守率。执行前需先完成 Rules 改造（预检/Plan 单源、删除状态文件、提交与推送收口独立成 tm-commit-push-checklist），再写入步骤门控。

| 序号 | 任务 | 状态 | 说明 |
|------|------|------|------|
| 3.0 | Rules 改造前置（预检/Plan 单源 + 提交推送收口独立） | ✅ | 已做：预检/Plan 以 tm-core/tm-plan 为准、codified 仅引用；已删 tm-core 状态文件条；已新增 tm-commit-push-checklist.mdc，已更新 team_memory、tm-web、codified。 |
| 3.1 | 在 tm-core 中写入「步骤门控」约定 | ✅ | 已做：按任务执行工作流时增加「步骤门控（必须）」：进入下一步前必须调用 tm_workflow_next_step，仅当返回的 next step 与当前意图一致时才执行该步。 |
| 3.2 | 在 execute-task-workflow-by-match.md 中强化门控 | ✅ | 已做：section 3 增加「步骤门控（必须）」：执行前或每步完成后先调用 tm_workflow_next_step，再执行其返回的 step。 |
| 3.3 | 关键 step 完成前自检与审计约定 | ✅ | 已做：tm-core 步骤审计条中增加「发送审计前必须自检该步 acceptance_criteria；可选在摘要中带 acceptance: 已满足/未满足」。 |

---

## 阶段四：可观测与 Web 展示（可选，按需排期）

**目标**：步骤审计在 Web 上按 step 聚合、组进度可一眼看到，不新增状态源，仅从 tm_message 解析或轻量 API 派生。

| 序号 | 任务 | 状态 | 说明 |
|------|------|------|------|
| 4.1 | Web 任务详情按 step 聚合展示 | ✅ | 已做：后端 parse_workflow_steps_from_messages + GET tasks/:id/workflow-steps；前端任务详情「工作流进度」区块，step 中文映射、空/加载状态，与 workflow_oracle 解析一致。 |
| 4.2 | 任务组「组进度」视图 | ✅ | 已做：GET task-groups/:id/workflow-progress 组内一次汇总；任务组卡片展示 x/y + 各任务当前 step 汇总与待复盘角标，窄屏适配。 |
| 4.3 | group_completed 时强提示组复盘 | ✅ | 已做：list/get 任务组返回 has_sediment；group_completed 且未复盘时展示「待执行组复盘」提示块与「组复盘」按钮，已复盘组弱化提示。 |

---

## 阶段五：审计格式与断点恢复（可选）

**目标**：统一步骤审计事件格式，便于解析与断点恢复；不新增存储表。

| 序号 | 任务 | 状态 | 说明 |
|------|------|------|------|
| 5.1 | 约定步骤审计的固定格式（含边界约定） | ✅ | 已在 tm-core「步骤审计（格式必须）」条下增补：单行前缀、每条 message 仅一条审计、摘要边界、解析仅首行；execute-task-workflow-by-match 已引用该格式。 |
| 5.2 | workflow_oracle 仅首行解析（防污染） | ✅ | 已做：parse_workflow_steps_from_messages 仅对 content 首行做正则匹配；单测增加空/仅换行/首行非审计等边界用例。可选 JSON 未引入。 |

---

## 阶段六：Step 独立成文件与 $ref 引用（可选）

**目标**：每个 step 可独立成文件，主 workflow 通过 $ref 引用组合；后端解析 $ref 后返回完整 step，为后续原子动作库预留能力。

| 序号 | 任务 | 状态 | 说明 |
|------|------|------|------|
| 6.1 | Step 独立成文件 + 主 workflow $ref 引用 | ✅ | 已做：steps/task-execution/*.yaml 已建，主 workflow 用 $ref 引用；action 仍为多行文本。 |
| 6.2 | workflow_oracle 实现 $ref 解析 | ✅ | 已做：_load_workflow 中 _resolve_step_ref 解析 steps 的 $ref，按相对路径加载并展开；返回给 MCP 的 step 含完整 action。 |
| 6.3 | （可选，后续排期）原子动作库 | ✅ | 已做：actions/ 目录与 run_ruff/run_tests.yaml；_resolve_action_ref、_merge_actions_into_step；step 内 actions 可 $ref。 |
| 6.4 | （原 6.1）YAML step 可选字段 | ✅ | 已做：step-execute/step-verify 增加 timeout_hint、retry_hint、idempotent。 |
| 6.5 | （原 6.2）返回 step 时带可选元数据 | ✅ | 已做：get_next_step 返回 timeout_hint、retry_hint、idempotent。 |
| 6.6 | （原 6.3）任务级验收字段 | ✅ | 已做：PersonalTask 增加 acceptance_criteria、acceptance_met；tm_task、Web API、任务 slideout 展示与编辑。 |

### 阶段六未完成任务规划（6.3～6.6）

**6.3 原子动作库**（后续排期，依赖 6.4 元数据稳定）

- 新建 `actions/` 目录，定义可复用的原子动作（如 `run_tests`、`commit`、`run_ruff`）
- step 内 `actions` 支持 `$ref: actions/xxx.yaml` 或 inline 文本
- `workflow_oracle._resolve_step_ref` 扩展为递归解析 actions 的 $ref

**6.4 YAML step 可选字段**

- 在 step schema 中增加可选字段：`timeout_hint`、`retry_hint`、`idempotent` 等
- 更新 `steps/task-execution/*.yaml` 中需声明的 step

**6.5 返回 step 时带可选元数据**

- `tm_workflow_next_step` 的返回中增加上述元数据字段，供 Agent 或 Web 展示

**6.6 任务级验收字段**（可单独排期）

- 任务模型增加验收相关字段；与 step 的 acceptance_criteria 关联

---

## 后续工作（OpenMemory vs TeamMemory 调研结论）

> 2025-03-02 调研：OpenMemory 记忆管理机制与 tm 对比，可借鉴的优化点。

### 差异摘要

| 维度 | OpenMemory (Mem0) | TeamMemory (tm) |
|------|-------------------|-----------------|
| 写入 | 对话 → LLM 抽取 → 自动冲突解决 | 结构化 API / tm_learn 抽取，保存前去重需人工确认 |
| 去重 | 写入时自动合并，以最新为准 | 保存前检测相似，用户确认或 skip_dedup |
|  metadata 过滤 | eq/gt/in/contains/AND/OR 等 | tags/project/visibility |
| 协作 | 无 | RBAC、审批、合并、反馈 |

### tm 可借鉴的优化点（按优先级）

| 优先级 | 优化点 | 说明 |
|--------|--------|------|
| P0 | Metadata 过滤扩展 | tm_search/API 支持 filter DSL（experience_type、quality_score 等） |
| P0 | 可选自动合并 | 配置 `auto_merge_on_save`，阈值内自动 update 而非仅 duplicate_detected |
| P1 | Query 优化（LLM 前置） | SearchPipeline 前可选 LLM 改写短/模糊 query |
| P1 | Dashboard 快速入口 | 记忆总览：最近添加、高频命中、待处理去重 |
| P2 | tm_add_raw | 支持原样存储对话片段（类似 infer=False） |
| P2 | 批量归档/删除 | 按 project/tag 批量归档或软删除 |

### 优化点 TODO

| 优先级 | 任务 | 状态 | 说明 |
|--------|------|------|------|
| P0 | Metadata 过滤扩展 | ⬜ | tm_search/API 支持 filter DSL（experience_type、quality_score 等） |
| P0 | 可选自动合并 | ⬜ | 配置 `auto_merge_on_save`，阈值内自动 update 而非仅 duplicate_detected |
| P1 | Query 优化（LLM 前置） | ⬜ | SearchPipeline 前可选 LLM 改写短/模糊 query |
| P1 | Dashboard 快速入口 | ⬜ | 记忆总览：最近添加、高频命中、待处理去重 |
| P2 | tm_add_raw | ⬜ | 支持原样存储对话片段（类似 infer=False） |
| P2 | 批量归档/删除 | ⬜ | 按 project/tag 批量归档或软删除 |

### 参考文档

- 完整调研与设计建议见对话记录；可沉淀为 `docs/plans/2025-03-02-openmemory-vs-tm-design.md` 供后续排期。

---

## 执行顺序与依赖

```
阶段〇（决策）
    ↓
阶段一（文档清理）  — 无依赖，建议最先
    ↓
阶段二（step ID 统一） — 依赖阶段〇确认
    ↓
阶段三（步骤门控）  — 可与阶段二并行，依赖阶段一完成更清晰
    ↓
阶段四（Web 可观测） — 可与阶段三并行
    ↓
阶段五（审计格式）  — 可与阶段四并行，为阶段四解析提供统一格式
    ↓
阶段六（Step $ref 与 YAML 扩展） — 可最后或与阶段五并行
```

---

## 状态图例

- ⬜ 未开始
- 🔄 进行中
- ✅ 已完成
- ⏸ 暂缓（依赖未满足或决策不执行）

---

## 变更记录

| 日期 | 变更 |
|------|------|
| 2025-03-02 | 初版：基于工作流问题梳理与三份借鉴方案整理阶段一～六及单文件任务清单。 |
| 2025-03-02 | 阶段一 1.2～1.5 已执行：1.2 workflow-writing-alternatives 示例更新并移入 workflows/expired；1.3 task-execution-workflow-design 泛化路径、删除过期内容；1.4 workflow-improvement-checklist 删除第 3 条；1.5 两篇任务详情移入 workflows/archive 并加归档注。 |
| 2025-03-02 | 阶段四 4.1～4.3 已完成：按「阶段四可观测与web展示_1d9ac00f.plan.md」实现工作流进度、组进度、组复盘强提示，状态已标为 ✅。 |
| 2025-03-02 | 阶段五 5.1～5.2 已完成：按「阶段五审计格式与断点恢复」plan 实现格式约定（tm-core + execute-task-workflow-by-match）、workflow_oracle 仅首行解析与单测边界、断点恢复文档化；状态已标为 ✅。 |
| 2025-03-02 | 阶段六更新：替换为「Step 独立成文件与 $ref 引用」方案；6.3 原子动作库后续排期，6.4～6.6 保留原 YAML 扩展与任务验收字段。 |
| 2025-03-02 | 阶段六 6.1～6.2 已完成：按 Workflow Step File Ref plan 实现 step 拆分、$ref 引用、workflow_oracle._resolve_step_ref 解析；状态已标为 ✅。 |
| 2025-03-02 | 补充后续工作：阶段六未完成任务规划（6.3～6.6）；OpenMemory vs tm 调研结论与可借鉴优化点。 |
| 2025-03-02 | 阶段六 6.3～6.5 已完成：按 phase_6_remaining_tasks plan 实现 YAML 可选字段、get_next_step 元数据、原子动作库。6.6 任务级验收字段可单独排期。 |
| 2025-03-02 | 阶段六 6.6 已完成：按 task_6.6_acceptance_fields plan 实现 PersonalTask 验收字段、MCP/API 支持、任务 slideout 验收标准展示与编辑。 |
| 2025-03-02 | 将 OpenMemory 可借鉴优化点拆成 TODO 表格，置于「后续工作」下。 |

# 工作流优化 — 实施计划

> 基于「工作流问题梳理 + 过期内容清理建议 + 三份借鉴方案」整理的可执行步骤。  
> 详见 [1-research/brief](../1-research/brief.md)、[1-research/options](../1-research/options.md)。

## 阶段〇：前置确认（决策）

- **决策**：step ID 是否在本轮统一为**英文语义 id**（如 step-coldstart、step-claim、step-execute 等）并统一格式（step-kebab-case）？
- **决策**：是否在本轮执行「步骤门控」（进入下一步前必须调用 tm_workflow_next_step）写入 Rules？

## 阶段一：文档清理与过期内容处理

**目标**：移除或收缩与「以 TM + tm_message 为准」矛盾的表述，统一步骤编号引用。

| 序号 | 任务 | 说明 |
|------|------|------|
| 1.1 | 处理 workflow-scheme.md | 已删除；执行以 TM 任务状态与 tm_message 为准 |
| 1.2 | 处理 workflow-writing-alternatives.md | 已归档至 archive/ |
| 1.3 | 修订 task-execution-workflow-design.md | 泛化路径表述，删除过期信息 |
| 1.4 | 清理 workflow-improvement-checklist.md | 删除状态文件相关条目 |
| 1.5 | 归档已完成任务详情文档 | 移入 workflows/archive/ |

## 阶段二：单源与一致性（step ID 与命名）

**前提**：阶段〇确认采用英文语义 step ID。

| 序号 | 任务 | 说明 |
|------|------|------|
| 2.1 | 统一 task-execution-workflow.yaml 的 step id | step-coldstart、step-claim、step-execute、step-complete、step-verify、step-retro |
| 2.2 | 统一 workflow-optimization-workflow.yaml 的 step id | 同上 |
| 2.3 | 同步更新设计文档与 Rules 中的 step 引用 | task-execution-workflow-design、tm-core、execute-task-workflow-by-match 等 |
| 2.4 | 同步 workflow_oracle 与解析逻辑 | _last_step_from_messages 支持英文语义 step id |

## 阶段三：步骤门控与 Rules 强化

| 序号 | 任务 | 说明 |
|------|------|------|
| 3.0 | Rules 改造前置 | 预检/Plan 单源、提交推送收口独立成 tm-commit-push-checklist |
| 3.1 | 在 tm-core 中写入「步骤门控」约定 | 进入下一步前必须调用 tm_workflow_next_step |
| 3.2 | 在 execute-task-workflow-by-match 中强化门控 | section 3 增加步骤门控 |
| 3.3 | 关键 step 完成前自检与审计约定 | acceptance_criteria 自检 |

## 阶段四：可观测与 Web 展示

| 序号 | 任务 | 说明 |
|------|------|------|
| 4.1 | Web 任务详情按 step 聚合展示 | parse_workflow_steps_from_messages + GET tasks/:id/workflow-steps |
| 4.2 | 任务组「组进度」视图 | GET task-groups/:id/workflow-progress |
| 4.3 | group_completed 时强提示组复盘 | has_sediment、待执行组复盘提示块 |

## 阶段五：审计格式与断点恢复

| 序号 | 任务 | 说明 |
|------|------|------|
| 5.1 | 约定步骤审计的固定格式 | 单行前缀、每条 message 仅一条审计、解析仅首行 |
| 5.2 | workflow_oracle 仅首行解析 | 防污染，单测边界用例 |

## 阶段六：Step 独立成文件与 $ref 引用

| 序号 | 任务 | 说明 |
|------|------|------|
| 6.1 | Step 独立成文件 + 主 workflow $ref 引用 | steps/task-execution/*.yaml |
| 6.2 | workflow_oracle 实现 $ref 解析 | _resolve_step_ref |
| 6.3 | 原子动作库 | actions/ 目录、_resolve_action_ref |
| 6.4 | YAML step 可选字段 | timeout_hint、retry_hint、idempotent |
| 6.5 | 返回 step 时带可选元数据 | get_next_step 返回元数据 |
| 6.6 | 任务级验收字段 | acceptance_criteria、acceptance_met |

## 执行顺序与依赖

```
阶段〇（决策）→ 阶段一（文档清理）→ 阶段二（step ID）→ 阶段三（门控）
→ 阶段四（Web 可观测）→ 阶段五（审计格式）→ 阶段六（$ref 与扩展）
```

## 后续工作（OpenMemory vs TM 调研）

- P0：Metadata 过滤扩展、可选自动合并
- P1：Query 优化（LLM 前置）、Dashboard 快速入口
- P2：tm_add_raw、批量归档/删除

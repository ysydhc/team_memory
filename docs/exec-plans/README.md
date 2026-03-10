# Exec Plans

执行计划存放目录。用于项目执行计划、阶段规划等。

## 查找指南

Plan 生命周期（wait / executing / completed）详见 [doc-maintenance-guide](../design-docs/harness/doc-maintenance-guide.md) 第六章。

**示例**：harness-phase 文档在 `completed/harness-phase/` 下。

> **说明**：`workflow_oracle` 从 `.tm_cursor/plans/workflows/*.yaml` 读取工作流定义，YAML 未迁移。

---

## wait/

### debug-migration/

.debug 文档迁移。

| 文档 | 说明 |
|------|------|
| [debug-migration-plan202503.md](wait/debug-migration/debug-migration-plan202503.md) | 迁移计划 |

---

## executing/

执行中计划的 execute 文档存放目录。一个 Plan 对应一个 `{plan-id}-execute.md`，格式见 [harness-spec](../design-docs/harness/harness-spec.md)。记录 Plan 执行日志，中断后重启仍写入同一文件；供 AI 加载、人类回溯。

（当前无执行中计划）

---

## completed/

### arch-node-search/

架构节点搜索功能（图 Tab 搜索、子串匹配、结果列表、图中高亮、范围切换）。

| 文档 | 说明 |
|------|------|
| [2-plan/plan.md](completed/arch-node-search/2-plan/plan.md) | 实施计划（来源 .cursor/plans） |
| [2-plan/execute.md](completed/arch-node-search/2-plan/execute.md) | 执行记录 |
| [3-retro/retro.md](completed/arch-node-search/3-retro/retro.md) | 项目复盘 |

### code-arch-viz-gitnexus/

GitNexus 配合代码架构可视化（1-research/2-plan/3-retro 规范结构）。

| 文档 | 说明 |
|------|------|
| [1-research/brief.md](completed/code-arch-viz-gitnexus/1-research/brief.md) | 调研任务书、问题定义、三视角分工 |
| [1-research/assessment.md](completed/code-arch-viz-gitnexus/1-research/assessment.md) | 评估（范围、风险、技术选型、数据映射） |
| [1-research/options.md](completed/code-arch-viz-gitnexus/1-research/options.md) | 方案对比与选型（均衡方案、GitNexus 接入） |
| [2-plan/plan.md](completed/code-arch-viz-gitnexus/2-plan/plan.md) | 实施计划（任务列表、验收标准） |
| [2-plan/execute.md](completed/code-arch-viz-gitnexus/2-plan/execute.md) | 执行记录 |
| [3-retro/retro.md](completed/code-arch-viz-gitnexus/3-retro/retro.md) | 项目复盘 |

### harness-phase/

Phase 0-1 至 Phase 5 实施计划及复盘（1-research/2-plan/3-retro 规范结构）。

| 文档 | 说明 |
|------|------|
| [1-research/brief.md](completed/harness-phase/1-research/brief.md) | 调研任务书、问题定义、Phase 演进脉络 |
| [1-research/assessment.md](completed/harness-phase/1-research/assessment.md) | 评估（Brownfield、可行性、风险、Blockers/High Risks 明细） |
| [2-plan/plan.md](completed/harness-phase/2-plan/plan.md) | 计划本体（Phase 3/4/5 任务拆分与执行顺序） |
| [2-plan/execute.md](completed/harness-phase/2-plan/execute.md) | 执行记录（Phase 4/5 合并） |
| [3-retro/retro.md](completed/harness-phase/3-retro/retro.md) | 复盘（时间线、问题追溯、改进建议、可复用实践） |
| [harness-follow-up-backlog.md](completed/harness-phase/harness-follow-up-backlog.md) | 后续待执行清单 |

### harness-methodology-follow-up/

方法论改进（通知、断点恢复、文档加载强制化等）。

| 文档 | 说明 |
|------|------|
| [plan.md](completed/harness-methodology-follow-up/plan.md) | 计划入口 |
| [harness-methodology-follow-up-implementation-plan.md](completed/harness-methodology-follow-up/harness-methodology-follow-up-implementation-plan.md) | 方法论跟进实施计划 |
| [harness-methodology-follow-up-execute.md](completed/harness-methodology-follow-up/harness-methodology-follow-up-execute.md) | 执行记录 |
| [harness-methodology-follow-up-observer-log.md](completed/harness-methodology-follow-up/harness-methodology-follow-up-observer-log.md) | Observer 日志 |

---

### workflow-visualization-v1/

工作流可视化 UI（1-research/2-plan/3-retro 规范结构）。

| 文档 | 说明 |
|------|------|
| [1-research/brief.md](completed/workflow-visualization-v1/1-research/brief.md) | 调研任务书、目标与范围、解析规范 |
| [1-research/assessment.md](completed/workflow-visualization-v1/1-research/assessment.md) | 评估（Blockers、High Risks、行动路线图、多角色评审） |
| [2-plan/plan.md](completed/workflow-visualization-v1/2-plan/plan.md) | 实施计划（Task 1～5、验收标准、风险缓解） |
| [2-plan/execute.md](completed/workflow-visualization-v1/2-plan/execute.md) | 执行记录（产出物、观察日志） |
| [3-retro/retro.md](completed/workflow-visualization-v1/3-retro/retro.md) | 复盘（Flow Observer 报告、协作质量评分、可改进点） |

### workflow-optimization/

工作流流程与任务管理优化（1-research/2-plan 规范结构）。

| 文档 | 说明 |
|------|------|
| [1-research/brief.md](completed/workflow-optimization/1-research/brief.md) | 调研任务书、背景与目标、任务组说明 |
| [1-research/options.md](completed/workflow-optimization/1-research/options.md) | 三方案选型（TM 唯一任务源、任务组交付单元） |
| [2-plan/plan.md](completed/workflow-optimization/2-plan/plan.md) | 实施计划（阶段〇～六任务定义、依赖、后续工作） |
| [2-plan/execute.md](completed/workflow-optimization/2-plan/execute.md) | 执行记录（各阶段完成状态、变更记录） |
| [workflow-optimization-tasks.md](completed/workflow-optimization/workflow-optimization-tasks.md) | 任务组清单 |
| [workflow-optimization-task1experience-comparison.md](completed/workflow-optimization/workflow-optimization-task1experience-comparison.md) | 冷启动任务经验对比 |
| [workflow-guide.md](completed/workflow-optimization/workflow-guide.md) | 工作流搭建指导 |
| [workflow-improvement-checklist.md](completed/workflow-optimization/workflow-improvement-checklist.md) | 改进建议清单 |
| [task-execution-workflow.md](completed/workflow-optimization/task-execution-workflow.md) | 任务执行工作流 |
| [task-execution-workflow-design.md](completed/workflow-optimization/task-execution-workflow-design.md) | 任务执行工作流设计 |

### dedup-group-misjudge/

| 文档 | 说明 |
|------|------|
| [dedup-group-misjudge-optimization-execution-plan.md](completed/dedup-group-misjudge/dedup-group-misjudge-optimization-execution-plan.md) | 去重组误判优化执行计划 |

### verification/

| 文档 | 说明 |
|------|------|
| [verification-plan-completion.md](completed/verification/verification-plan-completion.md) | 计划完成度验证方案 |
| [verification-report-completion.md](completed/verification/verification-report-completion.md) | 计划完成度验证报告 |

### mcp-io-debug-log/

| 文档 | 说明 |
|------|------|
| [mcp-io-debug-log-plan.md](completed/mcp-io-debug-log/mcp-io-debug-log-plan.md) | MCP IO 调试日志计划 |

### archive/

历史归档。

| 文档 | 说明 |
|------|------|
| [task-details-p43p25p37.md](completed/archive/task-details/task-details-p43p25p37.md) | P4-3 CI/CD、P2-5 工作流模板、P3-7 详细日志 |
| [task-details-p43p25p19.md](completed/archive/task-details/task-details-p43p25p19.md) | P4-3 CI/CD、P2-5 工作流模板、P1-9 提取配置 |
| [task-details-p16p15p23.md](completed/archive/task-details/task-details-p16p15p23.md) | P1-6 FTS、P1-5 检索解析、P2-3 请求日志 |
| [task-details-p110p19p22.md](completed/archive/task-details/task-details-p110p19p22.md) | P1-10 成功标准、P1-9 提取配置、P2-2 /metrics |
| [commit-task-mapping.md](completed/archive/commit-task-mapping.md) | 9 个 commit 与 9 个任务对应关系 |
| [task-p0group-completed-detail.md](completed/archive/task-p0group-completed-detail.md) | 子任务详情（已归档） |
| [workflow-optimization-task1cold-start-single-entry.md](completed/archive/workflow-optimization-task1cold-start-single-entry.md) | 冷启动只在一个入口 — 已归档 |
| [workflow-writing-alternatives.md](completed/archive/workflow-writing-alternatives.md) | 工作流写法可借鉴方案 — 已过期 |

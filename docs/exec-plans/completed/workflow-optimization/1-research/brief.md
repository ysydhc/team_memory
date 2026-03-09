# 工作流优化 — 调研任务书

## 背景与目标

- **现状问题**：覆盖范围有限、step ID 与文档不一致、可观测性弱、无步骤门控、扩展性未在 YAML 中声明。
- **优化方向**：单源与一致性 → 步骤门控与可观测 → 审计格式与断点恢复 → YAML/任务扩展性。
- **三方案可落地点**：步骤门控（进入下一步前调 tm_workflow_next_step）、Web 按 step 聚合与组进度视图、步骤审计格式统一（可选 JSON 行）、YAML step 可选元数据（timeout_hint/retry/idempotent）、验收自检写进 tm_message。

## 任务组说明

- **任务组名称**：workflow-optimization
- **推荐流程定义**：workflow-optimization-workflow.yaml（无代码收口：无 ruff、pytest、subagent 验收，仅 选活 → 执行 → 完成沉淀 → 组级复盘）
- **与 task-execution-workflow 的区别**：本流程用于「对工作流的优化」类任务，不跑代码收口与验收子 agent。

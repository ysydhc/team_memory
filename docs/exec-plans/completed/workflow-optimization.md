# 工作流优化任务组 — 使用说明

- **任务组名称**：workflow-optimization（已通过 tm_doc_sync 从 [workflow-optimization-tasks.md](workflow-optimization-tasks.md) 同步到 TM）
- **推荐流程定义**：[workflow-optimization-workflow.yaml](workflow-optimization-workflow.yaml)（**无代码收口**：无 ruff、pytest、subagent 验收，仅 选活 → 执行 → 完成沉淀 → 组级复盘）
- **第一项可执行建议**：[一] 冷启动只在一个入口（仅 step-0）。认领后按上述 YAML 的 step 执行即可。

与 [task-execution-workflow.yaml](task-execution-workflow.yaml) 的区别：本流程用于「对工作流的优化」类任务，不跑代码收口与验收子 agent。

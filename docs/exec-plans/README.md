# Exec Plans

执行计划存放目录。用于项目执行计划、阶段规划等。

## 目录结构

- `completed/`：已完成或进行中的执行计划（从 `.cursor/plans/` 及 `workflows/` 迁移）
- `archive/`：已归档或过期的计划（从 `workflows/archive/`、`workflows/expired/` 迁移）

> **说明**：`workflow_oracle` 仍从 `.cursor/plans/workflows/*.yaml` 读取工作流定义，YAML 未迁移。

---
description: Harness 补充规则（CLAUDE.md 未覆盖部分）
---

# harness-engineering

> 主规范在 `CLAUDE.md`，本文件仅补充其未覆盖的可执行规则。

## 任务开始前
1. 读 `CLAUDE.md`（结构、命令、规范入口）
2. 读 `README.md`、`AGENTS.md`；按需读 MCP（`docs/guide/mcp-server.md`、`src/team_memory/server.py`）、分层（`scripts/harness_import_check.py`、`docs/README.md`）
3. 确认「完成标准」再编码

## 反馈回路
1. 出错 → 更新 rules 或 docs，记根因
2. 未追踪改动 → 关键内容写入规则/文档
3. 可 `memory_save` 沉淀经验

## .harness/ 导航

| 文件 | 用途 |
|------|------|
| `.harness/harness-config.yaml` | 项目配置（质量门禁、超时、安全） |
| `.harness/orchestration/task-flow.md` | Phase 0-4 编排流程 |
| `.harness/orchestration/context-management.md` | Context rot 管理 |
| `.harness/orchestration/contracts/` | Plan / Task 契约模板 |
| `.harness/failure/failure-taxonomy.md` | 三级恢复 + 行为异常 + 安全分级 |
| `.harness/plans/progress.md` | 活跃计划状态 |
| `.harness/docs/harness-spec.md` | 框架规范（项目无关） |

## 不确定时
停下来问，不猜测。

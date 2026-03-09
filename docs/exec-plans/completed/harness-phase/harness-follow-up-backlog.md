# Harness 后续待执行清单

> 本清单为方法论对照后未完成项，待后续排期。Harness 方法论跟进计划（系统通知、notify 记录、文档加载、断点恢复）不包含本清单。
>
> **最后更新**：2025-03-07

---

## 一、CI / 外部系统

| 编号 | 内容 | 现状 | 难度 | 依赖 |
|------|------|------|------|------|
| 5 | CI 中 harness-check | 需确认 CI 配置是否在 PR 时自动跑 | 低 | GitHub Actions 等 |
| 6 | CI 中 doc-gardening | Phase 4 约定：CI 有 doc-gardening 独立 job | 低 | CI 配置 |
| 7 | CI 等价命令文档化 | 若 CI 无 Makefile，需文档化等价命令 | 低 | docs/README |
| 8 | gh PR 集成（Phase 2 可选） | 需 gh 已配置；无 gh 时保留手动 PR | 中 | gh CLI、GitHub |

---

## 二、tm / team_memory 可选

| 编号 | 内容 | 现状 | 难度 | 依赖 |
|------|------|------|------|------|
| 9 | tool_usage 基线 | Phase 0-1 报告：当前为 placeholder，需服务可用时补跑 | 中 | team_memory 服务 |
| 10 | harness_ref_scan 收口 | Phase 0-1：输出为空，需确认是否已完整执行 | 低 | 脚本执行 |
| 11 | Phase 6 经验库策略 | 预留：stale 经验判定、归档策略、清理频率 | 高 | 单独排期 |

---

## 三、Cursor 能力限制（无法实现）

| 编号 | 内容 | 说明 |
|------|------|------|
| 12 | 原生审批流 | Cursor 无内置 approval，只能靠对话确认 |
| 13 | 多 session 协同 | Cursor 无 session 编排 |
| 14 | tm_task 门控 | 依赖 team_memory 服务与 MCP |

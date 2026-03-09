# Harness Phase 系列调研任务书

> **范围**：Phase 0-1 至 Phase 5；**问题定义**：为 team_doc 建立 Harness 方法论与架构约束，实现可观测性与文档健康度治理。

---

## 一、问题定义

| 项目 | 内容 |
|------|------|
| **背景** | team_doc 需建立「人类掌舵、Agent 执行、尽量不手写」的 Harness 方法论，并约束架构分层、可观测性、文档维护 |
| **目标** | Phase 0-1 基线 → Phase 2 规则迁移 → Phase 3 架构约束 → Phase 4 可观测性 + doc-gardening → Phase 5 收尾与规则分离 |
| **边界** | 纯 Harness 优先，tm 叠加标注清晰；Brownfield 对齐，不破坏现有功能 |

---

## 二、Phase 演进脉络

| Phase | 主题 | 产出 |
|-------|------|------|
| 0-1 | 基线评估、规则迁移 | Brownfield、Rules 审计、harness-engineering、scripts |
| 2 | 先规划再执行、自审清单、功能验证、人类决策点 | （前置，Phase 3 依赖） |
| 3 | 架构约束 | 分层定义、import 检查、CI 集成、`make harness-check` |
| 4 | 可观测性 + 文档维护 | 日志 JSON、doc-gardening、Golden Set、白名单 |
| 5 | 收尾与规则分离 | Phase 3 收尾、Harness/tm 规则分离、step-0 模板、feedback-loop 回溯、Phase 6 预留 |

---

## 三、范围说明

- **调研来源**：Phase 0 Brownfield 评估、Rules 审计、Phase 0-1 监控报告、Phase 2 实施计划、Phase 3 协作回溯、Phase 4 全维度评审、Phase 4/5 Flow Observer 报告
- **执行模式**：Subagent-Driven（每 Task 派发 implementer，主 Agent 验收）
- **衔接**：Phase 3 预留 Phase 4；Phase 4 预留 Phase 6（原 Phase 5）；Phase 5 完成收尾与编号重排

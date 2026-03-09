# Harness Phase 系列实施计划

> **合并来源**：Phase 2/3/4/5 实施计划。执行模式：Subagent-Driven。

---

## 一、Phase 2：工作流增强

**Goal:** 在纯 Harness 工作流基础上，增强「先规划再执行」、自审步骤、功能验证，为 Phase 3 架构约束做准备。

| Task | 产出 |
|------|------|
| 1 | harness-engineering 中固化「先规划再执行」 |
| 2 | 创建 plan-self-review-checklist.md（Plan 执行自审清单） |
| 3 | tm-commit-push-checklist 中强化功能验证（端到端/行为验证） |
| 4 | 创建 human-decision-points.md（人类决策点约定） |
| 5 | 更新 AGENTS.md 与 docs 索引 |
| 6 | 全量验证（ruff、pytest、harness_ref_verify、AGENTS 行数 ≤ 120） |

**执行顺序**：Task 1 → Task 2 → Task 3 → Task 4 → Task 5 → Task 6

**与 Phase 3 衔接**：Phase 3 将包含分层定义、import 方向检查、CI 结构测试。

---

## 二、Phase 3：架构约束

**Goal:** 为 team_doc 定义分层与依赖方向，用脚本与 CI 强制约束。

| Task | 产出 |
|------|------|
| 1 | 编写分层定义文档 `docs/design-docs/architecture-layers.md` |
| 2 | 创建 import 方向检查脚本 `scripts/harness_import_check.py` |
| 3 | 接入 Makefile 与 CI |
| 4 | 更新 harness-engineering 与 AGENTS.md |
| 5 | 全量验证与反向依赖修复 |

**执行顺序**：Task 1 → Task 2（雏形）→ 摸底运行 → 修复反向依赖 → Task 2（完善）→ Task 3 → Task 4 → Task 5

---

## 三、Phase 4：可观测性 + 文档维护

**Goal:** 日志 JSON 结构化、doc-gardening 过时文档扫描。

| Task | 产出 |
|------|------|
| 1 | 设计日志 JSON 格式规范 `docs/design-docs/logging-format.md` |
| 2 | 实现日志 JSON 输出配置（bootstrap、config.LOG_FORMAT） |
| 3 | 创建 doc-gardening 设计文档 |
| 4 | 实现 doc-gardening 脚本 + Golden Set |
| 5 | 接入 Makefile 与 CI |
| 6 | 更新 harness-engineering 与文档索引 |

**step-0 强制**：统计 logger、确认 docs 结构、基线 ruff/pytest，产出基线报告后方可进入 Task 1。

---

## 四、Phase 5：收尾与规则分离

**Goal:** Phase 3 收尾验证、Harness 与 tm 规则分离、step-0 模板、feedback-loop 回溯、Phase 5→6 编号重排。

| Task | 产出 |
|------|------|
| 1 | Phase 3 Task 5 收尾验证（architecture-layers 4.3 已修复） |
| 2 | Harness 与 tm 规则分离（按 boundary 文档） |
| 3 | harness-workflow-execution 补充 Phase 4 step-0 模板 |
| 4 | feedback-loop 4.1、4.2 移入已完成区 |
| 5 | Phase 5→6 编号重排（全项目「Phase 5 预留」→「Phase 6 预留」） |
| 6 | 文档索引与 execute 记录 |

---

## 五、后续待执行（harness-follow-up-backlog）

见 [harness-follow-up-backlog](../harness-follow-up-backlog.md)：CI 中 harness-check/doc-gardening、tool_usage 基线、Phase 6 经验库策略等。

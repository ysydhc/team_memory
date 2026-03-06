# Harness Phase 5 实施计划：收尾与规则分离

> **前置**：Phase 0-1、Phase 2、Phase 3、Phase 4 已完成
> **来源**：Phase 0～4 执行过程自检发现的待办项
> **编号重排**：原「Phase 5 预留」（经验库策略、Plan 末尾预留节写法）更名为 **Phase 6**
> **修订**：已根据 plan-evaluator 报告完成 P1 修复与改进建议

**Goal:** 完成 Phase 3 收尾验证、Harness 与 tm 规则分离、step-0 模板补充、feedback-loop 回溯，并统一将「Phase 5」预留内容更名为「Phase 6」。

**Architecture:** 文档与规则调整为主，代码改动限于 Phase 3 反向依赖（若仍存在）；纯 Harness 优先，tm 叠加标注清晰。

---

## 一、Phase 5 范围（来自自检清单）

| 序号 | 待办项 | 产出 |
|------|--------|------|
| 1 | Phase 3 Task 5 收尾验证 | 反向依赖零违规或白名单已记录；architecture-layers 已知待修复节更新 |
| 2 | Harness 与 tm 规则分离 | harness-engineering、feedback-loop、tm-doc-maintenance 等按 boundary 分离 |
| 3 | step-0 模板补充 | harness-workflow-execution 增加 Phase 4 的 step-0 示例 |
| 4 | feedback-loop 回溯 | 4.1、4.2 已固化项移入「已完成」区 |
| 5 | Phase 5→6 编号重排 | 所有「Phase 5 预留」改为「Phase 6 预留」 |

---

## 二、step-0 摸底（强制）

| 动作 | 命令/产出 |
|------|-----------|
| 创建 execute | 创建 `docs/exec-plans/executing/harness-phase-5-execute.md` 并初始化（若 step-0 时尚未创建） |
| 运行 import 检查 | `python scripts/harness_import_check.py` |
| 运行 harness_ref_verify | `./scripts/harness_ref_verify.sh` |
| 检查当前违规 | 若有输出则记录；若无则确认 Phase 3 收尾已完成 |
| 基线 ruff/pytest | `ruff check src/`、`pytest tests/ -q` 通过 |
| 确认 docs 结构 | 列出 `docs/design-docs`、`docs/exec-plans` 目录 |
| 产出 | 基线报告（违规数、ruff/pytest 状态、docs 路径） |

**门控**：step-0 产出后，方可进入 Task 1。

**notify**：Plan 开始时调用 `./scripts/notify_plan_status.sh "Harness Plan" "harness-phase-5 已开始执行"`。

---

## 三、任务拆分

### Task 1：Phase 3 Task 5 收尾验证

**Files:**
- Modify: [docs/design-docs/architecture-layers.md](docs/design-docs/architecture-layers.md)（4.3 节）
- 若存在违规：按 Phase 3 Plan 修复 `architecture` → `web.architecture_models`、`auth` → `web.app`

**步骤：**
1. 运行 `python scripts/harness_import_check.py`，记录当前违规（若有）
2. 若零违规：将 architecture-layers 4.3 节「已知待修复」改为「已修复」或删除，注明完成时间
3. 若有违规：按 Phase 3 Task 5 方案修复（architecture_models 迁至 schemas、auth 按分层调整），或设白名单并记录理由与复审时机
4. 验收：`make harness-check` 通过；architecture-layers 与现状一致

---

### Task 2：Harness 与 tm 规则分离

**Files:**
- Modify: [.cursor/rules/harness-engineering.mdc](.cursor/rules/harness-engineering.mdc)
- Modify: [docs/design-docs/feedback-loop.md](docs/design-docs/feedback-loop.md)
- Modify: [.cursor/rules/tm-doc-maintenance.mdc](.cursor/rules/tm-doc-maintenance.mdc)（**若存在**；若不存在则跳过）
- Modify: [docs/design-docs/phase1-closure-checklist.md](docs/design-docs/phase1-closure-checklist.md)（**若存在**；若不存在则跳过）
- 参考: [docs/design-docs/harness-vs-tm-boundary.md](docs/design-docs/harness-vs-tm-boundary.md) 第四、五节

**调整方向（按 boundary 文档）：**

| 混合点 | 纯 Harness 做法 | tm 叠加（可选） |
|--------|-----------------|-----------------|
| 反馈回路 | 更新 rules 或 docs | 若启用 tm，可额外 tm_save |
| Subagent 审计 | 在回复中记录 `[subagent] task-N:` | 若与 tm_task 关联，可同时 tm_message |
| 收尾清单 | 引用校验、质量门禁；不含 tool_usage | tool_usage 作为「tm 项目可选」项 |

**harness-engineering 具体（按语义定位，避免行号）：**
- **引用 tm-doc-maintenance 处**：改为「见项目文档维护规则」或保留但标注「tm 项目可叠加」
- **反馈回路中 tm_save 表述处**：改为「可额外调用 tm_save（若启用 tm）」，主路径为「更新 rules/docs」
- **Subagent 审计处**：主路径为「在回复中记录」；tm_message 标为「可选（若与 tm_task 关联）」

**feedback-loop.md：**
- 全文已偏「纯 Harness + tm 可选」，确认 2.2 节「team_memory 项目可选」表述清晰；避免「必须 tm_save」的表述

**验收：** 纯 Harness 流程可在无 team_memory 服务、无 tm MCP 下理解并执行；tm 叠加处均有「可选」标注。

---

### Task 3：harness-workflow-execution 补充 Phase 4 step-0 模板

**Files:**
- Modify: [docs/design-docs/harness-workflow-execution.md](docs/design-docs/harness-workflow-execution.md)（3.2 节表格后或 3.4 节）

**内容：**
在「3.2 内容（按 Plan 类型可配置）」表格中增加一行，或在 3.4 节后新增「3.5 Phase 4 类 step-0 示例」：

| Plan 类型 | 摸底动作 | 产出 |
|-----------|----------|------|
| 可观测性类（Phase 4） | 统计 logger 数量、确认 docs 结构、基线 ruff/pytest | 基线报告（logger 数、docs 路径、ruff/pytest 状态） |

并补充 50～100 字说明：Phase 4 的 step-0 命令示例（`rg -l 'logging.getLogger' src/`、`ruff check src/`、`pytest tests/ -q`），产出格式要求。

**验收：** 新 Plan 编写时可直接引用该模板；与 Phase 4 Plan 的 step-0 节一致。

---

### Task 4：feedback-loop 回溯（已固化项移入已完成）

**Files:**
- Modify: [docs/design-docs/feedback-loop.md](docs/design-docs/feedback-loop.md)

**步骤：**
1. 确认 4.1（Web 静态缓存）、4.2（ruff 检查）已固化为 rules 或 team_memory 经验
2. **若尚未固化**：先完成固化（更新 rules 或执行 tm_save，参考 tm-commit-push-checklist、team_memory-codified-shortcuts），再移入已完成
3. 将 4.1、4.2 从「四、待完善项」移至「五、已完成（归档）」
4. 在已完成区注明：完成时间、固化位置（如 `tm-commit-push-checklist`、`team_memory-codified-shortcuts` 或 tm_save id）

**验收：** 待完善项区仅含未完成项；已完成区含 4.1、4.2 及固化位置说明。

---

### Task 5：Phase 5→6 编号重排

**Files:**
- Modify: [docs/exec-plans/harness-phase-4-implementation-plan.md](docs/exec-plans/harness-phase-4-implementation-plan.md)（第六节标题与内容）
- Modify: [docs/exec-plans/completed/harness-phase4-flow-observer-report-2025-03-07.md](docs/exec-plans/completed/harness-phase4-flow-observer-report-2025-03-07.md)（3.4 规则与文档更新建议）
- 检索并更新：所有提及「Phase 5 预留」的文档（**排除**：`workflow_oracle.py` 等代码注释中的 "phase 5" 为无关含义）

**调整：**
- 「Phase 4 可选」/「Phase 5 预留」→「Phase 6 预留」（标题与内容统一）
- 内容不变：经验库策略、stale 经验判定、归档策略（tm 项目）；Plan 末尾预留节写法
- flow-observer 报告 3.4 节「Phase 5 预留」→「Phase 6 预留」

**验收：** 全项目检索「Phase 5 预留」「Phase 5 可选」无遗漏（排除代码注释）；新 Phase 5 为本计划（收尾与规则分离），Phase 6 为原预留内容。

---

### Task 6：文档索引与 execute 记录

**Files:**
- Modify: [docs/exec-plans/index.md](docs/exec-plans/index.md)
- Create/Verify: [docs/exec-plans/executing/harness-phase-5-execute.md](docs/exec-plans/executing/harness-phase-5-execute.md)（若 step-0 时已创建则跳过创建，仅更新）
- Modify: [.cursor/rules/harness-engineering.mdc](.cursor/rules/harness-engineering.mdc)（若需增加 Phase 5 节）

**步骤：**
1. 在 exec-plans/index.md 的 active 区增加 `harness-phase-5-implementation-plan.md` 链接
2. 确认 harness-phase-5-execute.md 存在且格式正确（按 harness-workflow-execution 格式）
3. harness-engineering 若已有 Phase 4 节，可增加「Phase 5 收尾」简要说明（可选）

**验收：** 索引可导航至 Phase 5 计划；execute 文档存在且格式正确。

**notify**：Plan 完成时调用 `./scripts/notify_plan_status.sh "Harness Plan" "harness-phase-5 已完成"`。

---

## 四、执行顺序

```
Plan 开始 → notify + 创建 execute → step-0 摸底 → Task 1 → Task 2 → Task 3 → Task 4 → Task 5 → Task 6 → notify
```

Task 1～4 可部分并行（2 与 3 无依赖）；Task 5 依赖 2、3、4 完成（避免改完又改）；Task 6 最后。

---

## 五、Phase 6 预留（原 Phase 5）

Phase 6（tm 项目可选，后续排期）包含：

- **经验库策略**：stale 经验判定、归档策略、清理频率
- **Plan 末尾预留节写法**：在 Plan 末尾增加「Phase N+1 预留」节，延续 Phase 3→4→5 的写法

本计划不包含 Phase 6，完成 Phase 5 后可单独排期。

---

## 六、成功指标

| 指标 | 目标 |
|------|------|
| harness-check | 通过（零违规或白名单已记录） |
| 规则分离 | harness-engineering、feedback-loop 无强制 tm 依赖 |
| step-0 模板 | harness-workflow-execution 含 Phase 4 类示例 |
| feedback-loop | 4.1、4.2 已移入已完成区 |
| Phase 编号 | 全项目「Phase 5 预留」→「Phase 6 预留」 |

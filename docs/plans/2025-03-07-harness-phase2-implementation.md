# Harness Phase 2 实施计划：工作流增强

> **前置**：Phase 0-1 已完成（知识库重组、AGENTS.md 地图式、harness-engineering、feedback-loop、Harness/tm 边界分离）

**Goal:** 在纯 Harness 工作流基础上，增强「先规划再执行」、自审步骤、功能验证，为 Phase 3 架构约束做准备。

**Architecture:** Phase 2 聚焦工作流与规则增强，不依赖 tm_task/tm_message；可与 tm 叠加时再接入。

---

## 一、Phase 2 范围（来自 Harness 设计）

| 措施 | 验收标准 | 纯 Harness |
|------|----------|------------|
| **先规划再执行（强制）** | 复杂任务先输出计划，人工确认后再执行 | ✅ |
| **step-self-review** | Plan 执行中增加自审步骤（设计/代码/测试/文档） | ✅ |
| **功能验证强化** | 不只看 ruff/pytest，显式加入端到端/行为验证 | ✅ |
| **人类决策点（轻量）** | 在关键步骤写明「需用户确认」 | ✅ |
| **gh PR（可选）** | 需 gh 已配置；无 gh 时保留手动 PR | 可选 |

---

## 二、任务拆分

### Task 1：在 harness-engineering 中固化「先规划再执行」

**Files:** `.cursor/rules/harness-engineering.mdc`

**内容：**
- 新增一节：复杂任务（多文件、多步骤、涉及架构）须先输出实现计划，人工确认后再执行。
- 与 tm-core 中已有「遵循 harness-engineering：复杂任务先规划再执行」对齐，确保 harness 侧有完整表述。

**验收：** 规则可读、无 tm 依赖。

---

### Task 2：创建 Plan 执行自审清单（step-self-review）

**Files:** `docs/design-docs/plan-self-review-checklist.md`

**内容：**
- 每个 Plan Task 完成后，主 Agent 须自审：设计是否符合 Plan、代码是否整洁、测试是否覆盖、文档是否同步。
- 格式：勾选项清单，可被 Subagent-Driven Development 或纯 Harness 流程引用。
- 与 step-verify（tm 工作流）区分：本清单为**纯 Harness Plan 执行**的自审，不依赖 tm_task。

**验收：** 文档存在、可被 rules 或 prompt 引用。

---

### Task 3：在 tm-commit-push-checklist 中强化功能验证

**Files:** `.cursor/rules/tm-commit-push-checklist.mdc`

**内容：**
- 在「本次改动涉及 Web 时」中，显式要求：除 ruff、pytest 外，须有**端到端或行为验证**（如 Playwright、curl 关键接口）。
- 引用已有经验「功能改动后必须进行端到端验证（先API后页面）」。

**验收：** 规则更新、与 tm-web 衔接。

---

### Task 4：人类决策点约定

**Files:** `docs/design-docs/human-decision-points.md`

**内容：**
- 定义 Plan 执行中哪些节点需人类确认：如 Phase 完成、迁移前、AGENTS.md 改造后。
- 格式：节点 + 需确认内容 + 确认后动作。
- 纯 Harness：通过「主 Agent 暂停并汇报，等用户回复后再继续」实现，不依赖 Cursor 原生 approval。

**验收：** 文档存在、可被 Plan 或 rules 引用。

---

### Task 5：更新 AGENTS.md 与 docs 索引

**Files:** `AGENTS.md`, `docs/design-docs/README.md`

**内容：**
- 在 AGENTS.md 知识库导航中增加 plan-self-review-checklist、human-decision-points 链接。
- 确保 docs/design-docs/README.md 或索引包含 Phase 2 新增文档。

**验收：** 链接有效、AGENTS.md 行数 ≤ 120。

---

### Task 6：全量验证

**Files:** 无新增

**步骤：**
1. `ruff check src/`
2. `pytest tests/ -q`
3. `./scripts/harness_ref_verify.sh`
4. 检查 AGENTS.md 行数 ≤ 120

---

## 三、执行顺序

```
Task 1 → Task 2 → Task 3 → Task 4 → Task 5 → Task 6
```

---

## 四、与 Phase 3 的衔接

Phase 3（架构约束）将包含：
- 分层定义（schemas → services → routes → web）
- import 方向检查脚本
- CI 结构测试

Phase 2 完成后，可启动 Phase 3 计划编写与评审。

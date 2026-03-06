# Harness 与 tm (team_memory) 边界分析

> 目的：先跑通纯 Harness 工作流，再识别并分离与 tm 混合之处。

---

## 一、纯 Harness 工作流（无 tm 依赖）

Harness Engineering（OpenAI 提出）的核心是**通用**方法论，不依赖特定 MCP 或项目：

| 环节 | 纯 Harness 做法 | 不依赖 |
|------|-----------------|--------|
| **人类掌舵** | 人类设定目标、审计划、做关键决策 | tm_* |
| **Agent 执行** | Agent 生成代码/文档，人类验收 | tm_* |
| **知识库结构** | AGENTS.md 作为目录，docs/design-docs、docs/exec-plans | tm_* |
| **反馈回路** | 出错时**更新 rules 或 docs**，使同类错误不再发生 | tm_save |
| **Plan 执行** | 读 Plan → 派发子 Agent → 验收 → 下一 Task | tm_task |
| **Subagent 审计** | 在**回复中**记录 `[subagent] task-N: <摘要>` | tm_message |
| **引用校验** | harness_ref_scan.sh、harness_ref_verify.sh（脚本） | tm_* |
| **收尾清单** | 引用校验、质量门禁（ruff、pytest） | tm_* |

**纯 Harness 的反馈回路**：写入 `.cursor/rules/` 或 `docs/`，不要求调用任何 MCP。

---

## 二、tm (team_memory) 特有能力

| 能力 | 用途 | 与 Harness 关系 |
|------|------|-----------------|
| `tm_task` | 任务编排（task_id、group_id、workflow 门控） | 可选：Harness Plan 可与 tm_task 关联，也可不关联 |
| `tm_message` | 任务审计日志（需 task_id，存 DB） | 可选：纯 Harness 用「回复中记录」即可 |
| `tm_save` / `tm_learn` | 经验沉淀到 team_memory 经验库 | 可选：Harness 反馈回路可只更新 rules/docs |
| `tm_preflight` | 任务预检，触发经验检索 | 可选：Harness 不强制 |
| `harness_tool_usage_baseline.sh` | 调用 team_memory API `/api/v1/analytics/tool-usage` | **tm 特有**：依赖 team_memory 服务 |

---

## 三、当前混合点清单

### 1. harness-engineering.mdc

| 位置 | 内容 | 混合类型 |
|------|------|----------|
| 第 31 行 | 结构变更「见 tm-doc-maintenance」 | 引用 tm 规则 |
| 第 37 行 | 反馈回路「调用 tm_save 或 tm_learn」 | **核心混合**：Harness 通用是「更新 rules/docs」 |
| 第 45 行 | Subagent 审计「tm_message（若与 tm_task 关联）」 | **核心混合**：纯 Harness 应只写「在回复中记录」 |

### 2. tm-doc-maintenance.mdc

| 位置 | 内容 | 混合类型 |
|------|------|----------|
| 第 20.1 行 | Harness 文档迁移收尾、Tool usage 基线 | **反向混合**：tm 规则里塞入 Harness 收尾 |
| 第 20.1 行 | 「team_memory 服务可用」时跑 tool_usage | tm 特有（依赖 team_memory API） |

### 3. tm-commit-push-checklist.mdc

| 位置 | 内容 | 混合类型 |
|------|------|----------|
| 第三节 | harness_ref_scan、harness_ref_verify | 混合：Harness 验收放在 tm 收口清单 |

### 4. feedback-loop.md

| 位置 | 内容 | 混合类型 |
|------|------|----------|
| 全文 | 建议用 tm_save、tm_learn 沉淀 | **核心混合**：Harness 反馈回路可不用 tm |

### 5. phase1-closure-checklist.md

| 位置 | 内容 | 混合类型 |
|------|------|----------|
| 第二节 | 「team_memory 服务可用」时跑 tool_usage | tm 特有 |
| 第三节 | 「tm-doc-maintenance 已同步」 | 引用 tm 规则 |

### 6. tm-core.mdc

| 位置 | 内容 | 混合类型 |
|------|------|----------|
| 第 29 行 | 「遵循 harness-engineering」 | tm 引用 Harness，合理 |

---

## 四、分离建议

### 原则

- **Harness 规则**：只含通用方法论，不依赖 tm_*、不引用 tm 特有路径。
- **tm 规则**：可引用 Harness 原则，可「在启用 tm 时」叠加 tm 能力。
- **项目文档**：可同时描述「纯 Harness 流程」与「Harness + tm 叠加流程」。

### 具体调整方向

| 混合点 | 纯 Harness 做法 | tm 叠加（可选） |
|--------|-----------------|-----------------|
| 反馈回路 | 更新 rules 或 docs | 若启用 tm，可额外 tm_save |
| Subagent 审计 | 在回复中记录 `[subagent] task-N:` | 若与 tm_task 关联，可同时 tm_message |
| 收尾清单 | 引用校验、质量门禁；**不含** tool_usage | tool_usage 作为「tm 项目可选」项 |
| tm-doc-maintenance 20.1 | 移出或标注「仅当 Harness+tm 同时启用」 | 保留在 tm 侧 |
| phase1-closure-checklist | 第二节 tool_usage 标为「可选（tm 项目）」 | 同上 |

---

## 五、纯 Harness 工作流（可执行版）

1. **读 Plan** → 提取 Task 列表
2. **逐 Task**：派发子 Agent → 子 Agent 实现/测试/提交 → 主 Agent 验收
3. **审计**：主 Agent 在回复中写 `[subagent] task-N: 派发/完成`
4. **反馈**：出错时更新 rules 或 docs（不强制 tm_save）
5. **收尾**：harness_ref_scan、harness_ref_verify、ruff、pytest
6. **无**：tm_task、tm_message、tm_preflight、tm_save、tool_usage 基线

此流程可在**无 team_memory 服务、无 tm MCP** 的环境下跑通。

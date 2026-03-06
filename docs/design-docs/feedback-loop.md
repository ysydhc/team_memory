# 反馈回路机制

> Agent 出错时记录根因、修复与规则更新，形成闭环，避免同类问题重复发生。

---

## 一、机制概述（纯 Harness）

**核心流程**：出错 → 定位根因 → 修复 → **更新 rules 或 docs** → 必要时更新规则。

纯 Harness 的反馈回路**不依赖任何 MCP**：将根因、修复与规则更新步骤写入 `.cursor/rules/` 或 `docs/`，使同类错误不再发生。

---

## 二、沉淀方式

### 2.1 通用（纯 Harness）

| 方式 | 说明 |
|------|------|
| 更新 rules | 在 `.cursor/rules/` 中新增或修改规则，固化错误与修复 |
| 更新 docs | 在 `docs/design-docs/` 或 `docs/exec-plans/` 中记录经验 |

### 2.2 team_memory 项目可选：经验库叠加

若项目启用 team_memory MCP，可**额外**沉淀到经验库：

| 场景 | 推荐工具 | experience_type |
|------|----------|-----------------|
| Bug 修复、单点经验 | `tm_save` | bugfix / best_practice |
| 长对话、多步骤复盘 | `tm_learn` | 按内容 |
| 组级任务复盘 | `tm_save_group` | tech_design |

**建议**：单次出错与修复优先用 `tm_save`，结构清晰、检索高效。

---

## 三、沉淀内容要求

1. **根因**：问题产生的直接原因或触发条件。
2. **修复**：具体修复步骤、代码/配置变更、验证方式。
3. **规则更新**（若适用）：若需固化到 `.cursor/rules/` 或文档，说明更新点与理由。
4. **自包含**：若改动涉及未被 git 追踪的路径，须将关键内容写入 `solution` 或 `code_snippets`，不得仅写文件路径。

---

## 四、待完善项（待沉淀为 rules 或 docs）

> 上方为当前待办，下方为已完成归档。当某条已固化为 rules 后，移至「已完成」区，保持上方简洁。

（暂无。新待完善项可在此处添加。）

---

## 五、已完成（归档）

> 以下为已固化为 rules 或 docs 的项。从上方移入时，注明完成时间与固化位置。  
> **路径约定**：启用 tm 时规则在 `.tm_cursor/rules/`；纯 Harness 规则在 `.cursor/rules/`。

### 5.1 Web 静态缓存导致旧 JS 生效（2025-03-07）

- **固化位置**：`.tm_cursor/rules/tm-commit-push-checklist.mdc`、`tm-web.mdc`、`team_memory-codified-shortcuts.mdc`
- **经验**：Web 前端修改后需禁用静态缓存避免旧 JS 生效

### 5.2 ruff 检查未通过导致提交失败（2025-03-07）

- **固化位置**：`.tm_cursor/rules/tm-core.mdc`、`tm-commit-push-checklist.mdc`、`team_memory-codified-shortcuts.mdc`、`tm-quality.mdc`
- **经验**：Python 代码改动后必须通过 `ruff check src/` 再提交

---

## 六、与现有规则衔接

| 规则 | 衔接点 |
|------|--------|
| harness-engineering | 反馈回路为「出错时沉淀」的细化设计；纯 Harness 用 rules/docs |
| [harness-workflow-execution](harness-workflow-execution.md) | 工作流执行时加载本文档；某条待完善项固化后，移入「已完成」区 |
| tm-extraction-retrieval | 若启用 tm：Bug 修复、故障排查等场景触发经验提取 |
| tm-core | 若启用 tm：类型勿用 general；自包含写入 solution/code_snippets |

# 文档整理任务

**两阶段**：先扫描并请求用户确认，再按用户决策执行。

## 阶段 1：扫描 + 确认请求（无用户决策时）

### 0. 前置：读取并理解规范

**必须**先读取 `doc-maintenance-guide.md`、`plan-document-structure.md` 的完整内容。

### 1. 扫描

- 执行 `make harness-doc-check`、`make harness-plan-check`
- 按**主题**汇总问题（技术文档按文件，Plan 文档按主题）

### 2. 输出问题列表并直接提问

对每个有问题的主题，**直接提问**：「白名单 or 整理？」

输出格式：
```
## 需确认主题

| 主题 | 违规 | 请确认 |
|------|------|--------|
| docs/exec-plans/completed/harness-phase | DOC_PLAN_LEGACY_STRUCTURE | 白名单 / 整理？ |
| ... | ... | ... |

请直接回复每个主题的决策，格式：`主题: 白名单` 或 `主题: 整理`。主 Agent 将用 mcp_task(resume=本会话 agent_id, prompt=您的回复) 恢复本会话，我会解析并执行阶段 2。
```

若无问题，直接输出「无待确认项」并结束。

---

## 阶段 2：按用户决策执行（resume 恢复时）

当主 Agent 用 **resume** 恢复本会话且 prompt 为用户回复时：

### 白名单

- 将主题加入 `scripts/plan-structure-whitelist.txt` 或 `scripts/doc-gardening-whitelist.txt`

### 整理

- **Plan 文档**：按 [plan-document-structure](docs/design-docs/harness/plan-document-structure.md) 第一章（结构规范、合并原则）执行
  - 1-research/：brownfield、assessment 等 → brief.md、assessment.md、options.md
  - 2-plan/：implementation-plan、execute → plan.md、execute.md
  - 3-retro/：retrospective、flow-observer-report → retro.md
- 适当合并减少文档数量；**合并时须按「复盘必需保留清单」逐项核对**
- 从白名单移除该主题
- 结构变更后须按 [doc-maintenance-guide](docs/design-docs/harness/doc-maintenance-guide.md) 第五章同步 `docs/design-docs/README.md`、`AGENTS.md`

### 输出

- 整理前后对比、已执行操作清单

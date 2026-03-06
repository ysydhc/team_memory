# TeamMemory Agent 导航地图

> 项目概览：基于 MCP 的团队经验库，让 AI 拥有跨会话记忆。**Harness 原则**：人类掌舵、Agent 执行、尽量不手写。

---

## 双轨结构（R4）

| 轨道 | 路径 | 用途 |
|------|------|------|
| 设计/执行计划 | [docs/design-docs](docs/design-docs/) · [docs/exec-plans](docs/exec-plans/) | 设计文档、执行计划、归档 |
| 工作流 YAML | [.cursor/plans/workflows/](.cursor/plans/workflows/) | 可执行工作流定义 |

---

## GitNexus（代码理解）

项目索引：**team_doc**。任务开始先读 `gitnexus://repo/team_doc/context` 检查索引新鲜度。

| 任务 | Skill 文件 |
|------|------------|
| 理解架构 / "How does X work?" | [.claude/skills/gitnexus/exploring/SKILL.md](.claude/skills/gitnexus/exploring/SKILL.md) |
| 影响分析 / "What breaks if I change X?" | [.claude/skills/gitnexus/impact-analysis/SKILL.md](.claude/skills/gitnexus/impact-analysis/SKILL.md) |
| 调试 / "Why is X failing?" | [.claude/skills/gitnexus/debugging/SKILL.md](.claude/skills/gitnexus/debugging/SKILL.md) |
| 重构 / 重命名 / 拆分 | [.claude/skills/gitnexus/refactoring/SKILL.md](.claude/skills/gitnexus/refactoring/SKILL.md) |

工具：`query` · `context` · `impact` · `detect_changes` · `rename` · `cypher`。Schema 见 `gitnexus://repo/team_doc/schema`。

---

## 知识库导航

| 入口 | 路径 | 说明 |
|------|------|------|
| 设计文档 | [docs/design-docs](docs/design-docs/) | 架构、方案、设计决策 |
| 反馈回路 | [docs/design-docs/feedback-loop](docs/design-docs/feedback-loop.md) | Agent 出错时沉淀、规则更新 |
| 执行计划 | [docs/exec-plans](docs/exec-plans/) | 计划、任务、归档 |
| 入门 | [docs/GETTING-STARTED](docs/GETTING-STARTED.md) | 部署、接入、开发 |
| 扩展 | [docs/EXTENDED](docs/EXTENDED.md) | 任务管理、Skills、工作流 |

---

## 工作流

工作流 YAML 位于 [.cursor/plans/workflows/](.cursor/plans/workflows/)：

- `task-execution-workflow.yaml` · `task-execution-workflow-optimized.yaml` — 任务执行
- `workflow-optimization-workflow.yaml` — 工作流优化
- `steps/` · `actions/` · `checklists/` — 步骤、动作、检查清单

---

## 规则

规则位于 [.cursor/rules/](.cursor/rules/)：

- **tm-core** — MCP 核心、预检、任务管理
- **tm-extraction-retrieval** — 经验提取与检索
- **tm-commit-push-checklist** — 提交/推送前收口
- **tm-web** · **tm-quality** · **tm-plan** — Web、质量、Plan
- **team_memory-codified-shortcuts** — 固化结论快路径

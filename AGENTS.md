# TeamMemory Agent 导航地图

> 项目概览：基于 MCP 的团队经验库，让 AI 拥有跨会话记忆。**Harness 原则**：人类掌舵、Agent 执行、尽量不手写。

---

## 双轨结构（R4）

| 轨道 | 路径 | 用途 |
|------|------|------|
| 设计/执行计划 | [docs/design-docs](docs/design-docs/) · [docs/exec-plans](docs/exec-plans/) | 设计文档、执行计划、归档 |

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
| 架构分层 | [docs/design-docs/architecture-layers](docs/design-docs/architecture-layers.md) | 分层定义、import 约束、提交前 harness-check |
| 反馈回路 | [docs/design-docs/feedback-loop](docs/design-docs/feedback-loop.md) | Agent 出错时沉淀、规则更新 |
| Phase 1 收尾清单 | [docs/design-docs/phase1-closure-checklist](docs/design-docs/phase1-closure-checklist.md) | 文档迁移收尾清单 |
| Subagent 工作流 | [docs/design-docs/subagent-workflow](docs/design-docs/subagent-workflow.md) | 两阶段评审、逐步引入 |
| Harness vs tm 边界 | [docs/design-docs/harness-vs-tm-boundary](docs/design-docs/harness-vs-tm-boundary.md) | 纯 Harness 与 tm 叠加的分离说明 |
| Plan 自审清单 | [docs/design-docs/plan-self-review-checklist](docs/design-docs/plan-self-review-checklist.md) | Task 完成后自审 |
| 人类决策点 | [docs/design-docs/human-decision-points](docs/design-docs/human-decision-points.md) | 需用户确认的节点 |
| 工作流执行 | [docs/design-docs/harness-workflow-execution](docs/design-docs/harness-workflow-execution.md) | Plan 执行记录、摸底、通知、自检 |
| 日志格式 | [docs/design-docs/logging-format](docs/design-docs/logging-format.md) | JSON 行日志规范、生产/CI 切换 |
| doc-gardening | [docs/design-docs/doc-gardening](docs/design-docs/doc-gardening.md) | 文档健康度扫描、断裂链接、deprecated 引用 |
| 执行计划 | [docs/exec-plans](docs/exec-plans/) | 计划、任务、归档 |
| 入门 | [docs/GETTING-STARTED](docs/GETTING-STARTED.md) | 部署、接入、开发 |
| 扩展 | [docs/EXTENDED](docs/EXTENDED.md) | 任务管理、Skills、工作流 |

---

## 规则

规则位于 [.cursor/rules/](.cursor/rules/)：

- **tm-core** — MCP 核心、预检、任务管理
- **tm-extraction-retrieval** — 经验提取与检索
- **tm-commit-push-checklist** — 提交/推送前收口
- **tm-web** · **tm-quality** · **tm-plan** — Web、质量、Plan
- **team_memory-codified-shortcuts** — 固化结论快路径

# Subagent-Driven Development 工作流

本项目的 Plan 执行采用 Subagent-Driven Development，逐步引入完整两阶段评审。

---

## 一、完整流程（目标）

| 阶段 | Agent | 职责 |
|------|-------|------|
| 1. 实现 | implementer | 按任务实现、测试、提交、自审 |
| 2. 规格评审 | spec-reviewer | 核对实现与 Plan 一致，读代码验证，不信任报告 |
| 3. 质量评审 | code-quality-reviewer | 检查整洁度、可测试性、可维护性 |

**顺序**：spec 合规通过后，才进入 code quality 评审；不可颠倒或跳过。

---

## 二、Agent 定义

| Agent | 路径 | 触发 |
|-------|------|------|
| implementer | [implementer.md](../../.cursor/agents/implementer.md) | 派发实现任务 |
| spec-reviewer | [spec-reviewer.md](../../.cursor/agents/spec-reviewer.md) | 实现完成后 |
| code-reviewer | [code-reviewer.md](../../.cursor/agents/code-reviewer.md) | spec 通过后 |

---

## 三、逐步引入

- **当前**：implementer（或 generalPurpose）+ 主 Agent 完成度检查
- **目标**：implementer → spec-reviewer → code-reviewer 三阶段
- **路径**：先在高价值或复杂任务上启用 spec + code quality 评审，再推广到全部任务

---

## 四、Prompt 模板

详见 [subagent-driven-development](../../.claude/skills/subagent-driven-development/SKILL.md)：

- `implementer-prompt.md`
- `spec-reviewer-prompt.md`
- `code-quality-reviewer-prompt.md`

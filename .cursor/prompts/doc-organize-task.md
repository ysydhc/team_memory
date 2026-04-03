# 文档整理任务

## 0. 配置

Read `doc-harness.project.yaml` 与 doc-health skill（`.claude/skills/doc-health/SKILL.md`）。

## 1. 扫描

- 仅执行 `commands.doc_check`，按主题汇总。

## 2. 确认

| 主题 | rule_id | 白名单 / 修复？ |

## 3. 执行

- 白名单 → 只改 `whitelists.doc_gardening` 指向的文件。
- 修复 → 链接、`design_docs.index`、`sync_reminders` 入口；**不**做 Plan 目录 1-research/2-plan/3-retro 重组。

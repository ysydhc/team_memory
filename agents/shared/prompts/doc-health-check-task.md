# 文档健康巡检任务

## 0. 配置

Read 仓库根 `doc-harness.project.yaml`，再 Read `design_docs.index`。维护约定与 rule_id 见 **doc-health skill**（`.claude/skills/doc-health/SKILL.md`），**不要**查找 `doc-maintenance-guide.md`（已移除）。

## 1. 门禁

- 执行 `commands.doc_check`，解析 rule_id。
- 按 skill 内「维护约定」做脚本未覆盖的补充检查。

## 2. 报告

- 脚本检出、维护约定补充、索引与 `design_docs.index` 一致性、`sync_reminders` 提醒。

## 约束

- 默认只读；豁免见 `whitelists.doc_gardening`。

---
id: TASK-9
title: Obsidian vault 监听验证
status: Blocked
assignee: []
created_date: '2026-04-27 15:36'
labels: [obsidian, monitoring]
dependencies: []
---

## 阻塞原因
需要确认 vault 路径配置 + 文件变更事件能正确触发 TM 写入。代码已有但从未跑通过。

## 验证步骤
1. 确认 Obsidian vault 路径 (默认 ~/ObsidianVault?)
2. 手动创建/修改 vault 中的一个 md 文件
3. 检查 TM daemon 日志是否收到变更事件
4. 检查是否自动创建 draft 或 experience

## 相关文件
- (待确认具体文件路径)

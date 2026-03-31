# Task Contract 模板

> 在 Phase 2（Execution）中，每个任务开始前签署。

---

## Task: {task_name}

- **输入**：需要什么信息/文件
- **输出**：产出什么（文件、代码变更、配置）
- **验证**：如何确认完成（命令或检查项）
- **回退**：失败时如何恢复（git revert、手动还原、跳过）
- **超时**：预期执行时间（对照 `harness-config.yaml` timeouts）

# after-phase-0.md — 数据库 Migration 一致性检查

Context Loading 完成后，执行数据库迁移状态检查：

1. 运行 `alembic current` 获取当前数据库版本
2. 运行 `alembic heads` 获取代码中最新版本
3. 若两者不一致，**提醒用户**：
   - 列出待执行的 migration 文件
   - 建议执行 `alembic upgrade head`
   - **不自动执行**，等待用户确认后再开始 Task 1

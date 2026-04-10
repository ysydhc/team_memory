# after-phase-4.md — 新增 Migration 执行确认

Plan 收尾后，检查本次执行是否新增了 migration 文件：

1. 对比 `git diff --name-only` 中 `migrations/versions/` 下的新增文件
2. 若存在新增 migration，运行 `alembic current` 确认是否已执行
3. 若有未执行的 migration，**提醒用户**：
   - 列出未执行的 migration 及其描述
   - 建议执行 `alembic upgrade head`
   - 在 execute 文件中记录提醒

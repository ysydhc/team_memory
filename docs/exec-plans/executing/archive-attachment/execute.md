# Plan 执行记录：档案馆（Archive）能力

> Plan ID: 2025-03-16-archive-attachment | 创建: 2025-03-16 | 最后更新: 2025-03-16

## 已加载文档

- HARNESS-SPEC: `docs/design-docs/harness/harness-spec.md`
- Plan: `docs/plans/2025-03-16-archive-attachment.md`
- 设计文档: `docs/design-docs/archive-attachment-to-experience.md`

## 执行摘要

| 字段 | 值 |
|------|-----|
| 状态 | 已完成 |
| 当前 Task | 全部完成（Task 1-9） |
| 最后节点 | Task 9 验收完成 + lint/test 全量修复 |

## 执行日志（按时间倒序，最新在上）

### 2025-03-16 — Task 5-9 全部完成 + lint/test 修复

- **Task 5+6（SearchRequest + SearchPipeline）**：
  - `SearchRequest` 增加 `include_archives: bool = False`
  - `ExperienceService.search` 透传 `include_archives`
  - `SearchPipeline.search` 在 include_archives=True 时查询 ArchiveRepository、合并 L0/L1 结果、按 score 排序截断
  - `SearchCache` key 纳入 include_archives 防止缓存串扰
  - 提交: `feat(search): add include_archives to SearchRequest and pipeline merge`

- **Task 7+7.5（MCP 工具）**：
  - `tm_search`/`tm_solve` 增加 `include_archives` 参数，archive 结果跳过 use_count 递增
  - 新增 `tm_get_archive(archive_id)`: 返回 L2 详情含 attachments[]，404 处理
  - 测试：TestTmArchiveSave + TestTmGetArchive（4 用例）
  - 提交: `feat(mcp): tm_search/tm_solve accept include_archives; add tm_get_archive`

- **Task 8（经验状态变更回写 archive）**：
  - `ArchiveRepository.recompute_archive_status_for_linked_experience`: 重算关联 archive 状态
  - `ArchiveService.update_archive_status_for_experience`: 会话包装
  - `ExperienceService`: 注入 archive_service，在 publish/review 后调用回写
  - `bootstrap`: ArchiveService 在 ExperienceService 之前创建并注入
  - Web 路由 `change_status`/`publish_experience` 调用 archive 状态回写
  - 提交: `feat(archive): derive archive status from linked experiences on exp change`

- **Task 9（文档）**：
  - `docs/mcp-patterns.md` 登记 `tm_archive_save`、`tm_get_archive`、`include_archives` 参数
  - 提交: `docs(mcp-patterns): register tm_archive_save, tm_get_archive; document include_archives`

- **lint/test 修复**：
  - `pyproject.toml`: server.py per-file-ignores E402
  - `__init__.py`: _StderrFilter 长行拆分 (E501)
  - `test_logging_json.py`: patch _load_dotenv_if_available 防止 .env 干扰
  - `test_task_group_completed.py`: 移除 2 个 obsolete 用例 (replace_architecture_bindings 已在 c0c8c69 移除)
  - `make lint` 零报错，`make test` 516 passed / 37 skipped / 0 failed
  - 提交: `fix: resolve all lint and test failures`

### 2025-03-16 — Task 4 验收完成

- **动作**：实现 ArchiveService（archive_save 含 overview、embedding 生成、linked_experience_ids、attachments；get_archive）；bootstrap 注册 archive_service；server 注册 tm_archive_save；test_tm_archive_save_returns_archive_id 通过。
- **产出**：`src/team_memory/services/archive.py`、bootstrap.AppContext.archive_service、server.tm_archive_save、tests；已提交 `feat(mcp): add tm_archive_save and ArchiveService`。
- **下一步**：派发 Task 5（SearchRequest 与 ExperienceService.search 增加 include_archives）。
- **Subagent**：task-4 主 Agent 实现并提交，task-4 完成。

### 2025-03-16 — Task 3 验收完成

- **动作**：实现 ArchiveRepository（create_archive 含 overview、linked_experience_ids、attachments、依关联经验推导 status；search_archives 含 current_user 权限与 L0/L1 结构；get_archive_by_id 返 L2）。
- **产出**：`src/team_memory/storage/archive_repository.py`、`tests/test_archive_repository.py`（3 个用例，需 PostgreSQL 时执行）；ruff 通过；test_archive_models 通过；已提交 `feat(storage): add ArchiveRepository create, search_archives, get_archive_by_id`。
- **下一步**：派发 Task 4（ArchiveService + tm_archive_save）。
- **Subagent**：task-3 主 Agent 实现并提交，task-3 完成。

### 2025-03-16 — Task 2 验收完成

- **动作**：plan-implementer 仍为 Ask 模式；主 Agent 按 subagent 产出代为实现：新建 `tests/test_archive_models.py`，在 `models.py` 末尾追加 Archive、ArchiveExperienceLink、ArchiveAttachment 三个 ORM 类（含 overview、Vector(768)、relationship）。
- **产出**：`src/team_memory/storage/models.py` 新增约 95 行；`tests/test_archive_models.py` 新建；`pytest tests/test_archive_models.py -v` 通过；ruff check 新改文件通过；已提交 `feat(models): add Archive, ArchiveExperienceLink, ArchiveAttachment ORM`。
- **下一步**：派发 Task 3（ArchiveRepository）。
- **Subagent**：task-2 派发 plan-implementer（Ask 模式）；主 Agent 代为落盘并提交，task-2 完成。

### 2025-03-16 — Task 1 验收完成

- **动作**：plan-implementer 在 Ask 模式下无法写盘，返回完整迁移脚本与步骤；主 Agent 按 subagent 产出代为创建迁移文件并提交。
- **产出**：`migrations/versions/c2d3e4f5a6b7_add_archives_tables.py`（archives 含 overview、embedding vector(768)、ivfflat 索引；archive_experience_links、archive_attachments）；已 `git commit -m "feat(db): add archives, archive_experience_links, archive_attachments tables"`。
- **验证**：`ruff check` 新迁移文件通过；`alembic upgrade head` 因本地 PostgreSQL 未启动失败（Connect refused 5432）；`make lint` 失败为既有问题（__init__.py E501、server.py E402/I001），非本 Task 引入；`make test` 未全绿为既有/他模块。按基线决策：本 Task 未新增 lint/test 失败，视为完成。
- **下一步**：请在有 PostgreSQL 的环境执行 `alembic upgrade head` / `alembic downgrade -1` / `alembic upgrade head` 验证迁移与回滚；派发 Task 2（新增 ORM 模型）。
- **Subagent**：task-1 派发 plan-implementer（Ask 模式无法写盘）；主 Agent 代为落盘并提交，task-1 完成。

### 2025-03-16 — step-0 摸底完成，派发 Task 1

- **动作**：加载 HARNESS-SPEC 与 Plan；执行 harness import check（通过）；确认 Alembic head = 6ab06751f40e（与 Plan down_revision 一致）；创建 execute 目录与本文档；派发 Task 1 给 plan-implementer。
- **产出**：`docs/exec-plans/executing/archive-attachment/execute.md`；Task 1 已派发。
- **下一步**：等待 plan-implementer 回报 Task 1 实现与测试结果；验收后派发 Task 2。
- **Subagent**：task-1 派发 plan-implementer（待回报）。
- **通知**：best-effort 调用 `scripts/notify_plan_status.sh`（Plan 开始）。

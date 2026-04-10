# 生产运维手册（回滚与健康检查）

> 运维文档 | 发布回滚、迁移链、备份恢复、探活  
> 相关：[database-operations](../cmd/database-operations.md) · [troubleshooting](troubleshooting.md)

## 应用回滚

1. 停止当前部署进程或编排任务。
2. 将容器镜像 / 部署版本切回上一标签。
3. 按运维平台流程重新启动应用。

## 数据库回滚与迁移链

**重要**：迁移 **002（mvp_cleanup）** 为单向升级，**不得**再降级到 001 之前的数据形态。

在项目根目录、使用本仓库虚拟环境执行 Alembic（与 [database-operations](../cmd/database-operations.md) 一致；亦可用 `make migrate` 升级）：

```bash
# 查看当前 revision
alembic current

# 回退一步（慎用）
alembic downgrade -1

# 回退到指定 revision（慎用）
alembic downgrade <revision_id>
```

**Revision 链（新 → 旧）**

| Revision | 说明 |
|----------|------|
| 010_indexes_and_constraints | 当前最新 |
| 009_background_tasks | |
| 008_archive_raw_conversation | |
| 007_attachment_source_path | |
| 006_archive_knowledge_fields | |
| 005_archive_upload_failures | |
| 004_personal_memory_profile_kind | |
| 003_personal_memories_if_missing | |
| 002_mvp_cleanup | **单向**：不可再降到此 revision 之前 |
| 001_initial_mvp | |

## 应急：数据库备份与恢复

```bash
# 备份
docker compose exec postgres pg_dump -U developer team_memory > backup.sql

# 恢复（覆盖写入前请自行确认）
docker compose exec -T postgres psql -U developer team_memory < backup.sql
```

## 健康检查端点

| 路径 | 说明 |
|------|------|
| `/health` | 全量健康（数据库、Ollama、Embedding、LLM、缓存等，视配置参与检查） |
| `/ready` | 就绪探针（通常仅数据库） |

## 常见问题

| 现象 | 处理 |
|------|------|
| 端口 9111 被占用 | `make release-9111` |
| Embedding 失败 | `ollama pull nomic-embed-text` |
| 数据库连接被拒绝 | `docker compose up -d postgres` |
| 缓存异常 | 查看 Redis：`docker compose logs redis` |

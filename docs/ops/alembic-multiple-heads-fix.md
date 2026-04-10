# Alembic 多个 head 的成因与修复

> 运维文档 | 数据库迁移
> 相关：[database-operations 数据库操作](../cmd/database-operations.md)

## 现象

- `alembic heads` 报：**Revision XXX is present more than once**
- 或报：**Multiple head revisions**
- `alembic upgrade head` 报与某 revision 重叠、无法确定应用顺序

## 常见原因

1. **重复 revision id**：两个不同的迁移文件使用了同一个 `revision = "xxx"`
2. **分支未正确合并**：存在多条并行迁移链，合并迁移的 `down_revision` 指向了错误或重复的节点

## 修复步骤（通用）

1. **定位重复**：在 `migrations/versions/` 下搜索报错中的 revision id
2. **保留一条为主链**：选定一个文件保留原 revision
3. **另一条改为新 revision**：将第二个文件的 `revision` 改为新的唯一 id，`down_revision` 设为保留那条的 revision
4. **更新合并点**：若有合并迁移（`down_revision` 为元组），将其指向旧重复 id 的项改为**新** revision id
5. **验证**：`alembic heads` 应只显示一个 head；`alembic upgrade head` 确认可正常应用

## 相关

- 迁移目录：`migrations/versions/`
- 经验库可搜索「Alembic 多个 head」或标签 `alembic`、`migration`

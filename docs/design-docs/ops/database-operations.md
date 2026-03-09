# PostgreSQL 数据库操作

> 运维文档 | 数据库启停、查看、迁移、备份
> 技术概念：[设计文档索引](../README.md#tech-concepts) | 表结构：[database-schema](../backend-architecture/database-schema.md)

## 基本信息

| 项目 | 值 |
|------|---|
| 镜像 | `pgvector/pgvector:pg16` |
| 端口 | `5432` (本机) |
| 数据库名 | `team_memory` |
| 用户名 | `developer` |
| 密码 | `devpass` |
| 连接地址 | `postgresql://developer:devpass@localhost:5432/team_memory` |

## 启停操作

```bash
# 启动（后台运行，-d = detach）
docker compose up -d

# 查看运行状态
docker compose ps

# 正常输出示例：
#  NAME                COMMAND                  STATUS          PORTS
#  team_memory-postgres-1 "docker-entrypoint.s…"  Up (healthy)    0.0.0.0:5432->5432/tcp

# 查看数据库日志（最后 20 行）
docker compose logs --tail=20 postgres

# 停止（数据保留，下次 up 还在）
docker compose stop

# 完全删除（包括数据！慎用！）
docker compose down -v
```

## 查看数据

### 方法 1：用 docker exec 进入 psql

```bash
# 进入 PostgreSQL 命令行
docker compose exec postgres psql -U developer -d team_memory

# -- 常用 SQL --

# 查看所有表
\dt

# 查看经验列表（最近 10 条）
SELECT id, title, created_at FROM experiences ORDER BY created_at DESC LIMIT 10;

# 查看经验总数
SELECT COUNT(*) FROM experiences;

# 查看所有标签及数量
SELECT unnest(tags) AS tag, COUNT(*) AS cnt
FROM experiences GROUP BY tag ORDER BY cnt DESC;

# 查看某条经验的完整信息
SELECT * FROM experiences WHERE id = '这里粘贴ID';

# 查看反馈
SELECT * FROM experience_feedbacks ORDER BY created_at DESC LIMIT 10;

# 退出 psql
\q
```

### 方法 2：通过 Web API 查看

```bash
# 列出经验（需要先启动 Web 服务）
curl -s http://localhost:9111/api/experiences \
  -H "Authorization: Bearer 0D5007FEF6A90F5A99ED521327C9A698" | python3 -m json.tool

# 查看统计
curl -s http://localhost:9111/api/stats \
  -H "Authorization: Bearer 0D5007FEF6A90F5A99ED521327C9A698" | python3 -m json.tool
```

## 数据库迁移（schema 变更后）

```bash
# 应用所有未执行的迁移
alembic upgrade head

# 查看当前版本
alembic current

# 查看迁移历史
alembic history

# 回退一个版本（慎用）
alembic downgrade -1
```

## 重置数据库（从零开始）

```bash
# 停掉所有服务
docker compose down -v

# 重新启动
docker compose up -d

# 等几秒数据库就绪后，重新建表
sleep 3
alembic upgrade head
```

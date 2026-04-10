# 常见问题排查

> 运维文档 | 故障排查
> 相关：[quick-start 快速启动](../guide/quick-start.md) | [database-operations 数据库操作](../cmd/database-operations.md) | [web-server Web 服务](../guide/web-server.md)

## 1. 端口冲突（数据库 / 本机 PostgreSQL）

本仓库 `docker-compose.yml` 已将容器内 `5432` 映射到主机 **5433**（避免与宿主机默认 PostgreSQL 抢 **5432**）。**症状**：`docker compose ps` 里 postgres 不健康，或宿主机上已有服务占用你要映射的端口。

```bash
# 查看谁在用本机 5432、5433（按你 compose 里的映射改查）
lsof -i :5432
lsof -i :5433

# 若是本机安装的 PostgreSQL 占了口，可先停服务
brew services stop postgresql    # macOS Homebrew 安装的
sudo systemctl stop postgresql   # Linux

# 若改映射端口：同时改 compose 的 ports 与配置里的 `database.url`（或 `TEAM_MEMORY_DB_URL`），见 [database-operations](../cmd/database-operations.md)
```

## 2. 端口 9111 被占用（Web 服务启动失败）

**症状**：`Address already in use`

```bash
# 查看谁在用 9111 端口
lsof -i :9111

# 杀掉占用进程
kill $(lsof -t -i:9111)

# 或者换个端口启动
TEAM_MEMORY_WEB_PORT=9000 \
TEAM_MEMORY_API_KEY=0D5007FEF6A99ED521327C9A698 \
.venv/bin/python -m team_memory.web.app
```

## 3. ModuleNotFoundError: No module named 'team_memory'

**原因**：没有激活虚拟环境，或没有安装项目

```bash
# 确认虚拟环境已激活
source .venv/bin/activate
which python    # 应该显示 .venv/bin/python

# 重新安装
pip install -e ".[dev]"

# 或者直接用 .venv 下的 python（不需要 activate）
.venv/bin/python -m team_memory.web.app
```

## 4. 数据库连接失败

**症状**：`ConnectionRefusedError` 或连接字符串中的主机/端口拒绝（本仓库开发默认 **`localhost:5433`** + `postgresql+asyncpg://`，见 `config.development.yaml` / [database-operations](../cmd/database-operations.md)；报错信息里可能是 5432 或 5433，以配置为准）。

```bash
# 1. 确认 Docker 在运行
docker ps

# 2. 确认数据库容器在运行
docker compose ps

# 如果没启动
docker compose up -d

# 3. 等待数据库就绪（健康检查需要几秒）
docker compose logs --tail=5 postgres
# 看到 "database system is ready to accept connections" 就 OK

# 4. 手动测试连接（容器内 5432）
docker compose exec postgres psql -U developer -d team_memory -c "SELECT 1;"
```

## 5. Ollama 连接失败

**症状**：AI 文档解析返回 503 `Cannot connect to Ollama`

```bash
# 1. 确认 Ollama 在运行
ollama list

# 如果报错 "could not connect"，启动 Ollama
ollama serve    # 前台启动
# 或者打开 Ollama 应用（macOS 菜单栏图标）

# 2. 确认 embedding 模型已下载
ollama list | grep nomic-embed-text
# 如果没有，拉取：
ollama pull nomic-embed-text

# 3. 确认 LLM 模型可用（用于文档解析等，走 /api/chat）
# 须与当前配置一致；开发默认常为 llama3.2（见 config.development.yaml 的 llm.model）
ollama list | grep -E 'llama3\.2|你的模型名'
# 若没有，执行 ollama pull <模型名> 或改配置里的 llm.model

# 4. 测试 Ollama API
curl http://localhost:11434/api/tags
```

## 6. Alembic 迁移失败

**症状**：`alembic upgrade head` 报错

```bash
# 查看当前版本
alembic current

# 查看所有迁移
alembic history

# 如果数据库是全新的（刚 docker compose down -v），直接：
alembic upgrade head

# 如果迁移冲突，最简单的方式是重建数据库
docker compose down -v
docker compose up -d
sleep 3
alembic upgrade head
```

## 7. 语义搜索没有结果

**可能原因**：
- 数据库里经验太少
- Embedding 维度不匹配（换过 provider 但没迁移数据库）

```bash
# 检查经验数量（无 /api/v1/stats 时直接用 SQL）
docker compose exec -T postgres psql -U developer -d team_memory -tAc \
  "SELECT count(*) FROM experiences WHERE is_deleted = false;"

# 检查是否有经验的 embedding 为 null
docker compose exec postgres psql -U developer -d team_memory \
  -c "SELECT id, title, (embedding IS NULL) AS missing_embedding FROM experiences;"
```

## 8. Web 页面白屏 / JS 报错

```bash
# 打开浏览器开发者工具（Cmd+Option+I 或 F12）
# 查看 Console 标签页的错误信息

# 常见原因：
# - API Key 过期 → 刷新页面重新登录
# - 后端未启动 → 检查终端是否有报错
# - 数据库未启动 → 后端日志会显示连接错误
```

## 9. 如何查看后端日志

```bash
# 如果前台运行，日志直接在终端显示

# 如果后台运行（nohup），查看日志文件
tail -f .debug/web.log

# 更详细的应用日志：TEAM_MEMORY_DEBUG=1（bootstrap）；Uvicorn access 日志在 web.app 中为 log_level=info
TEAM_MEMORY_DEBUG=1 \
TEAM_MEMORY_API_KEY=你的Key \
.venv/bin/python -m team_memory.web.app
```

## 10. Web「架构 / GitNexus」

前端 **架构页已移除**（`src/team_memory/web/static/js/pages.js`）；当前 **无** `/api/v1/architecture/*`。GitNexus Bridge（如 `9321`）为可选独立进程，与 TM Web 路由无直接绑定。

## 11. （保留节号占位）

（原「架构搜索」小节已删除，避免与已移除前端不一致。）

## 12. 个人记忆（personal_memories）没有写入

**大致条件**：在 **`memory_save(..., content=...)`** 等长文/解析保存路径成功后，才可能异步抽取个人记忆；具体以 `ExperienceService` 与事件为准。

**怎么从日志判断**（`team_memory` logger，默认 INFO；MCP 看 Cursor MCP 日志或终端）：

| 日志片段 | 含义 |
|---------|------|
| `personal_memory: skipped — user is anonymous` | 当前 MCP 用户是 `anonymous`：检查 `TEAM_MEMORY_API_KEY` 能否通过鉴权；或设 `TEAM_MEMORY_USER`（MCP 侧回退用户名），与 Web 预注册用户名 `TEAM_MEMORY_AUTH__USER` 不同 |
| `personal_memory: no rows to save` | LLM 未抽出任何偏好条目（对话里没可抽内容、JSON 解析失败、或连不上 LLM——后者见 `llm_parser` 的 WARNING） |
| `personal_memory: saved N item(s) for user= ... (static=M dynamic=N)` | 已写入 N 条，含 static/dynamic 计数 |
| `personal_memory: saved N item(s) for user=`（旧日志） | 已写入 N 条 |
| `personal_memory: extract/save failed` | 写库或 embedding 异常；设 `TEAM_MEMORY_DEBUG=1` 可看栈 |

**数据库确认**（需有表，迁移见 `alembic upgrade head`）：

```bash
docker compose exec -T postgres psql -U developer -d team_memory -c \
  "SELECT id, user_id, left(content,60), scope, profile_kind, updated_at FROM personal_memories ORDER BY updated_at DESC LIMIT 10;"
```

**LLM**：个人记忆与正文解析共用 `llm.base_url` + `/api/chat`；`*-cloud` 模型会走云端，失败时先看网络与模型名。

## 速查：服务状态一览

```bash
# 一键检查所有服务状态
echo "=== Docker ===" && docker compose ps 2>/dev/null || echo "Docker not running"
echo ""
echo "=== Port 5433 (本仓库 PG 映射) ===" && lsof -i :5433 2>/dev/null || echo "Not in use"
echo "=== Port 5432 (本机 Postgres 常见) ===" && lsof -i :5432 2>/dev/null || echo "Not in use"
echo ""
echo "=== Port 9111 (Web) ===" && lsof -i :9111 2>/dev/null || echo "Not in use"
echo ""
echo "=== Ollama ===" && ollama list 2>/dev/null || echo "Ollama not running"
echo ""
echo "=== Port 9321 (可选 GitNexus Bridge) ===" && curl -s http://127.0.0.1:9321/context >/dev/null 2>&1 && echo "Bridge running" || echo "Bridge not running"
```

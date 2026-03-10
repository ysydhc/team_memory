# 常见问题排查

> 运维文档 | 故障排查
> 相关：[quick-start 快速启动](quick-start.md) | [database-operations 数据库操作](database-operations.md) | [web-server Web 服务](web-server.md)

## 1. 端口 5432 被占用（数据库启动失败）

**症状**：`docker compose up -d` 后 `docker compose ps` 显示状态不是 healthy

```bash
# 查看谁在用 5432 端口
lsof -i :5432

# 如果是本机安装的 PostgreSQL，先停掉
brew services stop postgresql    # macOS Homebrew 安装的
sudo systemctl stop postgresql   # Linux

# 或者修改 docker-compose.yml 改用其他端口，比如 5433
# ports:
#   - "5433:5432"
# 同时修改 config.yaml 中的数据库 URL 端口为 5433
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

**症状**：`ConnectionRefusedError` 或 `Connection to localhost:5432 refused`

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

# 4. 手动测试连接
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

# 3. 确认 LLM 模型可用（用于文档解析）
ollama list | grep gpt-oss
# 如果没有你配置的模型，修改 config.yaml 中的 llm.model

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
# 检查经验数量
curl -s http://localhost:9111/api/stats \
  -H "Authorization: Bearer 0D5007FEF6A90F5A99ED521327C9A698" | python3 -m json.tool

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

# 更详细的日志（DEBUG 级别）
LOG_LEVEL=debug \
TEAM_MEMORY_API_KEY=0D5007FEF6A90F5A99ED521327C9A698 \
.venv/bin/python -m team_memory.web.app
```

## 10. 架构页显示「未配置或不可用」

**症状**：Web 主导航「架构」进入后提示未配置或不可用。

```bash
# 1. 检查 config.yaml 中 architecture 配置
grep -A5 "architecture:" config.yaml
# 需有 provider: gitnexus 且 bridge_url 非空

# 2. 检查 Bridge 是否启动（默认 9321）
curl -s http://127.0.0.1:9321/context

# 3. 若 Bridge 未启动，从项目根目录执行
node tools/gitnexus-bridge/server.js

# 4. 确认目标仓库已索引（在目标仓库根目录）
npx gitnexus analyze

# 5. 验证 TM 架构 API
curl -s http://localhost:9111/api/v1/architecture/context \
  -H "Authorization: Bearer 你的API_KEY"
```

## 11. 架构搜索返回空结果（200 OK 但无节点）

**症状**：架构页「图」Tab 搜索返回 200 OK，但结果列表为空或显示「未找到匹配节点」。

**原因**：Bridge 未启动或不可达时，Provider 会返回空；现已改为返回 503，前端会提示「架构服务未配置」。

**排查**：

```bash
# 1. 确认 Bridge 已启动（默认 9321）
curl -s "http://127.0.0.1:9321/search?q=OllamaLLMRerankerProvider&scope=global"
# 应返回 {"nodes":[...]}，若 connection refused 则 Bridge 未启动

# 2. 启动 Bridge（从项目根目录）
node tools/gitnexus-bridge/server.js

# 3. 确认 GitNexus 已索引
npx gitnexus analyze
```

## 速查：服务状态一览

```bash
# 一键检查所有服务状态
echo "=== Docker ===" && docker compose ps 2>/dev/null || echo "Docker not running"
echo ""
echo "=== Port 5432 (PostgreSQL) ===" && lsof -i :5432 2>/dev/null || echo "Not in use"
echo ""
echo "=== Port 9111 (Web) ===" && lsof -i :9111 2>/dev/null || echo "Not in use"
echo ""
echo "=== Ollama ===" && ollama list 2>/dev/null || echo "Ollama not running"
echo ""
echo "=== Port 9321 (GitNexus Bridge) ===" && curl -s http://127.0.0.1:9321/context >/dev/null 2>&1 && echo "Bridge running" || echo "Bridge not running"
```

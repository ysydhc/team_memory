# Web 管理界面

> 运维文档 | Web 服务启动、配置、API 测试
> 相关：[quick-start 快速启动](quick-start.md) | [troubleshooting 故障排查](../ops/troubleshooting.md)

## 启动 Web 服务

```bash
cd /path/to/team_memory   # 或你的项目根目录

# 标准启动（前台运行，Ctrl+C 停止）
TEAM_MEMORY_API_KEY=xxx \
.venv/bin/python -m team_memory.web.app

# 可选：覆盖配置里的 auth.user（嵌套 env，见 pydantic-settings）
# TEAM_MEMORY_AUTH__USER=admin
```

启动成功后终端会显示：
```
Starting team_memory web server at http://0.0.0.0:9111
Open http://localhost:9111 in your browser
INFO:     Uvicorn running on http://0.0.0.0:9111 (Press CTRL+C to quit)
```

> 默认端口为 **9111**，可通过 YAML 中 `web.port` 或环境变量 `TEAM_MEMORY_WEB_PORT` 修改。

## 访问

- 打开浏览器：http://localhost:9111
- 输入 API Key：`xxx`
- 点击"登录"

## 功能说明

### 仪表盘
- 显示总经验数、近 7 天新增、标签数量
- 标签分布（点击标签可以筛选）
- 最近添加的经验列表

### 经验列表
- 分页浏览所有经验
- 标签筛选
- 点击卡片查看详情

### 语义搜索
- 输入自然语言描述（如"Docker 端口冲突怎么解决"）
- 基于向量相似度匹配，返回最相关的经验
- 展开"高级参数"面板可覆盖本次搜索的 max_count / top_k_children / min_avg_rating（placeholder 显示全局默认值）

### 设置（检索参数配置）
- 查看和修改检索参数：`max_tokens` / `max_count` / `trim_strategy` / `top_k_children` / `min_avg_rating` / `rating_weight` / `summary_model`
- 修改后即时生效（运行时），重启服务后恢复 YAML 文件中的默认值
- 也可通过 API 管理：`GET /api/v1/config/retrieval` 和 `PUT /api/v1/config/retrieval`（与配置中 `retrieval` 段对应）

### 新建经验

以 **手动填写**（标题 / 问题 / 方案等）为主；若界面另有 AI 辅助流程，以当前 `src/team_memory/web/static` 与 OpenAPI `/docs` 为准（后端路由以 `web/routes/experiences.py` 为准）。

## 环境变量

| 变量名 | 说明 | 默认值 |
|--------|------|--------|
| `TEAM_MEMORY_API_KEY` | API Key（登录凭证）；在 `api_key` 模式下会预注册到内存 | 无，通常必须设置 |
| `TEAM_MEMORY_AUTH__USER` | 预注册内存 Key 时的用户名（覆盖 YAML `auth.user`） | `admin`（来自 `AuthConfig.user`） |
| `TEAM_MEMORY_WEB_HOST` | 监听地址（覆盖 YAML web.host） | `0.0.0.0` |
| `TEAM_MEMORY_WEB_PORT` | 监听端口（覆盖 YAML web.port） | `9111` |
| `TEAM_MEMORY_DB_URL` | 数据库连接地址（覆盖 YAML database） | 无 |

### 修改端口

```bash
# 改为 9000 端口启动
TEAM_MEMORY_API_KEY=xxx \
TEAM_MEMORY_WEB_PORT=9000 \
.venv/bin/python -m team_memory.web.app
```

### 多用户

Web（`db_api_key`）可在界面注册多用户、每人多 API Key。环境变量 **`TEAM_MEMORY_API_KEY`** 仅将**一把** Key 预注册到内存；展示名用 **`TEAM_MEMORY_AUTH__USER`**（或 YAML `auth.user`），不是 `TEAM_MEMORY_USER`。

## 后台运行（不占终端）

```bash
# 方法 1：用 nohup
nohup env TEAM_MEMORY_API_KEY=xxx TEAM_MEMORY_AUTH__USER=admin \
  .venv/bin/python -m team_memory.web.app > .debug/web.log 2>&1 &

# 查看日志
tail -f .debug/web.log

# 停止
kill $(lsof -t -i:9111)
```

```bash
# 方法 2：用 screen（需要安装 screen）
screen -S team_memory_web
TEAM_MEMORY_API_KEY=xxx TEAM_MEMORY_AUTH__USER=admin \
  .venv/bin/python -m team_memory.web.app

# 按 Ctrl+A 然后按 D 脱离 screen（服务继续运行）
# 重新进入：screen -r team_memory_web
```

## 调试前端

Web 前端主体在 **`src/team_memory/web/static/`**：页面壳为 **`index.html`**，交互逻辑多在 **`js/`**（如 `app.js`、`pages.js`、`components.js`）。以当前仓库目录为准。

修改静态文件后：
1. 一般 **刷新页面**（Cmd+R / F5）即可；若后端对某路径做了强缓存，以实际响应头为准。
2. 用浏览器 **开发者工具**（Cmd+Option+I / F12）调试：Console / Network / Elements。

## API 快速测试

用 curl 直接测试后端接口（不需要浏览器）。API 路径为 `/api/v1/`，`/api/` 会自动重写兼容。

```bash
API_KEY="xxx"

# 登录
curl -s -X POST http://localhost:9111/api/v1/auth/login \
  -H "Content-Type: application/json" \
  -d "{\"api_key\": \"$API_KEY\"}" | python3 -m json.tool

# 健康检查（无需认证）
curl -s http://localhost:9111/health | python3 -m json.tool

# 列出经验
curl -s http://localhost:9111/api/v1/experiences \
  -H "Authorization: Bearer $API_KEY" | python3 -m json.tool

# 语义搜索
curl -s -X POST http://localhost:9111/api/v1/search \
  -H "Authorization: Bearer $API_KEY" \
  -H "Content-Type: application/json" \
  -d '{"query": "Docker 部署问题", "max_results": 5}' | python3 -m json.tool

# 新建经验
curl -s -X POST http://localhost:9111/api/v1/experiences \
  -H "Authorization: Bearer $API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "title": "测试经验",
    "problem": "测试问题描述",
    "solution": "测试解决方案",
    "tags": ["test"]
  }' | python3 -m json.tool

# （当前 OpenAPI 中无 /experiences/parse-document、parse-url；文档提取若以 UI 为准，请用浏览器内功能或 MCP / 自定义脚本。）
```

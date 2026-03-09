# Web 管理界面

> 运维文档 | Web 服务启动、配置、API 测试
> 相关：[quick-start 快速启动](quick-start.md) | [troubleshooting 故障排查](troubleshooting.md)

## 启动 Web 服务

```bash
cd /path/to/team_memory   # 或你的项目根目录

# 标准启动（前台运行，Ctrl+C 停止）
TEAM_MEMORY_API_KEY=xxx \
TEAM_MEMORY_USER=admin \
.venv/bin/python -m team_memory.web.app
```

启动成功后终端会显示：
```
Starting team_memory web server at http://0.0.0.0:9111
Open http://localhost:9111 in your browser
INFO:     Uvicorn running on http://0.0.0.0:9111 (Press CTRL+C to quit)
```

> 默认端口为 **9111**，可通过 `config.yaml` 中的 `web.port` 或环境变量 `TEAM_MEMORY_WEB_PORT` 修改。

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
- 修改后即时生效（运行时），重启服务后恢复 `config.yaml` 中的默认值
- 也可通过 API 管理：`GET /api/v1/config/retrieval` 和 `PUT /api/v1/config/retrieval`（与 config.yaml 的 `retrieval` 段对应）

### 新建经验（三种模式）
1. **手动填写** — 逐字段填写标题、问题、方案等
2. **从文档提取** — 粘贴 Markdown/纯文本，AI 自动提取结构化信息
3. **从链接导入** — 输入 URL，自动抓取网页内容并提取

> AI 提取后会自动切换到手动模式，所有字段已预填充，你可以检查修改后再保存。

## 环境变量

| 变量名 | 说明 | 默认值 |
|--------|------|--------|
| `TEAM_MEMORY_API_KEY` | API Key（登录凭证） | 无，必须设置 |
| `TEAM_MEMORY_USER` | 该 API Key 对应的用户名 | `admin` |
| `TEAM_MEMORY_WEB_HOST` | 监听地址（覆盖 config.yaml web.host） | `0.0.0.0` |
| `TEAM_MEMORY_WEB_PORT` | 监听端口（覆盖 config.yaml web.port） | `9111` |
| `TEAM_MEMORY_DB_URL` | 数据库连接地址（覆盖 config.yaml） | 无 |

### 修改端口

```bash
# 改为 9000 端口启动
TEAM_MEMORY_API_KEY=xxx \
TEAM_MEMORY_WEB_PORT=9000 \
.venv/bin/python -m team_memory.web.app
```

### 多用户

可以注册多个 API Key，每个团队成员一个：

```bash
# 同时设置多个用户（用逗号分隔暂不支持，需要代码扩展）
# 当前 MVP 阶段只支持一个 API Key
TEAM_MEMORY_API_KEY=你的key \
TEAM_MEMORY_USER=你的名字 \
.venv/bin/python -m team_memory.web.app
```

## 后台运行（不占终端）

```bash
# 方法 1：用 nohup
nohup env TEAM_MEMORY_API_KEY=xxx TEAM_MEMORY_USER=admin \
  .venv/bin/python -m team_memory.web.app > .debug/web.log 2>&1 &

# 查看日志
tail -f .debug/web.log

# 停止
kill $(lsof -t -i:9111)
```

```bash
# 方法 2：用 screen（需要安装 screen）
screen -S team_memory_web
TEAM_MEMORY_API_KEY=xxx TEAM_MEMORY_USER=admin \
  .venv/bin/python -m team_memory.web.app

# 按 Ctrl+A 然后按 D 脱离 screen（服务继续运行）
# 重新进入：screen -r team_memory_web
```

## 调试前端

Web 前端是纯 HTML/CSS/JS 单页应用，文件位置：

```
src/team_memory/web/static/index.html    ← 全部前端代码都在这里
```

修改这个文件后：
1. **不需要重启服务器**也不需要编译
2. 只需在浏览器中 **刷新页面**（Cmd+R / F5）即可看到更改
3. 用浏览器的 **开发者工具**（Cmd+Option+I / F12）调试：
   - Console 标签页：查看 JS 错误和日志
   - Network 标签页：查看 API 请求和响应
   - Elements 标签页：检查/修改 HTML 和 CSS

> **注意**：因为 HTML 是在 Python 启动时由 FastAPI 从文件系统读取后返回的，
> 如果修改了 index.html，刷新浏览器会直接读取最新文件，无需重启后端。

## API 快速测试

用 curl 直接测试后端接口（不需要浏览器）。API 路径为 `/api/v1/`，`/api/` 会自动重写兼容。

```bash
API_KEY="xxx"

# 登录
curl -s -X POST http://localhost:9111/api/v1/auth/login \
  -H "Content-Type: application/json" \
  -d "{\"api_key\": \"$API_KEY\"}" | python3 -m json.tool

# 查看统计
curl -s http://localhost:9111/api/v1/stats \
  -H "Authorization: Bearer $API_KEY" | python3 -m json.tool

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

# AI 解析文档
curl -s -X POST http://localhost:9111/api/v1/experiences/parse-document \
  -H "Authorization: Bearer $API_KEY" \
  -H "Content-Type: application/json" \
  -d '{"content": "# 问题\n端口被占用\n# 方案\n改端口映射"}' | python3 -m json.tool

# AI 解析链接
curl -s -X POST http://localhost:9111/api/v1/experiences/parse-url \
  -H "Authorization: Bearer $API_KEY" \
  -H "Content-Type: application/json" \
  -d '{"url": "https://example.com/some-article"}' | python3 -m json.tool
```

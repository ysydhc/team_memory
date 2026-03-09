# 一键启动全部服务

> 运维文档 | 新手使用
> 相关：[database-operations 数据库操作](database-operations.md) | [web-server Web 服务](web-server.md)

## 常用术语

| 术语 | 说明 |
|------|------|
| **tm_search** | MCP 工具，语义搜索经验库 |
| **tm_save** | MCP 工具，保存新经验 |
| **API Key** | 登录凭证，在 config.yaml 或 config.minimal.yaml 中配置 |
| **经验** | 问题-方案对，可被检索和复用 |
| **MCP** | Model Context Protocol，让 Cursor 等 AI 客户端调用 tm_search、tm_save 等工具 |

## 前提条件

| 依赖 | 确认方式 | 安装方式 |
|------|---------|---------|
| Docker Desktop | `docker --version` | https://www.docker.com/products/docker-desktop |
| Python 3.11+ | `python3 --version` | https://www.python.org |
| Ollama | `ollama --version` | https://ollama.com |
| Make | `make --version` | macOS/Linux 自带 |

## 第一次使用：一键初始化

打开终端，`cd` 到项目目录后执行：

```bash
# 一条命令完成全部初始化（启动 Docker + 安装依赖 + 初始化数据库）
make setup

# 启动 Web 管理界面
make web
```

浏览器打开 http://localhost:9111 ，输入 `config.yaml` 或 `config.minimal.yaml` 中配置的 API Key 登录。

> **简化配置**: 如果不想面对 100+ 行的完整配置，只需修改 `config.minimal.yaml` 中的 `auth.api_key` 即可启动。

**首次运行可能看到的输出**：`make setup` 会拉取 Docker 镜像、创建虚拟环境、执行 alembic 迁移；`make web` 启动后终端显示 `Uvicorn running on http://0.0.0.0:9111`。**常见首次错误**：端口 5432 或 9111 被占用，见 [troubleshooting](troubleshooting.md)。

## 手动初始化（不使用 Make）

```bash
# ---- 第 1 步：启动基础设施（PostgreSQL + Ollama + Redis）----
docker compose up -d

# 验证数据库是否就绪（看到 "accepting connections" 就 OK）
docker compose logs postgres | tail -5

# ---- 第 2 步：创建 Python 虚拟环境并安装依赖 ----
python3 -m venv .venv
source .venv/bin/activate       # macOS / Linux
# .venv\Scripts\activate        # Windows
pip install -e ".[dev]"

# ---- 第 3 步：初始化数据库表结构 ----
alembic upgrade head

# ---- 第 4 步：拉取 Embedding 模型（只需一次，约 274MB）----
ollama pull nomic-embed-text

# ---- 第 5 步：启动 Web 管理界面 ----
python -m team_memory.web.app
```

## 日常使用：快速启动

已经初始化过的话，每次只需要：

```bash
cd /path/to/team_memory   # 或你的项目根目录

# 方式一（推荐）
make dev

# 方式二（手动）
docker compose up -d          # 确保基础设施在跑
python -m team_memory.web.app    # 启动 Web 服务
```

## 健康检查

一键检查所有组件状态：

```bash
make health
# 或
./scripts/healthcheck.sh
```

输出示例：
```
===== team_memory Health Check =====

  PostgreSQL:           OK
  Ollama:               OK (model: nomic-embed-text)
  Redis (optional):     OK
  Web (port 9111):      healthy
  LLM model (optional): SKIP  (no LLM model configured)

=================================
Result: All critical services OK
```

## 常用 Make 命令

```bash
make help      # 查看所有可用命令
make setup     # 首次安装
make dev       # 启动全部服务（Docker + Web）
make web       # 仅启动 Web 界面
make mcp       # 仅启动 MCP Server
make test      # 运行全部测试
make lint      # Ruff 代码检查
make lint-fix  # Ruff 自动修复
make migrate   # 运行数据库迁移
make backup    # 备份数据库
make health    # 健康检查
make clean     # 清理缓存文件
```

## 停止所有服务

```bash
# 停止 Web 服务：在运行终端按 Ctrl+C

# 停止数据库（数据保留）
docker compose stop

# 停止数据库并删除数据（危险，从头开始）
docker compose down -v
```

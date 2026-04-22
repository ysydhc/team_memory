# CLI 快速上手：从 PyPI 发布到本地验证

> 完整路径：构建 → 发布 PyPI → 安装 → 启动服务 → 验证 CLI
> 相关：[mcp-pypi-local.md](mcp-pypi-local.md)（MCP PyPI 接入）· [mcp-server.md](mcp-server.md)（MCP 配置）· [quick-start.md](quick-start.md)（首次初始化）

---

## 0. 环境确认（开始前必查）

### 0.1 系统依赖

| 依赖 | 最低版本 | 检查命令 |
|------|---------|---------|
| Python | 3.11+ | `python3 --version` |
| pip | 最新 | `pip --version` |
| Docker | — | `docker --version` |
| Docker Compose | V2 | `docker compose version` |

### 0.2 项目依赖

```bash
# 确认在仓库根目录
cd /path/to/team_doc

# 确认 .env 存在且包含 API Key
cat .env | grep TEAM_MEMORY_API_KEY
# 应输出类似：TEAM_MEMORY_API_KEY="你的Key"
```

### 0.3 基础设施

```bash
# PostgreSQL 容器必须运行
docker compose up -d postgres

# 确认数据库就绪
docker compose exec -T postgres pg_isready -U developer -d team_memory
# 输出：/var/run/postgresql:5432 - accepting connections
```

### 0.4 构建工具

```bash
pip install build twine
```

---

## 1. 构建 Python 包

```bash
# 确认版本号（pyproject.toml 中 version 字段）
grep '^version' pyproject.toml
# 输出：version = "0.2.0"

# 如需升版本，修改 pyproject.toml 中的 version 字段

# 清理旧构建并重新打包
rm -rf dist/
python -m build
```

预期输出：

```
Successfully built team_memory-0.2.0.tar.gz and team_memory-0.2.0-py3-none-any.whl
```

验证包完整性：

```bash
twine check dist/*
# 两个 PASSED 即可
```

---

## 2. 发布到 PyPI

### 2.1 配置认证（首次）

前往 https://pypi.org/manage/account/token/ 创建 API Token，然后二选一：

**方式 A：环境变量（推荐）**

```bash
export TWINE_USERNAME=__token__
export TWINE_PASSWORD=pypi-你的token
```

**方式 B：配置文件**

```bash
cat > ~/.pypirc << 'EOF'
[pypi]
username = __token__
password = pypi-你的token
EOF
chmod 600 ~/.pypirc
```

### 2.2 上传

```bash
twine upload dist/*
```

预期输出：

```
Uploading team_memory-0.2.0-py3-none-any.whl
Uploading team_memory-0.2.0.tar.gz
View at: https://pypi.org/project/team-memory/0.2.0/
```

> **注意**：如果使用国内镜像源（如清华），新版本同步可能有 5–15 分钟延迟。可在安装时指定官方源。

---

## 3. 安装 / 更新本地包

### 3.1 开发环境（仓库内）

```bash
# 激活虚拟环境（如有）
source .venv/bin/activate

# 从 PyPI 安装最新版（指定官方源避免镜像延迟）
pip install --upgrade team-memory -i https://pypi.org/simple/

# 或安装指定版本
pip install team-memory==0.2.0 -i https://pypi.org/simple/
```

### 3.2 独立环境（无需克隆仓库）

```bash
python3.11 -m venv ~/.venvs/team-memory
source ~/.venvs/team-memory/bin/activate
pip install team-memory -i https://pypi.org/simple/
```

### 3.3 确认安装

```bash
# 检查版本
pip show team-memory | grep Version
# 输出：Version: 0.2.0

# 检查 CLI 入口
tm-cli --help
```

`tm-cli --help` 应显示 7 个子命令：

```
  archive      Create or update an archive
  upload       Upload a file attachment to an archive
  save         Save team knowledge (MCP save)
  recall       Search team knowledge (MCP recall)
  context      Profile + relevant knowledge (MCP context)
  get-archive  Fetch full archive L2 by id (MCP get_archive)
  feedback     Rate an experience (MCP feedback)
```

---

## 4. 启动服务

### 4.1 使用 Makefile（仓库内开发）

```bash
# 一键启动（Docker 基础设施 + Web 服务）
make dev
```

或分步：

```bash
docker compose up -d postgres redis   # 基础设施
make web                               # Web 服务（http://localhost:9111）
```

### 4.2 手动启动（独立环境）

```bash
# 确保 PostgreSQL 可访问，设置环境变量
export TEAM_MEMORY_DB_URL="postgresql+asyncpg://developer:devpass@localhost:5433/team_memory"
export TEAM_MEMORY_API_KEY="你的APIKey"
export TEAM_MEMORY_PROJECT="team_memory"

# 启动 Web 服务
team-memory-web
# 或
python -m team_memory.web.app
```

### 4.3 确认服务就绪

```bash
curl -s http://localhost:9111/health
# 应返回 JSON，包含 "status": "healthy"
```

---

## 5. 验证 CLI

### 5.1 设置环境变量

```bash
# 加载 .env（仓库内）
set -a && source .env && set +a

# 或手动设置
export TEAM_MEMORY_API_KEY="你的APIKey"
export TM_BASE_URL="http://localhost:9111"   # 默认值，可省略
```

### 5.2 逐个验证

**recall — 搜索经验**

```bash
tm-cli recall --query "test"
# 应返回 JSON（即使无结果也应是 {"experiences": [], ...} 结构）
```

**save — 保存经验**

```bash
tm-cli save --title "CLI 验证" --problem "验证 tm-cli save 是否正常" --solution "执行成功"
# 应返回包含 experience id 的 JSON
```

**recall — 验证刚保存的内容**

```bash
tm-cli recall --query "CLI 验证"
# 应能搜到刚才保存的经验
```

**context — 获取上下文**

```bash
tm-cli context --task-description "测试 CLI"
# 应返回 profile 和相关经验
```

**feedback — 评分**

```bash
# 用上面 save 返回的 experience id
tm-cli feedback --experience-id <刚才返回的id> --rating 5 --comment "验证通过"
```

**archive — 创建档案**

```bash
tm-cli archive --title "CLI 验证档案" --solution-doc "验证 tm-cli archive 功能正常"
# 应输出：Archive created: <archive_id>
```

**get-archive — 获取档案**

```bash
tm-cli get-archive --id <刚才返回的archive_id>
# 应返回档案完整内容
```

### 5.3 快速冒烟（一行验证）

```bash
set -a && source .env && set +a && \
  tm-cli recall --query "test" && \
  echo "--- CLI OK ---"
```

---

## 常见问题

| 问题 | 解决 |
|------|------|
| `tm-cli: command not found` | 确认 pip 安装路径在 `$PATH` 中；或用 `python -m team_memory.cli` |
| `Error: TEAM_MEMORY_API_KEY environment variable is not set` | `export TEAM_MEMORY_API_KEY="..."` 或 `source .env` |
| `Error: Cannot connect to http://localhost:9111` | Web 服务未启动，执行 `make dev` |
| `pip install` 找不到新版本 | 镜像源延迟，加 `-i https://pypi.org/simple/` 指定官方源 |
| 端口 9111 被占用 | `make release-9111` 或 `lsof -i :9111` 检查 |
| 数据库连接失败 | `docker compose up -d postgres` 确认容器运行 |

---

## 修订记录

| 日期 | 说明 |
|------|------|
| 2026-04-13 | 初稿：构建、发布、安装、启动、CLI 验证全流程 |

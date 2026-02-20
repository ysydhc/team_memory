# TeamMemory

基于 MCP 协议的团队经验数据库服务。让 AI Agent 自动查询和复用团队的历史解决方案，减少重复工作和 Token 消耗。

**适用场景**: 3-10 人小团队的 AI 辅助经验管理。

## 功能概览

- **语义搜索**: 基于向量嵌入的经验检索（支持混合搜索 + Reranker）
- **经验类型系统**: 7 种经验类型（通用/需求/Bug/技术方案/故障/最佳实践/学习笔记），每种类型有专属结构化字段
- **MCP 集成**: 11 个 MCP 工具（含 tm_save_typed 全量字段保存）+ 3 个 Resource + 4 个 Prompt
- **Web 管理界面**: 经验浏览、搜索、创建、审核、去重检测、多维度筛选（类型/严重等级/分类/进度）
- **父子层级存储**: 支持将相关经验组织为经验组（parent-children）
- **RBAC 权限**: admin / editor / viewer 三级角色
- **经验生命周期**: 草稿→审核→发布、过期检测、去重合并
- **版本历史**: 就地编辑 + 版本快照、回滚支持
- **工作流模板**: 7 种模板（含结构化字段定义、进度状态、严重等级选项）
- **完整度评分**: 0-100 分自动评分，鼓励团队逐步完善经验记录
- **AI 类型识别**: LLM 自动识别经验类型、提取结构化数据、解析 Git 引用
- **AI 类型推荐**: 创建经验时 AI 实时推荐合适的经验类型（debounce + 置信度）
- **一键部署**: Makefile 统一命令入口 + 最小配置模板 + Docker 自动拉取模型

## 快速开始

### 方式一：使用 Makefile（推荐）

```bash
# 首次安装：启动 Docker + 安装依赖 + 初始化数据库
make setup

# 启动 Web 管理界面（默认端口 9111）
make web

# 查看所有可用命令
make help
```

### 方式二：手动安装

```bash
# 1. 启动基础设施（PostgreSQL + Ollama + Redis）
docker compose up -d

# 2. 安装 Python 依赖
pip install -e ".[dev]"

# 3. 初始化数据库
alembic upgrade head

# 4. 准备 Embedding 模型（仅首次需要）
ollama pull nomic-embed-text

# 5. 启动服务
python -m team_memory.web.app    # Web 管理界面（http://localhost:9111）
python -m team_memory.server     # MCP Server（供 Cursor / Claude Desktop 使用）
```

### 配置

- **最小配置**: 修改 `config.minimal.yaml` 中的 `auth.api_key` 即可启动
- **完整配置**: `config.yaml` 包含所有选项，按 `[必改]` / `[可选]` / `[高级]` 分级标注
- **健康检查**: `make health` 或 `./scripts/healthcheck.sh`

## MCP 接入指南

### Cursor 配置

在项目的 `.cursor/mcp.json` 中添加：

```json
{
  "mcpServers": {
    "team_memory": {
      "command": "/path/to/team_memory/.venv/bin/python",
      "args": ["-m", "team_memory.server"],
      "cwd": "/path/to/team_memory",
      "env": {
        "TEAM_MEMORY_API_KEY": "your-api-key",
        "TEAM_MEMORY_USER": "your-name"
      }
    }
  }
}
```

### Claude Desktop 配置

在 Claude Desktop 设置中添加 MCP Server：

```json
{
  "mcpServers": {
    "team_memory": {
      "command": "/path/to/team_memory/.venv/bin/python",
      "args": ["-m", "team_memory.server"],
      "cwd": "/path/to/team_memory",
      "env": {
        "TEAM_MEMORY_API_KEY": "your-api-key"
      }
    }
  }
}
```

### MCP 工具列表

| 工具 | 功能 | 场景 |
|------|------|------|
| `tm_solve` | 智能问题求解 | 遇到问题时首先调用 |
| `tm_search` | 语义搜索 | 通用经验检索 |
| `tm_suggest` | 上下文推荐 | 基于当前文件/语言推荐 |
| `tm_learn` | 对话提取 | 对话后自动提取经验 |
| `tm_save` | 手动保存 | 保存单条经验 |
| `tm_save_group` | 保存经验组 | 保存父子层级经验 |
| `tm_feedback` | 反馈评分 | 对搜索结果评分 1-5 |
| `tm_update` | 更新经验 | 追加解决方案或标签 |

### MCP 资源

| 资源 URI | 说明 |
|----------|------|
| `experiences://recent` | 最近创建的经验 |
| `experiences://stats` | 经验库统计 |
| `experiences://stale` | 疑似过时的经验 |

### MCP Prompts

| Prompt | 说明 |
|--------|------|
| `summarize_experience` | 引导从对话中提取经验 |
| `submit_doc_experience` | 提交文档作为经验 |
| `review_experience` | 审核经验质量 |
| `troubleshoot` | 系统化故障排查 |

## 配置说明

配置分层加载（后者覆盖前者）：

| 层级 | 文件 | 用途 |
|------|------|------|
| 1 | `config.yaml` | 全量默认配置（按 `[必改]`/`[可选]`/`[高级]` 分级标注） |
| 2 | `config.minimal.yaml` | 用户简化配置（只需改 2 项） |
| 3 | `config.local.yaml` | 开发者高级覆盖 |
| 4 | `config.{env}.yaml` | 多环境叠加 |
| 5 | 环境变量 | 最高优先级 |

### 多环境配置

```bash
# 开发环境（默认）
TEAM_MEMORY_ENV=development  # 使用 config.yaml

# 生产环境
TEAM_MEMORY_ENV=production   # 叠加 config.production.yaml

# 测试环境
TEAM_MEMORY_ENV=test         # 叠加 config.test.yaml
```

### RBAC 角色权限

| 角色 | 权限 |
|------|------|
| admin | 全部操作（用户管理、配置修改、审计日志） |
| editor | 创建、编辑、删除、审核经验 |
| viewer | 只读（搜索、浏览、反馈） |

管理员通过 Web UI 设置页面管理 API Key 和角色分配。

### Embedding 配置

```yaml
embedding:
  provider: ollama  # ollama / openai / local
```

支持 Ollama（默认）、OpenAI API、本地 sentence-transformers。

## API 参考

启动 Web 服务后，访问：
- **Swagger UI**: `http://localhost:9111/docs`
- **ReDoc**: `http://localhost:9111/redoc`

## 运维

### 常用命令

```bash
make setup     # 首次安装
make dev       # 启动全部服务
make health    # 一键健康检查
make backup    # 备份数据库
make test      # 运行测试
make lint      # 代码检查
```

### 备份恢复

```bash
# 备份
make backup
# 或: ./scripts/backup.sh [output_dir]

# 恢复
./scripts/restore.sh backups/team_memory_20260209_120000.sql.gz
```

### Docker 部署

```bash
# 使用 docker-entrypoint.sh 自动化启动
# 自动等待 PG、运行迁移、自动拉取 Ollama 模型、生成 admin key
docker compose up -d
# 统一端口: 9111
```

### 监控

- 内置仪表盘: Web UI 首页
- 健康检查: `make health` 或 `GET /health`
- Prometheus: `GET /metrics`（需安装 `prometheus-client`）
- 就绪探针: `GET /ready`

## 开发

### 运行测试

```bash
# 全部测试
pytest -v

# 带覆盖率
pytest --cov=team_memory
```

### 技术栈

- **MCP**: FastMCP
- **Web**: FastAPI + Uvicorn
- **数据库**: PostgreSQL + pgvector
- **ORM**: SQLAlchemy 2.0 (async)
- **嵌入**: Ollama / OpenAI / 本地模型
- **搜索**: 向量 + 全文检索 + RRF 混合
- **缓存**: 内存 LRU / Redis
- **配置**: Pydantic Settings + YAML

## FAQ

**Q: 切换 Embedding 模型后需要做什么？**
A: 需要重新生成所有 embedding。使用 `scripts/migrate_embeddings.py`。

**Q: 如何从 ivfflat 切换到 HNSW 索引？**
A: 修改 `config.yaml` 中 `vector.index_type: hnsw`，然后运行迁移。建议在经验超过 10,000 条时切换。

**Q: 没有 Ollama 可以使用吗？**
A: 可以。将 `embedding.provider` 改为 `openai` 并配置 API Key，或使用 `local` 加载本地模型。

**Q: viewer 角色可以创建经验吗？**
A: 不可以。viewer 只能搜索和浏览。需要 editor 或 admin 角色才能创建。

**Q: 个人经验和团队经验有什么区别？**
A: 个人经验（scope=personal）只有创建者可见。团队经验（scope=team）所有人可见。个人经验可以"晋升"为团队经验。

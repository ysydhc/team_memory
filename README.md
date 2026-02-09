# team_doc

基于 MCP 协议的团队经验数据库服务。让 AI Agent 自动查询和复用团队的历史解决方案，减少重复工作和 Token 消耗。

## 快速开始

### 1. 启动数据库

```bash
docker compose up -d
```

### 2. 安装依赖

```bash
pip install -e ".[dev]"
```

### 3. 初始化数据库

```bash
alembic upgrade head
```

### 4. 配置

复制并修改配置文件：

```bash
cp config.yaml config.local.yaml
```

设置环境变量：

```bash
export OPENAI_API_KEY=your-api-key
export TEAM_DOC_API_KEY=your-team-doc-api-key
```

### 5. 启动 MCP Server

```bash
python -m team_doc.server
```

### 6. Cursor 集成

在项目的 `.cursor/mcp.json` 中添加：

```json
{
  "mcpServers": {
    "team_doc": {
      "command": "python",
      "args": ["-m", "team_doc.server"],
      "env": {
        "TEAM_DOC_API_KEY": "your-api-key",
        "TEAM_DOC_DB_URL": "postgresql+asyncpg://developer:devpass@localhost:5432/team_doc"
      }
    }
  }
}
```

## 开发

### 运行测试

```bash
pytest
```

### 运行测试（带覆盖率）

```bash
pytest --cov=team_doc
```

## 技术栈

- **MCP Server**: FastMCP
- **数据库**: PostgreSQL + pgvector
- **ORM**: SQLAlchemy 2.0 (async)
- **嵌入模型**: OpenAI API / 本地 bge-m3（可切换）
- **配置**: Pydantic Settings + YAML

详细技术文档见 [pm.md](pm.md)。

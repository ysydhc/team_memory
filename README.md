# TeamMemory

mcp-name: io.github.ysydhc/team-memory

**让 AI 拥有团队记忆** — 跨会话积累经验，像资深成员一样理解你的项目。

> 这是我学 AI 时萌生的一个想法。市面上已有类似产品，但总觉得不太贴合自己的使用习惯。做这个项目，既想通过和 AI 一起写代码来加深对大模型的理解，也希望能按自己的工作流，打磨出真正顺手的功能。

## 为什么需要 TeamMemory？

用 Cursor、Claude 等 AI 助手写代码时，往往会遇到三个问题：

| 盲区 | 现象 |
|------|------|
| **无记忆** | 上周刚帮你修过的 Bug，这周遇到类似的，它完全不记得 |
| **只见代码，不懂决策** | 能看懂「是什么」，却不知道「为什么这么写」「上次踩过什么坑」 |
| **静态知识不够用** | Rules、Skills 管得了规范，管不住每天冒出来的隐性经验（接口坑、故障根因、被否掉的方案） |

**TeamMemory 就是冲着这三个问题来的。** 通过 MCP 把语义可搜索的经验库接进 AI：遇到问题自动查历史方案，解决后自动提炼并存下来，下次谁遇到同类问题，直接就能命中。既适合 3–10 人的技术团队共享，也适合部署在本地个人使用，配合 Cursor / Claude Desktop。

## 快速开始（4 条命令 + 1 项配置）

**环境**：Docker Desktop、Python 3.11+、Make

```bash
# 1. 初始化（Docker + 依赖 + 数据库）
make setup

# 2. 设置 API Key（唯一必改项）
export TEAM_MEMORY_API_KEY=$(openssl rand -hex 16)
echo "API Key: $TEAM_MEMORY_API_KEY"

# 3. 拉取 Embedding 模型（仅首次需要）
ollama pull nomic-embed-text

# 4. 启动
make web
```

浏览器访问 http://localhost:9111 ，用上面的 API Key 登录即可。

## MCP 接入（Cursor / Claude）

安装：`pip install team_memory`，然后在 `.cursor/mcp.json` 里加上：

```json
{
  "mcpServers": {
    "team_memory": {
      "command": "python3",
      "args": ["-m", "team_memory.server"],
      "env": {
        "TEAM_MEMORY_DB_URL": "postgresql+asyncpg://developer:devpass@localhost:5432/team_memory",
        "TEAM_MEMORY_API_KEY": "你的 API Key"
      }
    }
  }
}
```

本机直连数据库时需要配 `TEAM_MEMORY_DB_URL`；从源码跑且项目里已有 config 的，可以不配。

## 更多

- **完整指南**：[docs/getting-started.md](docs/getting-started.md)
- **文档结构**：设计文档 [docs/design-docs](docs/design-docs/)、执行计划 [docs/exec-plans](docs/exec-plans/)，详见 [docs/getting-started.md](docs/getting-started.md)
- **扩展功能**（任务、Skills、工作流）：[docs/extended.md](docs/extended.md)

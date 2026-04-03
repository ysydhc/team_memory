# MCP Server（Cursor 集成）

> 运维文档 | MCP 启动、配置、验证  
> **入口**：`team_memory.server`（仅 `memory_*`）— 见 [mcp-lite-default.md](mcp-lite-default.md)  
> 相关：[quick-start 快速启动](quick-start.md) | [web-server Web 服务](web-server.md)

MCP 给 Cursor / Claude Desktop 里的 Agent 用，与 **Web** 是独立进程，可同时运行。

## MCP 启动

**本仓库**：`make mcp` → `python -m team_memory.server`。

本地手动启动示例：

```bash
cd /path/to/team_doc   # 项目根目录

TEAM_MEMORY_API_KEY=<你的 Key> \
/path/to/.venv/bin/python -m team_memory.server
```

### `memory_*` 工具（5 个）

| 工具名 | 用途（概要） |
|--------|----------------|
| `memory_context` | 当前任务上下文 + 画像 / 相关经验 |
| `memory_recall` | 按 problem / query / file 路径检索经验（可选档案预览） |
| `memory_save` | 直接保存或 `content` 长文解析保存（`scope=archive` 已移除） |
| `memory_get_archive` | 按 `archive_id` 拉取档案 L2 全文 |
| `memory_feedback` | 对检索结果评分 |

档案创建请使用 **`POST /api/v1/archives`**（或项目内 `/archive` skill），勿再传 `memory_save(scope=archive)`。

当前 MCP **未** 注册 Resources / Prompts；若客户端列表为空属正常。

## 配置 Cursor

在项目 `.cursor/mcp.json`（路径因项目而异）中将 `args` 设为：

```json
{
  "mcpServers": {
    "team_memory": {
      "command": "/path/to/.venv/bin/python",
      "args": ["-m", "team_memory.server"],
      "cwd": "/path/to/team_doc"
    }
  }
}
```

按需补充 `env`：`TEAM_MEMORY_API_KEY`、`TEAM_MEMORY_DB_URL`（本机直连数据库时）、`TEAM_MEMORY_PROJECT` 等。

### 在其他项目中使用

在对应仓库的 `.cursor/mcp.json` 里把 `command` / `cwd` 指到本仓库的 venv 与根目录，并配置环境变量。

## 验证 MCP

1. Cursor 已加载含 MCP 配置的项目。  
2. 新对话中让 Agent 使用 **`memory_context`** 或 **`memory_recall`**。  
3. 在 Cursor MCP 面板确认服务已连接。

若仍看到 **`tm_search`** 等旧工具名，说明配置仍指向已删除的入口或旧版本包，请改为 **`-m team_memory.server`** 并升级安装。

## 资源（Resources）

当前构建的 `server.py` **未** 暴露 MCP resources；以运行中服务实际列表为准。

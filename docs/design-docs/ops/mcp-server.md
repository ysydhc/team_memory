# MCP Server（Cursor 集成）

> 运维文档 | MCP 启动、配置、验证  
> **默认**：Lite（`memory_*`）— 见 [mcp-lite-default.md](mcp-lite-default.md)  
> 相关：[quick-start 快速启动](quick-start.md) | [web-server Web 服务](web-server.md)

MCP 给 Cursor / Claude Desktop 里的 Agent 用，与 **Web** 是独立进程，可同时运行。

## 推荐：Lite MCP（默认）

**本仓库默认**：`make mcp` → `python -m team_memory.server_lite`。

本地手动启动示例：

```bash
cd /path/to/team_doc   # 项目根目录

TEAM_MEMORY_API_KEY=<你的 Key> \
.venv/bin/python -m team_memory.server_lite
```

### Lite 工具（4 个）

| 工具名 | 用途（概要） |
|--------|----------------|
| `memory_context` | 当前任务上下文 + 画像 / 相关经验 |
| `memory_recall` | 按查询检索经验 |
| `memory_save` | 保存内容/经验 |
| `memory_feedback` | 对检索结果评分 |

## 遗留：完整 MCP（`tm_*`）

仅在与旧集成兼容时使用：

```bash
make mcp-full
# 或
team-memory-full
# 或
python -m team_memory.server
```

完整面 **计划择机移除**，新集成 **不要** 依赖 `tm_*`。详见 [mcp-lite-default.md](mcp-lite-default.md)。

## 配置 Cursor

在项目 `.cursor/mcp.json`（路径因项目而异）中将 `args` 设为 **Lite**：

```json
{
  "mcpServers": {
    "team_memory": {
      "command": "/path/to/.venv/bin/python",
      "args": ["-m", "team_memory.server_lite"],
      "cwd": "/path/to/team_doc"
    }
  }
}
```

### 在其他项目中使用

在对应仓库的 `.cursor/mcp.json` 里把 `command` / `cwd` 指到本仓库的 venv 与根目录，并配置 `TEAM_MEMORY_API_KEY`、`TEAM_MEMORY_DB_URL`（若需要）。

## 验证 MCP

1. Cursor 已加载含 MCP 配置的项目。  
2. 新对话中让 Agent 使用 **`memory_context`** 或 **`memory_recall`**（Lite）；若仍看到 `tm_search`，说明配置仍指向 **`team_memory.server`**，请改为 **`server_lite`**。  
3. 在 Cursor MCP 面板确认服务已连接。

## 资源（Resources）

若当前构建启用了 MCP resources，以运行中的 server / Lite 实际列表为准；完整面与 Lite 的 resources 集合可能不同。

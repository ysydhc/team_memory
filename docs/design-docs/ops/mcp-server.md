# MCP Server（Cursor 集成）

> 运维文档 | MCP 启动、配置、验证
> 相关：[quick-start 快速启动](quick-start.md) | [web-server Web 服务](web-server.md)

MCP Server 是给 Cursor 中的 AI Agent 用的，让 Agent 能自动搜索和保存经验。
它和 Web 服务是两个独立的进程，可以同时运行。

## 启动 MCP Server（手动测试用）

```bash
cd /path/to/team_memory   # 或你的项目根目录

TEAM_MEMORY_API_KEY=0D5007FEF6A90F5A99ED521327C9A698 \
.venv/bin/python -m team_memory.server
```

> 通常你不需要手动启动 MCP Server，Cursor 会根据 `.cursor/mcp.json` 自动启动。

## 配置 Cursor

项目已有 `.cursor/mcp.json` 文件：

```
<项目根目录>/.cursor/mcp.json
```

内容如下（将路径替换为你的项目根目录）：

```json
{
  "mcpServers": {
    "team_memory": {
      "command": "/path/to/team_memory/.venv/bin/python",
      "args": ["-m", "team_memory.server"],
      "cwd": "/path/to/team_memory"
    }
  }
}
```

### 在其他项目中使用 team_memory

如果你想在其他项目的 Cursor 中也能使用经验数据库，在那个项目下创建 `.cursor/mcp.json`：

```json
{
  "mcpServers": {
    "team_memory": {
      "command": "/path/to/team_memory/.venv/bin/python",
      "args": ["-m", "team_memory.server"],
      "cwd": "/path/to/team_memory",
      "env": {
        "TEAM_MEMORY_API_KEY": "0D5007FEF6A90F5A99ED521327C9A698"
      }
    }
  }
}
```

## MCP 提供的能力

### 工具 (Tools)

| 工具名 | 功能 | Agent 何时使用 |
|--------|------|---------------|
| `tm_search` | 语义搜索经验库 | 遇到问题时先搜索是否有类似经验 |
| `tm_save` | 保存新经验 | 解决完问题后保存方案 |

### 资源 (Resources)

| 资源名 | 功能 |
|--------|------|
| `experiences://recent` | 最近创建的经验 |
| `experiences://stats` | 经验库统计信息 |

## 验证 MCP 是否工作

### 步骤

1. 确认 Cursor 已加载项目（含 `.cursor/mcp.json`）
2. 打开新对话（Cmd+L 或 New Chat）
3. 输入：`请搜索经验库中关于 Docker 的经验`
4. 观察 Agent 是否调用 `tm_search` 工具

### 预期结果

- **配置正确**：Agent 会调用 `tm_search`，返回 JSON 格式的搜索结果（含 `results` 数组）
- **配置错误**：Agent 可能回复「无法访问经验库」或不会出现工具调用

### 其他验证方式

在 Cursor 的 MCP 面板（点击左侧边栏的 MCP 图标）中查看 `team_memory` 服务状态，应显示为已连接。

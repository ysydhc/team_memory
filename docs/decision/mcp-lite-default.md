# 决策：MCP 仅暴露 `memory_*`（单一入口）

> 状态：**已生效**  
> 日期：2026-04（随 `server_lite` 退场修订）  
> 相关：[mcp-server.md](../guide/mcp-server.md) · [README.md](../../README.md)（MCP 工具表）· [src/team_memory/server.py](../../src/team_memory/server.py)

## 背景

- 历史版本曾拆分 **`team_memory.server_lite`**（少量 `memory_*`）与完整 **`tm_*` 工具集**。  
- 当前仓库 **已统一**：唯一 MCP 模块为 **`team_memory.server`**，仅注册 **`memory_save`、`memory_recall`、`memory_context`、`memory_get_archive`、`memory_archive_upsert`、`memory_feedback`** 六个工具。

## 决议

| 项 | 约定 |
|----|------|
| **MCP 入口** | `make mcp`、`python -m team_memory.server`、控制台 **`team-memory`** → 均为同一实现（`memory_*`） |
| **`tm_*`** | **不再**作为 MCP 工具提供；旧文档或外部脚本若仍调用 `tm_search` 等需改为 **`memory_*`** 或 Web HTTP API |
| **档案写入** | `memory_save(scope=archive)` **已移除**；文案用 MCP **`memory_archive_upsert`**（或 HTTP `/archives`）；附件用 HTTP multipart 或 **`python -m team_memory.cli upload`**（见 [mcp-server.md](../guide/mcp-server.md)） |
| **扩展能力** | 任务看板、可安装目录等 **未** 随本仓库 MCP 暴露；以当前 Web 路由与 OpenAPI（`/docs`）为准 |

## 配置提示

- Cursor / Claude：`.cursor/mcp.json` 的 `args` 使用 **`-m team_memory.server`**（勿再写已删除的 `server_lite`）。

## 修订记录

| 日期 | 说明 |
|------|------|
| 2026-03-30 | 默认 `make mcp` 与 Lite 行为对齐（`memory_*`）。 |
| 2026-04 | 移除 `server_lite`；删去 `mcp-full` / `team-memory-full`；MCP 单文件 `server.py`。 |

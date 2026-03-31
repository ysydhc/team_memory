# 决策：默认 MCP 为 Lite（`memory_*`）

> 状态：**已生效（仓库默认）**  
> 日期：2026-03-30  
> 相关：[mcp-server.md](mcp-server.md) · Plan [personal-memory-profile](../../plans/2026-04-01-personal-memory-profile-supermemory-alignment.md)

## 背景

- 仓库长期并存 **`team_memory.server`**（`tm_*`，约 18 个工具）与 **`team_memory.server_lite`**（`memory_*`，4 个工具）。  
- 产品与集成成本上 **以 Lite 为唯一对外面**（工具少、命名清晰）；完整面仅为历史兼容。

## 决议

| 项 | 约定 |
|----|------|
| **默认入口** | `make mcp` → `python -m team_memory.server_lite` |
| **CLI 默认** | `team-memory` → **Lite**（与 `team-memory-lite` 等价） |
| **遗留入口** | `make mcp-full`、`team-memory-full` → `team_memory.server`（**`tm_*`**） |
| **移除计划** | 在 Lite 功能与文档对齐后 **择机删除** `server.py` 注册的对外 MCP（不设具体日期；执行前须公告 + 次要版本号）。 |

## 迁移提示

- Cursor / Claude：`.cursor/mcp.json`（或等价配置）应使用 **`-m team_memory.server_lite`**。  
- 仍依赖 `tm_search` / `tm_learn` 等的集成，在删除完整 MCP 前须 **迁到 Web API 或 Lite 组合调用**。

## PyPI / 发版公告（维护者清单）

下一次 **bump `pyproject.toml` `version`** 并打 tag 发布时：

1. 将根目录 **[CHANGELOG.md](../../../CHANGELOG.md)** 中 `[Unreleased]` 改为该版本号与日期。  
2. **GitHub Release**（或等价）：从 CHANGELOG 复制「Changed（破坏性）」摘要；首句标明 **Breaking**。  
3. **PyPI 长描述**（若单独维护）：可附一句 *`team-memory` defaults to Lite MCP; use `team-memory-full` for legacy `tm_*`.*  

自助用户入口：**CHANGELOG** + 本文 + [mcp-server.md](mcp-server.md)。

## `test_server.py` 与完整 MCP 移除（时间线 · 草案）

| 阶段 | 动作 | 说明 |
|------|------|------|
| **当前** | 保留 **`tests/test_server.py`** + **`team_memory.server`** | 回归遗留 `tm_*`；**默认产品路径**已为 Lite。 |
| **下一策** | 在 CHANGELOG / 发布说明中标注 *完整 MCP deprecated* | 不删代码，仅缩短承诺面。 |
| **删除前** | 下列条件**至少满足其一方可动议删 `server.py` MCP**：Lite 覆盖全部对外能力；或 `test_server_lite` + 集成测试已替代关键路径；且 **一版次要版本** 仅保留 `team-memory-full` 警告周期。 | 须单独 ADR + retro。 |

> 具体版本号不在本文锁定；由 **画像 Plan 完了** 或 **独立下线 Plan** 的 execute 勾选。

## 修订记录

| 日期 | 说明 |
|------|------|
| 2026-03-30 | 默认 `make mcp` / `team-memory` 切到 Lite；新增 `mcp-full` / `team-memory-full`。 |
| 2026-03-30 | 补充：PyPI 发版清单、`test_server.py`/完整 MCP 移除时间线草案；根目录 `CHANGELOG.md`。 |

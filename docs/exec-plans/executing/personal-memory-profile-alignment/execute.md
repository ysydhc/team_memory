# Execute：个人记忆 / 用户画像（Supermemory 对齐）

> **计划本体**： [2026-04-01-personal-memory-profile-supermemory-alignment.md](../../../plans/2026-04-01-personal-memory-profile-supermemory-alignment.md)  
> **任务清单（按条验收）**： [tasks.md](./tasks.md)

## 使用方式

1. 在 **tasks.md** 中按依赖勾选 **T-Axx … T-Vxx … T-Exx**；每条含 **工程验收 / 产品验收**。  
2. 在本文记录 **Plan §7 常量**（profile 条数上限、匿名返回、**`memory_recall(include_user_profile)` P0/P1**、跨 project、`valid_until` 清理 retention）。  
3. **MVP 发布前** 完成 **T-V01**（§9.3 一轮）与 **T-V02**（`make verify`）。  
4. Phase E 单独迭代或合并末周时，在本文注明 **分支/标签** 与 **发布说明**。

## Plan §7 常量（落地时填写）

| 常量 | 取值 | 填写日期 |
|------|------|----------|
| `profile` 每侧条数上限 N | 20（`mcp.profile_max_strings_per_side`） | 2026-03-30 |
| 匿名用户 `profile` 形态 | `{ static: [], dynamic: [] }` | 2026-03-30 |
| `memory_recall(include_user_profile)` | P1（已实现，`False` 默认） | 2026-03-30 |
| 同一用户跨 project 是否共享 profile | 是（仅 `user_id`，与现模型一致） | 2026-03-30 |
| 个人记忆语义合并阈值（余弦相似度） | `0.88`（`PERSONAL_MEMORY_OVERWRITE_THRESHOLD`，[repository.py](../../../../src/team_memory/storage/repository.py)） | 2026-03-30 |
| profile 条内去重策略 | 同侧按 `updated_at` 顺序，**首条保留**（`content` 小写去重） | 2026-03-30 |
| Phase E 物理清理 retention（若启用 E4） | **未启用**（随 Phase E 单独立项） |  |

**P2 / 后续**：见 [p2-backlog.md](./p2-backlog.md)。

## 进度速览

| 阶段 | 状态 | 备注 |
|------|------|------|
| Phase A | 已完成 | 迁移 `004_profile_kind`、ORM、Repository、`build_profile_for_user` |
| Phase B | 已完成 | `parse_personal_memory`、`_try_extract_*`、`upsert` 按 kind |
| Phase C | 已完成 | Lite `memory_context` / `include_user_profile`、Web 分组与筛选 |
| Phase D | 已完成 | Instructions、规则、`supermemory-comparison`、troubleshooting、本 execute 常量 |
| T-V01 / T-V02 | T-V02 绿 / T-V01 待人测 | `make verify`；§9.3 需人工 dogfood |
| Phase E | 未开始 | `valid_until` 续包 |

## 仓库默认 MCP（与画像 Plan 一致）

- **决策记录**：[mcp-lite-default.md](../../../design-docs/ops/mcp-lite-default.md)  
- **`make mcp`**、**`team-memory`** 默认已切到 **Lite**；完整 `tm_*` 仅 **`make mcp-full` / `team-memory-full`**。

## 变更记录

| 日期 | 说明 |
|------|------|
| 2026-03-30 | 初建 execute + tasks.md（PM 口径验收） |
| 2026-03-30 | 对齐 **全面 Lite MCP**：常量表与 Plan §7 一致，去掉 `tm_search` |
| 2026-03-30 | 链接 **mcp-lite-default**：make / pyproject 默认 Lite |
| 2026-03-30 | **CHANGELOG.md**；tasks **T-W01**（发版闭环）；mcp-lite-default：**PyPI 清单** + **test_server 时间线** |
| 2026-03-30 | **执行**：`004_profile_kind`；`build_profile_for_user`；Lite `memory_context` / `memory_recall(include_user_profile)`；解析与语义 upsert 按 kind 隔离 |
| 2026-03-30 | **Plan 收口**：execute 补充阈值/去重/E4；**p2-backlog.md**；parse_personal_memory 非法 kind 单测；getting-started/security 画像句 |

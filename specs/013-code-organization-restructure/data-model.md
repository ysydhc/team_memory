# Data Model: Phase 0 — 代码组织重组

**Branch**: `013-code-organization-restructure` | **Date**: 2026-05-13

## Schema Changes

**无数据库 schema 变更。** 无需 Alembic 迁移。

本次重组仅涉及 Python 文件的物理位置变更和 import 路径更新，不修改任何数据模型、表结构、或序列化格式。

## Affected Configuration

### `.cursor/hooks.json`

Hook 路径需从旧模块路径更新为新路径：

| 事件 | 旧路径 | 新路径 |
|------|--------|--------|
| afterAgentResponse | `team_memory.hooks.cursor_after_response` | `team_memory.hooks.platforms.cursor.after_response` |
| sessionStart | `team_memory.hooks.cursor_session_start` | `team_memory.hooks.platforms.cursor.session_start` |
| beforeSubmitPrompt | `team_memory.hooks.cursor_before_prompt` | `team_memory.hooks.platforms.cursor.before_prompt` |
| afterMCPExecution | `team_memory.hooks.cursor_after_mcp` | `team_memory.hooks.platforms.cursor.after_mcp` |

**注意**: 旧路径通过 re-export shim 仍有效（过渡期），但 hooks.json 应直接指向新路径。

### `harness_import_check.py` LAYER_MAP

**无需更新**。子包仍匹配现有前缀规则：
- `src/team_memory/services/search/` → 匹配 `src/team_memory/services/` → L2
- `src/team_memory/embedding/factory.py` → 匹配 `src/team_memory/embedding/` → L2

### Re-export Shim 清单

| 旧路径 | 新路径 | 消费者 |
|--------|--------|--------|
| `hooks/claude_session_start` | `hooks/platforms/claude/session_start` | Claude Code hooks.json |
| `hooks/cursor_*` (4 个) | `hooks/platforms/cursor/*` | .cursor/hooks.json |
| `hooks/hermes_*` (4 个) | `hooks/platforms/hermes/*` | Hermes 配置 |
| `daemon/<flat_module>` | `daemon/<subpkg>/<module>` | tests/, scripts/, cron |
| `services/<flat_module>` | `services/<subpkg>/<module>` | server.py, web/, tests/ |
| `bootstrap._create_embedding_provider` | `embedding.factory.create_embedding_provider` | daemon/, tests/ |

# Changelog

本仓库变更记录；**发版时**请将 `[Unreleased]` 下沉为带日期的版本号，并与 `pyproject.toml` 中 `version` 对齐。

格式参考 [Keep a Changelog](https://keepachangelog.com/zh-CN/1.1.0/)。

## [Unreleased]

## [0.1.3] — 2026-04-07

### Security

- **`tm_get_archive` / `ArchiveService.get_archive`**：按与向量检索一致的规则校验可见性（创建者或 `published` 且同 `project`）；**越权 ID 不再返回正文**（视为未找到）。

### Added

- **Lite MCP**：`memory_get_archive`；`memory_save(scope=archive)` 支持 `conversation_summary`、`attachments`、`archive_record_scope`、`archive_scope_ref`；`memory_recall` 命中档案时返回 **`overview_preview`**（L1）。
- **Web**：`GET /api/v1/archives`、`GET /api/v1/archives/{id}`；SPA **档案馆**页（`#archives`）与「搜索 / MCP」语义路径说明。

### Changed（破坏性 · CLI / 本地开发）

- **`make mcp`**：经 **`scripts/run_mcp_with_dotenv.sh`** 启动 MCP（需仓库根 **`.env`**；内部 **`python -m team_memory.server`**，**`memory_*`** 六工具；附件走 HTTP 或 **`python -m team_memory.cli upload`**，见 [mcp-server.md](docs/guide/mcp-server.md)）。
- **`team_memory.server_lite`**、**`make mcp-full`**、**`team-memory-full`**：**已移除**；统一入口为 **`team_memory.server`** / **`team-memory`**。
- **Console `team-memory`**：等同 **`python -m team_memory.server`**。

### Fixed

- **`tm-cli` / `python -m team_memory.cli archive`**：解析 **`POST /api/v1/archives`** 响应中的 **`item`** 包裹层，正确输出 `archive_id`（与 Web API 一致）。
- **测试**：`test_search_uses_default_project` 在无请求 `project` 时使用 **`monkeypatch.delenv("TEAM_MEMORY_PROJECT")`**，避免本机/CI 环境变量覆盖 `default_project` 断言。

### Changed（Lite MCP JSON）

- **`memory_context`**（及 `memory_recall(..., include_user_profile=True)`）返回的 **`profile`** 由对象列表改为 **`{ "static": string[], "dynamic": string[] }`**（对齐 Supermemory 形）。集成方若解析旧格式需更新。
- **数据库**：`personal_memories.profile_kind`（`static` / `dynamic`），迁移 **`004_profile_kind`**；存量 `scope=context` 行回填为 `dynamic`，其余 `static`。
- **配置**：`mcp.profile_max_strings_per_side`（默认 20），环境变量 `TEAM_MEMORY_MCP__PROFILE_MAX_STRINGS_PER_SIDE`。

### Documentation

- README、`docs/guide/mcp-server.md`：MCP 示例与方案 A（`.env` + 包装脚本）；决策见 [docs/decision/mcp-lite-default.md](docs/decision/mcp-lite-default.md)。

---

## [0.1.2] — 此前版本

未在此文件维护历史条目；自 **0.1.3** 起按上表累积。

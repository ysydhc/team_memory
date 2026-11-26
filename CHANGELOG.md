# Changelog

本仓库变更记录；**发版时**请将 `[Unreleased]` 下沉为带日期的版本号，并与 `pyproject.toml` 中 `version` 对齐。

格式参考 [Keep a Changelog](https://keepachangelog.com/zh-CN/1.1.0/)。

## [Unreleased]

## [0.3.8] — 2026-11-26

### Added

- **多租户项目别名隔离与去硬编码**：
  - 清理 `project_resolver.py` 中 `_BUILTIN_SHORTHAND_ALIASES` 的个人/业务项目硬编码，仅保留 TeamMemory 自身必要别名。
  - 支持从租户级配置 `tenants.settings_json["project_aliases"]`（服务端启动预加载 + `PUT /api/v1/tenants/current` 实时刷新）与客户端 `~/.config/tm/config.toml` `[project_aliases]`（`ClientConfig` 完整支持）加载别名，并按 `tenant_id` 严格隔离。
- **OpenAI Codex CLI 平台采集与接入**：
  - 新增 `hooks/core/transcript.py` 的 `parse_codex_transcript` / `resolve_codex_transcript_path` 解析 `~/.codex/sessions/**/rollout-*.jsonl`。
  - 新增 `hooks/platforms/codex/`（`session_start`、`user_prompt_submit`、`post_tool_use`、`stop`）与 `daemon/observe/codex_transcript_watcher.py`（支持 `tm-cli import --source codex`）。
  - 补齐 `tm-cli setup/check/uninstall --platform codex`（幂等读写 `~/.codex/hooks.json` 并保留第三方 Hook）与 Web v2「团队与接入」页 Codex CLI MCP 配置模板。

### Fixed

- 修复 `config/settings.py` 与 `client_config.py` 在受限目录下读取 `~/.config/tm/` 抛 `PermissionError` 以及 `.env` 错误使用 `override=True` 覆盖环境变量的问题。
- 修复 SQLite 测试下 PostgreSQL `UUID` 列因 NUMERIC 亲和力将纯数字 `DEFAULT_TENANT_ID` 转为 `float` 的问题（`@compiles(UUID, "sqlite") -> CHAR(32)`）。
- 修复 `server.py` HTTP MCP 上下文解析非 UUID `mcp_tenant_id` 导致阻断用户识别的问题，并补齐批量上下文加载、检索指标与重试单测。

## [0.3.0] — 2026-11-25

### Added

- **Multi-Tenant Hard Isolation (PostgreSQL RLS)**：
  - 新增 Alembic 迁移 `023_multitenant_rls`：建表 `tenants` / `tenant_members`，为 13 张核心表增加 `tenant_id NOT NULL` 并平滑回填至 `default` 租户（100% 向后兼容存量 API Key 与 MCP）。
  - 创建受限角色 `tm_app_user`（`NOLOGIN NOBYPASSRLS`），在 `get_session()` 事务内通过 `SET LOCAL ROLE tm_app_user` + `set_config('app.current_tenant_id', ...)` 强制启用 12 张业务表 PostgreSQL `FORCE ROW LEVEL SECURITY`。
  - 新增 `/api/v1/tenants/*` 租户切换、成员 RBAC 管理（`admin / member / viewer`）与每租户个人 API Key 管理接口；`SearchCache` 缓存键注入 `tenant_id` 防止跨租户缓存污染。
- **Web v2 极简人类控制台 (`web-ui/`)**：
  - 采用 Vite + React 19 + TypeScript + Tailwind CSS v4 重构前端，构建产物直出 `src/team_memory/web/static/dist/` 并由 FastAPI `:9111` 单进程静态托管（新增 `make web-build` / `make web-dev`）。
  - 交付左上角多租户 Org Switcher、全局 `⌘K / Ctrl+K` 语义搜索弹窗、统一「记忆库」工作台（经验/待审草稿/会话归档/我的画像 + Debug 4 路召回得分 + 560px Markdown 右侧详情抽屉）、10 秒接入页（Antigravity / Claude Code / Cursor / `tm-cli` 一键复制 + 连通性测试 + Key 轮换 + Admin 成员/重置密码）、「驾驶舱」（5 大子系统健康灯 + 30d/180d 线性/平方根折线图 + Golden Eval 消融摘要）与 Admin 二级「系统运维」页。
  - 新增 `POST /api/v1/experiences/{id}/supersede` 双时态废弃接口。

### Removed

- 下线旧版 4,300 行单文件前端（`src/team_memory/web/static/index.html` 与 `src/team_memory/web/static/js/*`）。

---

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

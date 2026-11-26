# Changelog

本仓库变更记录；**发版时**请将 `[Unreleased]` 下沉为带日期的版本号，并与 `pyproject.toml` 中 `version` 对齐。

格式参考 [Keep a Changelog](https://keepachangelog.com/zh-CN/1.1.0/)。

## [Unreleased]

## [0.4.2] — 2026-11-26

### Added

- **Coding Agent 逐条归因评分体系与 5 道稳定性护城河**：
  - 在 `src/team_memory/daemon/quality/faithfulness_judge.py` 新增 `AttributionJudge`（兼容 `FaithfulnessJudge`），废弃整篇回复 RAGAS 分母，改为逐条输出 `relevance / adopted / evidence_quote / contradicted`，并内置代码级 **Evidence-Required** 硬门禁（无 ≥4 字证据片段强降级 `adopted=False`）与 `usage_ledger` 每日预算熔断（默认 200 次/天）。
  - 修复 `src/team_memory/services/analysis/implicit_feedback.py` 与 `response_buffer.py` 在 PostgreSQL 强制 RLS 下的回写通道，实现 `search_logs.relevance_score / was_used`、`experience_feedbacks`、`experiences.implicit_positive_count / implicit_negative_count` 真实闭环回写；内置单日 ±2 限幅、`verified / is_pinned` 金标豁免与 `contradicted → possibly_stale` 自动标旧。
  - 移除 `hooks.py` 中 `subprocess.Popen(..., stderr=DEVNULL)` 黑洞子进程，升级为 Daemon 常驻异步评估 Worker（重试 ≤3 次 + 完整结构化日志）。
  - 新增 20 题裁判校准集 `evals/judge_calibration_set.json`、`scripts/calibrate_judge.py` 与 `make calibrate-judge`（Precision ≥ 85% 门禁，实测 `gbu-gemini-3.8-flash` Adopted Precision/Recall = 100%）。
  - 在 `src/team_memory/services/analysis/observability.py` 新增第 6 子系统 `scoring` 与 3 条不可造假自检红线（防空跑红灯、防判偏黄灯），完整接入 `/api/observability/overview` 与 `make ops-digest`。

## [0.4.1] — 2026-11-26

### Fixed

- **PostgreSQL + asyncpg `setweight(tsvector, varchar)` 类型不匹配热修 (`src/team_memory/storage/repository.py`)**：
  - 将 `_build_fts_tsvector_expr()` 中的权重参数由绑定参数改为 `literal_column("'A'")` / `"'B'"` / `"'C'"`，修复 `asyncpg` 下写入经验时报 `function setweight(tsvector, character varying) does not exist` 的 P0 缺陷。

## [0.4.0] — 2026-11-26

### Added

- **多租户安全 3 件套（P0）**：
  - 新增 `src/team_memory/utils/secret_redactor.py`：在 `memory_save`、`draft_save`、`archive_upsert` 等写入入口提供 0ms 凭证脱敏（覆盖 17 类常见 AK/SK、Token、JWT、私钥、Basic Auth URL 等，含 9 类零误伤边界）。
  - 新增 `src/team_memory/utils/prompt_guard.py`：在 MCP 召回返回体标注不可信参考数据边界并对 8 类 Prompt Injection 模式检出打标。
  - 新增 `src/team_memory/web/rate_limiter.py` 并升级 `web/app.py` 限流器为 Redis 按 `tenant_id / api_key` 滑动窗口限流（含防泄漏内存降级与 `Retry-After`）。
- **pgvector 4096 维二值量化 HNSW 索引与两阶段精排（P1）**：
  - 新增迁移 `028_binary_quantize_hnsw_indexes.py`，为 `experiences`、`archives`、`personal_memories` 建立 `binary_quantize(embedding)::bit(4096)` HNSW 部分表达式索引，检索启用 `hnsw.iterative_scan = relaxed_order` + 原始 4096 维余弦精确重排（原生 PG 实测 `2.201 ms` vs `15.422 ms`，提速 7.0x）。
- **精炼降损与多租户 LLM/Embedding 成本账本（P1）**：
  - 强化 `draft_refiner` / `refinement_worker` 结构化格式校验与轻量自动补全修复，区分噪音丢弃与格式丢弃。
  - 新增迁移 `029_llm_usage_logs.py`（带 `tenant_id` 强制 RLS）与 `src/team_memory/services/analysis/usage_ledger.py`，在统一 LLM/Embedding 层异步记录调用与 Token 估算，接入 `/api/observability/overview` 与 Janitor 90 天 TTL 清理。
- **本地 Cross-Encoder Reranker 适配与 Golden Eval 四组消融（P2）**：
  - `cross_encoder_provider.py` 支持 Apple Silicon `mps` 自动探测、进程单例懒加载与严格超时透传降级；`golden_eval.py` 新增第 4 组 `D (Full + Reranker)` 与 `Rerank Lift` 指标计算。

## [0.3.12] — 2026-11-26

### Added

- **本机原生定时自动灾备与可验证恢复体系（零 Docker 依赖）**：
  - 升级 `scripts/backup.sh`：自动读取 `.env` 与 `team_memory.config.load_settings()` 解析原生 PostgreSQL 连接参数，调用本机 `pg_dump --format=custom --compress=6` 备份完整数据库（含 schema、RLS 策略、pgvector 数据），并通过 `sqlite3 -init /dev/null -batch ... ".backup"` 热备 `~/.cache/tm-pipeline/drafts.db` 及 `PRAGMA integrity_check` 校验；生成 `manifest_<TS>.json`、`SHA256SUMS_<TS>.txt` 与 `backups/LATEST` 指针，支持 14 天日备（`backups/daily/`）+ 4 周周备（`backups/weekly/`）分层自动清理。
  - 升级 `scripts/restore.sh`：支持 `.dump`（`pg_restore`）并兼容 `.sql.gz`，恢复前自动验证 `SHA256SUMS`；新增 `--verify-only [backup_file]` 恢复演练模式（自动创建临时库导入 → 校验 `experiences` / `archives` / `entities` / `tenants` 核心表非零与 `pgvector`/RLS → 自动清理临时库），并使用 `psql -X` 屏蔽 `~/.psqlrc` 干扰。
  - 新增 `scripts/com.teammemory.backup.plist` 与 `Makefile` `backup-install` / `backup-uninstall` / `backup-verify` 目标（macOS `launchd` 每日 `03:30` 自动执行）。
  - 将「最近备份时间是否超过 48h」纳入 `scripts/healthcheck.sh` 与 `src/team_memory/services/analysis/observability.py`（`ops-digest` `infra` 子系统巡检与告警），补充 `tests/test_observability_and_eval.py` 单测，并在 `.gitignore` 忽略 `backups/`。

## [0.3.11] — 2026-11-26

### Added

- **本机原生 Homebrew PostgreSQL 16 (`:5432`) 免 Docker 双模式切换**：
  - `Makefile` 新增 `mode-native`、`mode-docker`、`mode-status` 目标，`make dev` 默认自动拉起本机原生 PostgreSQL (`:5432`)，零 Docker 内存开销。
  - `src/team_memory/storage/database.py` 新增 `_normalize_local_pg_port()`：当 `:5433`（Docker PG）未开启而 `:5432`（原生 PG）在线时自动路由，无缝切换。
  - `src/team_memory/bootstrap.py` 支持 `.env` `override=True` 与 `TEAM_MEMORY_DATABASE__URL` 环境变量优先解析。
  - `alembic.ini`、`config.development.yaml`、`scripts/healthcheck.sh` 默认对齐原生 PG `:5432` 并兼容 Homebrew `pg_isready`。

### Changed

- 更新 `evals/history.jsonl` 最新 Golden Eval 评测记录、`.cursor/hooks.json` Python 路径与 `uv.lock`。

## [0.3.10] — 2026-11-26

### Added

- **中英文技术标识符分词器升级 (`src/team_memory/utils/tokenizer.py`)**：
  - 零外部依赖重构分词管线（`jieba` 与 `_tokenize_simple` 共用），对英文/混合技术标识符采用**「保留小写原词 + 追加拆分词元」**策略：支持标准驼峰（`AdLifecycleMonitor`）、缩写词+驼峰（`ADXManager` / `HTTPServer`）、蛇形与扩展名（`use_worktree.sh`）、连字符错误码（`adx-sdk-518`）、版本号（`v0.3.5 → v0.3.5 0.3.5`）、字母数字边界（`Android16`）与路径分段；过滤单字符符号并保序去重。
  - 新增 `tests/test_tokenizer.py` 15 组技术标识符矩阵、中英混排、fallback 与 FTS 读写对齐单测。
- **存量 FTS 回填脚本与触发器迁移**：
  - 新增 Alembic 迁移 `027_fts_trigger_preserve_tokenized.py`，更新 PostgreSQL `experiences_fts_update()` 触发器以保留应用层分词后的 `NEW.fts`（直连 SQL 写入仍保留兜底逻辑）。
  - 新增 `scripts/migrate_fts_jieba.py`（支持 `--dry-run`、`--batch-size`、`--active-only` 与 `bypass_rls=True` 全租户覆盖）并更新 `Makefile` `migrate-fts-jieba` 入口。

### Fixed

- **FTS 写入与查询分词一致性 (`src/team_memory/storage/repository.py`)**：
  - `ExperienceRepository.create()` / `update()` 在 PostgreSQL 方言下通过 `_build_fts_tsvector_expr()` 使用统一 `tokenize()` 生成 A/B/C 加权 `tsvector`，并移除 `search_by_fts()` 中冗余的 `snake_tokens` 临时补丁，解决局部关键词（`Lifecycle` / `worktree` / `518` / `0.3.5`）FTS 召回盲区。

## [0.3.9] — 2026-11-26

### Fixed

- **Git Worktree 项目名误识别孤岛修复 (`project_resolver.py`)**：
  - 重构 `_git_repo_name(cwd)`：优先解析 `git rev-parse --path-format=absolute --git-common-dir`（兼容旧版 Git 自动回退 `--git-common-dir` + 绝对路径解析），当末级为 `.git` 时取主仓库目录名，避免 Worktree 临时目录名被误识别为独立 `project`。
  - 增加 bare repo / submodule 场景下的 `git remote get-url origin` 与 bare 目录名回退，最终兜底 `--show-toplevel`，保持 `.tm.toml` > Git 主仓名 > 目录名优先级不变。
  - 新增 `tests/test_project_resolver_worktree.py` 真实 `git init` + `git worktree add` 回归单测。

### Added

- **历史 Worktree 孤岛项目治理脚本 (`scripts/merge_worktree_projects.py`)**：
  - 支持 `--dry-run` 预览与 `--apply` 单事务安全归并，覆盖 `experiences`、`archives`、`entities`、`search_logs`、`personal_memories` 五张表。
  - 自动处理 `entities` 表 `(tenant_id, name, project)` 唯一约束冲突，合并重复实体的 `experience_entities`、`relationships` 出入边与 `evidence` 并消除自环边。

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

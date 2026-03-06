# 三项任务扩展详情（待确认后执行）

本批任务：**P1-6**、**P1-5**、**P2-3**。执行顺序：先 P1-6，再 P1-5，最后 P2-3；每完成一项经你确认后再进行下一项。

---

## 任务 1：P1-6 补齐存量迁移 Makefile 与 migrate_fts 脚本

**任务 ID**: `e20485bf-b2c0-418d-a45a-79e0a8369f71`

### 目标
为存量数据补齐 FTS（全文检索）字段，并统一通过 Makefile 与脚本入口做迁移，方便部署与运维执行。

### 背景
- `Experience.fts` 为 TSVECTOR 列（nullable），用于 `search_by_fts`；当前 `repository.create()` 未写入 `fts`，存量或早期数据可能为 `fts IS NULL`，导致全文检索无法命中。
- 需与现有检索一致：使用 `team_memory.services.tokenizer.tokenize` 做分词，再用 PostgreSQL `to_tsvector('simple', ...)` 生成向量；索引文本建议包含 title、description、solution、root_cause、code_snippets 等可检索字段。

### 执行步骤
1. **新增脚本 `scripts/migrate_fts.py`**
   - 从配置加载 `database.url`，连接 DB。
   - 查询 `Experience` 中 `is_deleted = false` 且 `fts IS NULL` 的记录（可加 `limit`/分批，避免一次加载过多）。
   - 对每条记录：将 title、description、solution、root_cause、code_snippets 等拼成一段文本，用 `tokenize(text)` 得到分词串，再执行 `UPDATE experiences SET fts = to_tsvector('simple', $1) WHERE id = $2`（或使用 SQLAlchemy 的 `text()`/`func.to_tsvector` 更新）。
   - 支持 `--dry-run`（只统计待更新条数、不写库）、`--batch-size N`、可选 `--limit N`。
   - 脚本内可复用 `tokenize` 与 DB session 方式，与 `scripts/migrate_embeddings.py` 风格保持一致。

2. **Makefile**
   - 新增目标 `migrate-fts`：执行 `python scripts/migrate_fts.py`（可带 `--dry-run` 的说明在 help 中）。
   - 在 `help` 的列表中加入 `migrate-fts` 的简短说明（例如：「补齐经验表 FTS 字段（存量迁移）」）。

3. **文档**
   - 在 `README.md` 的「数据库 / 迁移」或「运维」相关小节中，增加一行说明：存量库若需全文检索，可执行 `make migrate-fts` 或 `python scripts/migrate_fts.py`；并注明 `--dry-run` 用于预览。

### 改动文件（预期）
- 新建：`scripts/migrate_fts.py`
- 修改：`Makefile`（新增 `migrate-fts` 目标及 help）
- 修改：`README.md`（补充 migrate_fts 使用说明）

### 验收标准
- `ruff check src/` 通过；若脚本在 `scripts/` 且被 ruff 覆盖，则脚本风格符合项目要求。
- 在本地或测试库执行 `python scripts/migrate_fts.py --dry-run` 不报错且能统计到待更新数量（或 0）；执行一次不带 `--dry-run` 后，对应行的 `fts` 非 NULL。
- `make migrate-fts` 可执行；`make help` 中能看到 `migrate-fts`。

---

## 任务 2：P1-5 实现 LLM 查询扩展 fallback

**任务 ID**: `4a758f85-ae2f-4a21-8830-19e1903893e4`

### 目标
在搜索管线中增加「LLM 查询扩展」步骤：用 LLM 对用户 query 做关键词扩展或改写，以提升召回；当 LLM 不可用或超时时，回退到当前行为（仅同义词扩展或原 query），不阻塞搜索。

### 背景
- 当前 `SearchPipeline` 中已有 `_expand_query_synonyms(request.query, self._tag_synonyms)` 得到 `retrieval_query`，且已有 `llm_config` 注入。
- 需要新增可选阶段：在 synonym 扩展之前或之后，若配置启用且 LLM 可用，则调用 LLM 将原始 query 扩展为更利于检索的 query（如补充同义关键词、技术术语展开）；失败或超时则 fallback 到仅 synonym 扩展或原 query。

### 执行步骤
1. **配置**
   - 在 `config.yaml` 的 search 相关节点下（或新建 `search.llm_expansion`）增加开关与超时，例如：`query_expansion_enabled: false`、`query_expansion_timeout_seconds: 3`。若暂无独立节点，可放在现有 `search` 下。
   - 在 `config.py` 的 `SearchConfig`（或对应 Pydantic 模型）中增加字段：`query_expansion_enabled: bool = False`、`query_expansion_timeout_seconds: float = 3.0`。

2. **SearchPipeline**
   - 在 `search()` 中，在得到 `retrieval_query = _expand_query_synonyms(...)` 之前或之后，增加一步：
     - 若 `query_expansion_enabled` 且 `self._llm_config` 可用，则异步调用 LLM（prompt 如：「根据以下用户搜索意图，输出一行仅包含扩展后的搜索关键词，用空格分隔，不要解释」；输入为 `request.query`），设置 `query_expansion_timeout_seconds` 超时。
     - 若 LLM 返回有效字符串，则用该字符串与 `_expand_query_synonyms` 的结果合并或替代（策略可简化为：LLM 扩展结果 + 原 query 或 synonym 扩展结果，避免丢失原意）。
     - 若未启用、超时或异常，则 `retrieval_query` 保持为当前逻辑（仅 synonym 扩展或原 query）。
   - 缓存 key 仍使用 `request.query`（或现有 cache key 逻辑），不因 LLM 扩展结果改变 cache key，避免同一用户 query 因 LLM 结果不同而无法命中缓存。

3. **调用链**
   - 确保 ExperienceService 构建 SearchPipeline 时传入的 `llm_config` 与配置一致；若已有则仅增加上述配置项读取。

### 改动文件（预期）
- `config.yaml` — 增加 query_expansion 相关配置项
- `src/team_memory/config.py` — SearchConfig 增加 query_expansion_enabled、query_expansion_timeout_seconds
- `src/team_memory/services/search_pipeline.py` — 增加 LLM 查询扩展步骤与 fallback
- 若 LLM 调用需统一入口，可复用现有 LLM 客户端（如 llm_parser 或 config 中的 ollama/openai），避免重复造轮子

### 验收标准
- 配置 `query_expansion_enabled: false` 时，行为与当前一致（无 LLM 调用）。
- 配置为 `true` 且 LLM 可用时，检索使用的 query 能体现 LLM 扩展结果（可通过日志或单测 mock LLM 验证）。
- LLM 超时或异常时，搜索仍返回结果（fallback 到未扩展或仅 synonym 的 query）。
- `ruff check src/` 通过；现有搜索相关单测（如 `test_search_pipeline.py`）通过，必要时补充一条「LLM 不可用时 fallback」的测试。

---

## 任务 3：P2-3 日志与审计

**任务 ID**: `bc102e37-f23d-46ae-bce2-63b4f77a7b91`

### 目标
实现结构化请求日志与敏感操作审计：请求级日志（IP、用户、路径、耗时）便于排查与监控；敏感操作写入审计表，便于安全与合规追溯。

### 背景
- 项目已有 `AuditLog` 模型与 `GET /api/v1/.../audit-logs` 列表接口，但当前代码中未见对删除经验、修改经验、API Key 管理等敏感操作写入 `AuditLog`。
- 已有 `metrics_middleware` 记录请求与延迟，可在此基础上增加「请求日志」中间件，输出结构化日志（如 JSON 一行一条），包含：IP、method、path、status、duration_ms、user（若已认证）。

### 执行步骤
1. **请求日志中间件（结构化日志）**
   - 在 `app.py` 中新增一个 HTTP 中间件（可在 metrics_middleware 之后注册），在 `call_next` 前后计时，从 `request.client.host`（或 X-Forwarded-For）、`request.url.path`、`request.method`、`response.status_code`、duration 获取信息；若当前请求已解析出用户（如 API Key 对应用户名），一并写入。
   - 日志格式：推荐单行 JSON（如 `{"event":"request","path":"/api/v1/...","method":"GET","status":200,"duration_ms":12,"ip":"127.0.0.1","user":"optional"}`），便于 grep/ELK/Loki。使用 Python 标准 `logging` 即可，logger 名称如 `team_memory.web.request`，级别 INFO。
   - 可选：对 `/health`、`/metrics` 等高频、低价值路径做降噪（不记录或仅记录为 debug）。

2. **敏感操作写入 AuditLog**
   - 定义少量「写审计」辅助函数或 repository 方法，例如：`write_audit_log(session, user_name, action, target_type, target_id, detail=None, ip_address=None)`，向 `AuditLog` 表 insert 一条。
   - 在以下敏感操作完成后调用（在事务提交前或后均可，视是否与主业务同事务而定）：
     - **经验删除**：软删或硬删经验时，`action="delete"`，`target_type="experience"`，`target_id=experience_id`，`detail` 可含 title 或仅 id。
     - **经验重要修改**：如发布/驳回、状态变更、或「更新经验」的 API（若需审计更新，可先做「发布/驳回」与「删除」两类）。
     - **API Key 管理**：若存在创建/删除/禁用 API Key 的接口，则 `action="create"|"delete"|"disable"`，`target_type="api_key"`，`target_id=key_id 或 name`。
   - 从请求上下文获取 `user_name`（如当前 API Key 对应用户）、`ip_address`（request.client.host 或 X-Forwarded-For），传入审计写入。

3. **文档与配置**
   - 在 README 或 `.debug/` 中简短说明：请求日志为 JSON 行、字段含义；审计日志可通过 `GET /api/v1/.../audit-logs` 查询（已有则只补充说明）。
   - 可选：在 `config.yaml` 中增加 `logging.request_log_enabled: true`、`logging.audit_sensitive_ops: true`，便于关闭请求日志或审计写入（默认 true）。

### 改动文件（预期）
- `src/team_memory/web/app.py` — 请求日志中间件；若在路由层获取 user/ip，可能需在 middleware 或路由中传递上下文。
- `src/team_memory/storage/repository.py` 或 `src/team_memory/services/analytics.py` — 新增 `write_audit_log` 或等价方法；或单独 `audit.py` 模块。
- 经验删除/发布/驳回的调用处（如 experience service 或 web routes）— 调用审计写入。
- API Key 管理相关路由（若有）— 调用审计写入。
- `README.md` 或 `.debug/` — 日志与审计说明；可选 `config.yaml` + `config.py` 的 logging 开关。

### 验收标准
- 任意 API 请求后，日志中能看到一条 JSON 格式的 request 日志（含 path、method、status、duration_ms、ip）。
- 执行一次「删除经验」或「发布/驳回」后，`GET /api/v1/.../audit-logs` 能查到对应记录（action、target_type、target_id、user_name、ip 合理）。
- `ruff check src/` 通过；`pytest tests/test_web.py -v` 通过；若新增 audit 写入，可补充单测验证写入内容。

---

## 执行顺序与确认

1. **P1-6**（migrate_fts + Makefile）— 先做，无业务逻辑风险，仅脚本与文档。
2. **P1-5**（LLM 查询扩展 fallback）— 再做，需配置与单测验证 fallback。
3. **P2-3**（日志与审计）— 最后做，涉及中间件与多处调用审计写入。

每完成一项，我会汇报改动点与验收结果，等你确认后再进行下一项。

确认无误后请回复「可以开始执行」；执行将从 **P1-6** 开始。

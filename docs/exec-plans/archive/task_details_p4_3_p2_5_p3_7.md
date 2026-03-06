# 三项任务扩展详情（待确认后执行）

本批任务：**P4-3 CI/CD 流水线**、**P2-5 工作流模板**、**P3-7 详细使用日志**。执行顺序建议：P4-3 → P2-5 → P3-7；每完成一项经确认后再进行下一项。  
详细步骤见下文，确认无误后再开始执行。

---

## 任务 1：P4-3 CI/CD 流水线

**任务 ID**: `37b92230-a317-4be4-b8bb-e498eb896eef`

### 目标
完善 CI/CD：在现有 GitHub Actions 基础上查漏补缺，确保每次推送/PR 都跑 lint、测试、可选 Docker 构建；可选增加版本号/发布相关配置。

### 背景
- 项目已有 `.github/workflows/ci.yml`：含 lint（Ruff）、test（pytest + PostgreSQL service）、Web smoke、Docker build（push 时）。
- 需确认：分支/触发条件是否覆盖需求、是否缺少步骤（如 alembic 迁移、敏感信息扫描）、版本号管理是否需在 CI 中体现。

### 执行步骤
1. **审查并补全现有 CI**
   - 确认 `push`/`pull_request` 触发分支（当前为 main、develop 与 main）。
   - 确认 test job 中 `alembic upgrade head` 与 `pytest`、Web smoke 顺序与依赖正确。
   - 如需：在 push 到 main 时增加「敏感信息扫描」步骤（与固化规则「推送 GitHub 前敏感信息清理」对齐），或仅文档说明在本地执行。
2. **Docker 构建与版本**
   - 当前 Docker build 使用 `team_memory:${{ github.sha }}`；可选增加 tag 如 `latest`（仅 main）、或从 tag 打版本镜像。
   - 若需版本号管理：可在 CI 中读取 `git describe --tags` 或 `pyproject.toml` 的 version，写入镜像 tag 或构建产物；具体按团队习惯最小化实现。
3. **文档**
   - 在 README 或 `.debug/` 中说明：CI 在哪些分支/事件触发、包含哪些 job、如何查看结果；若存在敏感信息检查则说明在 PR 前本地执行方式。

### 改动文件（预期）
- `.github/workflows/ci.yml` — 按需增加步骤或 job（如敏感信息扫描、版本 tag）。
- `README.md` 或 `.debug/` — CI 说明与可选版本/发布说明。

### 验收标准
- 提交 PR 或 push 到目标分支后，GitHub Actions 中对应 workflow 运行成功（lint + test + 可选 Docker）。
- 文档中能明确看到触发条件与主要步骤说明。

---

## 任务 2：P2-5 工作流模板

**任务 ID**: `00488056-13bd-465f-a728-078f67f50685`

### 目标
预定义经验录入模板（如 Bug 修复、架构决策、故障排查、技术方案），在 Web 创建经验时可选模板并预填推荐字段/标签；MCP `tm_learn` 支持 `--template` 参数以应用模板提示。

### 背景
- 已有 `GET /api/v1/templates`：从 `config/templates/templates.yaml` 或 schema 自动生成模板列表；前端有「使用统计」等页面。
- 需：在 Web「创建经验」流程中增加模板选择；创建后根据模板预填 experience_type、推荐 tags、hints；MCP 侧 tm_learn 支持传入 template id 或 name。

### 执行步骤
1. **模板配置**
   - 确认或补充 `config/templates/templates.yaml`（若路径不同则以现有路由为准）：至少包含 bugfix、tech_design、incident、best_practice 等类型的模板定义（name、id、core_fields、suggested_tags、hints）。
   - 与现有 schema 的 experience_type、severity、category 等对齐，避免前端展示与后端校验不一致。
2. **Web 创建经验**
   - 在创建经验页（或弹窗）增加「选择模板」下拉或卡片；选择后预填：experience_type、tags（suggested_tags）、标题/问题/解决方案的 placeholder 或 hint。
   - 提交时仍走现有 `POST /api/v1/experiences`，仅请求体带上模板推荐的默认值。
3. **MCP tm_learn**
   - 在 `tm_learn` 工具中增加可选参数 `template: str | None = None`（模板 id 或 name）；若传入，则在调用解析/保存逻辑时带上模板对应的 type、推荐 tags 或 hints（具体以现有 llm_parser/experience 创建接口为准）。
4. **文档**
   - README 或 `.debug/` 中简短说明：如何配置模板、Web 与 MCP 如何使用模板。

### 改动文件（预期）
- `config/templates/templates.yaml` 或项目内模板配置路径 — 预定义模板内容。
- Web 前端：创建经验相关页面/组件 — 模板选择与预填。
- `src/team_memory/server.py` — tm_learn 增加 `template` 参数并向下传递。
- 文档 — 模板配置与使用说明。

### 验收标准
- `GET /api/v1/templates` 返回包含预定义类型的模板列表。
- Web 创建经验时可选模板，选择后表单有预填或 placeholder。
- `tm_learn` 支持 `--template`（或等价参数），传入后创建的经验带有对应 type/tags（或可观测到行为变化）。
- `ruff check src/` 与相关单测通过。

---

## 任务 3：P3-7 详细使用日志

**任务 ID**: `b75fb698-1552-4b7f-b1ea-1faa9d0c5c1c`

### 目标
在现有 `tool_usage_logs` 基础上，实现「按 API Key + 时间」的详细追踪与统计，并在 Web「使用统计」中可查，便于按 Key/用户/时间分析 MCP 工具调用情况。

### 背景
- 已有 `ToolUsageLog` 表与 `UsageTrackingHandler`（hooks）：记录 tool_name、user、project、duration_ms、success、created_at 等。
- 已有 Web「使用统计」页与 `GET /api/v1/analytics/tool-usage`、`/tool-usage/summary`（按 user、按 tool、按日等）。
- 当前日志未关联「哪个 API Key」发起的调用；若需按 Key 统计，需在记录时写入 api_key 标识，并在统计接口中支持按 Key 筛选与聚合。

### 执行步骤
1. **数据模型与写入**
   - 在 `ToolUsageLog` 中增加可选字段 `api_key_id: int | None`（或 `api_key_name: str | None`），与 ApiKey 表关联或仅存名称；若涉及表结构变更需 Alembic 迁移。
   - 在 MCP 调用链中（如 server 中执行工具时）：若当前请求能解析出 api_key 对应用户/Key 标识，则写入 ToolUsageLog 时带上 api_key_id 或 api_key_name；若无法解析则保持 null。
2. **统计接口**
   - 在 `GET /api/v1/analytics/tool-usage` 或新增接口中，支持按 `api_key_id` 或 `api_key_name` 筛选；支持按时间范围（已有 cutoff 等）聚合。
   - 返回结构可包含：按 Key 的调用次数、按 Key+日的分布、按 Key+工具名的分布等，便于前端展示。
3. **Web 使用统计页**
   - 在使用统计页增加「按 API Key」的筛选或 Tab；展示各 Key 的调用量、成功率、常用工具等（数据来源为上述统计接口）。
4. **文档**
   - 在 `.debug/` 或 README 中说明：使用统计包含按 API Key 的维度、数据来源为 tool_usage_logs。

### 改动文件（预期）
- `src/team_memory/storage/models.py` — ToolUsageLog 增加 api_key_id 或 api_key_name（可选）；若需迁移则新增 Alembic 版本。
- MCP 调用处（如 server.py 或 hooks）— 写入 ToolUsageLog 时传入 api_key 标识。
- `src/team_memory/web/routes/analytics.py` — 统计接口支持按 api_key 筛选与聚合。
- Web 前端：使用统计页 — 按 API Key 的筛选与展示。
- 文档 — 使用统计说明。

### 验收标准
- 使用某 API Key 调用 MCP 工具后，对应 tool_usage_logs 行能关联到该 Key（或 user 可区分）。
- Web 使用统计页能按 API Key（或用户）查看调用量/趋势。
- `ruff check src/`、`pytest tests/` 通过；若有迁移则本地升级后表结构正确。

---

## 执行顺序与确认

1. **P4-3**（CI/CD）— 先做，无业务逻辑风险，主要为配置与文档。
2. **P2-5**（工作流模板）— 再做，涉及配置、Web、MCP 三处，需联调验证。
3. **P3-7**（详细使用日志）— 最后做，涉及表结构（可选）、写入、统计与前端。

确认本详情无误后即可按上述顺序开始执行；每完成一项汇报改动与验收结果，经确认后再进行下一项。

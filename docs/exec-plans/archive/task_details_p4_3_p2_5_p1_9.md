# 三项任务扩展详情（待确认后执行）

本批任务：**P4-3 CI/CD 流水线**、**P2-5 工作流模板**、**P1-9 提取配置化 quality_gate / max_retries**。执行顺序建议：P4-3 → P2-5 → P1-9；每完成一项经确认后再进行下一项。  
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

## 任务 3：P1-9 提取配置化 quality_gate / max_retries

**任务 ID**: `42aed7d8-08e4-4e01-9480-254412e12efe`

### 目标
将经验提取的质量门控与重试策略从代码默认值改为配置驱动，便于按环境调优。

### 背景
- 当前 `parse_content()` 中 quality_min_score、重试次数为硬编码，无法按环境（开发/生产）或实验需求调整。
- 需在 config 中增加 extraction 相关配置，并在 llm_parser 与调用方统一读取。

### 执行步骤
1. **config.yaml**
   - 在合适位置（如与 `llm` 同级）新增 `extraction` 节点，包含：
     - `quality_gate: 2`（最低质量分，低于此值触发重试）
     - `max_retries: 1`（质量不足时最多重试次数）
     - 可选：`few_shot_examples: null`（自定义 few-shot 路径，null 表示使用内置）
2. **config.py**
   - 新增 `ExtractionConfig`（Pydantic Model），字段：`quality_gate: int = 2`, `max_retries: int = 1`, `few_shot_examples: str | None = None`。
   - 在顶层配置（如 `Settings` / `AppConfig`）中增加 `extraction: ExtractionConfig`，并从 YAML 加载。
3. **llm_parser.py**
   - 在 `parse_content()` 的调用链中，将当前硬编码的 `quality_min_score=2`、`quality_retry_once=True` 改为从配置读取：
     - `quality_min_score` ← `settings.extraction.quality_gate`
     - 重试次数逻辑：`max_retries` 控制最多重试次数（当前实现为「最多重试 1 次」即总调用 2 次，与 `max_retries: 1` 一致）。
4. **调用方**
   - 确认所有调用 `parse_content` 的地方（如 `server.py` 的 tm_learn、Web 解析接口）使用统一配置入口，而非局部硬编码。

### 改动文件（预期）
- `config.yaml` — 新增 `extraction` 节点
- `src/team_memory/config.py` — 新增 `ExtractionConfig`，主配置增加 `extraction`
- `src/team_memory/services/llm_parser.py` — `parse_content` 从配置读取 quality_gate / max_retries
- 若 Web 或 server 有单独传参，改为从 `get_settings().extraction` 读取

### 验收标准
- 修改 `config.yaml` 中 `extraction.quality_gate` / `max_retries` 后，行为随之变化（无需改代码）。
- `ruff check src/` 通过；现有单测（如 test_llm_parser、test_service 中与 parse 相关）通过。

---

## 执行顺序与确认

1. **P4-3**（CI/CD）— 先做，无业务逻辑风险，主要为配置与文档。
2. **P2-5**（工作流模板）— 再做，涉及配置、Web、MCP 三处，需联调验证。
3. **P1-9**（提取配置化）— 最后做，影响提取行为，需跑单测验证。

确认本详情无误后即可按上述顺序开始执行；每完成一项汇报改动与验收结果，经确认后再进行下一项。

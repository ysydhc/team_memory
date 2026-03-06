# 三项任务扩展详情（待确认后执行）

## 任务 1：P1-10 定义并落档 Phase 4 成功标准

**任务 ID**: `de3a5174-91f9-46f3-9fae-84b9dfce3ff4`

### 目标
将「Phase 4 工具调用率提升」的成功标准正式写入文档，便于验收与复盘。

### 执行步骤
1. **确定落档位置**
   - 在 `.debug/` 下新增或更新文档（如 `.debug/11-phase4-success-criteria.md`），或
   - 在 `README.md` 的「验收 / 效果指标」相关章节增加一小节。
2. **写入内容**
   - 成功标准：**2 周内 tm_search 日均调用量较基线提升 >30%**（或等价表述：如「2 周内 MCP 搜索类工具日均调用提升 >30%」）。
   - 可选：基线定义（如取落档前 7 日日均）、统计口径（仅 MCP / 含 Web）、验收时间窗口。
3. **可选**
   - 在 `.cursor/rules/` 或 `.cursor/plans/` 中若有「Phase 4 验收」相关描述，与之对齐并引用新文档。

### 改动文件（预期）
- 新建：`.debug/11-phase4-success-criteria.md`，或
- 修改：`README.md`（新增「Phase 4 成功标准」小节）

### 验收标准
- 文档中明确写出「2 周内」「tm_search（或 MCP 搜索）」「日均调用」「提升 >30%」四要素。
- 新读者能据此判断 Phase 4 是否达成。

---

## 任务 2：P1-9 提取配置化 quality_gate / max_retries

**任务 ID**: `42aed7d8-08e4-4e01-9480-254412e12efe`

### 目标
将经验提取的质量门控与重试策略从代码默认值改为配置驱动，便于按环境调优。

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

## 任务 3：P2-2 Prometheus /metrics 端点

**任务 ID**: `60303d82-fe41-46d6-b162-b78883a4e0a0`

### 目标
暴露 Prometheus 格式的指标端点，便于监控、告警与容量规划。

### 执行步骤
1. **端点与路由**
   - 在 Web 应用（如 `src/team_memory/web/app.py`）中新增 **GET /metrics**（或 **GET /api/v1/metrics**，与现有 API 版本一致）。
   - 该端点建议无需认证（或可配置为可选认证），便于 Prometheus 拉取。
2. **指标内容（最小可行集）**
   - **HTTP**：请求数（counter）、请求延迟（histogram 或 summary）、错误数（counter）。
   - **搜索**：搜索请求数、缓存命中数（可推导缓存命中率）。
   - **Embedding 队列**：队列深度或待处理数（若已有队列状态接口可复用）。
   - **经验库**：经验总数（或按 project 分布）— 可从现有统计接口或 DB 查询聚合。
3. **实现方式**
   - 优先使用 **prometheus_client**（`pip install prometheus-client`）：在应用启动时注册 default registry，在请求路径中打点（counter.inc(), histogram.observe()），在 `/metrics` 中调用 `generate_latest()` 返回 text format。
   - 若暂不引入依赖：可手写 Prometheus 文本格式（按规范输出 `# TYPE`、`# HELP` 及指标行），在 `/metrics` 中返回 `text/plain; charset=utf-8`。
4. **配置与文档**
   - 可选：在 `config.yaml` 中增加 `metrics.enabled: true`、`metrics.path: /metrics`，便于关闭或改路径。
   - 在 README 或 `.debug/` 中说明 `/metrics` 用途及主要指标含义；若有 Grafana 模板可一并提供或留占位说明。

### 改动文件（预期）
- `src/team_memory/web/app.py` — 注册 `/metrics` 路由及指标收集逻辑；或在单独模块中实现 handler 再挂载。
- 可选：`src/team_memory/metrics.py`（新建）— 集中定义 counter/histogram 与打点辅助函数。
- `config.yaml` — 可选 `metrics` 节点。
- `README.md` 或 `.debug/` — 补充监控说明。

### 验收标准
- `GET /metrics` 返回 200，响应体为 Prometheus 文本格式（含 `# TYPE` 及至少请求数/延迟/搜索相关指标）。
- 若使用 prometheus_client，需在 pyproject.toml 或 requirements 中声明依赖；`ruff check src/` 及现有测试通过。

---

## 执行顺序建议
1. **P1-10**（纯文档）— 先做，无代码风险。
2. **P1-9**（配置化）— 再做，影响提取行为，需跑单测。
3. **P2-2**（/metrics）— 最后做，涉及新依赖与路由，验收时需请求一次 /metrics。

确认无误后即可按上述顺序开始执行。

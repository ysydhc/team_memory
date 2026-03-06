# Harness Phase 4 实施计划：可观测性 + 文档维护

> **前置**：Phase 0-1、Phase 2、Phase 3 已完成
> **Goal:** 日志 JSON 结构化、doc-gardening 过时文档扫描，提升可观测性与文档健康度。纯 Harness 优先。
> **评审**：已根据《Harness Phase 4 全维度评审报告》完成调整。
> **执行模式**：Subagent-Driven（每个 Task 派发 implementer subagent，主 Agent 验收）

**Architecture:** 日志统一 → doc-gardening 脚本 → 规则/文档衔接；经验库策略为 tm 项目可选子项。

---

## 一、Phase 4 范围（来自 Phase 3 预留）

| 方向 | 内容 | 纯 Harness | 说明 |
|------|------|------------|------|
| **可观测性** | 日志格式统一（JSON 结构化） | ✅ | 便于日志聚合、检索、告警 |
| **文档维护** | doc-gardening（定期扫描过时文档） | ✅ | 扫描 deprecated、过期引用、断裂链接 |
| **经验库策略** | 若启用 tm 时的策略 | 可选 | tm 项目单独排期，本计划不包含 |

---

## 二、step-0 摸底（强制，通过后才能进入 Task 1）

| 动作 | 命令/产出 |
|------|-----------|
| 统计 logger 数量 | `rg -l 'logging\.getLogger' src/ --type py \| wc -l` 或等价 |
| 确认 docs 结构 | 列出 `docs/design-docs`、`docs/exec-plans` 目录 |
| 基线 ruff/pytest | `ruff check src/`、`pytest tests/ -q` 通过 |
| 产出 | 基线报告（logger 数量、docs 路径、ruff/pytest 状态） |

**门控**：step-0 产出后，方可进入 Task 1；否则视为流程不规范，中断并提醒。

---

## 三、任务拆分

### Task 1：设计日志 JSON 格式规范

**Files:** `docs/design-docs/logging-format.md`

**内容：**

- 定义 JSON 行日志格式（每行一个 JSON 对象，便于流式解析）
- 必填字段：`timestamp`、`level`、`logger`、`message`
- 可选字段：`request_id`、`module`、`extra`（键值对）
- 与现有 `logging.getLogger("team_memory.*")` 的衔接方式
- **request_logger**：是否走 JSON；若 web 有自定义 Formatter，需统一或保留单独配置
- **开发/生产切换**：环境变量 `LOG_FORMAT=json` 或 config 开关；开发默认 human-readable，生产/CI 默认 JSON
- 敏感字段脱敏约定（如 API Key、密码）

**验收：** 文档存在；格式可被 Python `structlog` 或 `logging` 配置实现；request_logger 与切换方式已写明。

---

### Task 2：实现日志 JSON 输出配置

**Files:** `src/team_memory/bootstrap.py`（日志初始化）、`src/team_memory/config.py`（仅 `LOG_FORMAT` 开关）

**架构约束**：日志 JSON 配置**仅放在 bootstrap（L3）**；config（L0）仅提供 `LOG_FORMAT=json` 开关，不直接依赖 structlog/python-json-logger，避免破坏 L0 无依赖约束。

**内容：**

- 引入 `python-json-logger` 或 logging 配置，统一 JSON 输出（优先最小侵入）
- 在 bootstrap 中根据 `config.LOG_FORMAT` 初始化 Handler/Formatter
- 保持现有 `logger = logging.getLogger("team_memory.web")` 等 logger 名称，仅替换输出格式
- 不改变现有调用方式（`logger.info(...)` 等）

**验收：** 启动后 `make dev` 或 `make web`，日志输出为 JSON 行；`ruff check src/`、`pytest tests/ -q` 通过；**至少对若干条日志样本做 JSON 解析，断言必填字段（timestamp、level、logger、message）存在且类型正确**（pytest 用例或启动后捕获样本断言）。

---

### Task 3：创建 doc-gardening 设计文档

**Files:** `docs/design-docs/doc-gardening.md`

**扫描范围（首版与 CI 统一）**：

| 范围 | 首版 | CI | 说明 |
|------|------|-----|------|
| `docs/design-docs`、`docs/exec-plans` | ✅ | ✅ | CI 只跑此范围，避免拖慢 |
| `README.md`、`AGENTS.md` | 可选 | ❌ | 由本地 `make harness-doc-check` 或定期任务扫描 |
| `.cursor/` | ❌ | ❌ | 若被 git 追踪可后续扩展 |

**内容：**

- 扫描项：deprecated 引用、404 链接、断裂路径、过时标注（如「部分内容已过时」）
- 输出格式：`file:line: rule_id: message`（与 harness_import_check 一致）
- **白名单规则**：`archive/` 内文档互相引用可豁免；白名单格式（路径或 rule_id）与复审时机（如每季度）须在文档中声明
- 与 tm-doc-maintenance 规则衔接：扫描结果可驱动人工或 Agent 修复；输出可被 Agent 解析，白名单规则可被脚本读取

**验收：** 文档存在；扫描项、范围、白名单定义清晰；可被脚本引用。

---

### Task 4：实现 doc-gardening 脚本

**Files:** `scripts/harness_doc_gardening.py`、`tests/fixtures/doc-gardening/expected.txt`（Golden Set）

**内容：**

- 解析 `docs/design-docs`、`docs/exec-plans` 下 markdown 的链接（`[text](url)`、`<url>`）
- 检查：内部链接是否指向存在文件；`docs/exec-plans/archive/`、`deprecated/` 是否被新文档引用（违反「deprecated 不引用」）；白名单按 Task 3 规则豁免
- 可选：扫描「部分内容已过时」等标注，产出待办清单
- 输出：`file:line: rule_id: message`；exit 0=无问题，非 0=有问题

**Golden Set**：在 `tests/fixtures/doc-gardening/expected.txt` 维护预期检出列表（含 `file:line: rule_id` 或等价格式）；脚本验收时用该列表做断言，建立自动化回归。

**验收：** 脚本可执行；对 Golden Set 中预期检出能正确输出；pytest 或 shell 用例覆盖 Golden Set 断言。

---

### Task 5：接入 Makefile 与 CI

**Files:** `Makefile`, `.github/workflows/`（若存在）

**内容：**

- Makefile 新增 `make harness-doc-check`：调用 doc-gardening 脚本（可扫描 `docs/` 全量或与 CI 一致）
- 可选：`make harness-check` 增加 doc-gardening 步骤（或单独 target，按需执行）
- **CI 执行方式**：doc-gardening 作为**独立 job**（可与 lint 并行），设置 `timeout-minutes: 5`；首版可 `continue-on-error: true` 试跑，误报收敛后再改为阻断

**验收：** `make harness-doc-check` 可执行；CI 配置正确（若有）；CI 中 doc-gardening job 独立、超时已设。

---

### Task 6：更新 harness-engineering 与文档索引

**Files:** `.cursor/rules/harness-engineering.mdc`, `AGENTS.md`, `docs/design-docs/README.md`

**内容：**

- harness-engineering 增加「Phase 4 可观测性」节：引用 logging-format.md；说明 doc-gardening 用途
- AGENTS.md 知识库导航增加 logging-format、doc-gardening 链接
- docs/design-docs/README.md 索引更新

**验收：** 规则可读；AGENTS.md 已链入新文档；路径正确、可跳转。

---

## 四、执行顺序

```
step-0 摸底 → Task 1 → Task 2 → Task 3 → Task 4 → Task 5 → Task 6
```

---

## 五、成功指标

| 指标 | 基线 | 目标 |
|------|------|------|
| 日志格式 | 混用 human-readable | 生产/CI 输出 JSON 行 |
| doc-gardening | 无 | 脚本可检出断裂链接、deprecated 引用 |
| 规则可被 Agent 引用 | 无 | logging-format、doc-gardening 已链入 AGENTS.md |

---

## 六、Phase 6 预留（原 Phase 5，tm 项目，后续排期）

- 经验库策略：若启用 tm，可定义 stale 经验判定、归档策略、清理频率等
- Plan 末尾预留节写法：在 Plan 末尾增加「Phase N+1 预留」节，延续 Phase 3→4→5 的写法
- 本计划不包含，完成 Phase 5 后可单独排期。

---

## 七、风险与缓解

| 风险 | 缓解 |
|------|------|
| structlog 引入改动面大 | 优先用 `python-json-logger` 或 logging 配置，最小侵入 |
| 日志配置破坏 L0 | 配置仅放 bootstrap；config 仅提供 `LOG_FORMAT` 开关 |
| doc-gardening 误报多 | 白名单机制（archive 内互相引用豁免）；规则先宽松，逐步收紧 |
| 文档量大会拖慢 CI | CI 只跑 `docs/design-docs`、`docs/exec-plans`；README/AGENTS 由本地或定期任务扫描 |

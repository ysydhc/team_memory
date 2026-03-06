# Harness Phase 3 实施计划：架构约束

> **前置**：Phase 0-1 已完成；Phase 2 已完成（先规划再执行、自审清单、功能验证、人类决策点）
> **评审**：已根据《Harness Phase 3 全维度评审报告》完成调整；已根据《Plan 细节评估报告》补充豁免规则、Task 边界、Brownfield 对齐等。
> **执行模式**：Subagent-Driven（每个 Task 派发 implementer subagent，主 Agent 验收）

**Goal:** 为 team_doc 定义分层与依赖方向，用脚本与 CI 强制约束，防止架构漂移。

**Architecture:** 分层定义 → import 检查脚本 → CI 集成；纯 Harness，不依赖 tm。

---

## 一、现状与目标

### 1.1 当前 src 结构（来自 Brownfield 评估）

```
src/team_memory/
├── schemas.py, config.py
├── storage/     # models, repository, database
├── services/    # experience, search_pipeline, hooks, cache
├── web/         # app, routes, middleware
├── auth/, embedding/, reranker/, architecture/
├── server.py, bootstrap.py, workflow_oracle.py
```

### 1.2 目标分层（Harness 风格）

**Harness 风格**：分层仅约束依赖方向，不引入新抽象；规则与脚本可独立于 tm 运行，便于 CI 与本地门禁。

| 层     | 模块                                                | 可依赖        | 禁止依赖                   |
| ----- | ------------------------------------------------- | ---------- | ---------------------- |
| L0 基础 | schemas, config                                   | 无          | -                      |
| L1 存储 | storage                                           | L0         | services, web          |
| L2 服务 | services, auth, embedding, reranker, architecture | L0, L1     | web, server, bootstrap |
| L3 入口 | web, server, bootstrap, workflow_oracle           | L0, L1, L2 | -                      |

**依赖方向**：只能向前依赖，禁止反向（如 services 不能 import web）。

**特殊规则**：bootstrap、server 为单向依赖汇聚点，**禁止被 L0–L2 引用**，避免循环依赖。

**L3 内部依赖**：web、server、bootstrap、workflow_oracle 之间允许互相引用（同属入口/组装层）；脚本不校验 L3 内部方向。

**已知待修复**：`architecture` → `web.architecture_models`、`auth` → `web.app` 为当前反向依赖，Phase 3 执行中需修复（如将 `architecture_models` 迁至 schemas）。

---

## 二、任务拆分

### Task 1：编写分层定义文档

**Files:** `docs/design-docs/architecture-layers.md`

**内容：**

- 分层表（L0～L3）与模块归属（含 auth、embedding、reranker、architecture、workflow_oracle）
- 允许的依赖矩阵（含同层横向依赖规则、L3 内部允许互相引用）
- 特殊规则：bootstrap、server 禁止被 L0–L2 引用
- **Brownfield 对齐**：Brownfield 指现有代码/遗留结构；对齐方式为：(1) 目录映射：`src/team_memory/<module>` 与分层表一一对应；(2) 新增模块须先归入某层再实现
- **豁免规则**（含示例）：
  - `# noqa: layer-check`：单行 import 后添加，该行豁免
  - `if TYPE_CHECKING:` 块内 import：仅类型注解用，豁免
  - 白名单：在 `architecture-layers.md` 或脚本配置中声明 `exclude_paths`（如某迁移脚本路径）；白名单须在文档中记录理由，定期复审，修复后移除
- **异常处理约定**：语法错误、无法解析文件、非 Python 文件、空文件 → 跳过并记录到 stderr，不中断整体检查
- 例外说明（如 server.py 同时挂 MCP 与 Web，bootstrap 跨层初始化）

**验收：** 文档存在；包含上述豁免示例与白名单格式；可被 import 检查脚本引用。

---

### Task 2：创建 import 方向检查脚本

**产出边界**：本 Task 产出为「可执行的 import 检查脚本 + 黄金用例通过」；不包含反向依赖修复。

**Files:** `scripts/harness_import_check.sh` 或 `scripts/harness_import_check.py`

**逻辑：**

- 解析 `src/team_memory/**/*.py` 的 import 语句（仅顶层 import，不解析动态 import）
- 按分层规则校验：若 L2 模块 import 了 L3 模块，则报错；L0–L2 不得 import bootstrap/server；L3 内部不校验
- 输出格式：`file:line: rule_id: message`（便于 CI 注释或 PR 状态展示）
- 豁免机制：`# noqa: layer-check` 或配置文件白名单；`if TYPE_CHECKING:` 块内 import 豁免
- 检查范围：默认不包含 `tests/`、迁移脚本；`tests/` 下对 `src/` 的 import 不纳入检查；可在配置中扩展
- 异常处理：语法错误、无法解析文件时跳过并记录，不中断整体检查

**黄金用例**（用于脚本自测与回归）：至少 2 个预期违规用例 + 1 个预期无违规用例，在 docs 或 tests 中声明。

**摸底运行**（Task 2 雏形完成后执行）：运行脚本输出当前违规清单；清单作为 Task 5 的输入，用于评估修复工作量与白名单范围。

**验收：** 脚本可执行；exit 0=通过、非 0=失败；黄金用例全部通过；摸底运行可产出违规清单。

---

### Task 3：接入 Makefile 与 CI

**Files:** `Makefile`, `.github/workflows/`（若存在）, `docs/` 或 `.debug/`（CI 等价命令）

**内容：**

- Makefile 新增 target：`make harness-check` 或 `make lint-harness`，调用 import 检查 + ruff + harness_ref_verify
- **CI 双轨策略**：优先使用 `make harness-check`；若 CI 环境无 Makefile，则直接调用等价命令（如 `python scripts/harness_import_check.py`），并在 docs 或 `.debug/` 中写出「CI 等价命令」清单，避免本地与 CI 行为不一致
- **与 harness_ref_verify 统一**：import 检查与 harness_ref_verify 均在 `pull_request` 和 `push` 时执行；可合并为同一 job（如 `lint` 或 `arch-check`），避免 job 过多
- 失败时阻断流水线

**验收：** `make harness-check` 可执行；CI 配置正确（若有）；CI 等价命令已文档化（若采用直接脚本方案）。

---

### Task 4：更新 harness-engineering 与 AGENTS.md

**Files:** `.cursor/rules/harness-engineering.mdc`, `AGENTS.md`

**内容：**

- harness-engineering 增加「架构约束」节：引用 architecture-layers.md，说明提交前须通过 import 检查
- AGENTS.md 知识库导航增加 architecture-layers 链接

**AGENTS.md 行数约束**：目标为 ≤ 120 行；若当前已超，本 Task 以「增加 architecture-layers 链接」为验收，行数作为后续优化目标；超出时可考虑拆分到子文档、归档低频入口、或放宽至 150 行并记录理由。

**验收：** 规则可读；AGENTS.md 已链入 architecture-layers；行数约束已按上述说明处理。

---

### Task 5：全量验证与反向依赖修复

**产出边界**：本 Task 产出为「全量零违规 + 反向依赖修复（或白名单）」；Task 2 负责脚本实现与自测，本 Task 负责修复与最终验收。

**步骤：**

1. `ruff check src/`
2. `pytest tests/ -q`
3. `./scripts/harness_ref_verify.sh`
4. `./scripts/harness_import_check.*`（新脚本）
5. 检查 AGENTS.md 行数（按 Task 4 约束）

**反向依赖修复**（必须完成）：

- 基于 Task 2 摸底产出的违规清单，评估修复工作量
- 至少修复已识别的反向依赖：`architecture → web.architecture_models`（将 architecture_models 迁至 schemas 或新建 `team_memory.schemas.architecture`）、`auth → web.app`（按分层文档方案调整）
- 若违规较多，可设白名单分阶段修复；白名单须在 architecture-layers.md 中记录路径与理由，并约定复审与移除时机

**验收标准**：import 检查脚本**零违规通过**（或白名单内违规已记录）；为 harness_import_check 增加至少 1 个自测（pytest 或 shell），约定 exit code（0=通过，非 0=失败）。

---

## 三、执行顺序

```
Task 1 → Task 2（雏形）→ 摸底运行（产出违规清单）→ 修复反向依赖 → Task 2（完善）→ Task 3 → Task 4 → Task 5
```

---

## 三.1 成功指标

| 指标     | 基线           | 目标                 |
| -------- | -------------- | -------------------- |
| import 违规数 | 摸底时输出     | Phase 3 后零违规     |
| CI 结构测试 | 无             | 通过（阻断失败）      |
| 规则可被 Agent 引用 | 无 | architecture-layers 已链入 AGENTS.md |

---

## 四、Phase 4 预留（后续排期）

Phase 4（可观测性 + 垃圾回收）包含：

- 日志格式统一（JSON 结构化）
- doc-gardening（定期扫描过时文档）
- 经验库策略（若启用 tm）

本计划不包含 Phase 4，完成 Phase 3 后可单独排期。

---

## 五、风险与缓解

| 风险              | 缓解                                 |
| --------------- | ---------------------------------- |
| 当前代码已存在反向依赖     | 摸底产出违规清单；至少修复 architecture/auth 已知违规；白名单须在文档中记录路径与理由，定期复审 |
| import 解析复杂度    | 用 `ast` 或 `grep` 简单实现；仅顶层 import；异常文件跳过并记录 |
| CI 环境无 Makefile | 文档化「CI 等价命令」清单，确保本地与 CI 行为一致       |

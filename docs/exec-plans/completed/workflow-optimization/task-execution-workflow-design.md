# 任务执行工作流 — 设计说明

> 本文档描述任务执行工作流的**设计决策、运行机制、优缺点、如何编辑**等与核心流程能力无关的说明。流程定义以 **.tm_cursor/plans/workflows/ 下对应工作流的 YAML** 为单源（如 task-execution-workflow.yaml）；YAML 设计思路与元信息见各工作流同名的 `.md`。

---

## 一、如何编辑本工作流

- **流程定义**：编辑 `.tm_cursor/plans/workflows/` 下对应工作流的 YAML 文件。字段含义：`id`、`name`、`action`、`acceptance_criteria`、`checkpoint`（打断点）、`allowed_next`（允许的下一步）、`when`（条件分支，若存在则优先于 allowed_next）。
- **Step 独立文件与 $ref**：每个 step 可单独成文件，主 workflow 用 `$ref` 引用，如 `steps: [ { $ref: steps/task-execution/step-coldstart.yaml }, ... ]`。路径相对于主 workflow 文件所在目录。后端（workflow_oracle）在加载时解析 `$ref`，返回给 MCP/Agent 的 step 含完整 action，对 Agent 透明。
- **增加节点**：在 YAML 的 `steps` 中新增一项，写好 `id`、`name`、`action`、`acceptance_criteria`、`checkpoint`、`allowed_next` 或 `when`；若插在中间可用新 id（如 step-execute-review）。若使用 $ref，新建 step 文件并追加引用。
- **删除节点**：从 YAML 的 `steps` 中删除对应项即可。
- **if/else 分支**：在需要条件分支的步骤使用 `when`，如 `when: [ { condition: group_completed, next: step-retro }, { condition: else, next: step-claim } ]`。
- **冷启动单入口**：冷启动仅在 step-coldstart 执行。step-claim 认领后若发现任务 description 为空或不足，须补做 step-coldstart 的冷启动动作并记为 step-coldstart 完成（或退回 step-coldstart），再进入 step-execute；不得在 step-claim 内定义另一套冷启动语义。

---

## 二、运行机制与保证手段

### 2.1 本工作流如何“工作”

- **本质**：流程定义在 **.tm_cursor/plans/workflows/ 下对应工作流的 YAML 文件**，描述从 step-coldstart 到 step-retro 的步骤、动作、验收标准、打断点与条件分支；**没有**独立的工作流引擎，由 Agent 解析 YAML 后按步执行。
- **执行顺序**：step-coldstart（可选）→ step-claim 选活与认领 → step-execute 执行与记录 → step-complete 完成并沉淀 → **step-verify 任务完成后验收（subagent，必须）** → 若 group_completed 则 step-retro 组级复盘，否则 step-claim 选活下一任务。**step-verify 不得跳过**。
- **执行依赖**：
  1. **用户指令**：例如「按任务执行工作流」「执行 TM 任务工作流」「只做下一步」「到下一个打断点为止」等。
  2. **Agent 行为**：Agent 在收到上述指令时，应**先读取**选流得到的该工作流 YAML（位于 `.tm_cursor/plans/workflows/`），再按步骤 ID 顺序执行；**执行以 TM 任务状态与 tm_message 为准**，按 YAML 步骤顺序推进；每步完成后可汇报；**step-complete 完成后必须执行 step-verify（subagent 验收），不得直接进入下一任务**。
  3. **Rules 约定**：在 `.cursor/rules/`（如 tm-core）中约定：执行 TM 任务工作流时，**必须**按选流结果读取对应工作流 YAML 并按 step 顺序执行，不得跳过未完成的步骤（含 step-verify）。

### 2.2 如何尽量保证 Agent 按此工作流执行

| 手段 | 说明 |
|------|------|
| **Rules 明确引用** | 在 tm-core 中写清：选流由 **subAgent** 完成，仅读各 YAML **开头**（使用场景 + meta），返回 workflow_id；主 Agent 仅加载该工作流执行，约定见 .cursor/prompts/execute-task-workflow-by-match.md。 |
| **入口 Prompt** | 在 `.cursor/prompts/` 中提供「执行任务工作流」类 prompt（如 execute-task-workflow-by-match.md）：主 Agent 获取任务后**调用 subAgent**，subAgent **仅读 .tm_cursor/plans/workflows/ 下各 \*.yaml 至 steps: 之前**，根据任务 title/description 与使用场景匹配，**仅返回 workflow_id**；主 Agent 根据返回的 workflow_id **只读该一个 YAML** 按 step 执行，控制主 Agent 上下文长度。 |
| **验收自检** | 每步执行后要求 Agent 自检该步「验收标准」是否满足，再进入下一步或暂停。 |
| **任务结束收口验证（必须）** | 每个任务完成后、提交前必须执行 **ruff 收口验证**：`ruff check src/`（若本任务涉及 tests 则加上 `tests/` 或对应测试文件）；未通过则修复直至通过，再执行任务级 git 提交。 |
| **任务完成后验收（必须，不得跳过）** | step-complete 完成后**必须先执行 step-verify**：使用 **subagent** 对当前已完成任务再次验收（人验证或模拟人验证）；**仅当验收通过后**才可进入 step-retro 或下一任务；未执行或验收未通过不得进入下一任务。详见 YAML 中 **step-verify**。 |

### 2.3 任务完成后验收（step-verify，subagent）执行约定

当主执行 Agent 完成某任务且全部检查项通过后，由主 Agent **启动 subagent（验收子 agent）** 对当前任务结果进行再次验收，约定如下：

1. **触发条件**：当前任务已 `status=completed`，且 ruff 收口、相关 pytest、任务级 git 提交均已完成。
2. **验收视角**：**人验证或模拟人验证**——从真实用户或模拟用户使用角度，验证本次交付是否满足预期、有无 bad case（如功能不可用、体验异常、与描述不符）。
3. **验收步骤**：
   - **人验证**：由真人按任务描述与验收标准操作（MCP/API/Web 或实际使用流程），确认通过或记录问题。
   - **模拟人验证**：由 subagent 模拟用户行为（调用 MCP、API、Web，或基于测试数据执行关键路径），检查结果与预期是否一致；若能力未对外暴露可构建测试数据，验收结束后**必须删除**该测试数据。
4. **结果处理**：
   - **存在 bad case**：验收 subagent 停止，向用户汇报 bad case 与复现方式，**等待用户确认**后再决定修复或进入下一任务。
   - **无 bad case**：验收 subagent 汇报「验收通过」，主 Agent 或用户可**自动进入下一任务**（如继续 step-claim 选活下一任务）。
5. **实施方式**：由主 Agent 通过可用机制（如 Cursor 的 mcp_task / subagent 等）启动验收 subagent，并传入「当前任务 ID、本次改动范围、验收标准与涉及路径」等上下文；验收 subagent 仅负责验收与结论，不修改代码（除非用户明确要求修复）。

#### 2.3.1 推荐启动方式与入参

- **推荐启动方式（二选一或组合）**：
  - **同会话内验收 checklist**：主 Agent 在同一会话内按固定清单执行（ruff、pytest、关键路径 + 人验证或模拟人验证），逐项打勾并汇报「验收通过/不通过」；适合改动范围小、验收项明确的任务；仅遇 bad case 需深度调试时再启独立 subagent。
  - **另启独立 subagent**：主 Agent 通过 mcp_task / subagent 等启动子 agent，将验收上下文传入，由子 agent 独立执行验收并回报结论；适合改动面大或需隔离环境的场景。
- **主 Agent 启动验收时建议传入的入参**（任一种启动方式均应尽量提供）：
  - `task_id`：当前已完成任务的 ID（TM-xxx 或 UUID）；
  - `改动范围`：本次任务涉及的文件/模块/API 列表；
  - `验收标准`：来自任务 description 或本工作流 step-verify 的通用标准（人验证/模拟人验证）；
  - `涉及路径`：代码或配置路径，便于验收时定位。

当前**没有**运行时强制：若 Rules 未加载或 Agent 未遵守约定，仍可能跳过某步。若要更强保证，可考虑：轻量脚本/MCP 工具返回「当前工作流 + 下一步 step」，或 Plan 模板中显式列出本工作流各 step 作为子步骤。

---

## 三、与开源工作流的对比

与 GitHub 上常见的工作流/自动化方案对比，本方案的特点如下。

### 3.1 对比对象简述

- **GitHub Actions**：YAML 定义、服务端调度、步骤顺序执行、与仓库/Issue/PR 集成。
- **GitHub Agentic Workflows**：自然语言/Markdown 描述意图，由 AI 解释执行、有沙箱与权限控制、与 Issues/PR 深度集成。
- **CCPM / Locus / LEO 等**：多以 GitHub Issue 为任务源，配合 worktree/CLI，有明确「下一步任务」与执行记录。

### 3.2 本工作流的优点

| 优点 | 说明 |
|------|------|
| **文件化、可版本管理** | 流程定义即 YAML，随仓库提交、diff 可见，便于评审与演进。 |
| **可编辑** | 增删 step（如 step-coldstart 冷启动）只需改对应工作流 YAML，无需改代码或额外引擎。 |
| **不依赖 GitHub** | 适合企业内网、无 GitHub Issue 场景；任务源为 TM（tm_ready / tm_task），与现有 MCP 一致。 |
| **与 TM 深度结合** | 选活、认领、完成、组级复盘与 tm_task / tm_save_group 一一对应，经验闭环统一在 TM。 |
| **无额外服务** | 不需要单独的工作流引擎或 CI runner，由现有 Cursor/Claude Code + MCP 即可运行。 |
| **人可读、可监督** | 进度与产出摘要通过 tm_message 与任务状态可查，便于人工或工具解析。 |

### 3.3 本工作流的缺陷

| 缺陷 | 说明 |
|------|------|
| **无硬性执行保证** | 依赖 Agent 遵守 Rules 与「读文件再执行」的约定；若未加载规则或模型自行跳过，无法从技术上强制。 |
| **无统一执行引擎** | 与 GitHub Actions / Agentic Workflows 不同，没有服务端 runner 或沙箱自动按步执行，需依赖对话内 Agent 行为。 |
| **进度推断依赖 tm_message** | 「只做下一步」「从断点继续」依赖对 tm_message 的解析与 Agent 推断，需约定步骤审计格式以保持一致。 |
| **无内置审计/可视化** | 执行记录分散在对话与 tm_message 中，没有统一的步骤级审计日志或仪表盘（可后续用 tm_message + 检索或 Web 聚合补齐）。 |
| **与 Issue 无直接同步** | 若团队同时使用 GitHub Issue，需自行做 TM 与 Issue 的映射或双写，本方案不提供开箱即用同步。 |

### 3.4 当前环境下缺陷的可解性

| 缺陷 | 可解性 | 在当前环境下可做的缓解/解法 |
|------|--------|----------------------------|
| **1. 无硬性执行保证** | **可缓解，不可根除** | 已做：Rules 强制按选流结果读取对应工作流 YAML 再按 step 执行。可追加：在 `.cursor/prompts/` 增加「执行任务工作流」入口 prompt；可选增加 MCP 工具（如 tm_workflow_next_step 返回「当前工作流 + 下一步 step_id + 验收标准」）作为单源真相，降低 Agent 自行跳过或乱序的可能。无法在对话内做到 100% 技术强制，只能尽量提高遵守概率。**跳出当前框架的可解方向**见下节。 |
| **2. 无统一执行引擎** | **可缓解** | 不引入新服务的前提下：可增加**轻量「步骤预言」**——例如 MCP 工具（如 tm_workflow_next_step），输入工作流名与 task_id，基于 tm_message 解析输出下一步 step_id、该步动作与验收标准；Agent 仍负责实际执行，但「下一步是什么」由该工具统一给出，减少推断歧义。本质是「无 runner，有 step oracle」。 |
| **4. 无内置审计/可视化** | **可解** | 已有 `tm_message(task_id, content)` 可挂到任务上。做法：在 Rules 中约定「每完成工作流一步，必须调用 tm_message，content 为约定格式」（例如包含 `[workflow] <workflow_id> step-<id>: <摘要>`（id 为 YAML 的 step id））。Web 端任务详情已展示消息列表；可在此基础上增加按「workflow/step」筛选或简单汇总页，即形成步骤级审计。无需新表，只需格式约定 + 现有消息列表/检索。 |
| **5. 与 Issue 无直接同步** | **按需可解** | 若团队**不用** GitHub Issue，此条可忽略。若**需要**同步：可在现有 MCP 上扩展（如 tm_task 创建/更新时可选填 issue_url 或 external_id），或单独写一个「TM ↔ Issue 同步」小工具/定时任务，双向或单向均可。属于需求明确后的增量开发，当前环境可支持，不做则保持现状。 |

**建议优先落地**：缺陷 4（每步 tm_message 约定 + Web 展示）仅靠 Rules 与现有能力即可完成；缺陷 1、2 的 MCP/入口 prompt 为可选增强。缺陷 5 按团队是否用 Issue 再决定是否做。

### 3.5 跳出当前框架：无硬性执行保证的可解方向

若**不局限于**「对话内 Agent + Rules + 读文件」的当前框架，从技术上逼近或实现「硬性执行保证」的常见方向如下。

| 方向 | 做法简述 | 保证程度 | 成本/前提 |
|------|----------|----------|-----------|
| **独立工作流引擎** | 引入 Temporal、Inngest、Prefect 等引擎，将 step 定义为引擎内的 activity/step；由引擎按 DAG 调度、重试、记录状态；Agent 仅作为某一 step 的「执行器」被引擎调用。 | 高：步骤顺序与状态由引擎强制，可审计、可重试。 | 需部署与维护引擎；需把本工作流「翻译」为引擎定义（YAML/代码）。 |
| **GitHub Actions / Agentic Workflows 驱动** | 用 Actions 或 Agentic Workflows 作为「外层编排」：每步一个 job 或 workflow run，步内通过 API/CLI 调用 Cursor/Agent 或 MCP；下一步是否执行由 job 成功与否决定。 | 高：步骤由 CI 调度，状态在 GitHub。 | 依赖 GitHub；需把任务源与 TM 做映射或双写。 |
| **专用 Runner 服务** | 自建轻量服务：按定时或 webhook 读取「下一步」、调用 Agent API（或 Cursor/Claude Code 的 headless 接口）执行单步、根据返回与验收标准更新状态并决定是否继续。 | 高：执行顺序与推进由服务控制，不依赖对话内自觉。 | 需开发与部署 Runner；需 Agent 可被 API/headless 调用。 |
| **Plan 模式固化 + 锁步** | 在 Cursor Plan 或 Claude Code Project 中，把本工作流各 step 写成不可跳过的子步骤；执行时仅「执行当前子步骤」并依赖 Plan 的 UI 锁步（不点下一步不展开下一项）。 | 中高：依赖编辑器 Plan 的锁步语义，若支持则顺序有保证。 | 依赖 Cursor/Claude Code 的 Plan 行为；需维护 Plan 模板与工作流定义同步。 |
| **步骤预言 MCP + 单入口** | 在现有框架内增加 MCP 工具「返回下一步 + 验收标准」；且**唯一入口**为「先调该 MCP，再执行其返回的 step」——Rules 与 prompt 强约束「禁止未调用该工具就执行任意 step」。 | 中：仍依赖 Agent 遵守「先调工具再执行」，但下一步由工具唯一决定，减少乱序与遗漏。 | 仅需开发 MCP + 强化 Rules/prompt；无新服务。 |

小结：要**真正硬性保证**，通常需要「步骤推进权」从对话内的 Agent 转移到**外部调度方**（引擎、CI、Runner、或至少锁步 Plan）；在纯对话 + Rules 框架内只能做到「尽量约束」，无法从进程层面强制。

### 3.6 后续任务（缺陷 2 落地）

以下已纳入后续实施（缺陷 5 已忽略）。

| 任务 | 描述 | 验收标准 |
|------|------|----------|
| **工作流步骤预言** | 实现轻量「步骤预言」：MCP 工具（如 tm_workflow_next_step），输入工作流名与 task_id，基于该任务的 tm_message 解析出当前进度，输出下一步 step_id、该步动作描述与验收标准（从对应工作流 YAML 解析）。Agent 执行前先调用该工具获取「下一步」，再执行。 | 存在可调用的 MCP；给定 task_id 可基于 tm_message 返回「下一步」及该步内容；文档或 help 说明用法。 |

### 3.7 小结

本工作流适合**以 TM 为任务源、在 Cursor/Claude Code 内由 Agent 按文档执行**的团队；强调可编辑、可版本化、与 TM 闭环一致。若需要**强制的、服务端驱动的步骤执行**或**与 GitHub Issue 深度联动**，可对接 GitHub Actions / Agentic Workflows 做混合使用。

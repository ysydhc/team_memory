# 任务执行工作流 — 改进建议清单

> 来源：完成 tm-workflow-support 任务组后，由多视角分析合并去重的建议，供逐项确认是否落地。  
> 流程定义见 [workflows/task-execution-workflow.yaml](workflows/task-execution-workflow.yaml)，设计说明见 [task-execution-workflow-design.md](task-execution-workflow-design.md)。

**任务组已入库**：改进项已按分类写入 TM 任务组 **workflow-optimization**（任务源： [workflows/workflow-optimization-tasks.md](workflows/workflow-optimization-tasks.md)，已通过 tm_doc_sync 同步）。

**边用工作流边优化**：执行 workflow-optimization 任务组时，请使用 **[workflow-optimization-workflow.yaml](workflows/workflow-optimization-workflow.yaml)**，该流程**不含代码收口**（无 ruff、pytest、subagent 验收），仅包含：选活 → 执行与记录 → 完成并沉淀 → 组级复盘。第一项待办建议为「[一] 冷启动只在一个入口」；可从 `tm_ready` 认领后按该 YAML 的 step 执行。

---

## 一、流程与 Rules 收口

| # | 建议 | 状态 |
|---|------|------|
| 1 | **step-complete 后显式禁止跳步**：流程定义中 step-complete 完成后唯一允许的下一步为 step-verify，禁止直接进入 step-claim 或 step-retro。 | ✅ 已做（YAML 中 step-complete 的 allowed_next 仅 [step-verify]） |
| 2 | **冷启动只在一个入口**：明确仅在 step-coldstart 做组/子任务冷启动；step-claim 认领后若 description 空，则退回 step-coldstart 或在本轮执行冷启动并记为 step-coldstart，避免两套语义。 | ✅ 已做 |
| 4 | **group_completed 后禁止先选活**：在 Rules「group_completed 时必调 tm_save_group」旁增加：在成功执行 tm_save_group 前，不得执行 tm_task_claim 或进入 step-claim 选活。 | ✅ 已做 |
| 5 | **步骤审计格式固定**：tm_message 的 content 必须含前缀 `[workflow] <workflow_id> step-<id>:`（id 为 YAML 的 step id，如 step-coldstart、step-claim），便于解析与 Web 按 step 筛选。 | ✅ 已做 |
| 6 | **step-verify 验收（subagent）推荐启动方式**：在流程或设计文档中写清 step-verify 的推荐启动方式与传入参数（同会话 checklist vs 另启 subagent），减少歧义。 | ✅ 已做（设计文档 2.3.1 + YAML step-verify 注释与 action 引用） |

---

## 二、Prompt 与入口

| # | 建议 | 状态 |
|---|------|------|
| 7 | **新增「执行任务工作流」入口 Prompt**：`.cursor/prompts/` 增加一条，**不直接指定工作流**，按任务内容与各工作流 meta（when_to_use、scope）匹配选择后再执行。 | ✅ 已做（execute-task-workflow-by-match.md + 两套 YAML 的 when_to_use/scope） |
| 8 | **关键节点自动汇报进度**：Rules/Prompt 约定在认领、完成、验收、组复盘后，Agent 主动输出一句进度摘要（如「本组 x/y，当前 TM-xxx；下一步 step-X」）。 | 待确认 |

---

## 三、工具与 MCP

| # | 建议 | 状态 |
|---|------|------|
| 9 | **步骤预言 MCP**：输入工作流名 + task_id（或 tm_message 历史），输出当前 run、下一步 step_id、该步动作与验收标准；执行工作流前先调该工具再执行。 | 已纳入设计文档「后续任务」 |
| 11 | **冷启动 description 由 LLM 生成模板**：根据 tm_task(get) 的 title（及可选 group 描述）用 LLM 生成「验收标准、风险点、关键步骤」的 description 初稿，Agent/人微调。 | 待确认 |
| 12 | **验收入参结构化**：任务增加可选字段（如 acceptance_criteria、affected_paths），或在完成时由 MCP 返回「建议验收范围」；验收 subagent 从结构化字段读取。 | 待确认 |

---

## 四、验收方式

| # | 建议 | 状态 |
|---|------|------|
| 13 | **step-verify 允许同会话内验收 checklist**：在同一会话内执行固定 checklist（ruff、pytest、关键路径 + 人验证/模拟人验证），主 Agent 按项打勾并汇报「验收通过/不通过」；仅 bad case 需深度调试时再启独立 subagent。 | 待确认 |
| 14 | **任务级验收 checklist 字段**：任务增加可选 `acceptance_checklist`（可勾选列表），Web + MCP 读写；step-verify 与 checklist 绑定，全部勾选才可「验收通过」。 | 待确认 |
| 15 | **验收 checklist 模板**：在流程或模板中约定某类任务默认验收项（如 ruff、pytest、文档更新）；创建/冷启动时可预填。 | 待确认 |

---

## 五、Web

| # | 建议 | 状态 |
|---|------|------|
| 16 | **组完成时强提示组复盘**：当 group_completed 时，Web 任务组视图显式展示「待执行组复盘」及 tm_save_group 入口或一键按钮。 | 待确认 |
| 17 | **任务消息按 step 聚合/筛选**：Web 任务详情或审计视图支持按 `[workflow] task-execution-workflow step-X` 筛选 tm_message，按 step 分组展示。 | 待确认 |
| 18 | **任务看板按状态分列**：任务页增加看板视图，列 = wait/plan/in_progress/completed，支持拖拽更新 status。 | 待确认 |
| 19 | **当前 step 与验收标准展示**：任务详情或工作流区块展示当前所处 step（如 step-coldstart～step-retro）及本步验收标准摘要，数据来自 tm_message。 | 待确认 |
| 20 | **组/项目进度摘要**：Web 或 API 提供组/项目进度摘要（本组 x/y 完成、进行中、逾期等）。 | 待确认 |

---

## 六、自动化与可观测

| # | 建议 | 状态 |
|---|------|------|
| 21 | **任务完成时可选 Webhook**：tm_task 更新为 completed 时可选触发 Webhook，便于 Slack/飞书/CI 联动。 | 待确认 |
| 23 | **步骤级完成率/耗时指标**：基于 tm_message 中 step-X 统计各 step 完成率/平均耗时，用于优化流程。 | 待确认 |

---

## 七、文档与恢复

| # | 建议 | 状态 |
|---|------|------|
| 25 | **维护「断点恢复与组复盘」索引**：在 .debug 或文档中写清：断点后怎么说、如何重置状态、组完成忘了复盘怎么办，与「手动恢复流程索引」衔接。 | 待确认 |

---

## 八、可选 / 后续

| # | 建议 | 状态 |
|---|------|------|
| 26 | **步骤预言 MCP 落地**：基于 tm_message 解析当前 step 与下一步，执行工作流前先调该工具再执行。 | 已纳入后续任务 |
| 27 | **与 GitHub Issue 同步**：外部 Issue 同步不在本期；若团队需要可在 tm_task 或单独工具中做 TM ↔ Issue 映射（按需）。 | 按需 |

---

## 已完成的改进（当前基线）

- **① step-complete 后禁止跳步**：YAML 中 step-complete 的 `allowed_next` 仅 `[step-verify]`，且设计说明与 Rules 已强调 step-verify 不得跳过。
- **验收步骤语义**：step-verify 为任务完成后验收（subagent + 人验证/模拟人验证），step-retro 为组级复盘。
- **流程定义 YAML 单源**：流程定义、条件分支、编辑方式均以 task-execution-workflow.yaml 为准；设计说明见 task-execution-workflow-design.md。
- **工作流描述精简**：task-execution-workflow.md 仅保留元信息与 YAML 设计思路；设计说明、优缺点、如何编辑等移至 task-execution-workflow-design.md。

---

## 使用说明

- **状态列**：✅ 已做 / 部分 / 待确认 / 已纳入后续任务 / 按需。
- 需要逐项落地时，可优先「一、流程与 Rules」和「二、Prompt 与入口」，再按需做三～八类。

# 工作流搭建指导

本文用于指导工作流搭建。执行与恢复以 **TM 任务状态 + tm_message** 为准，不使用状态文件。流程定义与运行状态分离，单源为 `.cursor/plans/workflows/*.yaml`。

---

## 一、定位与目标

| 能力 | 含义 | 实现要点 |
|------|------|----------|
| **可监督** | 随时看清当前进度、已完成/未完成步骤、关键产出 | 状态与产出写入 TM 任务状态 + tm_message，人/脚本可读 |
| **可打断** | 在任意步骤后暂停，不丢失已做工作 | 每步结束后写 tm_message 再继续；支持「只做下一步」或「到下一打断点为止」 |
| **可重启** | 中断后从下一未完成步骤继续，无需重头来 | 通过 tm_workflow_next_step(task_id) 或解析 tm_message 得到「最后完成步骤」与下一步 |
| **可重复** | 同一流程可再次跑或从某步重跑 | 流程定义不变；重跑即新任务或约定从某 step 再执行并写 tm_message |

---

## 二、流程定义与目录

- **位置**：`.cursor/plans/workflows/<工作流名称>.yaml`
- **步骤约定**：每步唯一 `id`；含 `name`、`action`、`acceptance_criteria`；可选 `checkpoint`（打断点）；流转用 `allowed_next` 或 `when`（条件分支）。
- **可编辑**：通过编辑 YAML 增删改步骤（如 step-coldstart 冷启动）；执行时 Agent 按选流结果读取对应 YAML，与 execute_task、Rules 衔接。
- **目录**：流程定义仅放在 `.cursor/plans/workflows/` 下，无状态文件。

---

## 三、执行协议与可监督性

### 3.1 指令与行为

| 指令（示例） | Agent 行为 |
|--------------|------------|
| 「只做下一步」 | 用 tm_workflow_next_step 或解析 tm_message 得到下一步 → 执行该步 → tm_message(step-X: 摘要) → 停止并汇报 |
| 「到下一个打断点为止」 | 逐步执行，遇带 checkpoint 的步骤完成后停止 |
| 「继续工作流 X」 | 从当前任务 + tm_message 推断第一个未完成步，执行该步 |
| 「执行全部」 | 从第一个未完成步执行到结束，每步写 tm_message |
| 「查看工作流 X 状态」 | 不执行；输出进度与下一步，数据来自 tm_workflow_next_step 或 tm_message 解析/Web |

### 3.2 单步自检与可打断/可重启

- **自检**：执行前确认该步验收标准可检查；执行后按验收自检，通过再发 step 完成消息，不通过则停止并记录原因。
- **可打断**：每步后主动停止。
- **可重启**：持久化靠 tm_message；继续时从解析结果取「下一步」。

### 3.3 可监督与可重复

- **可监督**：人通过 YAML + 任务详情中的 tm_message 或 Web 聚合看进度与下一步；脚本/API 通过 tm_workflow_next_step 或解析 tm_message 查当前 step、下一步、进度摘要。
- **可重复**：流程定义不变；重跑为新任务或从某 step 再执行并写 tm_message。

---

## 四、单任务与任务组流程

**单任务流**：选活（tm_ready → 展示 → tm_task_claim）→ 回复「当前认领任务：TM-xxx」→ 执行 + tm_message → 完成并沉淀（缺 summary 追问 → tm_task update completed）；服务端在 update(completed) 响应中返回 group_completed、group_id、hint（若该任务属 group 且组内全完成）。若响应含 group_completed，进入任务组完成流。

**任务组完成流**：group_completed: true → 当轮或下一轮必须调用 tm_save_group。先 tm_task(list, group_id) 获取组内任务与 group_progress；按组内 experience_type 判断：唯一→总-分，多种→总-分-分（类别名用 experience_type 或 labels）。调用 tm_save_group（总-分：parent + 组内单经验；总-分-分：parent 的 structured_data.grouped_children = { "类别": [exp_id, ...] }）。服务端幂等：同一 group_id 只允许一次组级复盘，已存在则拒绝或返回已有。可选 tm_message 记录「组级复盘已完成」。

---

## 五、依赖、恢复与手动恢复

- **依赖**：任务组创建后先用 tm_dependency 建齐 blocks，再依赖 tm_ready 取任务；Rules 写「任务组执行前确认依赖已建」。
- **恢复**：新会话用 tm_ready 或 tm_task(list, status=in_progress)；以 DB 为准。

**手动恢复能力**：

| 场景 | 手段 |
|------|------|
| 查看卡在哪 | tm_task(action=list, status=in_progress) |
| 查看任务组状态 | tm_task(action=list, group_id=xxx) |
| 恢复选活 | tm_ready(project=xxx) |
| 重新认领/继续某任务 | tm_task_claim(task_id) |
| 手动补完成单任务 | tm_task(action=update, task_id=xxx, status=completed, summary=...) |
| 新会话恢复上下文 | resume_project（MCP Prompt） |
| 按任务/组执行指引 | execute_task(task_id/group_id)（MCP Prompt） |
| 查看与编辑 | Web 看板 |

**推荐步骤**：  
- **情况 A（流程被打断）**：resume_project 或 tm_task list(in_progress) + tm_ready → 继续认领或继续开发并完成。  
- **情况 B（漏做组复盘）**：tm_task(list, group_id) 确认全完成 → 手动调用 tm_save_group 或 Web「组完成→沉淀」。

---

## 六、Rules 与编辑器落地

**Rules 固化**：选活→认领入口句式（如「执行下一个就绪任务」「给我一个能做的任务」→ tm_ready → 展示 → tm_task_claim）；完成并沉淀句式，缺 summary 则追问再 tm_task(update, completed, summary=...)；group_completed: true 时当轮或下一轮必须调用 tm_save_group；认领后回复附带「当前认领任务：TM-xxx（标题）」；对「我的进度」返回进行中数、任务组未完成摘要。

**编辑器与 MCP**：选活、认领、完成、组沉淀均通过 MCP（tm_task、tm_ready、tm_task_claim、tm_save_group）；Rules（tm-core/tm-plan）固化组完成与入口；Plan 模板显式写 tm_task(update, completed, summary)、若 group_completed 则 tm_save_group；Subagents 用同一 MCP，主 Agent 用 tm_task(get/list) 看状态。Cursor 与 Claude Code 共用同一 MCP 配置；项目级 .mcp.json 可提交以共享。

---

## 七、多视角要点与子任务

**项目风险**：组进度不清→group_completed/group_progress 显式返回 + Rules 固化组复盘；依赖未建→先 tm_dependency 再 tm_ready；组级经验漏写/总-分-分选错→experience_type 驱动 + 幂等；数据不一致→DB 为唯一事实源；Agent 漏做→Rules 固化 + 组完成时返回 hint。

**技术选型**：总-分沿用 parent_id + tm_save_group；总-分-分用 structured_data.grouped_children；组完成触发以 Agent 内联 + Rules 为主；紧相关 vs 类型混杂按 experience_type 唯一→总-分、多种→总-分-分；状态以 DB 为准，新会话用 tm_ready/tm_task(list) 恢复。

**使用角度**：固化选活→认领、完成并沉淀入口；缺 summary 追问；组完成 Agent 建议 + 用户确认，Web/Cursor 提供组完成→沉淀入口；当前任务提示与「我的进度」摘要。

**可执行子任务（tm-workflow-support）**：1) 服务端 P0：update 响应返回 group_completed/group_id/hint；2) 服务端 P1：list(group_id) 返回 group_progress；3) 服务端 P1：组级经验幂等；4) 服务端 P1：总-分-分 structured_data.grouped_children；5) 服务端 P1：紧相关/类型混杂判断与 tm_save_group 参数；6) Rules 固化选活/完成入口与 group_completed 时必调 tm_save_group；7) Rules：当前任务提示与「我的进度」摘要；8) Plan/execute_task 模板加入完成与组复盘步骤；9) 文档 .debug/README 同步；10) 文档：手动恢复流程索引；11) Web P2：组完成→沉淀入口；12) Rules：任务组执行前确认依赖已建。

---

## 八、交付物与小结

| 文档 | 说明 |
|------|------|
| 本文 | 工作流搭建指导 |
| workflows/task-execution-workflow.yaml | 可编辑任务执行工作流（YAML 单源，含 step-coldstart 冷启动） |
| workflow-pm-final-three-solutions.md | 方案一/二/三与总-分/总-分-分约束 |
| task-execution-workflow-design.md | 设计说明与运行机制 |

**小结**：目标能力为可监督、可打断、可重启、可重复，实现以 TM + tm_message 为准，不使用状态文件。流程以 YAML 单源、可编辑。固化选活→认领、完成并沉淀、group_completed 必做组复盘。手动恢复见第五节表与推荐步骤。落地时优先服务端 group_completed 响应与 Rules 固化，再 group_progress、总-分-分、幂等，最后 Plan 模板、当前任务提示、Web 组完成入口。

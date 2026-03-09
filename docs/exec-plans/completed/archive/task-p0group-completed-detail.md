# 子任务详情：服务端 P0 — update 响应返回 group_completed / group_id / hint

> **【已归档】** 已完成的子任务详情，仅作历史参考。
>
> 对应 TM 任务：tm-workflow-support 第 1 项。本文档为细节分析、最小可执行单元、单测设计与验收标准。

---

## 一、任务目标

在 `tm_task(action=update, task_id=xxx, status=completed, summary=...)` 的**响应**中，当该任务属于某任务组且该组内所有任务均为 completed 或 cancelled 时，返回：
- `group_completed: true`
- `group_id`: 该任务组的 UUID 字符串
- `group_completed_hint`: 建议调用 tm_save_group 做组级复盘的提示文案

---

## 二、需要改动的点

| 序号 | 改动位置 | 改动内容 |
|------|----------|----------|
| 1 | `src/team_memory/server.py`：`tm_task` 的 `action == "update"` 分支 | 在 `status == "completed"` 且 `summary` 存在、且已完成 sediment（及可选 reflection）之后，`await session.commit()` 之前，增加「组完成判定」逻辑。 |
| 2 | 同上 | 若 `task.group_id` 非空，则用 `repo.list_tasks(project=task.project, group_id=task.group_id)` 获取该组全部任务；若所有任务 `status` 均属于 `("completed", "cancelled")`，则置 `group_completed = True`，并记录 `group_id` 与 hint 文案。 |
| 3 | 同上 | 在构建返回给前端的 `result` 时，若 `group_completed` 为 True，则向 `result` 中写入 `group_completed`、`group_id`、`group_completed_hint`；否则不写入（或可写 `group_completed: false`，以明确语义）。 |

---

## 三、最小可执行单元（保证每次执行质量最优化）

每个单元可在一次小提交内完成，且可单独验证。

| 单元 ID | 描述 | 验收方式 |
|---------|------|----------|
| **U1** | **组完成判定纯逻辑**：在 repository 或 server 内实现「给定 group_id 与 project，判断该组内是否所有任务均为 completed/cancelled」的纯函数或 async 函数（仅读 DB，不写）。入参：session/repo, project, group_id；返回：bool。 | 单测：构造组内 0/1/多任务、不同 status 组合，断言返回值。 |
| **U2** | **update 分支中调用 U1**：在 `tm_task` update 且 `status=completed` 且 `summary` 且 `task.group_id` 非空时，在同一 session 内调用 U1；根据返回值设置 `group_completed`、`group_id`、`group_completed_hint` 三个变量（先不写回响应）。 | 单测：通过 tm_task MCP 或 HTTP 调用 update completed，mock 或真实 DB，断言某次调用后内部变量正确（或通过响应反推）。 |
| **U3** | **响应体写入**：在构建 `result` 时，若上一步得到 `group_completed is True`，则 `result["group_completed"] = True`，`result["group_id"] = str(task.group_id)`，`result["group_completed_hint"] = "建议调用 tm_save_group 做组级复盘"`；否则不添加这些键（或 `group_completed: false`）。 | 单测：端到端或集成测，调用 update 完成某组最后一条任务，断言响应 JSON 包含上述键值。 |
| **U4** | **边界与文档**：无 group_id、组内尚有 wait/in_progress/plan 时均不返回 group_completed true；在 .debug 或 README 中补充「组完成时响应字段」说明。 | 单测：边界用例；文档审阅。 |

---

## 四、单测设计（优先于实现）

### 4.1 测试层级与目标

- **单元级**：组完成判定逻辑（U1），不依赖完整 MCP，可仅用 TaskRepository + 内存/临时 DB）。
- **集成级**：`tm_task` update 的 HTTP 或 MCP 调用，真实 DB（或 testcontainers），验证响应 JSON。

### 4.2 用例表

| 用例 ID | 场景简述 | 前置条件 | 操作 | 预期结果 |
|---------|----------|----------|------|----------|
| **T1** | 任务无 group_id | 存在任务 A，group_id=None，status=wait | tm_task update A status=completed summary="x" | 响应无 group_completed 或 group_completed=false；有 sediment_experience_id。 |
| **T2** | 有 group_id，组内未全完成 | 组 G 下有任务 A(completed)、B(wait) | 将 A 再次 update completed（或完成 B 前先完成 A 的请求） | 响应无 group_completed=true。 |
| **T3** | 有 group_id，组内全部 completed/cancelled | 组 G 下仅有 A、B，A 已 completed，B 当前 wait | 将 B update status=completed summary="y" | 响应含 group_completed=true, group_id=str(G.id), group_completed_hint 含「tm_save_group」或「组级复盘」。 |
| **T4** | 组内多任务，最后一条完成触发 | 组 G 下有 A/B/C，A/B 已 completed，C 已 cancelled | 任意一条已完成的再发一次 update（幂等）或新完成一条 | 若本次 update 后组内全部 completed/cancelled，则响应 group_completed=true。 |
| **T5** | 组内仅 1 条任务，完成即组完成 | 组 G 下仅任务 A | tm_task update A status=completed summary="z" | 响应 group_completed=true, group_id=str(G.id)。 |

### 4.3 实现建议

- 组完成判定逻辑（U1）建议放在 `server.py` 内为 async 函数（如 `_is_group_completed(session, project, group_id) -> bool`），或放在 `TaskRepository` 中（如 `group_all_completed_or_cancelled(project, group_id)`），便于单测直接测该函数。
- 集成测可复用现有 `tests/test_server.py` 或 `tests/test_integration.py` 的 fixture（DB、client），对 `/api/v1/mcp` 或直接对 `tm_task` 调用进行 POST，解析 JSON 断言。

---

## 五、验收标准

满足以下全部条件视为本子任务通过。

| 编号 | 验收项 | 判定方式 |
|------|--------|----------|
| AC1 | 当且仅当「本次 update 为 status=completed 且带 summary」且「任务属于某任务组」且「该组内所有任务 status 均为 completed 或 cancelled」时，响应中包含 `group_completed: true`、`group_id`（字符串）、`group_completed_hint`（含「tm_save_group」或「组级复盘」）。 | 自动化测试 T3/T4/T5 通过；人工抽查 1 次。 |
| AC2 | 当任务无 group_id，或组内存在非 completed/cancelled 状态时，响应中不出现 `group_completed: true`（可无该键或为 false）。 | 自动化测试 T1/T2 通过。 |
| AC3 | 组完成判定不改变现有「单任务完成 → 沉淀经验」行为，即 sediment_experience_id 等仍按原逻辑返回。 | 现有相关单测/集成测仍通过；T1/T3 中检查 sediment 字段存在。 |
| AC4 | 代码通过 `ruff check src/`；新增或修改的单测通过 `pytest tests/ -v`。 | CI/本地执行。 |
| AC5 | .debug 或 README 中有一处说明：tm_task update 在组完成时会返回 group_completed、group_id、group_completed_hint。 | 文档审阅。 |

---

## 六、任务结束收口验证清单

- [x] 所有最小可执行单元 U1～U4 已完成并提交。
- [x] 单测 T1～T5 已实现且通过（见 tests/test_task_group_completed.py）。
- [x] 验收标准 AC1～AC5 均已满足。
- [x] `ruff check src/` 通过。
- [x] `pytest tests/test_task_group_completed.py tests/test_server.py tests/test_lifecycle.py` 通过。
- [ ] 文档已更新（任务管理相关文档）；TM 任务待你确认效果后更新为「已完成」。

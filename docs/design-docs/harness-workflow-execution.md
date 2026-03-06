# Harness 工作流执行设计

> 纯 Harness 的 Plan 执行流程：摸底步骤、执行记录、自动找文档、系统通知、与 feedback-loop 衔接。目标：工作流可复用、Agent 自动执行与自检、人把控大方向。

---

## 一、设计目标

| 目标 | 说明 |
|------|------|
| **工作流可复用** | 通用流程（含摸底）适用于后续 Plan/Task，保证项目不跑偏 |
| **减少人看文档** | Agent 按任务自动到指定目录找文档，人把控大方向 |
| **执行可追溯** | 一个 Plan 对应一个 execute 文档，中断后重启仍写入同一文档 |
| **流程可自检** | Agent 能检查自己的流程是否规范，不规范则中断并提醒 |
| **通知可感知** | 系统通知 + 回复中记录提示，执行人可及时知晓 |

---

## 二、Plan 执行记录

### 2.1 路径与命名

| 项目 | 约定 |
|------|------|
| **目录** | `docs/exec-plans/executing/` |
| **命名** | `{plan-id}-execute.md`，如 `harness-phase-3-execute.md` |
| **一个 Plan 一个文件** | 同一 Plan 的多次执行（含中断后重启）均追加写入同一文件 |

### 2.2 文档格式

```markdown
# Plan 执行记录：{Plan 名称}

> Plan ID: {plan-id} | 创建: {date} | 最后更新: {date}

## 执行摘要

| 字段 | 值 |
|------|-----|
| 状态 | 进行中 / 已中断 / 已完成 |
| 当前 Task | Task N |
| 最后节点 | {step 或 人类决策点} |

## 执行日志（按时间倒序，最新在上）

### {YYYY-MM-DD HH:mm} — {节点摘要}

- **动作**：{做了什么}
- **产出**：{文件、结果}
- **下一步**：{待执行或需确认}

---

（更多日志条目...）
```

### 2.3 追加写入规则

- 每次执行动作（Task 完成、人类决策点、中断、重启）后，**追加**一条日志到「执行日志」区。
- **最新在上**：新条目插入在「执行日志」区顶部，便于 AI 快速读取当前状态。
- **中断后重启**：读取同一 execute 文档，解析「当前 Task」「最后节点」，从断点继续；新日志仍追加到同一文件。

### 2.4 给 AI 的回复格式

Agent 在关键节点（Task 完成、中断、需确认）的回复中**必须**包含：

```
Plan 执行记录已更新：docs/exec-plans/executing/{plan-id}-execute.md
- 状态：{进行中/已中断/已完成}
- 当前 Task：Task N
- 请查看上述文档获取完整执行日志。
```

该格式供 AI 在后续对话中加载并理解执行进度，不依赖人类转述。

---

## 三、摸底步骤（通用 step-0）

### 3.1 定位

- 作为 **Plan 执行流程的固定 step-0**，在进入 Task 1 之前执行。
- 适用于所有 Plan，不限于 Phase 3。

### 3.2 内容（按 Plan 类型可配置）

| Plan 类型 | 摸底动作 | 产出 |
|-----------|----------|------|
| 架构约束类 | `python scripts/harness_import_check.py` | 违规清单（若有） |
| 文档迁移类 | 扫描目标路径、统计文件数 | 迁移范围清单 |
| 可观测性类（Phase 4） | 统计 logger 数量、确认 docs 结构、基线 ruff/pytest | 基线报告（logger 数、docs 路径、ruff/pytest 状态） |
| 通用 | 执行 Plan 中声明的「摸底命令」或跳过 | 基线报告 |

**Phase 4 类 step-0 示例**：`rg -l 'logging.getLogger' src/ --type py \| wc -l` 统计 logger；`ruff check src/`、`pytest tests/ -q` 基线；列出 `docs/design-docs`、`docs/exec-plans` 确认结构。产出为基线报告（logger 数量、docs 路径、ruff/pytest 状态）。

### 3.3 门控

- 摸底**必须**执行并通过（或产出可接受的基线报告），才能进入 Task 1。
- 若 Plan 未声明摸底，step-0 可简化为「确认 Plan 已加载、execute 文档已创建」。

### 3.4 自动化

- 在 harness 规则或 Plan 模板中**强制** step-0 存在。
- Agent 执行 Plan 时，**先**执行 step-0，**再**进入 Task 1；否则视为流程不规范，中断并提醒。

---

## 四、Agent 自动找文档

### 4.1 指定目录

| 目录 | 用途 |
|------|------|
| `docs/design-docs/` | 设计文档、机制说明、反馈回路 |
| `docs/exec-plans/` | 执行计划、execute 记录 |
| `docs/exec-plans/executing/` | 当前 Plan 的 execute 文档 |

### 4.2 查找策略

- **任务开始前**：根据 Plan 或任务描述，到 `docs/design-docs/` 查找相关文档（如 architecture-layers、feedback-loop、human-decision-points）。
- **任务执行中**：若涉及反馈回路、人类决策点，加载 `feedback-loop.md`、`human-decision-points.md`。
- **断点恢复**：加载 `docs/exec-plans/executing/{plan-id}-execute.md` 获取上次执行状态。

### 4.3 规则约定

- 在 harness-engineering 或专用规则中约定：**执行 Plan 前，Agent 须根据 Plan 内容自动加载 `docs/design-docs/` 下相关文档**。
- 不依赖人类事先阅读或转述。

---

## 五、系统通知

### 5.1 脚本接口

**路径**：`scripts/notify_plan_status.sh`

**用法**：
```bash
./scripts/notify_plan_status.sh "标题" "正文"
```

**实现**（跨平台）：
- macOS：`osascript -e 'display notification "正文" with title "标题"'`
- Linux：`notify-send "标题" "正文"`（需 libnotify）
- Windows：可选用 PowerShell 或第三方工具

### 5.2 调用时机

| 时机 | 标题 | 正文示例 |
|------|------|----------|
| Plan 开始 | Harness Plan | {plan-id} 已开始执行 |
| 人类决策点 | Harness 需确认 | {plan-id} Task N：{需确认内容}，请回复 |
| 中断 | Harness 已中断 | {plan-id} 已中断，原因：{简要} |
| Plan 完成 | Harness Plan | {plan-id} 已完成 |

### 5.3 与执行记录的关系

- 通知为**即时提醒**，供执行人感知。
- 执行记录为**持久日志**，供 AI 与人类回溯。
- 二者互补：通知触发查看，记录提供详情。

---

## 六、流程自检与中断

### 6.1 自检项

| 检查项 | 说明 |
|--------|------|
| step-0 已执行 | 摸底（或等效）已完成 |
| 人类决策点已确认 | 若当前为决策点，须收到用户肯定回复 |
| harness-check 通过 | 若涉及代码改动，提交前须 `make harness-check` |
| execute 已更新 | 每次关键节点后，execute 文档已追加 |

### 6.2 不规范时的动作

1. **停止执行**：不再进入下一 Task。
2. **写入 execute**：记录「流程不规范：{原因}」。
3. **回复中提示**：明确写出「Plan 执行记录已更新：xxx，流程不规范已中断，请确认后继续」。
4. **系统通知**：调用 `notify_plan_status.sh`，标题「Harness 流程中断」，正文含原因。

---

## 七、与 feedback-loop 的衔接

### 7.1 依赖关系

- harness 工作流**显式依赖** `docs/design-docs/feedback-loop.md`。
- 执行 Plan 时，Agent 须加载 feedback-loop，按其中「待完善项」与「沉淀方式」推进。

### 7.2 feedback-loop 结构（见该文档）

- **上方**：待完善项、待沉淀示例（最新、可执行）。
- **分界线**：`---` 或 `## 已完成（归档）`。
- **下方**：已完成的项（从上方移入，保持上方简洁）。

### 7.3 回溯修改

- 当某条待完善项已固化为 rules 或 docs 后，Agent 须：
  1. 将该条从 feedback-loop **上方**移至**下方**（已完成区）；
  2. 在下方添加完成时间、固化位置（如 rules 路径）；
  3. 保持上方永远为待办，下方为归档。

---

## 八、与现有文档的衔接

| 文档 | 衔接点 |
|------|--------|
| [feedback-loop](feedback-loop.md) | 待完善项结构、回溯修改规则、待完善项固化后移入已完成区 |
| [human-decision-points](human-decision-points.md) | 决策点清单、中断与确认 |
| [plan-self-review-checklist](plan-self-review-checklist.md) | Task 完成后自审 |
| [harness-engineering](.cursor/rules/harness-engineering.mdc) | 规则中引用本设计、自动找文档约定 |

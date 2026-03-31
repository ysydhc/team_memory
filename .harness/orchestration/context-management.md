# Context Management — Context 管理策略

> 来源：GSD 框架的 "Context Rot" 解法。长任务中 Agent 质量随 context 累积而下降，需主动管理。

## Layer 1: Context 监控与预警

| Context 使用率 | 状态 | 动作 |
|---|---|---|
| < `context.warning_threshold` | 正常 | 继续执行 |
| `warning_threshold` ~ `critical_threshold` | WARNING | 提醒尽快完成当前任务，准备保存状态到 `progress.md` |
| > `context.critical_threshold` | CRITICAL | 立即保存状态，完成当前步骤后停止；后续任务在新 session 中继续 |

阈值在 `harness-config.yaml` 的 `context` 块中配置。

---

## Layer 2: 任务拆分原则

在 Phase 1（Planning）中评估是否需要拆分。触发条件（满足任一）：

- 预估涉及文件 > `context.split_file_threshold` 个
- 涉及多个子系统 / 模块
- 预估 context 消耗 > 50%
- 任务包含可独立执行的并行子任务

### Wave-Based 并行模型

```
Wave 1（并行）：无依赖的子任务
  ├── Task A: 修改 models      → Subagent 1
  └── Task B: 修改 schemas     → Subagent 2

Wave 2（依赖 Wave 1）：
  └── Task C: 修改 services    → Subagent 3

Wave 3（依赖 Wave 2）：
  └── Task D: 修改 API routes  → Subagent 4
```

优先**垂直切片**（独立功能并行）而非**水平分层**（model → service → API 串行）。

---

## Layer 3: Subagent Context 隔离

当使用 subagent 执行子任务时，每个 subagent 只加载：

**必须加载**：
- 当前子任务的 Plan 描述
- `progress.md`（当前状态）
- 项目入口指令（CLAUDE.md / AGENTS.md）
- `read_first` 清单中指定的文件

**不加载**：
- 主 session 的对话历史
- 其他子任务的执行记录
- 已完成任务的详细输出

这确保每个 subagent 拥有 fresh context window，避免 context rot。

---

## Layer 4: Analysis Paralysis 防护

检测条件：连续 `context.max_consecutive_reads` 次 Read/Grep/Glob 操作而未执行任何 Edit/Write。

恢复动作：
1. 输出当前已收集的信息摘要
2. 提示 "请基于已有信息开始行动，或说明为何需要继续搜索"
3. 若再连续 3 次仍无写入 → 建议停下询问用户

**注意**：此防护仅在 Phase 2（Execution）中生效。Phase 0（Context Loading）和 Phase 1（Planning）中的密集读取是正常的。

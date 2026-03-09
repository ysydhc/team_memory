# tm 经验提取能力构建指南

> **目标读者**：将 Harness 工作流纳入自己项目的学习者  
> **目标**：最大化 tm 能力，实现 Agent 主动提取有用经验，无需用户显式说「任务完成」

---

## 一、当前项目缺什么

### 1.1 触发条件依赖 LLM 理解，不可控

现有规则（tm-extraction-retrieval 2.2）用自然语言描述触发条件，如「对话轮数 ≥ 10」「对话持续 > 30 分钟」。LLM 难以可靠执行：

- 无法准确统计对话轮数
- 无法获知会话开始时间
- 易遗漏或误判

### 1.2 缺少可读写的状态文件

没有持久化结构供 Agent 读取「当前轮数」「上次触发时间」等，无法做程序化判断。

### 1.3 流程到头未显式纳入规则

「Plan 完成」「tm_task group_completed」「flow-observer 结束」等自然结束点未在规则中明确，Agent 易忽略。

### 1.4 防抖机制缺失

若 24 小时内已触发提取，应避免重复。现有规则无此约束。

---

## 二、需要构建的组件

| 组件 | 路径 | 职责 |
|------|------|------|
| **状态文件** | `.cursor/extraction-state.json` | 存储 session_id、会话名称、轮数、上次触发时间等；Agent 读/写 |
| **规则补充** | `.cursor/rules/tm-extraction-retrieval.mdc` | 2.2 补充「流程到头」；新增 2.4 状态文件检查流程；6.1 强化 |

---

## 三、状态文件设计

### 3.1 路径与格式

- **路径**：`.cursor/extraction-state.json`
- **格式**：JSON
- **倒序**：`sessions` 数组顶部为最新会话

### 3.2 结构

```json
{
  "sessions": [
    {
      "session_id": "sess_20250307_001",
      "session_name": "工作流可视化迁移",
      "session_started_at": "2025-03-07T09:00:00Z",
      "last_message_at": "2025-03-07T10:30:00Z",
      "current_round": 15,
      "total_records": 3,
      "last_trigger_at": "2025-03-07T09:30:00Z"
    }
  ]
}
```

| 字段 | 说明 |
|------|------|
| session_id | 会话唯一标识，新会话时生成（如 sess_YYYYMMDD_NNN） |
| session_name | 会话名称，可从首条用户消息摘要或默认「未命名会话」 |
| session_started_at | 会话开始时间（ISO 8601），用于「对话持续 > 30 分钟」判定 |
| last_message_at | 最近一条消息时间（ISO 8601），用于新会话判定（距当前 > 24h 则新会话） |
| current_round | 当前对话轮数（1 轮 = 1 用户消息 + 1 Agent 回复） |
| total_records | 本会话已触发提取的次数 |
| last_trigger_at | 上次触发提取的时间，用于 24 小时防抖；null 表示未触发过 |

### 3.3 新会话判定

当 `last_message_at` 距当前超过 **24 小时**，或文件不存在时，视为新会话：新建 session 并置于 `sessions` 顶部，`current_round` 从 0 开始。

### 3.4 谁更新

**Agent** 在每次回复末尾负责：

1. 读取 `.cursor/extraction-state.json`
2. 定位当前会话（`sessions[0]`）
3. 更新 `current_round += 1`、`last_message_at = now`
4. 若本次触发了提取，更新 `last_trigger_at`、`total_records += 1`
5. 写回文件，保持 `sessions` 倒序（顶部最新）

---

## 四、规则修改要点

### 4.1 2.2 触发条件补充

新增：

| 条件 | 说明 |
|------|------|
| Plan 完成 | 执行的 Plan 全部 Task 完成且验收通过 |
| 流程到头 | tm_task step-complete 返回 group_completed、flow-observer 输出报告、用户说「当前任务结束」等 |
| 状态文件满足 | 读取 `.cursor/extraction-state.json`，`current_round >= 10` 或 `last_message_at` 距会话开始 > 30 分钟，且 `last_trigger_at` 距现在 > 24 小时（防抖） |

### 4.2 新增 2.4 状态文件检查流程

**每次回复前**：

1. 读取 `.cursor/extraction-state.json`
2. 若无文件或 `sessions` 为空，创建初始结构
3. 若 `last_message_at` 超过 24 小时，新建会话并置于顶部

**每次回复后**：

1. 更新 `current_round`、`last_message_at`
2. 若本次执行了提取，更新 `last_trigger_at`、`total_records`
3. 写回文件

### 4.3 6.1 项目专属强化

- 明确：**每次回复前必须读取状态文件**，结合 2.2 条件判断是否评估提取
- 明确：**每次回复后必须更新状态文件**（轮数、时间、触发记录）

---

## 五、有价值经验的场景（纳入 2.1）

以下场景应提取（与 2.1 表一致，此处强调）：

1. **Bug 修复**：根因、修复步骤、验证方式
2. **技术选型**：备选方案、选择理由、约束条件
3. **架构/配置变更**：变更内容、原因、回滚方式
4. **Plan 执行**：关键决策、踩坑、解法
5. **工作流/协作改进**：流程优化、规则更新建议
6. **用户偏好**：语言、表达风格、工具习惯（存为 user_preference）

---

## 六、分步实施清单

| 步骤 | 动作 | 验收 |
|------|------|------|
| 1 | 创建 `.cursor/extraction-state.json` 初始文件 | 文件存在，结构正确 |
| 2 | 修改 `tm-extraction-retrieval.mdc`：2.2 补充、新增 2.4、6.1 强化 | 规则完整可执行 |
| 3 | 确保规则与本文档一致 | 规则可执行、无矛盾 |
| 4 | 将 `.cursor/extraction-state.json` 加入 `.gitignore`（若不想提交会话状态） | 可选 |
| 5 | 人工验证：多轮对话后，Agent 是否读取状态、是否在满足条件时触发提取 | 行为符合预期 |

---

## 七、注意事项

1. **session_id 来源**：Cursor 不提供 Chat ID，Agent 可用 `sess_` + 日期 + 序号生成；新会话判定依赖 24h 时间差。
2. **session_name**：可从首条用户消息截取，或默认「未命名会话」。
3. **并发**：同一 workspace 通常单 Chat 活跃，状态文件单写即可；若未来多 Agent 并行，需考虑锁或合并策略。
4. **.gitignore**：会话状态含时间等本地信息，建议忽略；若需团队共享「轮数阈值」等配置，可拆出 `extraction-config.json` 并提交。

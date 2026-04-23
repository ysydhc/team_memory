---
name: memory-pipeline
description: Hermes 记忆管线 skill — 自动检索、草稿缓冲、收敛检测、评估标注
triggers:
  - 每轮对话开始时自动执行
  - 用户提到历史经验时
---

# Hermes 记忆管线

## 触发条件

每轮对话自动执行（不需要用户手动调用）

## 执行步骤

### 步骤 1：自动检索

每轮对话开始时，根据用户消息判断是否需要检索记忆：

- 如果用户消息包含关键词（"之前"/"上次"/"经验"/"踩坑"/"remember"/"previously"）
  → 调用 TM MCP 的 memory_recall 工具进行全量检索
- 否则 → 调用 TM MCP 的 memory_context 工具获取项目级上下文

### 步骤 2：标注记忆来源

在回复中引用检索到的记忆时，必须保留 [mem:xxx] 标记：
- 正确："根据你之前的经验 [mem:exp-a1b2]，Docker 网络用 TUN 模式"
- 错误："根据你之前的经验，Docker 网络用 TUN 模式"（丢失了标记）

这个标记用于评估系统自动判定记忆是否被使用。

### 步骤 3：收敛检测

在以下情况判断当前任务已收敛：
- 用户说"解决了"/"完成了"/"先这样"/"搞定"
- 执行了 git commit
- 执行了测试且通过

### 步骤 4：草稿写入

收敛时 → 调用 TM MCP 的 memory_draft_publish 发布草稿
未收敛 → 调用 TM MCP 的 memory_draft_save 保存草稿

### 步骤 5：会话结束

会话结束时，如果有未发布的草稿 → 强制发布

## 管线脚本

Hermes 可以直接调用 hermes_pipeline.py 脚本：
- on_turn_start(user_message, project) → 检索记忆
- on_turn_end(session_id, agent_response, project, recent_tools) → 更新草稿
- on_session_end(session_id) → 强制发布

## 评估

系统自动通过 [mem:xxx] 标记评估记忆使用情况：
- 使用率 > 40% → 系统有效
- 使用率 20-40% → 需要调整
- 使用率 < 20% → 系统无效

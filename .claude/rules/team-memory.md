---
description: Team Memory 使用规则（Lite 模式）
---

## Team Memory

> 如果 MCP 不可用（如 Agent 无 MCP 支持），所有工具均可通过 `tm-cli` 命令行等价调用。
> 前提：`TEAM_MEMORY_API_KEY` 环境变量已设置，Web 服务运行中（`make dev`）。

6 个工具，2 条核心规则：

1. **遇到问题 → 先 `memory_recall`，再动手**
2. **解决问题 / 做了决定 / 发现坑 → `memory_save`**

| 场景 | MCP 工具 | CLI 等价 |
|------|---------|---------|
| 开始新任务 | `memory_context(file_paths=[...])` | `tm-cli context --file-paths "a.py,b.py"` |
| 遇到 bug / 错误 | `memory_recall(problem="...")` | `tm-cli recall --problem "..."` |
| 探索性搜索 | `memory_recall(query="...")` | `tm-cli recall --query "..."` |
| recall 命中档案需全文 | `memory_get_archive(archive_id="...")` | `tm-cli get-archive --id "..."` |
| 会话/计划归档（文案） | `memory_archive_upsert(title=..., solution_doc=..., ...)` | `tm-cli archive --title "..." --solution-doc "..."` |
| 不熟悉的代码 | `memory_recall(file_path="...")` | `tm-cli recall --file-path "..."` |
| 解决了问题 | `memory_save(title=..., problem=..., solution=...)` | `tm-cli save --title "..." --problem "..." --solution "..."` |
| 长对话有价值内容 | `memory_save(content="...", scope="project")` | `tm-cli save --content "..." --scope project` |
| 个人偏好 | `memory_save(title=..., problem=..., solution=..., scope="personal")` | `tm-cli save --title "..." ... --scope personal` |
| 搜索结果有帮助 | `memory_feedback(experience_id=..., rating=5)` | `tm-cli feedback --experience-id "..." --rating 5` |

## 自动触发检查点

在以下时刻，评估是否需要 `memory_save`：

1. **TaskUpdate(completed)** — 这个 task 中是否遇到了非预期错误并修复？是否做了选型决策？
2. **用户纠正你的做法** — 存 feedback（what you did wrong → correct approach）
3. **用户提供凭据/配置** — 存 reference（scope=personal）
4. **修复了测试失败或 lint 报错** — 存 problem/solution

不存的：正常完成无波折的 task、已在代码/commit 中体现的改动、临时调试信息。

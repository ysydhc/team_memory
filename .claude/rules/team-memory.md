---
description: Team Memory 使用规则（Lite 模式）
---

## Team Memory

6 个工具，2 条核心规则：

1. **遇到问题 → 先 `memory_recall`，再动手**
2. **解决问题 / 做了决定 / 发现坑 → `memory_save`**

| 场景 | 工具 | 示例 |
|------|------|------|
| 开始新任务 | `memory_context(file_paths=[...])` | 返回 `profile.static` / `profile.dynamic`（字符串数组）及相关团队经验 |
| 遇到 bug / 错误 | `memory_recall(problem="...")` | 检查团队是否已有方案；只 recall 不调 context 时可加 `include_user_profile=True` |
| 探索性搜索 | `memory_recall(query="...")` | 搜索相关经验；需要含档案时加 `include_archives=True` |
| recall 命中档案需全文 | `memory_get_archive(archive_id="...")` | `type=archive` 仅为预览，按需拉 L2 |
| 会话/计划归档（文案） | `memory_archive_upsert(title=..., solution_doc=..., ...)` | 与 `POST /api/v1/archives` 等价；大文件用 HTTP 或 **`python -m team_memory.cli upload`**（见 [docs/guide/mcp-server.md](../../docs/guide/mcp-server.md)） |
| 不熟悉的代码 | `memory_recall(file_path="...")` | 获取文件相关经验 |
| 解决了问题 | `memory_save(title=..., problem=..., solution=...)` | 保存到团队知识库 |
| 长对话有价值内容 | `memory_save(content="...", scope="project")` | LLM 自动提取保存 |
| 个人偏好 | `memory_save(title=..., problem=..., solution=..., scope="personal")` | 保存个人偏好 |
| 搜索结果有帮助 | `memory_feedback(experience_id=..., rating=5)` | 提升未来排名 |

## 自动触发检查点

在以下时刻，评估是否需要 `memory_save`：

1. **TaskUpdate(completed)** — 这个 task 中是否遇到了非预期错误并修复？是否做了选型决策？
2. **用户纠正你的做法** — 存 feedback（what you did wrong → correct approach）
3. **用户提供凭据/配置** — 存 reference（scope=personal）
4. **修复了测试失败或 lint 报错** — 存 problem/solution

不存的：正常完成无波折的 task、已在代码/commit 中体现的改动、临时调试信息。

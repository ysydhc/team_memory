---
name: memory_experience
description: 引导式保存 TeamMemory「经验 Experience」：tm-cli save + 去重 recall；整段会话档案馆请用 /memory_archive
user-invocable: true
allowed-tools:
  - Read
  - Shell
---

# /memory_experience

引导用户把**一条可语义检索的经验（Experience）**写入 TeamMemory。
与 **`/memory_archive`（档案馆 L0/L1/L2）分工不同**：本流程只做 **经验条目的规范保存**，不做档案馆结构化归档。

## 工具与边界

> **优先使用 `tm-cli`**（Shell 调用），前提：`TEAM_MEMORY_API_KEY` 环境变量已设置，Web 服务运行中（`make dev`）。
> MCP 工具（`memory_save` / `memory_recall`）作为备选，仅在 CLI 不可用时使用。

| 目标 | CLI 命令 | MCP 备选 |
|------|---------|---------|
| 单条经验（标题 + 问题 + 方案） | `tm-cli save --title "..." --problem "..." --solution "..."` | `memory_save(title=..., problem=..., solution=...)` |
| 整段长文，让服务端解析 | `tm-cli save --content "..." --scope project` | `memory_save(content=..., scope=...)` |
| 去重搜索 | `tm-cli recall --problem "..."` | `memory_recall(problem=...)` |
| 探索搜索 | `tm-cli recall --query "..."` | `memory_recall(query=...)` |
| **团队默认可见** | `--scope project`（默认） | `scope="project"` |
| **个人偏好** | `--scope personal` | `scope="personal"` |
| **档案馆** | `/memory_archive` skill | `memory_archive_upsert` |

经验类型 **`--experience-type`**（可选）：`general`、`feature`、`bugfix`、`tech_design`、`incident`、`best_practice`、`learning`。可与用户口语（「踩坑」「决策」「最佳实践」）映射。

可选参数：`--tags "t1,t2"`、`--project "..."`、`--group-key "..."`。

## 步骤

1. **分辨需求**
   若用户要的是「整段会话进档案馆、要 L1 overview / L2 全文、要附件」→ 引导使用 **`/memory_archive`**，**不要**用本 skill 代替档案馆流程。

2. **确认 scope**
   默认 **`project`**；仅当用户明确是个人偏好/私人注记时用 **`personal`**。

3. **引导填字段（直接保存）**
   - **title**：简短标题
   - **problem**：背景 / 遇到的问题
   - **solution**：方案、结论或行动项
   - **tags**：可选，可你方建议后由用户定稿
   - **experience_type**：可选，从上表选

4. **去重**
   执行 `tm-cli recall --problem "<问题描述>"` 或 `tm-cli recall --query "<关键词>"`。若已有高度相似条目，提示用户：放弃、补充差异后再存，或仍坚持新增。

5. **写入**
   - 常规：
     ```bash
     tm-cli save --title "<标题>" --problem "<问题>" --solution "<方案>" \
       --tags "tag1,tag2" --scope project
     ```
   - 大段粘贴：
     ```bash
     tm-cli save --content "<长文内容>" --scope project --tags "tag1,tag2"
     ```
   - 如需指定类型/分组：加 `--experience-type bugfix --group-key "sprint-42"`

6. **反馈**
   阅读命令输出的 JSON：成功则告知用户 **经验 id**、`status` 等；若出现 `error`、`duplicate_detected`、`validation_error` 等，根据 `message` / `code` 向用户说明下一步。

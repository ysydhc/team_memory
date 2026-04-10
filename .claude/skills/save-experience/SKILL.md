---
name: save-experience
description: 引导式保存 TeamMemory「经验 Experience」：memory_save + 去重 recall；整段会话档案馆请用 /archive
user-invocable: true
allowed-tools:
  - Read
  - mcp__team_memory__memory_save
  - mcp__team_memory__memory_recall
---

# /save-experience

引导用户把**一条可语义检索的经验（Experience）**写入 TeamMemory。  
与 **`/archive`（档案馆 L0/L1/L2、`memory_archive_upsert`）分工不同**：本流程只做 **经验条目的规范保存**，不做档案馆结构化归档。

## 与当前 MCP 设计的边界

| 目标 | 工具 / 路径 |
|------|-------------|
| 单条经验（标题 + 问题/背景 + 方案或结论） | **`memory_save`**，**`title` + `problem` 必填**；**`solution`** 可缺省（`None`），仍建议填写便于检索 |
| 整段会话/长文，让服务端解析成经验 | **`memory_save(content=..., scope=...)`** |
| **团队默认可见** | `scope="project"`（**默认**） |
| **个人偏好、与工作无关的个人注记** | `scope="personal"` |
| **档案馆**（overview、`solution_doc`、关联经验 ID、附件等） | **`/archive`** 或 **`memory_archive_upsert`** / `POST /api/v1/archives` |
| **禁止** | **`memory_save(scope="archive")`** — 已移除，将返回 `code=scope_removed` |

经验类型 **`experience_type`**（可选，与实现一致）：`general`、`feature`、`bugfix`、`tech_design`、`incident`、`best_practice`、`learning`。可与用户口语（「踩坑」「决策」「最佳实践」）映射到上列枚举。

可选参数：**`tags`**、**`project`**（租户）、**`group_key`**（如迭代名/需求名），与 `memory_save` 一致。

## 步骤

1. **分辨需求**  
   若用户要的是「整段会话进档案馆、要 L1 overview / L2 全文、要附件」→ 引导使用 **`/archive`**，**不要**用本 skill 代替档案馆流程。

2. **确认 scope**  
   默认 **`project`**；仅当用户明确是个人偏好/私人注记时用 **`personal`**。

3. **引导填字段（直接保存）**  
   - **title**：简短标题  
   - **problem**：背景 / 遇到的问题  
   - **solution**：方案、结论或行动项  
   - **tags**：可选，可你方建议后由用户定稿  
   - **experience_type**：可选，从上表选  

4. **去重**  
   调用 **`memory_recall`**（`problem=` 或 `query=`；一般不必 `include_archives`，除非用户明确要对照档案）。若已有高度相似条目，提示用户：放弃、补充差异后再存，或仍坚持新增。

5. **写入**  
   - 常规：`memory_save(...)` 传入上述字段及 `scope` / `experience_type` / `project` / `group_key`（若有）。  
   - 大段粘贴：优先 **`memory_save(content=..., scope=..., tags=...)`**。

6. **反馈**  
   阅读工具返回的 JSON：成功则告知 **经验 id**、`status` 等；若 **`error`**、**`duplicate_detected`**、**`validation_error`** 等，用返回里的 `message` / `code` 向用户说明下一步。

## MCP 工具名前缀

若会话中 MCP 服务名不是 `team_memory`，`mcp__team_memory__...` 在你的环境里可能不同；以客户端实际注册的 **`memory_save` / `memory_recall`** 为准，逻辑不变。

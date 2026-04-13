---
name: memory_archive
description: 归档当前会话的知识到团队档案馆
user_invocable: true
---

# /memory_archive — 会话知识归档

将当前会话中的决策、方案、结论归档到团队档案馆，供未来 Agent 以最小 Token 成本检索定位。

> **优先使用 `tm-cli`**（Shell 调用），前提：`TEAM_MEMORY_API_KEY` 环境变量已设置，Web 服务运行中（`make dev`）。

## 归档流程

### 1. 确定主题

如果用户未指定归档主题，基于当前对话推断并询问确认。

### 2. 提取 L0（搜索元数据，~100 tok）

- **title**: 简洁标题（<100 字符）
- **content_type**: `session_archive` | `tech_design` | `incident_review` | `decision_record` | 自定义
- **value_summary**: 一句话价值描述（<200 字符），回答"这个归档对未来有什么用？"
- **tags**: 3-8 个标签

### 3. 生成 L1 overview（判断依据，~500-2000 tok）

按以下模板生成结构化内容：

```text
## 核心决策
| 决策 | 选定方案 | 理由 |
|------|----------|------|
| ... | ... | ... |

## 关键结论
- ...
- ...

## 适用条件与边界
- 适用于：...
- 不适用于：...
```

### 4. 生成 L2（完整知识体）

- **solution_doc**: 完整叙述（问题背景 → 方案选型 → 实施要点 → 验证结果）
- **conversation_summary**: 对话关键转折点摘要（500-1000 tok）

### 5. 关联经验

列出本次 session 中通过 `tm-cli save` 保存的 experience IDs。

### 6. 检测去重

调用 API 前告知用户：如果同名档案已存在，将被覆盖更新。

### 7. 调用 API 完成归档

**通路 A（推荐）：tm-cli**

```bash
tm-cli archive \
  --title "<标题>" \
  --content-type "<类型>" \
  --value-summary "<一句话价值>" \
  --tags "tag1,tag2" \
  --overview "<L1 内容>" \
  --solution-doc "<L2 内容>" \
  --summary "<对话摘要>" \
  --linked-experience-ids "uuid1,uuid2" \
  --project "<项目名>"
```

L1/L2 内容较长时，可写入临时文件后使用 `--overview-file` 和 `--solution-file`：

```bash
tm-cli archive \
  --title "<标题>" \
  --overview-file "/tmp/overview.md" \
  --solution-file "/tmp/solution.md" \
  --tags "tag1,tag2" \
  --project "<项目名>"
```

**通路 B（备选）：HTTP API**

```bash
curl -s -X POST "${TM_BASE_URL:-http://localhost:9111}/api/v1/archives" \
  -H "Authorization: Bearer ${TEAM_MEMORY_API_KEY}" \
  -H "Content-Type: application/json" \
  -d '{
    "title": "<标题>",
    "content_type": "<类型>",
    "value_summary": "<一句话价值>",
    "tags": ["tag1", "tag2"],
    "overview": "<L1 结构化内容>",
    "solution_doc": "<L2 完整叙述>",
    "conversation_summary": "<对话摘要>",
    "linked_experience_ids": ["uuid1", "uuid2"],
    "project": "<项目名>"
  }'
```

**通路 C（备选）：MCP**

使用 **`memory_archive_upsert`**（仅当 MCP 可用且已配置时）。拿到 `archive_id` 后，附件仍需走 CLI 或 HTTP。

> 其余 memory_* 操作（save、recall、context、feedback、get-archive）也可通过 `tm-cli` 等价调用，详见 `tm-cli --help`。

### 8. 上传附件（如有 Plan 文件）

```bash
tm-cli upload \
  --archive-id "<archive_id>" \
  --file "<本地文件路径>" \
  --kind plan_doc \
  --project "<项目名>"
```

或：

```bash
curl -X POST "${TM_BASE_URL:-http://localhost:9111}/api/v1/archives/<archive_id>/attachments/upload?project=<项目名>" \
  -H "Authorization: Bearer ${TEAM_MEMORY_API_KEY}" \
  -F "file=@<本地文件路径>" \
  -F "kind=plan_doc"
```

### 9. 清理临时文件（成功归档后）

若写过**仅用于本次归档**的本地草稿（例如 `/tmp/overview.md`、`/tmp/solution.md`），在 **API 返回成功**（`Archive created` / `updated`）且**已用这些文件做完附件上传**（若有）之后，**删除**它们，避免残留。

- 若归档 **失败**（HTTP 错误、连接失败、未拿到 `archive_id`）：**保留**临时文件，便于重试或排查。

### 10. 本地归档

将已归档的 Plan 文件移至 `.harness/plans/completed/`：

```bash
mv .harness/plans/<plan-file>.md .harness/plans/completed/
```

## 注意事项

- L0/L1 由你（Agent）生成，不依赖后端 LLM
- 文件直接上传，不通过 MCP 传输大体积内容
- 同名档案（title+project）会覆盖更新，API 响应中 `action` 字段标识操作类型
- 归档完成后告知用户 archive_id 和操作结果；成功路径下勿忘 **§9删除临时文件**

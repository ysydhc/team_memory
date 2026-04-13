---
description: MCP 工具层 — 修改 server.py 时遵守
paths:
  - "src/team_memory/server.py"
---

# mcp-tools

修改/新增 `server.py` 中 MCP 工具时遵守：

## 签名
```python
@mcp.tool(name="memory_xxx", description="...")
@track_usage
async def memory_xxx(...) -> str:
```
- 命名空间：`memory_` 前缀
- 返回 `str`（JSON）
- 禁止异常透传

## 响应格式

### 错误响应
所有错误统一格式：
```json
{"error": true, "message": "Human readable message", "code": "error_code"}
```
常见 code 值：`validation_error`、`not_found`、`internal_error`、`content_too_long`、`scope_removed`、`embedding_failed`

### 成功响应
使用 `data` 包装返回数据：
```json
{"message": "Knowledge saved.", "data": {"id": "...", "title": "...", "status": "..."}}
```
- 列表结果使用 `{"results": [...]}` 格式
- 重复检测：`{"message": "...", "duplicate_detected": true, "data": {"candidates": [...]}}`

### memory_recall profile 字段
`memory_recall` 响应始终包含 `profile` 字段：
- `include_user_profile=True` 时返回 `{"profile": {"static": [...], "dynamic": [...]}}`
- `include_user_profile=False`（默认）时返回 `{"profile": null}`

## 当前工具（6 个）

| 工具 | 用途 |
|------|------|
| `memory_save` | 保存经验（直接或 LLM 提取） |
| `memory_recall` | 搜索团队知识库（solve/search/suggest 三模式） |
| `memory_context` | 获取用户画像 + 相关经验 |
| `memory_get_archive` | 按 ID 获取归档全文（L2） |
| `memory_archive_upsert` | 创建/更新档案馆记录（与 `POST /api/v1/archives` 一致） |
| `memory_feedback` | 对搜索结果评分（1-5） |

## 参数
- 必填无默认值；可选有默认值；`project` 用 `_resolve_project(project)`

## 测试
- 正常、空结果、service 报错返回 error 格式
- 测试在 `tests/test_server.py`

## CLI 兼容端点

每个 MCP 工具都有对应的 REST API 和 CLI 命令，行为与返回格式一致：

| MCP 工具 | REST 端点 | CLI 命令 |
|----------|----------|---------|
| `memory_save` | `POST /api/v1/mcp/save` | `tm-cli save` |
| `memory_recall` | `POST /api/v1/mcp/recall` | `tm-cli recall` |
| `memory_context` | `POST /api/v1/mcp/context` | `tm-cli context` |
| `memory_get_archive` | `GET /api/v1/mcp/archive/{id}` | `tm-cli get-archive` |
| `memory_archive_upsert` | `POST /api/v1/mcp/archive-upsert` | `tm-cli archive` |
| `memory_feedback` | `POST /api/v1/mcp/feedback` | `tm-cli feedback` |

新增 MCP 工具时须同步添加 REST 端点和 CLI 子命令。

详见 [README.md](../../README.md)、[docs/guide/mcp-server.md](../../docs/guide/mcp-server.md)。

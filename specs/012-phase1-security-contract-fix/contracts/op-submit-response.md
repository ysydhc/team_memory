# Contract: `memory_operations.op_submit_response`

**Module**: `services/memory_operations.py`
**Status**: New function

## Signature

```python
async def op_submit_response(
    user: str,
    *,
    query: str,
    response: str,
    result_ids: list[str] | None = None,
    project: str | None = None,
) -> dict:
    """Buffer an agent response for faithfulness evaluation.

    将 agent 的响应暂存到 response_buffer 表，供后续 faithfulness_batch 异步评估。
    """
```

## Input Validation

| Parameter | Type | Required | Validation |
|-----------|------|----------|------------|
| `user` | `str` | Yes | 非空 |
| `query` | `str` | Yes | 截断至 2000 字符 |
| `response` | `str` | Yes | 截断至 10000 字符 |
| `result_ids` | `list[str] \| None` | No | 每项须为有效 UUID 字符串 |
| `project` | `str \| None` | No | 走 `resolve_project` 推断 |

## Return Value

### Success

```python
{
    "status": "buffered",
    "buffer_id": "550e8400-e29b-41d4-a716-446655440000"
}
```

### Error

```python
{
    "error": True,
    "code": "internal_error",
    "message": "Failed to buffer response: ..."
}
```

## Implementation Sketch

```python
async def op_submit_response(user, *, query, response, result_ids=None, project=None):
    project = _resolve_project(project)
    buffer_id = str(uuid_mod.uuid4())

    truncated_query = query[:2000]
    truncated_response = response[:10000]
    ids_json = json.dumps(result_ids or [])

    try:
        async with get_session() as session:
            await session.execute(
                text("""
                    INSERT INTO response_buffer (id, query, agent_response, result_ids, project, created_at)
                    VALUES (:id, :query, :response, :result_ids::jsonb, :project, now())
                """),
                {
                    "id": buffer_id,
                    "query": truncated_query,
                    "response": truncated_response,
                    "result_ids": ids_json,
                    "project": project,
                },
            )
            await session.commit()
        logger.info("response buffered: id=%s project=%s", buffer_id, project)
        return {"status": "buffered", "buffer_id": buffer_id}
    except Exception as e:
        logger.error("op_submit_response failed: %s", e)
        return {"error": True, "code": "internal_error", "message": f"Failed to buffer response: {e}"}
```

## Callers

1. **`server.py`** (`memory_submit_response` MCP tool) — local mode
2. **`remote_client.py`** (`RemoteMCPClient.op_submit_response`) — remote mode via HTTP
3. **`mcp_compat.py`** (`POST /mcp/submit-response`) — HTTP endpoint for remote mode

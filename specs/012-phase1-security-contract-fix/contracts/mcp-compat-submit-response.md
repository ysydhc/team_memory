# Contract: POST /api/v1/mcp/submit-response

**Module**: `web/routes/mcp_compat.py`
**Status**: New endpoint

## Request

```
POST /api/v1/mcp/submit-response
Content-Type: application/json
Authorization: Bearer <API_KEY>
```

### Request Body

```json
{
  "query": "string (required, max 2000 chars)",
  "response": "string (required, max 10000 chars)",
  "result_ids": ["uuid-1", "uuid-2"],
  "project": "string (optional)"
}
```

### Pydantic Model

```python
class McpSubmitResponseRequest(BaseModel):
    query: str
    response: str
    result_ids: list[str] | None = None
    project: str | None = None
```

## Response

### 200 OK

```json
{
  "status": "buffered",
  "buffer_id": "uuid"
}
```

### 400 Bad Request

```json
{
  "error": true,
  "code": "validation_error",
  "message": "query and response are required"
}
```

### 500 Internal Server Error

```json
{
  "error": true,
  "code": "internal_error",
  "message": "Failed to buffer response: <detail>"
}
```

## Implementation

```python
@router.post("/submit-response")
async def mcp_submit_response(
    body: McpSubmitResponseRequest,
    user: User = Depends(get_current_user),
):
    uid = user.name
    try:
        result = await memory_operations.op_submit_response(
            uid,
            query=body.query,
            response=body.response,
            result_ids=body.result_ids,
            project=body.project,
        )
        return _json_response(result, user=uid, tool_name="submit_response")
    except HTTPException:
        raise
    except Exception as e:
        logger.exception("mcp_compat submit_response failed: %s", e)
        raise HTTPException(status_code=500, detail=str(e)) from e
```

## Alignment

此路由与现有 6 个 MCP compat 路由保持一致的模式：
- Request body → Pydantic model
- Auth via `get_current_user`
- Delegate to `memory_operations.op_*`
- Response via `_json_response` helper

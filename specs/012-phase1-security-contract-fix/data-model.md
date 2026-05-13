# Data Model: Phase 1 — Security & Contract Fix

**Branch**: `012-phase1-security-contract-fix` | **Date**: 2026-05-13

## Schema Changes

**无数据库 schema 变更。** 无需 Alembic 迁移。

所有修改均为代码层（数据通路 / 序列化），表结构保持不变。

---

## Entity: `response_buffer` (existing table)

### Current Schema

| Column | Type | Nullable | Default | Notes |
|--------|------|----------|---------|-------|
| `id` | UUID | NOT NULL | `gen_random_uuid()` | PK |
| `query` | TEXT | NOT NULL | — | 截断 2000 字符 |
| `agent_response` | TEXT | NOT NULL | — | 截断 10000 字符 |
| `result_ids` | JSONB | NULL | — | 搜索结果 UUID 列表 |
| `project` | VARCHAR(255) | NULL | — | 项目名 |
| `created_at` | TIMESTAMPTZ | NOT NULL | `now()` | 创建时间 |

### Data Path Change

```
Before (C2):
  server.py → asyncpg.connect(hardcoded DSN) → INSERT INTO response_buffer

After:
  server.py → memory_operations.op_submit_response()
            → get_session() → session.execute(text(INSERT))
```

### `result_ids` Format Change

```json
// Before (server.py 当前代码)
[{"": "uuid-1"}, {"": "uuid-2"}]

// After (规范化)
["uuid-1", "uuid-2"]
```

**Impact**: `daemon/faithfulness_batch.py` 在 `fetch_pending` 时 `json.loads(result_ids)` 返回类型从 `list[dict]` 变为 `list[str]`。需确认下游兼容。

---

## Entity: `SearchPipelineResult` (dataclass, in-memory)

### Current Definition (`services/search_pipeline.py`)

```python
@dataclass
class SearchPipelineResult:
    results: list[dict[str, Any]]
    total_candidates: int
    search_type: str
    reranked: bool
    cached: bool
    duration_ms: int
    stage_metrics: dict[str, int] = field(default_factory=dict)
```

### Serialization Change

```
Before (C5):
  cache.put(key, result)   # result 是 SearchPipelineResult dataclass
  json.dumps(result)       # ❌ TypeError: Object of type SearchPipelineResult is not JSON serializable

After:
  from dataclasses import asdict
  cache.put(key, asdict(result))  # dict, json.dumps 正常
  cache.get(key)                  # 返回 dict，需标记 cached=True
```

### Validation Rules

- `results`: list of dicts，每个 dict 含 `id`, `title`, `description`, `score`, `tags` 等
- `total_candidates`: >= 0
- `search_type`: `"hybrid"` | `"vector"` | `"fulltext"`
- `reranked`: bool
- `cached`: bool（从缓存取出时设为 True）
- `duration_ms`: >= 0
- `stage_metrics`: 可选，各阶段耗时

---

## Entity: `tokenizer` (module relocation)

### Migration

```
Before: src/team_memory/services/tokenizer.py
After:  src/team_memory/utils/tokenizer.py
```

### Interface (unchanged)

```python
def tokenize(text: str) -> str:
    """jieba 分词，返回空格分隔的分词结果。"""
```

### Consumer Update

| Consumer | Old Import | New Import |
|----------|-----------|------------|
| `storage/repository.py` | `from team_memory.services.tokenizer import tokenize` | `from team_memory.utils.tokenizer import tokenize` |
| `storage/query_builders.py` | (if used) | `from team_memory.utils.tokenizer import tokenize` |
| `tests/` | `from team_memory.services.tokenizer import tokenize` | `from team_memory.utils.tokenizer import tokenize` |

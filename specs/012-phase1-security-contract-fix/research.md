# Research: Phase 1 — Security & Contract Fix

**Branch**: `012-phase1-security-contract-fix` | **Date**: 2026-05-13

## R1: `op_submit_response` 实现方案

### Decision
在 `memory_operations.py` 新增 `async def op_submit_response(user, *, query, response, result_ids, project)` 函数，使用 `get_session` + `session.execute(text(INSERT SQL))` 写入 `response_buffer` 表。

### Rationale
- `response_buffer` 表未在 SQLAlchemy ORM models.py 中映射，走 raw SQL 是最低侵入路径
- MCP 工具调用频率远低于 Daemon 批处理，`get_session`（SQLAlchemy async session）的连接池足够
- 与 Daemon `ResponseBuffer.add` 逻辑对齐：截断 query[:2000] / response[:10000]、生成 UUID、写入 created_at
- 统一 `result_ids` 为 `list[str]`（纯 UUID 字符串列表），修正当前 `[{"": rid}]` 的可疑格式

### Alternatives Considered
| Alternative | Rejected Because |
|---|---|
| 复用 Daemon `ResponseBuffer` 实例 | 需在 MCP 进程初始化 asyncpg 池，与 bootstrap 的 SQLAlchemy session 不一致 |
| 为 `response_buffer` 添加 ORM Model + Repository | 改动面大，daemon 侧不会改为 ORM；与现有 `search_log_writer` 的 raw asyncpg 模式一致 |
| 直接在 server.py 修复连接池但保持分层违背 | 不解决根本问题，仍绕过 Services 层 |

---

## R2: tokenizer 迁移目标位置

### Decision
`services/tokenizer.py` → `utils/tokenizer.py`

### Rationale
- `tokenizer.py` 是纯文本处理工具（jieba 分词），不含任何业务/存储/网络依赖
- `utils/` 包已存在（含 `__init__.py` 和 `project.py`），位于 L0 层，所有层可安全 import
- 消费者：`storage/repository.py`（FTS 写入）和 `storage/query_builders.py`（FTS 查询）均可引用 `utils`

### Alternatives Considered
| Alternative | Rejected Because |
|---|---|
| `storage/tokenizer.py` | 限制了 services 层的访问权（若 search_pipeline 未来需要） |
| 新建 `text/` 顶层包 | 仅一个文件，过度设计 |

### Migration Impact
- `storage/repository.py`: 2 处 import 需更新（行 29, 352）
- `tests/test_tokenizer.py`: 1 处 import 需更新
- `harness_import_check.py`: `utils/` 已在 L0，无需更新 LAYER_MAP

---

## R3: Redis 缓存序列化方案

### Decision
在 `SearchPipeline` 存入 Redis 前，用 `dataclasses.asdict()` 将 `SearchPipelineResult` 转为 dict，取出时用 dict 直接返回（不重建 dataclass）。

### Rationale
- `SearchPipelineResult` 字段：`results: list[dict]`, `total_candidates: int`, `search_type: str`, `reranked: bool`, `cached: bool`, `duration_ms: int`, `stage_metrics: dict[str, int]`
- 全为基础类型，`dataclasses.asdict()` 递归转换后天然 JSON 友好
- Redis `get` 返回 dict，`SearchPipeline` 在消费时只读取字段，不依赖 dataclass 方法

### Alternatives Considered
| Alternative | Rejected Because |
|---|---|
| 自定义 `to_dict()` / `from_dict()` | `asdict` 已满足，额外代码无收益 |
| `pydantic.dataclasses.dataclass` | 影响 `SearchPipelineResult` 的全部消费者 |
| `pickle` | 跨版本不安全，调试不友好 |

---

## R4: `result_ids` 规范化

### Decision
统一为 `list[str]`（纯 UUID 字符串列表）。`json.dumps(result_ids)` 直接序列化入库。

### Rationale
- 当前 `server.py` 生成 `[{"": rid}]`（key 为空字符串的 dict），语义不明，下游解析困难
- Daemon `ResponseBuffer.add` 接收 `list[dict[str, Any]]`，但实际只需 UUID
- 简化为 `["uuid1", "uuid2"]` 最直观

### Breaking Change Assessment
- `daemon/faithfulness_batch.py` 在 `fetch_pending` 时 `json.loads(result_ids)` → 现在返回 `list[str]` 而非 `list[dict]`
- 需要检查下游是否依赖 dict 结构（预评估：不依赖，仅传递）

# Implementation Plan: Phase 1 — Security & Contract Fix

**Branch**: `012-phase1-security-contract-fix` | **Date**: 2026-05-13 | **Spec**: [spec.md](spec.md)
**Input**: Feature specification from `specs/012-phase1-security-contract-fix/spec.md`

## Summary

修复三位专家审查发现的 5 个 Critical 级问题：消除硬编码凭据、统一 `memory_submit_response` 数据通路（MCP→Services→Storage）、补齐 Remote 模式 HTTP 路由与 monkey-patch、修复 Redis 缓存序列化、迁移 `tokenizer.py` 消除 Storage→Services 分层倒依赖。

## Technical Context

**Language/Version**: Python 3.11+
**Primary Dependencies**: FastMCP, FastAPI, asyncpg, SQLAlchemy (async), httpx, Pydantic v2
**Storage**: PostgreSQL 15+ (pgvector), SQLite (daemon buffers), Redis (可选缓存后端)
**Testing**: pytest (async), `make verify` = lint + 全量测试, `make harness-check` = import 门禁
**Target Platform**: Linux/macOS server (MCP stdio + HTTP)
**Project Type**: MCP server + Web service + CLI + Daemon
**Performance Goals**: MCP 工具响应 <500ms p95；缓存命中 <50ms
**Constraints**: 严格分层 Server/Web→Services→Storage→Models；零硬编码凭据
**Scale/Scope**: 现有 ~150 个 Python 源文件, ~180 个测试文件

## Constitution Check

*GATE: Must pass before Phase 0 research. Re-check after Phase 1 design.*

| Gate | Status | Notes |
|------|--------|-------|
| I. 分层架构 — 禁止向上 import | **VIOLATION → FIX TARGET** | `storage/repository.py` → `services/tokenizer` (C1); `server.py` 直接 asyncpg (C2) |
| II. 异步优先 | ✅ PASS | 新增 `op_submit_response` 将使用 async `get_session` |
| III. 类型安全 | ✅ PASS | 新增路由/op 使用 Pydantic 模型 |
| IV. 项目隔离 | ✅ PASS | 不涉及查询过滤变更 |
| V. 双语 | ✅ PASS | 新增代码遵循中英注释 |
| VI. 可观测性 | ✅ PASS | 使用 logger 而非 print |
| Non-Negotiable #1 — 禁止硬编码密钥 | **VIOLATION → FIX TARGET** | `server.py` / `response_buffer.py` 含 `devpass` (C3) |
| Non-Negotiable #2 — 禁止向上 import | **VIOLATION → FIX TARGET** | 同 Gate I |
| Non-Negotiable #5 — 禁止提交破坏测试的代码 | ✅ PASS | `make verify` 作为验收门 |

**Gate Result**: 3 个 VIOLATION 正是本次修复目标。修复后全部 PASS。

## Project Structure

### Documentation (this feature)

```text
specs/012-phase1-security-contract-fix/
├── plan.md              # This file
├── spec.md              # Feature specification
├── research.md          # Phase 0: technical decisions
├── data-model.md        # Phase 1: entity/data changes
├── contracts/           # Phase 1: API contract changes
│   ├── mcp-compat-submit-response.md
│   └── op-submit-response.md
└── tasks.md             # Phase 2 output (by /speckit-tasks)
```

### Source Code (affected files)

```text
src/team_memory/
├── server.py                          # C2/C3: 重写 memory_submit_response
├── remote_client.py                   # C6: 补 op_submit_response + setup_remote_ops
├── utils/
│   └── tokenizer.py                   # C1: 从 services/ 迁入 (NEW)
├── services/
│   ├── memory_operations.py           # C2: 新增 op_submit_response
│   ├── cache.py                       # C5: 修复 Redis 序列化
│   └── tokenizer.py                   # C1: 删除 (迁移到 utils/)
├── storage/
│   └── repository.py                  # C1: 更新 import 路径
├── daemon/
│   └── response_buffer.py             # C3: 移除硬编码 DSN
├── web/routes/
│   └── mcp_compat.py                  # C6: 新增 submit-response 路由
tests/
├── test_server.py                     # 更新工具数/mock
├── test_tokenizer.py                  # 更新 import
├── test_response_buffer.py            # 更新 DSN 测试
└── test_cache_redis.py                # 新增 Redis 序列化测试
```

**Structure Decision**: 保持现有分层结构不变，仅做文件迁移（tokenizer）和新增函数/路由。

## Complexity Tracking

| Violation | Why Needed | Simpler Alternative Rejected Because |
|-----------|------------|-------------------------------------|
| 无新增违规 | N/A | N/A |

---

## Phase 0: Research

### R1: `op_submit_response` 的实现方案

**Decision**: 在 `memory_operations.py` 新增 `op_submit_response` 函数，使用 `get_session` + 原生 SQL（与 `daemon/response_buffer.py` 的 `ResponseBuffer.add` 逻辑对齐）。

**Rationale**:
- `response_buffer` 表未在 SQLAlchemy ORM `models.py` 中定义（仅有 raw DDL），走 `session.execute(text(...))` 是最低侵入路径
- Daemon 的 `ResponseBuffer` 使用 asyncpg 连接池是出于性能考虑（高频批量），MCP 侧频率低，`get_session` 够用
- 保持 `result_ids` 格式为 `list[str]`（纯 UUID 列表），修正当前 `[{"": rid}]` 的可疑格式

**Alternatives considered**:
- 复用 Daemon `ResponseBuffer` 实例 → 需要在 MCP 进程中初始化 asyncpg 池，与 bootstrap 的 SQLAlchemy session 不一致，增加复杂度
- 为 `response_buffer` 添加 ORM Model + Repository → 改动面大，且 daemon 侧不会改为 ORM

### R2: tokenizer 迁移目标

**Decision**: 迁移到 `utils/tokenizer.py`。

**Rationale**:
- `tokenizer` 是纯文本处理工具，不含任何业务/存储依赖
- `utils/` 已存在且在 L0 层（所有层均可安全 import）
- `storage/repository.py` 和 `services/search_pipeline.py`（若需要）都可引用

**Alternatives considered**:
- `storage/tokenizer.py` → 限制了 services 层的访问
- 新建 `text/` 包 → 过度设计，仅一个文件

### R3: Redis 序列化方案

**Decision**: 使用 `dataclasses.asdict()` 将 `SearchPipelineResult` 和 `SearchResultItem` 转为 dict 后 `json.dumps`，取出时用 `SearchPipelineResult(**data)` 重建。

**Rationale**:
- `SearchPipelineResult` 字段全为基础类型（list[dict], int, str, bool, dict）
- `dataclasses.asdict` 递归转换，天然 JSON 友好
- 无需引入 pydantic 或自定义 encoder
- `SearchResultItem.data` 已是 dict，无嵌套 dataclass 问题

**Alternatives considered**:
- 自定义 `to_dict()/from_dict()` → 额外代码，且 `asdict` 已满足
- 切换到 Pydantic dataclass → 影响面大
- pickle → 跨版本不安全

### R4: `result_ids` 规范化格式

**Decision**: 统一为 `list[str]`（纯 UUID 字符串列表）。

**Rationale**:
- 当前 `server.py` 中 `parsed_ids = [{"": rid}]` 生成 `{"": "uuid"}` 结构，语义不明
- Daemon `ResponseBuffer.add` 接收 `list[dict[str, Any]]`，但下游 `fetch_pending` 做 `json.loads` 后也是 list
- 统一为最简格式，`json.dumps(result_ids)` 直接序列化

**Alternatives considered**:
- 保持 `[{"id": rid}]` 结构 → 过度包装
- 使用 PostgreSQL array 类型 → 需要 schema 迁移

---

## Phase 1: Design & Contracts

### Data Model

> 详见 [data-model.md](data-model.md)

**无 schema 变更**。`response_buffer` 表结构保持不变。改变的是：
- 写入路径：从 `server.py` 直接 asyncpg → 通过 `memory_operations.op_submit_response` → `get_session` + SQL
- `result_ids` 字段的 JSON 值格式：从 `[{"": "uuid"}]` → `["uuid"]`

### Contracts

> 详见 [contracts/](contracts/)

**新增 HTTP 端点**:
- `POST /api/v1/mcp/submit-response` — 镜像 `memory_submit_response` MCP 工具

**新增内部 API**:
- `memory_operations.op_submit_response(user, *, query, response, result_ids, project)` — 统一编排函数

### Implementation Sequence

```
Task 1: 迁移 tokenizer.py (C1)               — 独立，无外部依赖
    └→ 更新 import：repository.py, test_tokenizer.py
    └→ 验证：make harness-check

Task 2: 移除硬编码凭据 (C3)                   — 独立
    └→ server.py: 删除 fallback DSN
    └→ response_buffer.py: 构造函数 dsn 必传
    └→ 验证：rg "devpass" src/ → 零命中

Task 3: 新增 op_submit_response (C2)          — 依赖 Task 2
    └→ memory_operations.py: 新增函数
    └→ server.py: 重写 memory_submit_response
    └→ 验证：make test

Task 4: 补齐 Remote 路由 (C6)                 — 依赖 Task 3
    └→ mcp_compat.py: 新增 POST /mcp/submit-response
    └→ remote_client.py: 补 op_submit_response + setup_remote_ops
    └→ 验证：Remote 模式测试

Task 5: 修复 Redis 缓存序列化 (C5)            — 独立
    └→ cache.py: SearchPipelineResult → asdict → json
    └→ search_pipeline.py: 存取适配
    └→ 新增测试

验收：make verify + make harness-check
```

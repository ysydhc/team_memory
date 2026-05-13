# Feature Specification: Phase 1 — Security & Contract Fix

**Feature Branch**: `012-phase1-security-contract-fix`
**Created**: 2026-05-13
**Status**: Draft
**Input**: 三位专家（架构师/后台开发/RAG工程师）并行审查发现的 Critical 级问题修复

## Background

基于对 TeamMemory 系统的全面架构审查，Phase 1 聚焦修复 6 个 Critical 级问题中的 5 项——涵盖安全隐患、分层违背、契约断裂和缓存失效。

### 问题清单

| # | 问题 | 源文件 | 严重度 |
|---|------|--------|--------|
| C1 | `repository.py` Storage 层 import `services.tokenizer`，分层倒置 | `storage/repository.py` | Critical |
| C2 | `memory_submit_response` MCP 入口层直接 asyncpg 写 DB | `server.py` | Critical |
| C3 | `server.py` / `response_buffer.py` 硬编码默认 DSN 含开发账号密码 | `server.py`, `daemon/response_buffer.py` | Critical |
| C5 | Redis 缓存 `json.dumps` 对 `SearchPipelineResult` dataclass 序列化失败 | `services/cache.py` | Critical |
| C6 | Remote 模式 `op_submit_response` HTTP 路由未注册 | `remote_client.py`, `web/routes/mcp_compat.py` | Critical |

## User Scenarios & Testing

### User Story 1 — 消除硬编码凭据 (Priority: P1)

运维/安全工程师需要确保源码中不含硬编码的数据库凭据，生产部署不会因默认密码暴露而产生安全风险。

**Why this priority**: 硬编码 `developer:devpass` 出现在 MCP 入口和 Daemon 中，属于安全红线

**Independent Test**: 在不设置任何环境变量的情况下启动 MCP server，应当报错而非回退到硬编码 DSN

**Acceptance Scenarios**:

1. **Given** 源码中 `server.py` 和 `daemon/response_buffer.py`, **When** 搜索 `devpass` 或 `developer:`, **Then** 零命中
2. **Given** 未设置 `TEAM_MEMORY_DB_URL` 环境变量, **When** `memory_submit_response` 被调用, **Then** 返回明确的配置缺失错误而非回退默认
3. **Given** 设置了正确的 `TEAM_MEMORY_DB_URL`, **When** `memory_submit_response` 被调用, **Then** 使用该 URL 正常连接

---

### User Story 2 — 统一 `memory_submit_response` 数据通路 (Priority: P1)

`memory_submit_response` 需走 Services/Storage 分层而非 MCP 层直连 asyncpg，与其余 8 个工具保持一致的分层架构。

**Why this priority**: 分层违背 + 每次新建连接无池化 + 与 Daemon ResponseBuffer 逻辑重复

**Independent Test**: `memory_submit_response` 工具调用成功写入 `response_buffer` 表，且通过 Services 层执行

**Acceptance Scenarios**:

1. **Given** MCP 本地模式, **When** 调用 `memory_submit_response(query, response)`, **Then** 通过 `memory_operations` 写入 `response_buffer` 表
2. **Given** MCP 远程模式, **When** 调用 `memory_submit_response(query, response)`, **Then** 通过 `_remote_client` 委派到远程 Web API
3. **Given** `server.py` 源码, **When** 检查 `memory_submit_response` 函数, **Then** 不包含 `asyncpg.connect` 直接调用
4. **Given** 数据库连接, **When** 高频调用 `memory_submit_response`, **Then** 使用连接池而非每次新建连接

---

### User Story 3 — 补齐 Remote 模式契约 (Priority: P1)

Remote MCP 部署下，`submit_response` 路由和 `setup_remote_ops` patch 需完整覆盖，使 faithfulness 评估在远程模式下可用。

**Why this priority**: 远程部署时 faithfulness 相关路径完全不可用，属于功能断裂

**Independent Test**: Remote 模式下调用 `memory_submit_response`，请求正确 POST 到 Web API 并返回 `buffered` 状态

**Acceptance Scenarios**:

1. **Given** `web/routes/mcp_compat.py`, **When** 检查路由列表, **Then** 包含 `POST /api/v1/mcp/submit-response`
2. **Given** Remote 模式, **When** 调用 `memory_submit_response`, **Then** 通过 HTTP POST 到 `/api/v1/mcp/submit-response`
3. **Given** `remote_client.py` 的 `setup_remote_ops`, **When** 初始化远程模式, **Then** `op_submit_response` 被正确 patch

---

### User Story 4 — 修复 Redis 缓存序列化 (Priority: P2)

配置 Redis 缓存后端时，`SearchPipelineResult` dataclass 需能正确序列化/反序列化，搜索结果缓存在 Redis 模式下正常工作。

**Why this priority**: 配置 Redis 后端时缓存形同虚设，但内存后端下可用，属于有条件触发

**Independent Test**: 配置 `backend=redis` 后执行搜索，第二次相同查询命中缓存返回正确结果

**Acceptance Scenarios**:

1. **Given** Redis 缓存后端, **When** `SearchPipelineResult` 存入缓存, **Then** `json.dumps` 不抛异常
2. **Given** Redis 缓存后端, **When** 从缓存取回结果, **Then** 反序列化后数据与原始结果一致
3. **Given** 内存缓存后端, **When** 存取 `SearchPipelineResult`, **Then** 行为不变（回归保护）

---

### User Story 5 — 迁移 `tokenizer.py` 消除分层倒依赖 (Priority: P2)

`storage/repository.py` 当前 import `services/tokenizer.py` 导致 Storage→Services 的依赖倒置，需将 tokenizer 迁移到 Storage 层或 utils 层可访问的位置。

**Why this priority**: 破坏核心分层契约，但运行时影响较小（只用于 FTS 分词）

**Independent Test**: `make harness-check` 的 import 检查通过，且 `repository.py` 不再 import `services.*`

**Acceptance Scenarios**:

1. **Given** `storage/repository.py` 源码, **When** 检查 import 语句, **Then** 不包含 `from team_memory.services` 的任何 import
2. **Given** FTS 写入路径, **When** 保存带中文内容的 experience, **Then** jieba 分词仍正常工作
3. **Given** `make harness-check`, **When** 执行 import 检查, **Then** 零违规（Storage→Services）
4. **Given** 搜索管线, **When** 对中文内容执行 FTS 搜索, **Then** 分词召回质量不变

---

### Edge Cases

- `memory_submit_response` 在数据库不可达时应返回结构化错误 JSON 而非 traceback
- Redis 连接断开时缓存操作应静默降级，不影响搜索主流程
- tokenizer 迁移后，原有 `from team_memory.services.tokenizer import tokenize` 的其他调用方是否需要更新
- Remote 模式下 Web API 返回非 200 时，MCP 工具应返回用户可读的错误信息

## Requirements

### Functional Requirements

- **FR-001**: `server.py` 和 `daemon/response_buffer.py` 中 MUST NOT 包含硬编码数据库 URL 或凭据
- **FR-002**: `memory_submit_response` MUST 通过 `memory_operations.op_submit_response` 委派执行，遵循 MCP → Services → Storage 分层
- **FR-003**: `memory_submit_response` MUST 支持 `_remote_client` 分支，与其余工具一致
- **FR-004**: `web/routes/mcp_compat.py` MUST 注册 `POST /api/v1/mcp/submit-response` 路由
- **FR-005**: `remote_client.py` 的 `setup_remote_ops` MUST 覆盖 `op_submit_response`
- **FR-006**: `RedisCacheBackend.put` MUST 能序列化 `SearchPipelineResult` dataclass
- **FR-007**: `RedisCacheBackend.get` MUST 能反序列化回等价的 `SearchPipelineResult`
- **FR-008**: `tokenizer.py` MUST 迁移到 `services/` 之外的位置（如 `utils/text.py` 或 `storage/tokenizer.py`）
- **FR-009**: 迁移后所有原有调用方 MUST 更新 import 路径
- **FR-010**: `memory_submit_response` MUST 使用连接池或 `get_session`，不得每次新建连接
- **FR-011**: DSN 解析 MUST 统一使用 Settings 层的 `_resolve_db_url` 或等价机制

### Key Entities

- **response_buffer**: 存储 agent 响应文本用于 faithfulness 评估的表；字段含 id, query, agent_response, result_ids, created_at
- **SearchPipelineResult**: 搜索管线输出的 dataclass，含 results list、metadata、scores 等
- **tokenizer**: jieba 中文分词工具，提供 `tokenize(text) -> str` 函数，被 FTS 写入和查询路径使用

## Success Criteria

### Measurable Outcomes

- **SC-001**: `rg "devpass|developer:" src/` 零命中（凭据清除）
- **SC-002**: `make verify`（lint + 全量测试）全绿
- **SC-003**: `make harness-check` 零 import 违规
- **SC-004**: `memory_submit_response` 在本地和远程模式下均可正常调用
- **SC-005**: Redis 缓存后端下搜索结果缓存存取正常（需测试覆盖）
- **SC-006**: 无新增裸 `asyncpg.connect` 调用出现在 `server.py` 中

## Assumptions

- 现有测试套件（`tests/`）覆盖了受影响模块的核心路径，可作为回归保护
- `response_buffer` 表的 schema 不需要变更，只是写入路径变化
- jieba 分词库的 import 开销可接受，放在 utils 层不影响启动时间
- Redis 后端当前非默认配置，修复后需新增测试覆盖但不影响现有用户
- `setup_remote_ops` 的 monkey-patch 模式暂时保留，架构改进留到 Phase 3

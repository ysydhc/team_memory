# Tasks: Phase 1 — Security & Contract Fix

**Input**: Design documents from `specs/012-phase1-security-contract-fix/`
**Prerequisites**: plan.md (required), spec.md (required), research.md, data-model.md, contracts/

**Organization**: Tasks are grouped by user story to enable independent implementation and testing of each story.

## Format: `[ID] [P?] [Story] Description`

- **[P]**: Can run in parallel (different files, no dependencies)
- **[Story]**: Which user story this task belongs to (e.g., US1, US2, US3)
- Include exact file paths in descriptions

---

## Phase 1: Setup

**Purpose**: No project initialization needed — this is an existing codebase. Verify baseline.

- [ ] T001 Run `make verify` to confirm baseline tests pass before modifications
- [ ] T002 Run `make harness-check` to record current import violations as baseline

**Checkpoint**: Baseline green (除已知 Storage→Services 违规)

---

## Phase 2: Foundational (Blocking Prerequisites)

**Purpose**: No shared foundational tasks needed. US1 and US5 are self-contained prerequisites that enable downstream stories.

**⚠️ NOTE**: US2 depends on US1 completion; US3 depends on US2 completion. US4 and US5 are fully independent.

**Checkpoint**: Foundation confirmed — user story implementation can begin

---

## Phase 3: User Story 5 — 迁移 tokenizer.py 消除分层倒依赖 (Priority: P2) 🎯 Independent

**Goal**: 将 `services/tokenizer.py` 迁移到 `utils/tokenizer.py`，消除 Storage→Services 的分层违规

**Independent Test**: `make harness-check` 零 import 违规；`repository.py` 不再 import `services.*`

### Implementation for User Story 5

- [ ] T003 [P] [US5] Move `src/team_memory/services/tokenizer.py` to `src/team_memory/utils/tokenizer.py` (preserve file content unchanged)
- [ ] T004 [US5] Update import in `src/team_memory/storage/repository.py`: change `from team_memory.services.tokenizer import tokenize` to `from team_memory.utils.tokenizer import tokenize` (all occurrences)
- [ ] T005 [P] [US5] Search `src/team_memory/storage/query_builders.py` for tokenizer imports and update if found
- [ ] T006 [US5] Search all files under `tests/` for `from team_memory.services.tokenizer` and update imports to `from team_memory.utils.tokenizer`
- [ ] T007 [US5] Search all files under `src/team_memory/services/` for `from team_memory.services.tokenizer` (self-references) and update to `from team_memory.utils.tokenizer`
- [ ] T008 [US5] Add re-export shim in `src/team_memory/services/tokenizer.py` (keep file with `from team_memory.utils.tokenizer import tokenize  # noqa: F401` and deprecation comment) for backward compatibility
- [ ] T009 [US5] Run `make harness-check` — verify zero Storage→Services violations
- [ ] T010 [US5] Run `make test` — verify all tokenizer-dependent tests pass

**Checkpoint**: `harness-check` 零 import 违规，FTS 分词功能不变

---

## Phase 4: User Story 1 — 消除硬编码凭据 (Priority: P1) 🎯 MVP

**Goal**: 源码中零硬编码数据库凭据；缺失环境变量时报明确错误而非回退默认

**Independent Test**: `rg "devpass|developer:" src/` 零命中

### Implementation for User Story 1

- [ ] T011 [P] [US1] Remove hardcoded DSN fallback in `src/team_memory/server.py` — in `memory_submit_response`, replace `os.environ.get("TEAM_MEMORY_DB_URL", "postgresql://developer:devpass@...")` with `os.environ["TEAM_MEMORY_DB_URL"]` (or use Settings), add proper error handling for missing env var
- [ ] T012 [P] [US1] Remove hardcoded DSN default in `src/team_memory/daemon/response_buffer.py` — change `dsn: str = "postgresql://developer:devpass@..."` to require explicit DSN parameter (no default value), update callers to pass DSN from Settings/env
- [ ] T013 [US1] Search entire `src/` for remaining occurrences of `devpass` or `developer:devpass` and remove any found
- [ ] T014 [US1] Update `tests/test_response_buffer.py` — ensure tests provide explicit DSN via fixture or mock, not relying on hardcoded default
- [ ] T015 [US1] Verify: run `rg "devpass" src/` and confirm zero matches
- [ ] T016 [US1] Run `make test` — verify all tests still pass with explicit DSN configuration

**Checkpoint**: 源码零凭据，测试全绿

---

## Phase 5: User Story 2 — 统一 memory_submit_response 数据通路 (Priority: P1)

**Goal**: `memory_submit_response` MCP 工具通过 `memory_operations.op_submit_response` 走 Services 层写入，不再直接 `asyncpg.connect`

**Independent Test**: 调用 `memory_submit_response` 成功写入 `response_buffer` 表，server.py 中无 `asyncpg.connect`

**Depends on**: US1 (hardcoded DSN already removed)

### Implementation for User Story 2

- [ ] T017 [US2] Add `import json` and `from sqlalchemy import text` to `src/team_memory/services/memory_operations.py` (if not already present)
- [ ] T018 [US2] Implement `async def op_submit_response(user, *, query, response, result_ids=None, project=None) -> dict` in `src/team_memory/services/memory_operations.py` — follow contract in `contracts/op-submit-response.md`: truncate query[:2000]/response[:10000], generate UUID, INSERT via `get_session` + `session.execute(text(...))`
- [ ] T019 [US2] Rewrite `memory_submit_response` function in `src/team_memory/server.py` — replace direct `asyncpg.connect` + raw SQL with delegation to `memory_operations.op_submit_response(user, query=query, response=response, result_ids=result_ids, project=project)`
- [ ] T020 [US2] Remove `import asyncpg` from `src/team_memory/server.py` if no other function uses it
- [ ] T021 [US2] Add `_remote_client` branch in `memory_submit_response` in `src/team_memory/server.py` — check `if _remote_client:` and delegate to `_remote_client.op_submit_response(...)`, matching the pattern of other MCP tools
- [ ] T022 [US2] Update `tests/test_server.py` — mock `memory_operations.op_submit_response` for `memory_submit_response` tool tests; verify it's called with correct parameters
- [ ] T023 [US2] Verify: `rg "asyncpg.connect" src/team_memory/server.py` returns zero matches
- [ ] T024 [US2] Run `make test` — verify all server tests pass

**Checkpoint**: server.py 零直连 DB，分层完整

---

## Phase 6: User Story 3 — 补齐 Remote 模式契约 (Priority: P1)

**Goal**: Remote MCP 部署下 `submit_response` 路由和 monkey-patch 完整，faithfulness 评估可用

**Independent Test**: Remote 模式下调用 `memory_submit_response`，请求 POST 到 Web API 并返回 `buffered`

**Depends on**: US2 (`op_submit_response` must exist in `memory_operations.py`)

### Implementation for User Story 3

- [ ] T025 [P] [US3] Add `McpSubmitResponseRequest` Pydantic model in `src/team_memory/web/routes/mcp_compat.py` — fields: `query: str`, `response: str`, `result_ids: list[str] | None = None`, `project: str | None = None`
- [ ] T026 [US3] Add `POST /mcp/submit-response` route in `src/team_memory/web/routes/mcp_compat.py` — follow contract in `contracts/mcp-compat-submit-response.md`, delegate to `memory_operations.op_submit_response`
- [ ] T027 [P] [US3] Add `op_submit_response` method to `RemoteMCPClient` class in `src/team_memory/remote_client.py` — `async def op_submit_response(self, user, **kwargs)` posting to `/api/v1/mcp/submit-response`
- [ ] T028 [US3] Add `op_submit_response` to `setup_remote_ops` function in `src/team_memory/remote_client.py` — add `memory_operations.op_submit_response = client.op_submit_response` alongside existing patches
- [ ] T029 [US3] Add test: verify `POST /api/v1/mcp/submit-response` returns 200 with `{"status": "buffered", "buffer_id": "..."}` in `tests/test_web.py` or `tests/test_mcp_callback.py`
- [ ] T030 [US3] Add test: verify `setup_remote_ops` patches `op_submit_response` onto `memory_operations` module in `tests/test_remote_client.py`
- [ ] T031 [US3] Run `make test` — verify all remote client and web tests pass

**Checkpoint**: Remote 模式 faithfulness 通路完整

---

## Phase 7: User Story 4 — 修复 Redis 缓存序列化 (Priority: P2) 🎯 Independent

**Goal**: `SearchPipelineResult` 在 Redis 缓存后端下正确序列化/反序列化

**Independent Test**: Redis 后端下存取 `SearchPipelineResult`，`json.dumps` 不报错，取回数据与原始一致

### Implementation for User Story 4

- [ ] T032 [P] [US4] In `src/team_memory/services/search_pipeline.py`, find where `SearchPipelineResult` is stored into cache — change `cache.put(key, result)` to `cache.put(key, asdict(result))` (add `from dataclasses import asdict`)
- [ ] T033 [US4] In `src/team_memory/services/search_pipeline.py`, find where cache result is retrieved — ensure the returned dict is converted back to `SearchPipelineResult` with `cached=True`, or document that downstream consumers accept both dict and dataclass
- [ ] T034 [US4] Add test in `tests/test_search_orchestrator.py` or new `tests/test_cache_serialization.py` — verify `SearchPipelineResult` round-trips through `RedisCacheBackend.put` / `get` without error
- [ ] T035 [P] [US4] Add test — verify `MemoryCacheBackend` behavior unchanged (regression protection) for `SearchPipelineResult`
- [ ] T036 [US4] Run `make test` — verify all cache and search tests pass

**Checkpoint**: Redis 缓存功能正常，内存缓存回归通过

---

## Phase 8: Polish & Cross-Cutting Concerns

**Purpose**: Final verification across all stories

- [ ] T037 Run `make lint` — zero errors
- [ ] T038 Run `make verify` — lint + all tests pass
- [ ] T039 Run `make harness-check` — zero import violations
- [ ] T040 Run `rg "devpass|developer:" src/` — zero hits (final credential check)
- [ ] T041 Run `rg "asyncpg.connect" src/team_memory/server.py` — zero hits (final layer check)
- [ ] T042 Verify `daemon/faithfulness_batch.py` handles `result_ids` as `list[str]` (check `fetch_pending` downstream usage)

---

## Dependencies & Execution Order

### Phase Dependencies

- **Setup (Phase 1)**: No dependencies — verify baseline
- **US5 (Phase 3)**: No dependencies — can start immediately, parallel with US1
- **US1 (Phase 4)**: No dependencies — can start immediately, parallel with US5
- **US2 (Phase 5)**: Depends on US1 completion (hardcoded DSN must be removed from server.py first)
- **US3 (Phase 6)**: Depends on US2 completion (`op_submit_response` must exist)
- **US4 (Phase 7)**: No dependencies — can start immediately, parallel with anything
- **Polish (Phase 8)**: Depends on all user stories being complete

### User Story Dependencies

```
US5 (tokenizer) ──────────────────────────────────┐
US1 (credentials) → US2 (data path) → US3 (remote)├→ Polish
US4 (Redis cache) ────────────────────────────────┘
```

- **US5**: Independent — can run in parallel with all other stories
- **US1**: Independent — enables US2
- **US2**: Depends on US1 — enables US3
- **US3**: Depends on US2 — end of chain
- **US4**: Independent — can run in parallel with all other stories

### Within Each User Story

- Implementation tasks in logical order (model → service → route)
- Verification at end of each story
- Story complete before moving to dependents

### Parallel Opportunities

- **Immediate parallel**: US5, US1, US4 can all start simultaneously
- **After US1 completes**: US2 starts
- **After US2 completes**: US3 starts
- T011 and T012 within US1 can run in parallel (different files)
- T025 and T027 within US3 can run in parallel (different files)
- T032 and T035 within US4 can run in parallel (different files)

---

## Parallel Example: Three Agents

```
Agent A: US5 (tokenizer migration)
  T003 → T004 → T005 → T006 → T007 → T008 → T009 → T010

Agent B: US1 → US2 → US3 (credential chain)
  T011,T012 (parallel) → T013 → T014 → T015 → T016
  → T017 → T018 → T019 → T020 → T021 → T022 → T023 → T024
  → T025,T027 (parallel) → T026 → T028 → T029 → T030 → T031

Agent C: US4 (Redis cache)
  T032 → T033 → T034,T035 (parallel) → T036
```

---

## Implementation Strategy

### MVP First (US1 Only)

1. Complete Phase 1: Verify baseline
2. Complete Phase 4: US1 — Remove hardcoded credentials
3. **STOP and VALIDATE**: `rg "devpass" src/` → zero hits, `make test` green
4. This alone resolves the #1 security issue

### Incremental Delivery

1. US1 (credentials) → Deploy/Validate — security hardened
2. US5 (tokenizer) → `make harness-check` clean — layering fixed
3. US2 (data path) → server.py clean architecture — layer compliance
4. US3 (remote route) → faithfulness in remote mode — feature complete
5. US4 (Redis cache) → cache correctness — reliability improved
6. Each story adds value without breaking previous stories

### Single Agent Sequential

1. T001–T002 (baseline)
2. T003–T010 (US5, independent)
3. T011–T016 (US1)
4. T017–T024 (US2)
5. T025–T031 (US3)
6. T032–T036 (US4)
7. T037–T042 (polish)

---

## Notes

- [P] tasks = different files, no dependencies
- [Story] label maps task to specific user story for traceability
- US5 placed before US1 in sequence because it's fully independent and quick
- No schema migrations needed — all changes are code-level
- Existing test suite provides regression protection
- `re-export shim` in T008 ensures backward compatibility for any external consumers of `services.tokenizer`

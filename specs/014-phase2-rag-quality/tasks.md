# Tasks: Phase 2 — RAG 质量提升

**Input**: Design documents from `specs/014-phase2-rag-quality/`
**Prerequisites**: plan.md (required), spec.md (required for user stories), research.md, data-model.md, contracts/

**Tests**: Tests are OPTIONAL - only include them if explicitly requested in the feature specification.

**Organization**: Tasks are grouped by user story to enable independent implementation and testing of each story.

## Format: `[ID] [P?] [Story] Description`

- **[P]**: Can run in parallel (different files, no dependencies)
- **[Story]**: Which user story this task belongs to (e.g., US1, US2, US3)
- Include exact file paths in descriptions

## Phase 1: Setup

**Purpose**: Project initialization and baseline verification

- [x] T001 Run `make verify` to confirm baseline tests pass before modifications
- [x] T002 Run `make harness-check` to record current import violations as baseline

---

## Phase 2: Foundational (Blocking Prerequisites)

**Purpose**: Core infrastructure that MUST be complete before ANY user story can be implemented
*(Note: Phase 2 stories in this spec are largely independent optimizations, so there are no blocking shared foundational tasks required.)*

**Checkpoint**: Foundation ready - user story implementation can now begin in parallel

---

## Phase 3: User Story 1 - 废弃并删除 `[mem:]` 标记 (Priority: P1) 🎯 MVP

**Goal**: 移除现有的 `[mem:]` 标记机制，清理关联代码。

**Independent Test**: Can be fully tested by submitting a query to the agent and verifying that the returned context/results no longer contain `[mem:xxxx]` tokens, and evaluation jobs complete successfully.

### Implementation for User Story 1

- [x] T003 [P] [US1] Remove marker injection logic (`inject_markers`, `check_was_used`) in `src/team_memory/services/analysis/evaluation.py`
- [x] T004 [P] [US1] Remove marker injection calls (`_maybe_inject_markers`) in `src/team_memory/services/search/search_orchestrator.py`
- [x] T005 [P] [US1] Remove `usage_hint` containing `[mem:xxxx]` from `src/team_memory/services/memory/memory_operations.py`
- [x] T006 [P] [US1] Remove marker string prefixes (`[mem:{eid}]`) in `src/team_memory/hooks/platforms/claude/session_start.py`
- [x] T007 [P] [US1] Remove marker string prefixes (`[mem:{rid}]`) in `src/team_memory/hooks/platforms/claude/user_prompt_submit.py`
- [x] T008 [US1] Update evaluation tests in `tests/test_evaluation.py` and `tests/test_evaluation_was_used.py` to remove `[mem:]` assertions
- [x] T009 [US1] Update hook and orchestrator tests (e.g., `tests/test_search_orchestrator.py`, `tests/test_hook_cursor_session_start.py`) to align with marker removal

**Checkpoint**: User Story 1 should be fully functional and testable independently

---

## Phase 4: User Story 2 - TS 查询转义保障安全 (Priority: P1)

**Goal**: 在全文本搜索构建查询条件时，转义/过滤 PostgreSQL `tsquery` 元字符。

**Independent Test**: Can be fully tested by searching for queries with special characters (e.g., "Error: 500 & timeout!") and verifying that the database executes the search successfully.

### Implementation for User Story 2

- [x] T010 [P] [US2] Implement regex stripping/escaping for `tsquery` meta-characters in `_build_hybrid_tsquery` within `src/team_memory/storage/query_builders.py`
- [x] T011 [US2] Add unit test cases for special characters in `tests/test_repository_unit.py` (or related query builder tests)

**Checkpoint**: User Story 2 works independently

---

## Phase 5: User Story 3 - Cross-Encoder 推理异步化 (Priority: P1)

**Goal**: 将同步执行的 Cross-Encoder 推理隔离到异步线程中，防止阻塞事件循环。

**Independent Test**: Can be fully tested by verifying that `asyncio.to_thread` is correctly utilized for the prediction call.

### Implementation for User Story 3

- [x] T012 [P] [US3] Wrap `self._model.predict(pairs)` with `asyncio.to_thread` in `src/team_memory/reranker/cross_encoder_provider.py`
- [x] T013 [US3] Update reranker unit tests (if mocks are used) to support the async execution path

**Checkpoint**: User Story 3 works independently

---

## Phase 6: User Story 4 - 优化自适应截断与重排顺序 (Priority: P2)

**Goal**: 将自适应截断逻辑移到 Cross-Encoder 重排之后，并扩大初筛传递给重排器的候选集。

**Independent Test**: Can be fully tested by verifying the pipeline execution logs and verifying the adaptive filter applies its threshold logic on final reranked scores.

### Implementation for User Story 4

- [x] T014 [US4] Increase intermediate candidate pool passed to the reranker in `src/team_memory/services/search/search_pipeline.py`
- [x] T015 [US4] Move adaptive filtering logic (`self._apply_adaptive_filter`) to run AFTER the reranking stage in `src/team_memory/services/search/search_pipeline.py`
- [x] T016 [US4] Update search pipeline tests in `tests/test_search_pipeline.py` to reflect new ordering and candidate pool sizes

**Checkpoint**: User Story 4 works independently

---

## Phase 7: User Story 5 - 落实 `grouped` 和 `top_k_children` 结果聚合 (Priority: P2)

**Goal**: 当 `grouped=True` 时，返回聚合后的结果结构，且每个分组最多保留 `top_k_children` 个匹配项。

**Independent Test**: Can be fully tested by calling the search API with `grouped=True` and `top_k_children=2`, verifying results are grouped by parent ID.

### Implementation for User Story 5

- [x] T017 [US5] Implement grouping logic (e.g., using `itertools.groupby` by parent identifier) and `top_k_children` slicing at the end of the pipeline in `src/team_memory/services/search/search_pipeline.py`
- [x] T018 [US5] Ensure the output array flattens back cleanly to satisfy the search API contract defined in `contracts/search-api-grouped.md`
- [x] T019 [US5] Add unit tests for `grouped=True` and `top_k_children` logic in `tests/test_search_pipeline.py`

**Checkpoint**: All user stories should now be independently functional

---

## Final Phase: Polish & Cross-Cutting Concerns

**Purpose**: Improvements and validations that affect multiple user stories

- [x] T020 [P] Run `make lint` to ensure no code style regressions
- [x] T021 [P] Run `make harness-check` to ensure no import/architectural layer violations
- [x] T022 Run `make verify` to ensure the entire test suite passes successfully

---

## Dependencies & Execution Order

### Phase Dependencies

- **Setup (Phase 1)**: No dependencies - can start immediately
- **Foundational (Phase 2)**: Depends on Setup completion - BLOCKS all user stories (N/A for this spec)
- **User Stories (Phase 3+)**: All depend on Foundational phase completion
  - User stories can then proceed in parallel (US1, US2, US3 are independent)
  - Or sequentially in priority order (P1 → P2)
- **Polish (Final Phase)**: Depends on all desired user stories being complete

### User Story Dependencies

- **User Story 1 (P1)**: Independent
- **User Story 2 (P1)**: Independent
- **User Story 3 (P1)**: Independent
- **User Story 4 (P2)**: Independent, but touches the same file (`search_pipeline.py`) as US5.
- **User Story 5 (P2)**: Independent, but touches the same file (`search_pipeline.py`) as US4.

### Parallel Opportunities

- T003 through T007 (US1 removal tasks) can all run in parallel as they touch different layers.
- US1, US2, and US3 can be executed entirely in parallel by different developers.
- US4 and US5 both modify `search_pipeline.py`, so they should ideally be done sequentially to avoid merge conflicts.

---

## Parallel Example: User Story 1

```bash
# Launch all marker removal modifications together:
Task: "Remove marker injection logic in src/team_memory/services/analysis/evaluation.py"
Task: "Remove usage_hint in src/team_memory/services/memory/memory_operations.py"
Task: "Remove marker string prefixes in src/team_memory/hooks/platforms/claude/session_start.py"
```

## Implementation Strategy

### Incremental Delivery

1. Complete Setup
2. Add User Story 1 (Remove `[mem:]`) → Verify tests pass
3. Add User Story 2 (TS escape) → Verify DB queries
4. Add User Story 3 (Async predict) → Verify throughput/performance
5. Add User Story 4 (Filter ordering) → Validate recall quality
6. Add User Story 5 (Grouped logic) → Validate API contract
7. Complete Polish phase (Lint & verify)
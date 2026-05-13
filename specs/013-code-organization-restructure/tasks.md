# Tasks: Phase 0 — 代码组织重组

**Input**: Design documents from `specs/013-code-organization-restructure/`
**Prerequisites**: plan.md (required), spec.md (required), research.md, data-model.md

**Organization**: Tasks are grouped by user story to enable independent implementation and testing of each story.

## Format: `[ID] [P?] [Story] Description`

- **[P]**: Can run in parallel (different files, no dependencies)
- **[Story]**: Which user story this task belongs to (e.g., US1, US2, US3, US4)
- Include exact file paths in descriptions

---

## Phase 1: Setup

**Purpose**: Verify baseline before any file moves.

- [ ] T001 Run `make verify` to confirm all tests pass before modifications
- [ ] T002 Run `make harness-check` to record current import violations as baseline
- [ ] T003 Run `rg "team_memory.hooks\." tests/` to catalog all test imports referencing hooks modules (save output for shim reference)
- [ ] T004 Run `rg "team_memory.daemon\." tests/` to catalog all test imports referencing daemon modules (save output for shim reference)

**Checkpoint**: Baseline green, import catalog captured

---

## Phase 2: Foundational (Blocking Prerequisites)

**Purpose**: No blocking shared infrastructure needed. Each US creates its own sub-package structure independently.

**⚠️ NOTE**: US1 (hooks), US2 (daemon), US3 (embedding) are all independent. US4 (services) should run last due to widest impact.

**Checkpoint**: Foundation confirmed — user story implementation can begin

---

## Phase 3: User Story 1 — hooks/ 适配器重组 (Priority: P1) 🎯 MVP

**Goal**: hooks/ 从 22 个平铺文件 → core/ + platforms/{claude,cursor,hermes} + integrations/，消除 12 处 DAEMON_URL 重复和 3×65 行 MCP callback 克隆

**Independent Test**: `make test` 全绿；`.cursor/hooks.json` 所有 Hook 可正常调用；`make harness-check` 通过

### Step 1: 创建 core/ 共享基础设施

- [ ] T005 [US1] Create directory `src/team_memory/hooks/core/` with `__init__.py`
- [ ] T006 [US1] Create `src/team_memory/hooks/core/daemon_http.py` — extract `DAEMON_URL` resolution (env `TM_DAEMON_URL` > default `http://127.0.0.1:3901`) and `post_hook(hook_name: str, payload: dict, timeout: float = 5.0) -> httpx.Response` sync helper; also `post_hook_fire_and_forget(hook_name, payload)` that swallows `ConnectError`
- [ ] T007 [P] [US1] Create `src/team_memory/hooks/core/stdin.py` — extract `parse_stdin_json() -> dict` from `common.py:parse_hook_input` pattern (try `json.load(sys.stdin)`, on failure return `{}`)
- [ ] T008 [P] [US1] Create `src/team_memory/hooks/core/mcp_callback.py` — extract shared recall detection logic: `RECALL_PATTERNS: list[str]`, `is_recall_tool(tool_name: str) -> bool`, `extract_callback_payload(hook_input: dict, platform: str) -> dict | None`, `send_mcp_callback(payload: dict) -> None` (uses `daemon_http.post_hook_fire_and_forget`)
- [ ] T009 [P] [US1] Create `src/team_memory/hooks/core/output.py` — platform-specific stdout helpers: `print_claude_context(text: str)`, `print_cursor_action(data: dict)`, `print_hermes_context(text: str)`, `print_empty()` — each writes JSON to stdout and flushes

### Step 2: 创建 platforms/ 子包并迁移 Hook 脚本

- [ ] T010 [US1] Create directories `src/team_memory/hooks/platforms/__init__.py`, `platforms/claude/__init__.py`, `platforms/cursor/__init__.py`, `platforms/hermes/__init__.py`
- [ ] T011 [US1] Move `src/team_memory/hooks/claude_session_start.py` → `src/team_memory/hooks/platforms/claude/session_start.py` and refactor to use `core.daemon_http.post_hook` + `core.stdin.parse_stdin_json` + `core.output.print_claude_context`
- [ ] T012 [P] [US1] Move `src/team_memory/hooks/claude_user_prompt_submit.py` → `src/team_memory/hooks/platforms/claude/user_prompt_submit.py` and refactor to use core modules
- [ ] T013 [P] [US1] Move `src/team_memory/hooks/claude_stop.py` → `src/team_memory/hooks/platforms/claude/stop.py` and refactor to use core modules
- [ ] T014 [P] [US1] Move `src/team_memory/hooks/claude_post_tool_use.py` → `src/team_memory/hooks/platforms/claude/post_tool_use.py` and refactor to use `core.mcp_callback`
- [ ] T015 [P] [US1] Move `src/team_memory/hooks/cursor_session_start.py` → `src/team_memory/hooks/platforms/cursor/session_start.py` and refactor to use core modules
- [ ] T016 [P] [US1] Move `src/team_memory/hooks/cursor_before_prompt.py` → `src/team_memory/hooks/platforms/cursor/before_prompt.py` and refactor to use core modules
- [ ] T017 [P] [US1] Move `src/team_memory/hooks/cursor_after_response.py` → `src/team_memory/hooks/platforms/cursor/after_response.py` and refactor to use core modules
- [ ] T018 [P] [US1] Move `src/team_memory/hooks/cursor_after_mcp.py` → `src/team_memory/hooks/platforms/cursor/after_mcp.py` and refactor to use `core.mcp_callback`
- [ ] T019 [P] [US1] Move `src/team_memory/hooks/hermes_pipeline.py` → `src/team_memory/hooks/platforms/hermes/pipeline.py` and refactor DAEMON_URL to use `core.daemon_http`
- [ ] T020 [P] [US1] Move `src/team_memory/hooks/hermes_stop.py` → `src/team_memory/hooks/platforms/hermes/stop.py` and refactor to use core modules
- [ ] T021 [P] [US1] Move `src/team_memory/hooks/hermes_user_prompt_submit.py` → `src/team_memory/hooks/platforms/hermes/user_prompt_submit.py` and refactor to use core modules
- [ ] T022 [P] [US1] Move `src/team_memory/hooks/hermes_post_tool_call.py` → `src/team_memory/hooks/platforms/hermes/post_tool_call.py` and refactor to use `core.mcp_callback`

### Step 3: 迁移 integrations/ 和清理

- [ ] T023 [US1] Create directory `src/team_memory/hooks/integrations/` with `__init__.py`
- [ ] T024 [P] [US1] Move `src/team_memory/hooks/weekly_report.py` → `src/team_memory/hooks/integrations/weekly_report.py`
- [ ] T025 [P] [US1] Move `src/team_memory/hooks/git_watcher.py` → `src/team_memory/hooks/integrations/git_watcher.py`
- [ ] T026 [P] [US1] Move `src/team_memory/hooks/initial_index.py` → `src/team_memory/hooks/integrations/initial_index.py`
- [ ] T027 [P] [US1] Move `src/team_memory/hooks/retrieval_trigger.py` → `src/team_memory/hooks/integrations/retrieval_trigger.py`
- [ ] T028 [US1] Move `src/team_memory/hooks/session_timeout.py` → `src/team_memory/daemon/integration/session_timeout.py` (belongs to daemon, not hooks; create shim at old path)

### Step 4: Re-export shims 和配置更新

- [ ] T029 [US1] Create re-export shims at ALL 12 old hook paths (e.g. `src/team_memory/hooks/claude_session_start.py` → `from team_memory.hooks.platforms.claude.session_start import *; from ... import main`) with `if __name__ == "__main__": main()` for CLI compatibility
- [ ] T030 [US1] Update `src/team_memory/hooks/__init__.py` — add re-exports for common names if any external code imports from `team_memory.hooks` directly
- [ ] T031 [US1] Update `.cursor/hooks.json` — change all 4 cursor hook module paths from `team_memory.hooks.cursor_*` to `team_memory.hooks.platforms.cursor.*`
- [ ] T032 [US1] Clean up `src/team_memory/hooks/common.py` — remove unused `call_mcp_tool` if no production code uses it; keep `parse_hook_input` as alias to `core.stdin.parse_stdin_json`
- [ ] T033 [US1] Clean up `src/team_memory/hooks/shared.py` — remove dead `HookInput` and `PipelineConfig` classes; keep `TMClient` (used by integrations/)

### Step 5: 更新测试

- [ ] T034 [US1] Update all test files under `tests/` that import `team_memory.hooks.claude_*` / `cursor_*` / `hermes_*` — change to new paths or verify shims work
- [ ] T035 [US1] Update `tests/test_hook_common.py` — adjust imports if `common.py` internals changed
- [ ] T036 [US1] Run `make verify` — all tests pass
- [ ] T037 [US1] Run `make harness-check` — zero new import violations

**Checkpoint**: hooks/ 重组完成，所有 Hook 可调用，测试全绿

---

## Phase 4: User Story 3 — embedding/ 工厂迁出 (Priority: P2) 🎯 Independent

**Goal**: embedding/ 拥有 `factory.py`，bootstrap.py 不再包含 `_create_embedding_provider`

**Independent Test**: `make test` 通过；`rg "_create_embedding_provider" src/team_memory/bootstrap.py` 零命中

### Implementation for User Story 3

- [ ] T038 [US3] Create `src/team_memory/embedding/__init__.py` if not exists — add re-exports: `EmbeddingProvider`, `ConcurrencyLimitedEmbedding`, `create_embedding_provider`
- [ ] T039 [US3] Create `src/team_memory/embedding/factory.py` — move `_create_embedding_provider` function body (~90 lines, L216-302) from `src/team_memory/bootstrap.py`, rename to `create_embedding_provider(settings: Settings) -> EmbeddingProvider`; keep all internal lazy imports (httpx, provider classes) and fallback logic intact
- [ ] T040 [US3] Update `src/team_memory/bootstrap.py` — replace `_create_embedding_provider(settings)` call with `from team_memory.embedding.factory import create_embedding_provider; create_embedding_provider(settings)`, remove the old function definition; keep `ConcurrencyLimitedEmbedding` wrapping
- [ ] T041 [P] [US3] Update `src/team_memory/daemon/embedding_backfill.py` — change `from team_memory.bootstrap import _create_embedding_provider` to `from team_memory.embedding.factory import create_embedding_provider`
- [ ] T042 [P] [US3] Search `src/team_memory/daemon/app.py` and `src/team_memory/daemon/pipeline.py` for `_create_embedding_provider` imports — update to `team_memory.embedding.factory.create_embedding_provider`
- [ ] T043 [US3] Update all test files that patch `team_memory.bootstrap._create_embedding_provider` — change mock target to `team_memory.embedding.factory.create_embedding_provider`
- [ ] T044 [US3] Run `make verify` — all tests pass
- [ ] T045 [US3] Verify: `rg "_create_embedding_provider" src/` returns zero matches (only tests if they kept old patch target as comment)

**Checkpoint**: embedding factory 自给自足，bootstrap 瘦身

---

## Phase 5: User Story 2 — daemon/ 子域分包 (Priority: P2) 🎯 Independent

**Goal**: daemon/ 从 36 个平铺文件 → 8 个功能域子包（runtime, buffers, refinement, backfill, quality, wiki, observe, integration）

**Independent Test**: `make test` 全绿；`python -m team_memory.daemon` 正常启动；`python -m team_memory.daemon.faithfulness_batch` CLI 入口有效

### Step 1: 创建子包目录

- [ ] T046 [US2] Create 8 sub-package directories under `src/team_memory/daemon/`: `runtime/`, `buffers/`, `refinement/`, `backfill/`, `quality/`, `wiki/`, `observe/`, `integration/` — each with `__init__.py`

### Step 2: 迁移 runtime/

- [ ] T047 [US2] Move `src/team_memory/daemon/app.py` → `src/team_memory/daemon/runtime/app.py` — update internal imports (config, other daemon modules now under subpackages)
- [ ] T048 [P] [US2] Move `src/team_memory/daemon/__main__.py` → `src/team_memory/daemon/runtime/__main__.py` — update `from .app import create_app` to `from team_memory.daemon.runtime.app import create_app`; keep a shim `daemon/__main__.py` that delegates to `runtime.__main__`
- [ ] T049 [P] [US2] Move `src/team_memory/daemon/config.py` → `src/team_memory/daemon/runtime/config.py`
- [ ] T050 [P] [US2] Move `src/team_memory/daemon/reloader.py` → `src/team_memory/daemon/runtime/reloader.py`
- [ ] T051 [P] [US2] Move `src/team_memory/daemon/health_check.py` → `src/team_memory/daemon/runtime/health_check.py`
- [ ] T052 [P] [US2] Move `src/team_memory/daemon/structured_logger.py` → `src/team_memory/daemon/runtime/structured_logger.py`

### Step 3: 迁移 buffers/

- [ ] T053 [P] [US2] Move `src/team_memory/daemon/conversation_buffer.py` → `src/team_memory/daemon/buffers/conversation_buffer.py`
- [ ] T054 [P] [US2] Move `src/team_memory/daemon/draft_buffer.py` → `src/team_memory/daemon/buffers/draft_buffer.py`
- [ ] T055 [P] [US2] Move `src/team_memory/daemon/response_buffer.py` → `src/team_memory/daemon/buffers/response_buffer.py`
- [ ] T056 [P] [US2] Move `src/team_memory/daemon/pending_queue.py` → `src/team_memory/daemon/buffers/pending_queue.py`
- [ ] T057 [P] [US2] Move `src/team_memory/daemon/payload_capture.py` → `src/team_memory/daemon/buffers/payload_capture.py`

### Step 4: 迁移 refinement/

- [ ] T058 [P] [US2] Move `src/team_memory/daemon/pipeline.py` → `src/team_memory/daemon/refinement/pipeline.py` — update imports from `tm_sink`, `config` to new subpackage paths
- [ ] T059 [P] [US2] Move `src/team_memory/daemon/draft_refiner.py` → `src/team_memory/daemon/refinement/draft_refiner.py`
- [ ] T060 [P] [US2] Move `src/team_memory/daemon/refinement_worker.py` → `src/team_memory/daemon/refinement/refinement_worker.py`

### Step 5: 迁移 backfill/, quality/, wiki/, observe/, integration/

- [ ] T061 [P] [US2] Move 4 backfill files: `embedding_backfill.py`, `entity_backfill.py`, `solution_backfill.py`, `entity_dedup.py` → `src/team_memory/daemon/backfill/`
- [ ] T062 [P] [US2] Move 4 quality files: `faithfulness_batch.py`, `faithfulness_judge.py`, `search_eval.py`, `convergence_detector.py` → `src/team_memory/daemon/quality/`
- [ ] T063 [P] [US2] Move 4 wiki files: `llm_wiki.py`, `wiki_cli.py`, `wiki_compiler.py`, `markdown_indexer.py` → `src/team_memory/daemon/wiki/`
- [ ] T064 [P] [US2] Move 4 observe files: `metrics_collector.py`, `search_log_writer.py`, `tm_stats.py`, `watcher.py` → `src/team_memory/daemon/observe/`
- [ ] T065 [P] [US2] Move 5 integration files: `tm_sink.py`, `auto_dream.py`, `personal_memory_extractor.py`, `intent_classifier.py`, `test_llm.py` → `src/team_memory/daemon/integration/`

### Step 6: 更新内部 imports

- [ ] T066 [US2] Update all cross-subpackage imports within `src/team_memory/daemon/` — e.g. `runtime/app.py` imports from `buffers/`, `refinement/`, `quality/`, etc. using absolute paths `team_memory.daemon.<subpkg>.<module>`
- [ ] T067 [US2] Update `runtime/app.py` lifespan — ensure all module references use new subpackage paths (this is the composition root, imports ~20 modules)

### Step 7: Re-export shims

- [ ] T068 [US2] Create re-export shims at ALL old flat paths (e.g. `src/team_memory/daemon/app.py` → `from team_memory.daemon.runtime.app import *`) — critical for `python -m team_memory.daemon.faithfulness_batch` and test imports
- [ ] T069 [US2] Keep `src/team_memory/daemon/__main__.py` as thin launcher delegating to `runtime.__main__` (ensures `python -m team_memory.daemon` still works)
- [ ] T070 [US2] Update `src/team_memory/daemon/__init__.py` — add re-exports for commonly imported names

### Step 8: 更新测试

- [ ] T071 [US2] Update all test files under `tests/` that import `team_memory.daemon.*` modules — adjust import paths or verify shims work
- [ ] T072 [US2] Run `make verify` — all tests pass
- [ ] T073 [US2] Verify `python -m team_memory.daemon` starts without error (ctrl+c after startup)
- [ ] T074 [US2] Verify `python -m team_memory.daemon.faithfulness_batch --help` works (CLI entry via shim)

**Checkpoint**: daemon/ 子域分包完成，所有入口点有效

---

## Phase 6: User Story 4 — services/ 领域子包 (Priority: P3)

**Goal**: services/ 从 23 个平铺文件 → 8 个领域子包（search, memory, llm, entity, archive, maintenance, analysis, infra）

**Independent Test**: `make verify` 全绿；`make harness-check` 通过；所有 MCP 工具正常调用

**Depends on**: US1/US2/US3 已稳定

### Step 1: 创建子包目录

- [ ] T075 [US4] Create 8 sub-package directories under `src/team_memory/services/`: `search/`, `memory/`, `llm/`, `entity/`, `archive/`, `maintenance/`, `analysis/`, `infra/` — each with `__init__.py`

### Step 2: 迁移文件到子包

- [ ] T076 [P] [US4] Move search domain: `search_pipeline.py`, `search_orchestrator.py`, `cache.py` → `src/team_memory/services/search/`
- [ ] T077 [P] [US4] Move memory domain: `memory_operations.py`, `experience.py`, `personal_memory.py` → `src/team_memory/services/memory/`
- [ ] T078 [P] [US4] Move llm domain: `llm_client.py`, `llm_parser.py`, `llm_provider.py` → `src/team_memory/services/llm/`
- [ ] T079 [P] [US4] Move entity domain: `entity_extractor.py`, `entity_search.py` → `src/team_memory/services/entity/`
- [ ] T080 [P] [US4] Move archive domain: `archive.py` → `src/team_memory/services/archive/`
- [ ] T081 [P] [US4] Move maintenance domain: `janitor.py`, `janitor_scheduler.py`, `promotion_compiler.py` → `src/team_memory/services/maintenance/`
- [ ] T082 [P] [US4] Move analysis domain: `evaluation.py`, `contradiction_detector.py`, `pattern_extractor.py`, `topic_discovery.py` → `src/team_memory/services/analysis/`
- [ ] T083 [P] [US4] Move infra domain: `event_bus.py`, `task_runner.py`, `tokenizer.py`, `intent_router.py` → `src/team_memory/services/infra/`

### Step 3: 更新 imports 和 re-export shims

- [ ] T084 [US4] Update all cross-subpackage imports within `src/team_memory/services/` — e.g. `search/search_pipeline.py` imports `cache` from `search/`, `llm_client` from `llm/`, etc.
- [ ] T085 [US4] Create re-export shims at ALL old flat paths (e.g. `src/team_memory/services/search_pipeline.py` → `from team_memory.services.search.search_pipeline import *`)
- [ ] T086 [US4] Update `src/team_memory/services/__init__.py` — re-export key names for backward compatibility
- [ ] T087 [US4] Update imports in `src/team_memory/server.py` — adjust `memory_operations` import path
- [ ] T088 [US4] Update imports in `src/team_memory/web/routes/` — adjust service import paths
- [ ] T089 [US4] Update imports in `src/team_memory/bootstrap.py` — adjust service import paths
- [ ] T090 [US4] Update imports in `src/team_memory/daemon/` — adjust service import paths (especially `memory_operations`, `llm_parser`, `personal_memory`)
- [ ] T091 [US4] Update `src/team_memory/storage/repository.py` — change `from team_memory.services.tokenizer` to `from team_memory.services.infra.tokenizer` (or use shim)

### Step 4: 更新测试

- [ ] T092 [US4] Update all test files under `tests/` that import `team_memory.services.*` modules — adjust import paths or verify shims work
- [ ] T093 [US4] Run `make lint` — zero errors
- [ ] T094 [US4] Run `make verify` — all tests pass
- [ ] T095 [US4] Run `make harness-check` — zero import violations

**Checkpoint**: services/ 领域子包完成，分层完整

---

## Phase 7: Polish & Cross-Cutting Concerns

**Purpose**: Final verification and documentation

- [ ] T096 Run `make lint` — zero errors across entire codebase
- [ ] T097 Run `make verify` — lint + all tests pass (final gate)
- [ ] T098 Run `make harness-check` — zero import violations (final gate)
- [ ] T099 Verify `.cursor/hooks.json` paths point to new locations and work
- [ ] T100 Verify `python -m team_memory.daemon` starts correctly
- [ ] T101 Verify `python -m team_memory.daemon.faithfulness_batch --help` works
- [ ] T102 Verify `rg "_create_embedding_provider" src/team_memory/bootstrap.py` returns zero matches
- [ ] T103 Update `scripts/harness_import_check.py` LAYER_MAP if any new top-level paths were introduced (verify: current subpackages should auto-match existing prefix rules)
- [ ] T104 Update `AGENTS.md` architecture section — reflect new sub-package structure

---

## Dependencies & Execution Order

### Phase Dependencies

- **Setup (Phase 1)**: No dependencies — verify baseline
- **US1 hooks/ (Phase 3)**: No dependencies — can start immediately
- **US3 embedding/ (Phase 4)**: No dependencies — can start immediately, parallel with US1
- **US2 daemon/ (Phase 5)**: No dependencies — can start immediately, parallel with US1/US3
- **US4 services/ (Phase 6)**: Depends on US1/US2/US3 being stable (widest impact)
- **Polish (Phase 7)**: Depends on all user stories complete

### User Story Dependencies

```
US1 (hooks, P1) ──────────────────────┐
US3 (embedding, P2) ──────────────────├→ US4 (services, P3) → Polish
US2 (daemon, P2) ─────────────────────┘
```

- **US1**: Independent, highest priority, largest deduplication gain
- **US3**: Independent, smallest scope (~90 lines move)
- **US2**: Independent, largest file count (36 files)
- **US4**: Should run after US1-3 stabilize (touches server.py, web/, daemon/ imports)

### Within Each User Story

- Create directories → Move files → Update imports → Re-export shims → Update tests → Verify

### Parallel Opportunities

- **3-way parallel**: US1, US2, US3 can all start simultaneously
- **Within US1**: T007/T008/T009 parallel (core modules); T011-T022 parallel (platform hooks)
- **Within US2**: T048-T052 parallel (runtime); T053-T057 parallel (buffers); T061-T065 parallel (leaf packages)
- **Within US4**: T076-T083 all parallel (8 domain moves touch different files)

---

## Parallel Example: Three Agents

```
Agent A: US1 — hooks/ restructure (Phase 3)
  T005-T009 (core/) → T010-T022 (platforms/) → T023-T028 (integrations/)
  → T029-T033 (shims/cleanup) → T034-T037 (tests/verify)

Agent B: US3 + US2 — embedding factory then daemon/ (Phase 4 → Phase 5)
  T038-T045 (embedding/) → T046 (dirs) → T047-T065 (moves)
  → T066-T070 (imports/shims) → T071-T074 (tests/verify)

Agent C: US4 — services/ (Phase 6, after Agents A+B done)
  T075 (dirs) → T076-T083 (moves) → T084-T091 (imports/shims)
  → T092-T095 (tests/verify)

All: T096-T104 (Polish)
```

---

## Implementation Strategy

### MVP First (US1 Only)

1. Complete Phase 1: Verify baseline
2. Complete Phase 3: US1 — hooks/ restructure
3. **STOP and VALIDATE**: `make verify` green, hooks callable
4. This alone delivers the highest-value restructure (most code deduplication)

### Incremental Delivery

1. US1 (hooks/) → 代码重复消除，按平台分目录 — **最高收益**
2. US3 (embedding/) → factory 迁出，bootstrap 瘦身 — **最小风险**
3. US2 (daemon/) → 36 文件分包，可发现性提升 — **中等规模**
4. US4 (services/) → 领域分层，God module 物理隔离 — **最大范围**
5. Each story adds organizational value without breaking previous stories

### Single Agent Sequential

1. T001–T004 (baseline)
2. T005–T037 (US1 hooks)
3. T038–T045 (US3 embedding)
4. T046–T074 (US2 daemon)
5. T075–T095 (US4 services)
6. T096–T104 (polish)

---

## Notes

- [P] tasks = different files, no dependencies on incomplete tasks
- [Story] label maps task to specific user story for traceability
- **Re-export shims are critical** — they guarantee all existing `python -m`, test imports, and external references continue working during transition
- Shims should include deprecation comments with planned removal version
- **Do NOT modify business logic** in any moved file — pure structural refactor
- Each step should be verified with `make test` before proceeding to the next step
- `git mv` preferred over manual copy+delete for history preservation

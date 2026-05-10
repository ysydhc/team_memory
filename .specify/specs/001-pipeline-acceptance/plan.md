# Implementation Plan: Pipeline Acceptance Test System

**Spec**: spec.md
**Status**: Ready for implementation
**Updated**: 2026-05-10 (incorporated 5-role review findings)

## Review Summary

5 roles reviewed: Backend Dev, QA, Architect, DevOps, End User.
4 CRITICAL bugs fixed in commit 565a5cc:
- C1: Convergence re-trigger prevention (session.status check)
- C3: count_messages() SQL COUNT instead of fetch-all
- C4: monitoring.yaml secrets moved to env vars

Remaining HIGH/MEDIUM items incorporated into test strategy below.

## Architecture

### Test Tiers (from Architect review)

```
Tier 1: Unit Tests (已有，通过)
  每个模块独立测试，mock 上下游
  运行: pytest -m "not integration"

Tier 2: Integration Tests (本次新增)
  真实 daemon + 真实 SQLite + 真实 PostgreSQL
  LLM 用 deterministic fixture mock
  运行: pytest -m integration

Tier 3: End-to-End Tests (本次新增)
  模拟真实 hook payload，验证完整数据流
  LLM 用 deterministic fixture mock
  运行: pytest -m e2e
```

### External Dependencies Strategy (from QA review)

| Dependency | Tier 2 | Tier 3 | Strategy |
|-----------|--------|--------|----------|
| SQLite | Real (in-memory) | Real (in-memory) | Fresh per test |
| PostgreSQL | Real (test DB) | Real (test DB) | TRUNCATE between tests |
| Ollama | Mock embedding | Mock embedding | conftest fixture |
| LiteLLM | Mock LLM response | Mock LLM response | conftest fixture |

### Modules Involved

| Module | File | Verification Point |
|--------|------|-------------------|
| HookPayload | daemon/app.py | Accept all platform formats (lenient) |
| pipeline.process_before_prompt | daemon/pipeline.py | User message storage + recall |
| pipeline.process_after_response | daemon/pipeline.py | Assistant message + convergence |
| ConversationBuffer | daemon/conversation_buffer.py | SQLite CRUD + count_messages |
| RefinementWorker | daemon/refinement_worker.py | LLM call + publish to PG |
| SearchLogWriter | daemon/search_log_writer.py | Log creation + was_used |
| EvaluationService | services/evaluation.py | Marker injection + detection |
| FaithfulnessJudge | daemon/faithfulness_judge.py | LLM scoring |
| ResponseBuffer | daemon/response_buffer.py | Batch threshold trigger |

## Implementation Steps

### Step 1: Test Infrastructure (conftest.py)
- File: tests/conftest.py (add fixtures)
- Fixtures needed:
  - `mock_embedding`: returns deterministic vectors
  - `mock_llm_response`: returns fixed refinement JSON
  - `fresh_conversation_buffer`: in-memory SQLite per test
  - `clean_postgres`: TRUNCATE between tests
  - `test_daemon_app`: FastAPI TestClient with mocked deps
- Verify: fixtures importable, tests discoverable

### Step 2: Hook Payload Acceptance
- File: tests/test_pipeline_acceptance.py
- Tests:
  - Cursor format (agent_response field) → accepted
  - Claude format (response_text field) → accepted
  - Hermes format (prompt field) → accepted
  - Unknown fields → accepted (lenient), logged
  - Missing all response fields → skipped (action=ok)
- Verify: all 5 tests pass

### Step 3: Message Storage Flow
- File: tests/test_pipeline_acceptance.py
- Tests:
  - before_prompt stores user message in ConversationBuffer
  - after_response stores assistant message in ConversationBuffer
  - Message count increments correctly
  - project=None uses "default" (H1 fix from review)
- Verify: messages visible in SQLite

### Step 4: Convergence Detection
- File: tests/test_pipeline_acceptance.py
- Tests:
  - 2 assistant messages → no convergence
  - 3 assistant messages → convergence triggers
  - Session status changes to needs_refinement
  - Re-trigger prevention: already needs_refinement → skip (C1 fix)
  - Per-session isolation: count is per session, not global
- Verify: session status in SQLite

### Step 5: RefinementWorker Processing
- File: tests/test_pipeline_acceptance.py
- Tests:
  - _tick() picks up needs_refinement sessions
  - _refine_one calls LLM (mocked) and saves to PG
  - draft_save + draft_publish succeed
  - Experience visible in PostgreSQL with correct fields
  - mark_published updates session status
- Risk: depends on LLM mock fixture quality
- Verify: experience in PostgreSQL

### Step 6: Search Retrieval
- File: tests/test_pipeline_acceptance.py
- Tests:
  - Save experience → search finds it
  - Search filtered by project
  - Results contain expected fields (title, description, score)
  - [mem:xxx] markers injected into results
- Verify: search returns saved experience

### Step 7: Evaluation Pipeline
- File: tests/test_pipeline_acceptance.py
- Tests:
  - search_log created on recall
  - was_used detection (marker in response)
  - fuzzy_match detection (keyword overlap)
  - ResponseBuffer batch threshold (5 → triggers judge)
  - FaithfulnessJudge returns score (mocked LLM)
- Verify: search_logs entries in PostgreSQL

## Addressing Review Findings

### From Backend Dev
- [H1] project=None → test that default project is used
- [C1] convergence re-trigger → test that already-processed sessions are skipped
- [C3] count_messages → test uses count_messages (not len(get_messages))

### From QA
- [P0] No integration test file → this plan creates it
- [P0] LLM mock strategy → conftest fixtures with deterministic responses
- [P0] PostgreSQL isolation → TRUNCATE fixture
- [P1] 30s tick → call _tick() directly in tests
- [P1] Multi-session isolation → test with 2 sessions same project
- [P2] mcp_callback → add to Step 2

### From Architect
- [MEDIUM] Monolithic lifespan → test uses FastAPI TestClient
- [MEDIUM] Dual config → test uses DaemonConfig directly
- TIERED approach: unit → integration → e2e

### From DevOps
- [HIGH-6] Worker liveness → test verifies /status includes worker state
- [HIGH-7] Retry counter loss → not addressed in this plan (future)
- [MEDIUM-9] Graceful shutdown → not addressed in this plan (future)

### From End User
- [HIGH-1] Silent failure → test verifies hook returns meaningful status
- [HIGH-2] User-critical paths → test verifies search returns relevant results
- [MEDIUM-4] Daemon dependency → test verifies daemon start/stop

## Risks

| Risk | Impact | Mitigation |
|------|--------|------------|
| LLM mock not realistic | RefinementWorker test misses bugs | Use actual LLM response JSON as fixture |
| PostgreSQL test pollution | Tests interfere | TRUNCATE fixture per test |
| Ollama unavailable | embedding tests skip | Mock embedding fixture |
| Test slowness | CI timeout | Tier 2+3 only run on demand (not every commit) |

## Output Files
- tests/test_pipeline_acceptance.py — main acceptance test file
- tests/conftest.py — shared fixtures (add to existing)
- Makefile — add `make test-acceptance` target

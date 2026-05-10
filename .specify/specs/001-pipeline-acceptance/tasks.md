# Tasks: Pipeline Acceptance Test System

**Plan**: plan.md
**Total tasks**: 12

## Task 1: Create shared test fixtures in conftest.py
- **File(s)**: tests/conftest.py
- **Action**: Add fixtures: mock_embedding, mock_llm_response, fresh_conversation_buffer, clean_postgres, test_daemon_app
- **Test**: pytest --collect-only tests/test_pipeline_acceptance.py discovers tests
- **Depends on**: none
- **Status**: [ ] pending

## Task 2: Test HookPayload accepts all platform formats
- **File(s)**: tests/test_pipeline_acceptance.py
- **Action**: Test Cursor (agent_response), Claude (response_text), Hermes (prompt), unknown fields, missing fields
- **Test**: All 5 payload format tests pass
- **Depends on**: Task 1
- **Status**: [ ] pending

## Task 3: Test before_prompt stores user message
- **File(s)**: tests/test_pipeline_acceptance.py
- **Action**: Call process_before_prompt with real ConversationBuffer, verify user message in SQLite
- **Test**: Message count = 1, role = "user", content matches
- **Depends on**: Task 1
- **Status**: [ ] pending

## Task 4: Test after_response stores assistant message
- **File(s)**: tests/test_pipeline_acceptance.py
- **Action**: Call process_after_response, verify assistant message in SQLite
- **Test**: Message count increments, role = "assistant"
- **Depends on**: Task 3
- **Status**: [ ] pending

## Task 5: Test convergence threshold (3 messages)
- **File(s)**: tests/test_pipeline_acceptance.py
- **Action**: Send 2 messages (no convergence) then 3rd (convergence triggers)
- **Test**: After 2: status=active. After 3: status=needs_refinement
- **Depends on**: Task 4
- **Status**: [ ] pending

## Task 6: Test convergence re-trigger prevention
- **File(s)**: tests/test_pipeline_acceptance.py
- **Action**: Send 4th message to already-converged session
- **Test**: No duplicate refinement, status unchanged
- **Depends on**: Task 5
- **Status**: [ ] pending

## Task 7: Test project=None defaults correctly
- **File(s)**: tests/test_pipeline_acceptance.py
- **Action**: Call process_after_response with no project
- **Test**: Message stored with project="default"
- **Depends on**: Task 4
- **Status**: [ ] pending

## Task 8: Test RefinementWorker processes needs_refinement sessions
- **File(s)**: tests/test_pipeline_acceptance.py
- **Action**: Mark session needs_refinement, call _tick(), verify experience in PostgreSQL
- **Test**: Experience exists in PG with correct title/project/content
- **Depends on**: Task 1, Task 5
- **Status**: [ ] pending

## Task 9: Test search retrieval finds saved experience
- **File(s)**: tests/test_pipeline_acceptance.py
- **Action**: Save experience via sink.draft_save + draft_publish, then recall
- **Test**: Search returns the experience, project filter works
- **Depends on**: Task 8
- **Status**: [ ] pending

## Task 10: Test search_log creation on recall
- **File(s)**: tests/test_pipeline_acceptance.py
- **Action**: Call recall via daemon endpoint, check search_logs table
- **Test**: search_log entry exists with correct query/project
- **Depends on**: Task 9
- **Status**: [ ] pending

## Task 11: Test evaluation marker injection and was_used
- **File(s)**: tests/test_pipeline_acceptance.py
- **Action**: Search with inject_markers, send response with [mem:xxx], check was_used
- **Test**: [mem:xxx] in results, was_used=True in search_logs
- **Depends on**: Task 10
- **Status**: [ ] pending

## Task 12: Add make test-acceptance target and run full suite
- **File(s)**: Makefile
- **Action**: Add `make test-acceptance` target that runs pytest -m e2e
- **Test**: `make test-acceptance` passes with 0 failures
- **Depends on**: Tasks 1-11
- **Status**: [ ] pending
